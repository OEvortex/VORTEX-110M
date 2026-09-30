from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chat_template as CT
from pretrain import (build_optimizer, cosine_lr, fetch_hub_checkpoint,
                      load_checkpoint, save_checkpoint)


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────
def normalize_messages(raw):
    if isinstance(raw, dict) and "messages" in raw and raw["messages"]:
        msgs = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in raw["messages"]
            if isinstance(m, dict) and m.get("content")
        ]
        # Must have at least one user and one assistant turn to be trainable.
        roles = {m["role"] for m in msgs}
        if "user" in roles and "assistant" in roles:
            # Fold a leading system turn to the front if the dataset has one.
            return msgs
        return None

    # Dolly / Alpaca style: instruction / context / input -> response
    if isinstance(raw, dict) and raw.get("instruction"):
        instr = raw["instruction"]
        extra = raw.get("context") or raw.get("input")
        if extra:
            instr = f"{instr}\n\n{extra}"
        resp = raw.get("response") or raw.get("output") or ""
        if not resp:
            return None
        return [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": resp},
        ]

    return None


class SFTDataset(IterableDataset):

    def __init__(self, dataset, tokenizer, block_size=1024, seed=0,
                 max_tokens=None, epochs=1, val_frac=0.02):
        super().__init__()
        self.ds = dataset
        self.tok = tokenizer
        self.block_size = block_size
        self.seed = seed
        self.max_tokens = max_tokens
        self.epochs = epochs
        self.val_frac = val_frac
        self._epoch = 0
        self.stats = {"seen": 0, "kept": 0, "too_long": 0, "no_target": 0, "bad": 0}

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        emitted = 0
        for _ in range(self.epochs):
            idxs = list(range(len(self.ds)))
            rng.shuffle(idxs)  # per-epoch shuffle, deterministic given the seed
            for i in idxs:
                if self.max_tokens and emitted >= self.max_tokens:
                    return
                self.stats["seen"] += 1
                msgs = normalize_messages(self.ds[i])
                if not msgs:
                    self.stats["bad"] += 1
                    continue
                out = CT.encode_example(self.tok, msgs, self.block_size)
                if out is None:
                    self.stats["too_long"] += 1
                    continue
                enc, labels = out
                if not any(l != -100 for l in labels):
                    self.stats["no_target"] += 1
                    continue
                self.stats["kept"] += 1
                emitted += 1
                yield (torch.tensor(enc, dtype=torch.long),
                       torch.tensor(labels, dtype=torch.long))


def collate(batch):
    maxlen = max(len(ids) for ids, _ in batch)
    pad = 0
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        n = maxlen - len(ids)
        input_ids.append(torch.cat([ids, torch.full((n,), pad, dtype=torch.long)]))
        labels.append(torch.cat([lab, torch.full((n,), -100, dtype=torch.long)]))
        attn.append(torch.cat([torch.ones(len(ids), dtype=torch.long),
                               torch.zeros(n, dtype=torch.long)]))
    return (torch.stack(input_ids), torch.stack(labels), torch.stack(attn))


# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────
def build_tokenizer(tok_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    return tok


def add_chat_tokens(tok):
    new_ids = tok.add_special_tokens({"additional_special_tokens": CT.SPECIAL_TOKENS})
    im_end_id = tok.convert_tokens_to_ids(CT.IM_END)
    return new_ids, im_end_id


def load_base_model(base, device, tok, verbose=True):
    from model import VortexForCausalLM
    # Try the Hub id directly; if it is a local path or already downloaded,
    # from_pretrained handles it. The model config carries the real vocab.
    model = VortexForCausalLM.from_pretrained(base)
    model = model.to(device)

    # The embedding table must grow to fit the added chat tokens, and it is
    # TIED to lm_head, so both are resized together. New rows are init'd from
    # the pretraining distribution (see resize_token_embeddings).
    old_vocab = model.model.embed_tokens.weight.shape[0]
    target = len(tok)
    if target > old_vocab:
        model.model.resize_token_embeddings(target)
        model.lm_head.weight = model.model.embed_tokens.weight  # keep tied
        model.cfg.vocab_size = target
        model.config.vocab_size = target
        if verbose:
            print(f"[sft] embedding {old_vocab} -> {target} "
                  f"(+{target - old_vocab} chat tokens), re-tied", flush=True)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Base checkpoint: local dir or Hub id (e.g. VTXAI/vortex-50m-16k)")
    p.add_argument("--tokenizer", required=True, help="Tokenizer dir or Hub id")
    p.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--max-tokens", type=int, default=30_000_000,
                   help="Approx SFT tokens to train on (30M is the sweet spot)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--block", type=int, default=1024,
                   help="Max sequence length; longer conversations are skipped")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5, help="CRITICAL: keep ~1e-5..5e-5")
    p.add_argument("--min-lr", type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hub-repo", default=None)
    p.add_argument("--push-every", type=int, default=0, help="0 = only at the end")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--resume-from-hub", default=None,
                   help="Resume a prior SFT run from this Hub repo")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override the computed step count (for smoke tests)")
    args = p.parse_args()

    print(CT.describe(), flush=True)
    print(f"[sft] args: {vars(args)}\n", flush=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sft] device={device}", flush=True)

    # ── Tokenizer + chat tokens ──────────────────────────────────────
    tok = build_tokenizer(args.tokenizer)
    n_before = len(tok)
    new_ids, im_end_id = add_chat_tokens(tok)
    tok.chat_template = CT.build_chat_template()
    if args.tokenizer and not str(args.tokenizer).startswith((".", "/")):
        # Persist the chat-enabled tokenizer next to the SFT output so the
        # pushed model is loadable with the same control tokens.
        pass
    print(f"[sft] tokenizer {n_before} -> {len(tok)} tokens; "
          f"{CT.IM_END} = id {im_end_id}", flush=True)

    # ── Data ─────────────────────────────────────────────────────────
    from datasets import load_dataset
    print(f"[sft] loading {args.dataset} ...", flush=True)
    ds = load_dataset(args.dataset, args.dataset_config, split="train")
    print(f"[sft] {len(ds):,} raw examples", flush=True)

    train_ds = SFTDataset(ds, tok, block_size=args.block, seed=args.seed,
                          max_tokens=args.max_tokens, epochs=args.epochs)
    loader = DataLoader(train_ds, batch_size=args.batch,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=torch.cuda.is_available())
    it = iter(loader)

    eff = args.batch * args.grad_accum
    tok_per_step = eff * args.block  # upper bound (most seqs are shorter)
    steps = args.max_steps or max(1, (args.max_tokens * args.epochs) // tok_per_step)
    print(f"[sft] batch={args.batch} x accum={args.grad_accum} = {eff} seq/step",
          flush=True)
    print(f"[sft] training for {steps} steps (~{steps * tok_per_step / 1e6:.0f}M "
          f"token-slots, target {args.max_tokens / 1e6:.0f}M real)", flush=True)

    # ── Model ────────────────────────────────────────────────────────
    model = load_base_model(args.base, device, tok)
    model.gradient_checkpointing_enable()
    n = sum(p.numel() for p in model.parameters())
    print(f"[sft] model: {n / 1e6:.2f}M params", flush=True)

    optim = build_optimizer(model, argparse.Namespace(
        lr=args.lr, weight_decay=args.weight_decay, beta1=0.9, beta2=0.95,
        fused_adamw=True))

    start_step = 0
    if args.resume_from_hub:
        print(f"[sft] resuming SFT from {args.resume_from_hub}", flush=True)
        ck, step = fetch_hub_checkpoint(args.resume_from_hub, token=args.token,
                                        save_dir=args.out_dir)
        if ck:
            res = load_checkpoint(ck, model, optim, device)
            start_step = res["step"]

    # ── Train ────────────────────────────────────────────────────────
    model.train()
    losses = []
    val_history = []
    best_val = float("inf")
    t0 = time.time()
    log_every = 5

    for step in range(start_step, steps):
        lr = cosine_lr(step, args.warmup, steps, args.lr, args.min_lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.zero_grad()
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                ids, labels, _ = next(it)
            except StopIteration:
                it = iter(loader)
                ids, labels, _ = next(it)
            ids = ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, labels=labels, chunk_size=1024)
                loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        losses.append(accum_loss)

        if (step + 1) % log_every == 0 or step == 0:
            avg = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
            el = time.time() - t0
            print(f"[sft] step={step + 1}/{steps} loss={avg:.4f} lr={lr:.2e} "
                  f"({(step + 1 - start_step) / el:.2f} it/s)", flush=True)

        # Validation
        if args.val_every and (step + 1) % args.val_every == 0:
            model.eval()
            vloss, nb = 0.0, 0
            with torch.no_grad():
                for _ in range(args.val_batches):
                    try:
                        ids, labels, _ = next(it)
                    except StopIteration:
                        break
                    ids, labels = ids.to(device), labels.to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        vloss += model(input_ids=ids, labels=labels,
                                       chunk_size=1024).loss.item()
                    nb += 1
            model.train()
            if nb:
                vl = vloss / nb
                val_history.append((step + 1, avg, vl))
                flag = ""
                if vl < best_val:
                    best_val = vl
                    flag = "  *best"
                print(f"[sft] step={step + 1}  val={vl:.4f}{flag}", flush=True)

        if (step + 1) % args.save_every == 0 or (step + 1) == steps:
            ck = save_checkpoint(model, optim, step + 1,
                                 argparse.Namespace(save_dir=args.out_dir,
                                                     hub_repo=args.hub_repo),
                                 losses, val_history, best_val)
            # Save the chat-enabled tokenizer beside the weights.
            tok.save_pretrained(os.path.join(ck, "tokenizer"))
            if args.hub_repo and args.push_every and \
                    (step + 1) % args.push_every == 0:
                from huggingface_hub import HfApi
                HfApi(token=args.token).upload_folder(
                    folder_path=ck, repo_id=args.hub_repo,
                    commit_message=f"sft step {step + 1}", token=args.token)
                print(f"[sft] pushed -> {args.hub_repo}", flush=True)

    # ── Final: report + always push the last checkpoint ──────────────
    print(f"\n[sft] DONE {steps} steps in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    st = train_ds.stats
    print(f"[sft] data: {st['kept']:,} kept / {st['seen']:,} seen "
          f"({st['too_long']:,} too long, {st['bad']:,} malformed)", flush=True)
    if val_history:
        print(f"[sft] best val {min(r[2] for r in val_history):.4f}", flush=True)

    if args.hub_repo:
        ck = os.path.join(args.out_dir, f"step_{steps}")
        from huggingface_hub import HfApi
        api = HfApi(token=args.token)
        api.create_repo(args.hub_repo, token=args.token, exist_ok=True)
        api.upload_folder(folder_path=ck, repo_id=args.hub_repo,
                          commit_message="final SFT weights", token=args.token)
        print(f"[sft] pushed final -> {args.hub_repo}", flush=True)


if __name__ == "__main__":
    main()
