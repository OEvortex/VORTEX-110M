"""
Vortex-110M pretraining script.

Trains the custom Vortex-110M model on pre-tokenized memory-mapped data
using bf16 mixed precision + torch.compile + AdamW. Pushes checkpoints
to the configured Hub repo on a schedule.

Run on a single GPU (A100-80GB recommended for batch=8).

Usage:
    python pretrain.py --config pretrain_config.json
    python pretrain.py --steps 40000 --batch 8 --grad-accum 8 --block 2048 \
        --shards path/to/shard_0000.bin path/to/shard_0001.bin ...
"""
from __future__ import annotations
import os
import sys
import math
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Trackio for monitoring
try:
    import trackio
    HAS_TRACKIO = True
except Exception:
    HAS_TRACKIO = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="JSON config file")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--beta1", type=float, default=None)
    p.add_argument("--beta2", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--shards", nargs="+", default=None, help="Shard file paths (bin); default pulls from Hub")
    p.add_argument("--hub-repo", default=None)
    p.add_argument("--trackio-space", default=None)
    p.add_argument("--trackio-project", default=None)
    p.add_argument("--push-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--compile", action="store_true", default=None)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


DEFAULTS = dict(
    steps=40000, warmup=1000, lr=1.2e-3, min_lr=1.2e-4, weight_decay=0.1,
    beta1=0.9, beta2=0.95, grad_clip=1.0, batch=64, grad_accum=2,
    block=8192, seed=42, shards=None, hub_repo="VTXAI/vortex-110m",
    trackio_space="VTXAI/vortex-110m-trackio", trackio_project="vortex-110m",
    push_every=3000, log_every=20, save_dir="/tmp/vortex_ckpt", compile=True,
)


def resolve_args():
    raw = parse_args()
    cfg = dict(DEFAULTS)
    if raw.config:
        with open(raw.config) as f:
            cfg.update(json.load(f))
    for k in DEFAULTS:
        v = getattr(raw, k.replace("-", "_"), None)
        if v is not None:
            cfg[k] = v
    if raw.no_compile:
        cfg["compile"] = False
    if raw.compile:
        cfg["compile"] = True
    return argparse.Namespace(**cfg)


def cosine_lr(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_default_shards():
    """Pull shard paths from the configured data Hub dataset."""
    from huggingface_hub import snapshot_download
    print("[pretrain] downloading data shards from Hub...", flush=True)
    local_data = snapshot_download(
        repo_id="VTXAI/vortex-110m-data",
        repo_type="dataset",
        allow_patterns=["data/*.bin"],
    )
    shard_dir = f"{local_data}/data"
    shards = sorted([f"{shard_dir}/{f}" for f in os.listdir(shard_dir) if f.endswith(".bin")])
    return shards


def main():
    args = resolve_args()
    print(f"[pretrain] args: {vars(args)}", flush=True)

    if args.shards is None or args.shards == "AUTO" or (isinstance(args.shards, list) and args.shards == ["AUTO"]):
        args.shards = get_default_shards()

    # Trackio init (no-op if not configured)
    if HAS_TRACKIO and os.environ.get("TRACKIO_SPACE_ID"):
        trackio.init(
            project=os.environ.get("TRACKIO_PROJECT", args.trackio_project),
            space_id=os.environ.get("TRACKIO_SPACE_ID", args.trackio_space),
            config=vars(args),
        )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pretrain] device={device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})", flush=True)

    # ── Data ───────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import MMapDataset

    ds = MMapDataset(args.shards, block_size=args.block, seed=args.seed)
    print(f"[pretrain] dataset: {len(args.shards)} shards, {ds.total_tokens/1e6:.1f}M tokens", flush=True)

    def collate(batch):
        x = torch.stack(batch)
        return x[:, :-1], x[:, 1:]

    loader = DataLoader(
        ds, batch_size=args.batch, num_workers=8, pin_memory=True,
        collate_fn=collate, persistent_workers=True, prefetch_factor=4,
    )
    it = iter(loader)

    # ── Model ──────────────────────────────────────────────────────────
    from config import VortexArch, TokenizerProfile
    from model import VortexForCausalLM, VortexConfig

    # Profile tokenizer so vocab covers all special tokens (EOS, etc.)
    try:
        prof = TokenizerProfile.from_hub()
        vocab_size = prof.vocab_size
        print(f"[pretrain] tokenizer: {prof.tokenizer_id} vocab={vocab_size} eos={prof.eos_token_id}", flush=True)
    except Exception as e:
        print(f"[pretrain] WARN: tokenizer profile failed: {e}", flush=True)
        vocab_size = 151670   # safe default (Qwen3-4B max id+1)

    arch = VortexArch(max_position_embeddings=args.block, vocab_size=vocab_size)
    cfg = VortexConfig(**{**vars(arch)})
    model = VortexForCausalLM(cfg).to(device)
    n = sum(p.numel() for p in model.parameters())
    n_no_embed = n - model.model.embed_tokens.weight.numel()
    print(f"[pretrain] model: {n/1e6:.1f}M params ({n_no_embed/1e6:.1f}M non-embed)", flush=True)

    if args.compile:
        model = torch.compile(model, mode="default")
        print(f"[pretrain] torch.compile enabled (mode=default)", flush=True)

    # ── Optimizer ──────────────────────────────────────────────────────
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith("bias") or "norm" in name or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8,
    )
    print(f"[pretrain] optimizer: AdamW, decay={sum(p.numel() for p in decay)/1e6:.1f}M no_decay={sum(p.numel() for p in no_decay)/1e6:.1f}M", flush=True)

    # ── Train ──────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    accum = args.grad_accum
    eff_bs = args.batch * accum
    tok_per_step = eff_bs * args.block
    print(f"[pretrain] effective batch={eff_bs}, tokens/step={tok_per_step:,}", flush=True)
    print(f"[pretrain] total tokens: {args.steps * tok_per_step / 1e9:.2f}B", flush=True)

    model.train()
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.zero_grad()
        accum_loss = 0.0
        for _ in range(accum):
            try:
                inp, tgt = next(it)
            except StopIteration:
                it = iter(loader)
                inp, tgt = next(it)
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=inp, labels=tgt, chunk_size=1024)
                loss = out.loss / accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        losses.append(accum_loss)
        if (step + 1) % args.log_every == 0 or step == 0:
            dt = time.time() - t0
            tok_s = (step + 1) * tok_per_step / dt
            avg = sum(losses[-args.log_every:]) / max(1, len(losses[-args.log_every:]))
            log_line = (
                f"step={step+1}/{args.steps} loss={avg:.4f} lr={lr:.2e} "
                f"tok/s={tok_s/1e3:.1f}k  dt={dt:.1f}s"
            )
            print(log_line, flush=True)
            if HAS_TRACKIO and os.environ.get("TRACKIO_SPACE_ID"):
                trackio.log({
                    "step": step + 1,
                    "loss": avg,
                    "lr": lr,
                    "tok_per_sec": tok_s,
                    "tokens": (step + 1) * tok_per_step,
                })

        if (step + 1) % args.push_every == 0 or (step + 1) == args.steps:
            ckpt = os.path.join(args.save_dir, f"step_{step+1}")
            os.makedirs(ckpt, exist_ok=True)
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_model.save_pretrained(ckpt)
            if HAS_TRACKIO and os.environ.get("TRACKIO_SPACE_ID"):
                trackio.log({"checkpoint_step": step + 1})
            if args.hub_repo:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=ckpt, repo_id=args.hub_repo,
                    commit_message=f"step {step+1}",
                )
                print(f"[pretrain] pushed to {args.hub_repo}", flush=True)

    print(f"[pretrain] DONE: {args.steps} steps in {(time.time()-t0)/60:.1f} min", flush=True)
    if HAS_TRACKIO and os.environ.get("TRACKIO_SPACE_ID"):
        trackio.finish()


if __name__ == "__main__":
    main()
