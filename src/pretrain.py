"""
VTX-300M pretraining script.

Trains the custom VTX-300M model (GQA, SwiGLU, RoPE, RMSNorm) on
pre-tokenized memory-mapped data using bf16 mixed precision +
torch.compile + AdamW. Pushes checkpoints to the configured Hub repo.

Optimized for single-GPU training on RTX 5090 Blackwell (32GB VRAM).

Usage:
    python pretrain.py --config pretrain_config.json
    python pretrain.py --steps 30000 --batch 8 --grad-accum 4 --block 2048 \
        --shards path/to/shard_0000.bin path/to/shard_0001.bin ...
    python pretrain.py --no-resume   # ignore checkpoints, start fresh
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


class Lion(torch.optim.Optimizer):
    """LION optimizer — only stores momentum (half the VRAM of AdamW).

    Paper: https://arxiv.org/abs/2302.06675
    Recommended: lr 3-10x smaller than AdamW, betas=(0.9, 0.99)
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # State: just momentum
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

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
    p.add_argument("--rope-theta", type=float, default=None, help="RoPE base frequency (default 1M, use 50M for 128k context)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--shards", nargs="+", default=None, help="Shard file paths (bin); default pulls from Hub")
    p.add_argument("--hub-repo", default=None)
    p.add_argument("--trackio-space", default=None)
    p.add_argument("--trackio-project", default=None)
    p.add_argument("--push-every", type=int, default=None)
    p.add_argument("--token", type=str, default=None, help="HuggingFace token for pushing checkpoints")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--compile", action="store_true", default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="Start fresh, ignore existing checkpoints")
    return p.parse_args()


# Defaults tuned for RTX 5090 Blackwell (32GB VRAM) — Lion optimizer
DEFAULTS = dict(
    steps=76000, warmup=1000, lr=3e-4, min_lr=3e-5, weight_decay=0.1,
    beta1=0.9, beta2=0.99, grad_clip=1.0, batch=4, grad_accum=8,
    block=2048, seed=42, shards=None, hub_repo="VTXAI/vtx-300m",
    trackio_space="VTXAI/vtx-300m-trackio", trackio_project="vtx-300m",
    push_every=3000, token=None, log_every=25, save_dir="/tmp/vtx_300m_ckpt", compile=True,
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


def find_latest_checkpoint(save_dir):
    """Find the latest checkpoint directory in save_dir.

    Looks for step_N directories and returns (path, step_number) of the
    highest-numbered one, or (None, 0) if nothing is found.
    """
    if not os.path.isdir(save_dir):
        return None, 0
    candidates = []
    for name in os.listdir(save_dir):
        if name.startswith("step_") and os.path.isdir(os.path.join(save_dir, name)):
            try:
                step_num = int(name.split("_", 1)[1])
                candidates.append((step_num, os.path.join(save_dir, name)))
            except ValueError:
                continue
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0])
    best_step, best_path = candidates[-1]
    # Verify it has both model weights and training state
    has_model = any(f.endswith(".safetensors") or f.endswith(".bin")
                    for f in os.listdir(best_path))
    has_state = os.path.exists(os.path.join(best_path, "training_state.pt"))
    if not has_model:
        print(f"[pretrain] WARN: checkpoint at {best_path} has no model weights, skipping", flush=True)
        return None, 0
    if not has_state:
        print(f"[pretrain] WARN: checkpoint at {best_path} has no training_state.pt, "
              f"will resume from model weights only (optimizer reset)", flush=True)
    return best_path, best_step


def save_checkpoint(model, optim, step, args, losses):
    """Save model + optimizer + training state to a checkpoint directory."""
    ckpt = os.path.join(args.save_dir, f"step_{step}")
    os.makedirs(ckpt, exist_ok=True)
    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_model.save_pretrained(ckpt)
    # Save optimizer + step + RNG states for faithful resume
    torch.save({
        "step": step,
        "optim": optim.state_dict(),
        "rng": {
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        "config": vars(args),
    }, os.path.join(ckpt, "training_state.pt"))
    print(f"[pretrain] saved checkpoint: {ckpt}", flush=True)
    return ckpt


def load_checkpoint(ckpt_path, model, optim, device):
    """Load optimizer state and step number from a checkpoint.

    Model weights are assumed already loaded by from_pretrained.
    Returns the step to resume from (next step after the saved one).
    """
    state_file = os.path.join(ckpt_path, "training_state.pt")
    if not os.path.exists(state_file):
        return 0
    state = torch.load(state_file, map_location=device, weights_only=False)
    optim.load_state_dict(state["optim"])
    rng = state.get("rng", {})
    if rng.get("torch") is not None:
        torch.random.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["cuda"])
    resumed_step = state["step"]
    print(f"[pretrain] loaded training state from step {resumed_step}", flush=True)
    return resumed_step + 1  # resume from next step


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

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[pretrain] VRAM: {vram_gb:.1f}GB", flush=True)

    # ── Checkpoint auto-discovery ──────────────────────────────────────
    ckpt_path, ckpt_step = find_latest_checkpoint(args.save_dir)
    resume = ckpt_path is not None and not getattr(args, "no_resume", False)
    if resume:
        print(f"[pretrain] found checkpoint: {ckpt_path} (step {ckpt_step})", flush=True)
    elif ckpt_path:
        print(f"[pretrain] --no-resume set, ignoring checkpoint at {ckpt_path}", flush=True)

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
        vocab_size = 151670   # safe default (Qwen3 max id+1)

    arch_kwargs = dict(max_position_embeddings=args.block, vocab_size=vocab_size)
    if getattr(args, "rope_theta", None) is not None:
        arch_kwargs["rope_theta"] = args.rope_theta
    arch = VortexArch(**arch_kwargs)
    cfg = VortexConfig(**{**vars(arch)})
    if getattr(args, "rope_theta", None) is not None:
        print(f"[pretrain] rope_theta={args.rope_theta:.0f} (context extension enabled)", flush=True)

    if resume:
        try:
            print(f"[pretrain] loading model from checkpoint: {ckpt_path}", flush=True)
            model = VortexForCausalLM.from_pretrained(ckpt_path).to(device)
        except Exception as e:
            print(f"[pretrain] WARN: failed to load checkpoint ({e}), training from scratch", flush=True)
            resume = False
            model = VortexForCausalLM(cfg).to(device)
    else:
        model = VortexForCausalLM(cfg).to(device)

    n = sum(p.numel() for p in model.parameters())
    n_no_embed = n - model.model.embed_tokens.weight.numel()
    print(f"[pretrain] model: {n/1e6:.1f}M params ({n_no_embed/1e6:.1f}M non-embed)", flush=True)
    print(f"[pretrain] architecture: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} kv_heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)} "
          f"intermediate={cfg.intermediate_size} context={cfg.max_position_embeddings}", flush=True)

    if args.compile:
        import shutil
        cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
        if cc:
            if not os.environ.get("CC"):
                os.environ["CC"] = cc
            model = torch.compile(model, mode="default")
            print(f"[pretrain] torch.compile enabled (mode=default, CC={cc})", flush=True)
        else:
            print(f"[pretrain] WARN: no C compiler found, disabling torch.compile", flush=True)

    # ── Optimizer ──────────────────────────────────────────────────────
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith("bias") or "norm" in name or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    optim = Lion(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(args.beta1, args.beta2),
    )
    print(f"[pretrain] optimizer: Lion (1 state, ~50% VRAM vs AdamW), decay={sum(p.numel() for p in decay)/1e6:.1f}M no_decay={sum(p.numel() for p in no_decay)/1e6:.1f}M", flush=True)

    # Load optimizer/RNG state if resuming
    start_step = 0
    if resume:
        start_step = load_checkpoint(ckpt_path, model, optim, device)
        print(f"[pretrain] resuming from step {start_step}", flush=True)

    # Enable gradient checkpointing to save VRAM (~40% less activation memory)
    model.gradient_checkpointing_enable()
    print(f"[pretrain] gradient checkpointing enabled", flush=True)

    # ── Train ──────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    accum = args.grad_accum
    eff_bs = args.batch * accum
    tok_per_step = eff_bs * args.block
    print(f"[pretrain] effective batch={eff_bs}, tokens/step={tok_per_step:,}", flush=True)
    print(f"[pretrain] total tokens: {args.steps * tok_per_step / 1e9:.2f}B", flush=True)
    if start_step > 0:
        print(f"[pretrain] remaining tokens: {(args.steps - start_step) * tok_per_step / 1e9:.2f}B", flush=True)

    model.train()
    losses = []
    t0 = time.time()
    steps_this_run = 0
    for step in range(start_step, args.steps):
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
        steps_this_run += 1
        if (step + 1) % args.log_every == 0 or step == 0:
            dt = time.time() - t0
            tok_s = steps_this_run * tok_per_step / dt
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
            ckpt = save_checkpoint(model, optim, step + 1, args, losses)
            if HAS_TRACKIO and os.environ.get("TRACKIO_SPACE_ID"):
                trackio.log({"checkpoint_step": step + 1})
            if args.hub_repo:
                from huggingface_hub import HfApi
                api = HfApi(token=args.token)
                api.upload_folder(
                    folder_path=ckpt, repo_id=args.hub_repo,
                    commit_message=f"step {step+1}", token=args.token,
                )
                print(f"[pretrain] pushed to {args.hub_repo}", flush=True)

    print(f"[pretrain] DONE: {args.steps} steps in {(time.time()-t0)/60:.1f} min", flush=True)
    if HAS_TRACKIO and os.environ.get("TRACKIO_SPACE_ID"):
        trackio.finish()


if __name__ == "__main__":
    main()
