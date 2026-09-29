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
import random
import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Matches a checkpoint directory name exactly: "step_3000" -> 3000.
# Anchored so a stray "step_3000_old" or "step_abc" is ignored rather than
# crashing the scan.
_STEP_DIR_RE = re.compile(r"step_(\d+)")


def build_optimizer(model, args):
    """AdamW with decoupled weight decay, param groups, and no WD on 1-D params.

    Why AdamW and not Lion: Lion halves optimizer state but its update is
    `sign(momentum)`, which throws away gradient magnitude. For a 2B-token run
    on a 50M model AdamW's per-parameter scaling is the better-behaved default,
    and the 2x state cost is irrelevant here -- fp32 moments for 49.5M params
    is ~400MB against 24GB of VRAM.

    Three settings that matter, and are easy to get wrong:

    1. `decoupled_weight_decay=True`. Folding L2 into the gradient biases the
       effective LR per parameter; the decoupled form is what weight_decay=0.1
       actually means and it behaves correctly alongside the no-decay group.

    2. No weight decay on 1-D params (norm gains). Decaying a gain vector
       pulls it toward zero, fighting the RMSNorm rescale. Standard practice
       (GPT-2 / LLaMA) and very easy to miss.

    3. `fused=True`. The fused CUDA kernel runs the whole update in one pass
       with no per-parameter Python loop -- a real win across ~130 tensors.
       Falls back automatically on CPU.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith("bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    kwargs = dict(lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8)
    fused = bool(getattr(args, "fused_adamw", True)) and torch.cuda.is_available()
    if fused:
        kwargs["fused"] = True

    optim = torch.optim.AdamW(groups, **kwargs)

    n_dec = sum(p.numel() for p in decay)
    n_nod = sum(p.numel() for p in no_decay)
    print(f"[pretrain] optimizer: {'AdamW (fused)' if fused else 'AdamW'} "
          f"betas=({args.beta1},{args.beta2}) eps=1e-8", flush=True)
    print(f"[pretrain]   decay    {n_dec/1e6:7.1f}M  wd={args.weight_decay}", flush=True)
    print(f"[pretrain]   no_decay {n_nod/1e6:7.1f}M  wd=0.0 (norm gains)", flush=True)
    print(f"[pretrain]   state ~{(n_dec+n_nod)*8/1e9:.2f} GB fp32 moments", flush=True)
    return optim



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
    p.add_argument("--arch", type=str, default=None,
                   help="architecture preset, e.g. vortex-50m (see config.PRESETS)")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="tokenizer dir or Hub id matching the shards on disk")
    p.add_argument("--rope-theta", type=float, default=None,
                   help="RoPE base (default 10k; use 1M when extending context)")
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
    p.add_argument("--no-fused-adamw", dest="fused_adamw", action="store_false", default=None,
                   help="Disable the fused AdamW CUDA kernel")
    p.add_argument("--auto-batch", action="store_true", default=None,
                   help="Probe the GPU for the largest batch that fits, then set "
                        "grad_accum to hit --target-batch tokens")
    p.add_argument("--no-auto-batch", dest="auto_batch", action="store_false", default=None,
                   help="Disable GPU probing; use --batch / --grad-accum verbatim")
    p.add_argument("--target-batch", type=int, default=None,
                   help="effective batch (in sequences) to hold constant when "
                        "--auto-batch picks the per-device batch")
    p.add_argument("--val-every", type=int, default=None,
                   help="Run validation every N steps (0 disables)")
    p.add_argument("--val-tokens", type=int, default=None,
                   help="Tokens to use for each validation pass")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader worker processes (0 = load in the main process)")
    p.add_argument("--resume-from-hub", type=str, default=None, metavar="REPO",
                   help="Hub model repo to pull a resumable checkpoint from when "
                        "save_dir has none (e.g. VTXAI/vortex-50m-16k). "
                        "Defaults to --hub-repo when that is set.")
    p.add_argument("--hub-only", action="store_true",
                   help="Ignore local checkpoints entirely and resume from the Hub")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Auto batch sizing
# ──────────────────────────────────────────────────────────────────────
def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def find_max_batch(build_step, params, block: int, cap: int = 256) -> tuple:
    """Binary-search the largest per-device batch that completes a fwd+bwd step.

    `build_step(bs)` must return a scalar loss for a batch of `bs` sequences.
    An OOM is caught and treated as data rather than a crash, and
    `empty_cache()` runs between probes -- a fragmented allocation would
    otherwise make a size that actually fits look like a failure.

    Returns (best_batch, log_lines). Returns 0 if even batch=1 OOMs, which
    means --block is too long for this card.
    """
    log = [f"probing batch 1..{cap} at block={block} on "
           f"{torch.cuda.get_device_name(0)} ({_vram_gb():.1f} GB)"]

    def fits(bs: int) -> bool:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            loss = build_step(bs)
            loss.backward()
            ok = bool(torch.isfinite(loss).item())
            del loss
            return ok
        except torch.cuda.OutOfMemoryError:
            return False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            raise
        finally:
            for p in params:
                p.grad = None
            torch.cuda.empty_cache()

    if not fits(1):
        log.append("  batch=1 FAILED -- lower --block or free VRAM")
        return 0, log

    lo, hi, best = 2, cap, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    log.append(f"  max batch that fits: {best}  (peak {peak:.1f} GB of "
               f"{_vram_gb():.1f} GB)")
    return best, log


# Defaults tuned for a 24GB card (RTX 4090) with fused AdamW.
# 2B tokens at block=2048, effective batch 32 sequences => 65,536 tok/step
# => 30,517 steps. batch/grad_accum are placeholders; --auto-batch overrides.
DEFAULTS = dict(
    steps=30517, warmup=305, lr=6e-4, min_lr=6e-5, weight_decay=0.1,
    beta1=0.9, beta2=0.95, grad_clip=1.0, batch=4, grad_accum=8,
    block=2048, seed=42, shards=None, hub_repo="VTXAI/vortex-50m",
    trackio_space="VTXAI/vortex-50m-trackio", trackio_project="vortex-50m",
    push_every=3000, token=None, log_every=25, save_dir="/tmp/vortex_50m_ckpt",
    compile=True, arch="vortex-50m-16k", tokenizer=None, rope_theta=None,
    fused_adamw=True, auto_batch=True, target_batch=32,
    val_every=500, val_tokens=2_000_000, num_workers=8,
    resume_from_hub=None, hub_only=False,
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
        repo_id=os.environ.get("VORTEX_DATA_REPO", "VTXAI/vortex-50m-data"),
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
        m = _STEP_DIR_RE.fullmatch(name)
        if m and os.path.isdir(os.path.join(save_dir, name)):
            candidates.append((int(m.group(1)), os.path.join(save_dir, name)))
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


def read_state_step(path):
    """Read the `step` field out of a training_state.pt.

    A 50M-param AdamW checkpoint carries ~400MB of fp32 moments and
    `torch.load` has no lazy-field mode, so the payload is materialized
    whether or not the moments are wanted. All this saves is the optimizer
    reconstruction -- it keeps the call site honest about needing one integer.
    Returns None if the file is unreadable.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        return int(state["step"])
    except Exception as e:
        print(f"[pretrain] WARN: could not read step from {path} ({e})", flush=True)
        return None


def _weights_present(d):
    return any(f.endswith(".safetensors") or f.endswith(".bin")
               for f in os.listdir(d))


def fetch_hub_checkpoint(repo_id, token=None, save_dir=None):
    """Download a resumable checkpoint from a Hub model repo.

    Checkpoints are pushed to the Hub as a FLAT snapshot -- `model.safetensors`,
    `config.json`, `training_state.pt` and the tokenizer files all sit at the
    repo root, because `api.upload_folder(folder_path=ckpt, ...)` preserves no
    directory structure. There is therefore no `step_3000/` prefix to read a
    step number from; the step lives INSIDE `training_state.pt`, so it is
    fetched first and used to name the destination directory.

    Returns (path, step) for a checkpoint safe to resume from, or (None, 0)
    if the repo is unreachable or unusable as a resume point.
    """
    import shutil

    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        files = list_repo_files(repo_id, repo_type="model", token=token)
    except Exception as e:
        print(f"[pretrain] WARN: cannot reach Hub repo {repo_id} ({e})", flush=True)
        return None, 0

    if "training_state.pt" not in files:
        print(f"[pretrain] WARN: {repo_id} has no training_state.pt, so it "
              f"cannot be used to RESUME -- it only holds weights. Pass "
              f"--no-resume to fine-tune from its weights instead.", flush=True)
        return None, 0
    if not any(f.endswith(".safetensors") for f in files):
        print(f"[pretrain] WARN: {repo_id} has no model.safetensors", flush=True)
        return None, 0

    # 1. training_state.pt first: its step number decides where everything
    #    lands. Downloaded to a scratch dir, then moved into place once the
    #    step is known.
    if not save_dir:
        return None, 0
    os.makedirs(save_dir, exist_ok=True)

    # 1a. Never re-download what is already staged. A resumed run re-enters
    #     here on every restart, and the step number needed to identify the
    #     staged dir lives INSIDE the 400MB state file -- so without this
    #     check each restart pays a full re-download to learn it already has
    #     the answer. The step dir name is the only place the step is recorded.
    staged_path, staged_step = find_latest_checkpoint(save_dir)
    if staged_path is not None:
        print(f"[pretrain] Hub checkpoint step {staged_step} already staged at "
              f"{staged_path}; not re-downloading", flush=True)
        return staged_path, staged_step

    staging = os.path.join(save_dir, ".hub_staging")
    os.makedirs(staging, exist_ok=True)

    print(f"[pretrain] fetching training_state.pt from {repo_id} "
          f"(~400MB)...", flush=True)
    try:
        state_path = hf_hub_download(repo_id, "training_state.pt",
                                     repo_type="model", token=token,
                                     local_dir=staging)
    except Exception as e:
        print(f"[pretrain] WARN: download of training_state.pt failed ({e})", flush=True)
        return None, 0

    step = read_state_step(state_path)
    if step is None:
        print("[pretrain] WARN: training_state.pt is unreadable; cannot "
              "determine the resume step", flush=True)
        return None, 0

    # 2. Land it as save_dir/step_N/, exactly the layout save_checkpoint()
    #    writes, so every later local resume finds it with no special-casing.
    dest = os.path.join(save_dir, f"step_{step}")
    os.makedirs(dest, exist_ok=True)
    print(f"[pretrain] resuming from Hub {repo_id} at step {step} -> {dest}",
          flush=True)

    # 3. Weights + config + tokenizer files alongside it. `local_dir` keeps the
    #    repo's own directory structure, and the repo root is flat, so these
    #    land directly in `staging` and are then moved into `dest`.
    wanted = [f for f in files
              if f.endswith((".safetensors", ".json", ".model", ".txt"))]
    for fname in wanted:
        try:
            hf_hub_download(repo_id, fname, repo_type="model", token=token,
                            local_dir=staging)
            print(f"[pretrain]   + {fname}", flush=True)
        except Exception as e:
            print(f"[pretrain] WARN: could not fetch {fname} ({e})", flush=True)

    for entry in os.listdir(staging):
        src = os.path.join(staging, entry)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(dest, entry))
    try:
        os.rmdir(staging)
    except OSError:
        pass

    if not _weights_present(dest):
        print(f"[pretrain] WARN: fetched checkpoint at {dest} has no model "
              f"weights -- cannot resume", flush=True)
        return None, 0

    print(f"[pretrain] Hub checkpoint ready: {dest} (step {step})", flush=True)
    return dest, step


def resolve_resume_point(args):
    """Decide which checkpoint to resume from, local first, then the Hub.

    Local wins by default: a local step_N is either this machine's own newer
    work or the same Hub snapshot already staged, and re-downloading 600MB to
    end up at an older step would be a regression, not a resume. The Hub is
    consulted exactly when the local directory cannot supply a usable
    checkpoint -- a fresh box, a wiped volume, or a preemption that lost the
    scratch disk.

    `--no-resume` short-circuits everything here, including the Hub: it means
    "do not continue anyone's run", not "continue from somewhere else".
    """
    if getattr(args, "no_resume", False):
        path, _ = find_latest_checkpoint(args.save_dir)
        if path is not None:
            print(f"[pretrain] --no-resume set, ignoring checkpoint at {path}", flush=True)
        else:
            print("[pretrain] --no-resume set, not fetching from the Hub", flush=True)
        return None, 0, None

    if getattr(args, "hub_only", False):
        print("[pretrain] --hub-only: ignoring local checkpoints", flush=True)
        return None, 0, None

    path, step = find_latest_checkpoint(args.save_dir)
    if path is not None:
        return path, step, None

    repo = getattr(args, "resume_from_hub", None) or args.hub_repo
    if not repo:
        return None, 0, None

    # Make the staging directory exist before the (slow) download, so a full
    # disk fails in seconds rather than after 600MB of transfer.
    os.makedirs(args.save_dir, exist_ok=True)
    path, step = fetch_hub_checkpoint(repo, token=args.token, save_dir=args.save_dir)
    if path is None:
        return None, 0, None
    return path, step, repo


def _detach_rng_state(obj):
    """Coerce anything RNG-shaped into a contiguous CPU uint8 ByteTensor.

    torch.save/load round-trips the state correctly, but the LOAD side is
    the trap: `torch.load(..., map_location=device)` on a GPU box
    deserializes the saved CPU ByteTensor as a CUDA tensor, and
    `torch.random.set_rng_state` accepts only a CPU ByteTensor. Callers get
    `TypeError: RNG state must be a torch.ByteTensor` from deep inside
    torch/random.py, which reads like a corrupted checkpoint rather than a
    device-placement mismatch.
    """
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    try:
        return torch.as_tensor(np.asarray(obj, dtype=np.uint8),
                               dtype=torch.uint8).contiguous()
    except Exception:
        return None


def restore_rng(rng):
    """Best-effort restore of every RNG stream a checkpoint carries.

    Each stream is independent and non-fatal: failing to restore one costs
    exact reproducibility of the next few steps, not the run itself. That
    matters because the torch stream is a ByteTensor whose device depends on
    the loader, and checkpoints written by older builds may not carry the
    python/numpy streams at all.
    """
    if not rng:
        return

    cpu_state = _detach_rng_state(rng.get("torch"))
    if cpu_state is not None:
        try:
            torch.random.set_rng_state(cpu_state)
        except Exception as e:
            print(f"[pretrain] WARN: torch RNG state not restored ({e}); "
                  f"resume will not be bit-identical", flush=True)

    cuda_state = _detach_rng_state(rng.get("cuda"))
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state(cuda_state)
        except Exception as e:
            print(f"[pretrain] WARN: CUDA RNG state not restored ({e})", flush=True)

    py_state = rng.get("python")
    if py_state is not None:
        try:
            random.setstate(py_state)
        except Exception:
            pass

    np_state = rng.get("numpy")
    if np_state is not None and len(np_state) >= 3:
        try:
            np.random.set_state(tuple(np_state))
        except Exception:
            pass


def save_checkpoint(model, optim, step, args, losses, val_history=None,
                    best_val=None):
    """Save model + optimizer + RNG + history to a checkpoint directory."""
    ckpt = os.path.join(args.save_dir, f"step_{step}")
    os.makedirs(ckpt, exist_ok=True)
    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_model.save_pretrained(ckpt)
    # Save optimizer + step + RNG states for faithful resume. RNG states are
    # explicitly pulled to CPU here so the file is portable across machines
    # and independent of the map_location used at load time.
    rng = {
        "torch": torch.random.get_rng_state().cpu(),
        "cuda": (torch.cuda.get_rng_state().cpu()
                 if torch.cuda.is_available() else None),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    torch.save({
        "step": step,
        "optim": optim.state_dict(),
        "rng": rng,
        "losses": list(losses),
        "val_history": list(val_history or []),
        "best_val": best_val,
        "config": vars(args),
    }, os.path.join(ckpt, "training_state.pt"))
    print(f"[pretrain] saved checkpoint: {ckpt}", flush=True)
    return ckpt


def load_checkpoint(ckpt_path, model, optim, device):
    """Load optimizer state, RNG streams, history, and step from a checkpoint.

    Model weights are assumed already loaded by from_pretrained.
    Returns a dict with the step to resume from (the step AFTER the saved one)
    plus the restored loss/validation history, so the final train-vs-val
    report and the overfit streak counter span the whole run instead of
    starting over blank at the resume point.
    """
    empty = {"step": 0, "losses": [], "val_history": [], "best_val": None}
    state_file = os.path.join(ckpt_path, "training_state.pt")
    if not os.path.exists(state_file):
        return empty
    state = torch.load(state_file, map_location="cpu", weights_only=False)
    optim.load_state_dict(state["optim"])
    restore_rng(state.get("rng") or {})
    resumed_step = int(state["step"])
    print(f"[pretrain] loaded training state from step {resumed_step}", flush=True)
    return {
        "step": resumed_step + 1,  # resume from the next step
        "losses": list(state.get("losses") or []),
        "val_history": [tuple(r) for r in (state.get("val_history") or [])],
        "best_val": state.get("best_val"),
    }


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

    # ── Checkpoint auto-discovery (local, then Hub) ─────────────────────
    ckpt_path, ckpt_step, hub_src = resolve_resume_point(args)
    resume = ckpt_path is not None
    if resume:
        src = f" (from Hub {hub_src})" if hub_src else ""
        print(f"[pretrain] found checkpoint: {ckpt_path} (step {ckpt_step}){src}", flush=True)

    # ── Data ───────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import MMapDataset

    ds = MMapDataset(args.shards, block_size=args.block, seed=args.seed)
    print(f"[pretrain] dataset: {len(args.shards)} shards, {ds.total_tokens/1e6:.1f}M tokens", flush=True)

    # The model config's vocab is known only after the tokenizer loads, so the
    # shard/vocab sanity check runs later, right after the model is built.
    def collate(batch):
        x = torch.stack(batch)
        return x[:, :-1], x[:, 1:]

    def make_loader(epoch: int):
        """Build the training loader for a given data-stream epoch.

        `worker_init_fn` is load-bearing, not boilerplate. MMapDataset seeds a
        single RNG in __init__, and forked workers inherit that state verbatim
        -- so all 8 workers draw the IDENTICAL sample sequence. The interleaved
        batches are then the same block 8 times over: an 8x cut in the unique
        data actually seen, completely silent. `set_epoch` already folds the
        worker id into its seed, so re-seeding per worker is all it takes.

        The loader is built lazily (after the checkpoint load) so the epoch can
        be seeded from the resumed step. Workers fork and freeze their RNG at
        first iteration, so a loader seeded before the resume would replay the
        blocks the interrupted run already trained on.
        """
        ds.set_epoch(epoch)

        def _init_worker(_):
            ds.set_epoch(epoch)

        nw = max(0, args.num_workers)
        return DataLoader(
            ds, batch_size=args.batch, num_workers=nw, pin_memory=True,
            collate_fn=collate, persistent_workers=nw > 0,
            worker_init_fn=_init_worker if nw > 0 else None,
            prefetch_factor=4 if nw > 0 else None,
        )

    # ── Model ──────────────────────────────────────────────────────────
    from config import VortexArch, TokenizerProfile, PARAM_BUDGET
    from model import VortexForCausalLM, VortexConfig

    # Architecture preset is the single source of truth for the shape.
    arch = VortexArch.from_name(args.arch)
    arch.max_position_embeddings = args.block
    if args.rope_theta is not None:
        arch.rope_theta = args.rope_theta
    arch.validate()

    # The tokenizer's real vocab is authoritative. It must match the shards
    # on disk -- a mismatch here silently trains on garbage ids.
    # An explicit --tokenizer wins; otherwise fall back to DEFAULT_TOKENIZER_ID,
    # so a fresh box resumes without needing a hand-placed tokenizer directory.
    tok_source = args.tokenizer
    if tok_source is None and hub_src:
        # The pushed checkpoint ships its own tokenizer files, so resuming
        # outside the original box reproduces the original tokenization.
        tok_source = hub_src
    try:
        prof = TokenizerProfile.from_pretrained(tok_source)
        if prof.vocab_size != arch.vocab_size:
            print(f"[pretrain] NOTE: tokenizer vocab {prof.vocab_size:,} != preset "
                  f"{arch.vocab_size:,}; using the tokenizer's value so the model "
                  f"can represent every id in the shards.", flush=True)
        arch.vocab_size = prof.vocab_size
        print(f"[pretrain] tokenizer: {prof.tokenizer_id} vocab={prof.vocab_size:,} "
              f"eos={prof.eos_token_id}", flush=True)
    except Exception as e:
        print(f"[pretrain] WARN: could not load tokenizer ({e});", flush=True)
        print(f"[pretrain]       falling back to the preset vocab "
              f"{arch.vocab_size:,}. Data MUST have been tokenized with it.", flush=True)

    over = arch.n_params() - PARAM_BUDGET
    if over > 0:
        print(f"[pretrain] ERROR: {arch.name_or_path} is {arch.n_params()/1e6:.2f}M, "
              f"{over/1e6:.2f}M OVER the {PARAM_BUDGET/1e6:.0f}M budget.", flush=True)
        sys.exit(1)
    print(arch.summary(), flush=True)

    cfg = VortexConfig(**arch.to_dict())
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
    print(f"[pretrain] model: {n/1e6:.2f}M params ({n_no_embed/1e6:.2f}M non-embed)", flush=True)

    # Every id in the shards must be representable. If the data was tokenized
    # with a DIFFERENT (larger) vocab than the model has, ids >= vocab_size
    # reach the embedding and either crash or silently wrap -- so check the
    # real maximum instead of trusting the config.
    try:
        import numpy as _np
        max_id = 0
        for p in args.shards[:3]:
            arr = _np.memmap(p, dtype=_np.uint32, mode="r")
            if len(arr):
                max_id = max(max_id, int(arr[:min(len(arr), 5_000_000)].max()))
        if max_id >= arch.vocab_size:
            print(f"[pretrain] ERROR: shards contain token id {max_id} but the model's "
                  f"vocab is only {arch.vocab_size:,}.", flush=True)
            print(f"[pretrain]        The data was tokenized with a different "
                  f"tokenizer. Re-tokenize with `retokenize.py` or point "
                  f"--tokenizer at the right one.", flush=True)
            sys.exit(1)
        print(f"[pretrain] shard id range: 0..{max_id} (fits vocab {arch.vocab_size:,})", flush=True)
    except FileNotFoundError:
        pass
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

    # ── Auto batch sizing ──────────────────────────────────────────────
    # Runs BEFORE the optimizer so the probe's gradients/state never mix with
    # training state, and before torch.compile so a probe compile is not paid
    # for twice. The largest batch that fits is the fastest per-token option;
    # grad_accum then tops the effective batch back up to the target.
    if args.auto_batch and torch.cuda.is_available() and not resume:
        probe_model = model
        probe_model.gradient_checkpointing_enable()
        params = [p for p in probe_model.parameters() if p.requires_grad]

        def build_probe_step(bs: int):
            ids = torch.randint(0, arch.vocab_size, (bs, args.block), device=device)
            tgt = torch.randint(0, arch.vocab_size, (bs, args.block), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = probe_model(input_ids=ids, labels=tgt, chunk_size=1024)
                return out.loss / bs

        print(f"[pretrain] auto-batch: probing {torch.cuda.get_device_name(0)} "
              f"({_vram_gb():.1f} GB)...", flush=True)
        best, log = find_max_batch(build_probe_step, params, args.block)
        for line in log:
            print(f"[pretrain] {line}", flush=True)

        if best == 0:
            print(f"[pretrain] ERROR: cannot fit batch=1 at block={args.block}. "
                  f"Lower --block.", flush=True)
            sys.exit(1)

        # Leave headroom: probing to the exact ceiling leaves no margin for a
        # longer document, allocator fragmentation, or Trackio. 90% of the
        # discovered max is the largest size that stays reliably stable.
        chosen = max(1, int(best * 0.9))
        accum = max(1, args.target_batch // chosen)
        if chosen * accum != args.target_batch:
            print(f"[pretrain] note: target_batch={args.target_batch} is not a "
                  f"multiple of batch={chosen}; using "
                  f"{chosen * accum} sequences/step", flush=True)
        args.batch, args.grad_accum = chosen, accum
        print(f"[pretrain] auto-batch -> batch={chosen} x grad_accum={accum} "
              f"= {chosen * accum} seq/step "
              f"({chosen * accum * args.block:,} tok/step)", flush=True)

        # Reclaim everything the probe allocated.
        for p in params:
            p.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif args.auto_batch:
        print(f"[pretrain] auto-batch skipped "
              f"({'no CUDA' if not torch.cuda.is_available() else 'resuming'})", flush=True)

    # ── Optimizer ──────────────────────────────────────────────────────
    optim = build_optimizer(model, args)

    # Load optimizer/RNG state if resuming
    start_step = 0
    losses = []
    val_history = []          # (step, train_loss, val_loss)
    best_val = float("inf")
    overfit_strikes = 0
    if resume:
        ck = load_checkpoint(ckpt_path, model, optim, device)
        start_step = ck["step"]
        losses = ck["losses"]
        val_history = ck["val_history"]
        # A fresh run has no `best_val` to beat, so seed it from the checkpoint
        # rather than re-triggering the overfit warning on the first val step
        # that merely ties the pre-crash best.
        best_val = ck["best_val"] if ck["best_val"] is not None else float("inf")
        print(f"[pretrain] resuming from step {start_step}", flush=True)

    # Data stream is seeded from the resumed step so the worker RNGs do not
    # replay blocks the interrupted run already consumed. Built after the
    # checkpoint load for that reason -- workers freeze their seed at fork.
    data_epoch = start_step // max(1, args.batch * args.grad_accum)
    loader = make_loader(data_epoch)
    it = iter(loader)
    if resume and data_epoch:
        print(f"[pretrain] data stream reseeded to epoch {data_epoch} "
              f"(past {start_step:,} steps already consumed)", flush=True)

    # Enable gradient checkpointing to save VRAM (~40% less activation memory)
    model.gradient_checkpointing_enable()
    print(f"[pretrain] gradient checkpointing enabled", flush=True)

    # ── Held-out validation split (the overfit guard) ──────────────────
    # A 50M model on 2B tokens (~40 epochs of a Chinchilla-optimal budget) can
    # memorize. Holding out the tail of the corpus gives a held-out loss curve;
    # if train loss keeps falling while val loss rises, that is overfitting and
    # the run is wasting the rest of its budget.
    val_it = None
    if args.val_every and ds.total_tokens > args.block * 100:
        import numpy as np

        n_val_blocks = max(args.batch, min(64, args.val_tokens // args.block))
        val_rng = np.random.default_rng(args.seed + 999)
        val_blocks = []
        for _ in range(n_val_blocks):
            start = int(val_rng.integers(ds.total_tokens - args.block * 2,
                                         ds.total_tokens - args.block - 2))
            mm, local = ds._locate(start)
            val_blocks.append(torch.from_numpy(
                np.asarray(mm[local:local + args.block + 1], dtype=np.int64)))
        val_batches = [
            torch.stack(val_blocks[i:i + args.batch])
            for i in range(0, len(val_blocks) - args.batch + 1, args.batch)
        ]
        if val_batches:
            val_it = iter(val_batches)
            n_val_tok = len(val_batches) * args.batch * args.block
            print(f"[pretrain] validation: {len(val_batches)} held-out batches "
                  f"x {args.batch} x {args.block} = {n_val_tok:,} tokens "
                  f"(every {args.val_every} steps)", flush=True)
            # Guard against a pathological config: fewer than 8 batches makes
            # the val loss too noisy to read a trend from.
            if len(val_batches) < 8:
                print(f"[pretrain] WARNING: only {len(val_batches)} val batches; "
                      f"raise --val-tokens for a readable val curve", flush=True)
        else:
            print(f"[pretrain] validation disabled (corpus too small)", flush=True)
    else:
        print(f"[pretrain] validation disabled", flush=True)

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
    steps_this_run = 0
    t0 = time.time()
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

        # ── Validation + overfit detection ──────────────────────────────
        if val_it is not None and args.val_every and (step + 1) % args.val_every == 0:
            model.eval()
            vloss, nb = 0.0, 0
            with torch.no_grad():
                for vb in val_batches:
                    v_in = vb[:, :-1].to(device, non_blocking=True)
                    v_tg = vb[:, 1:].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        vloss += model(input_ids=v_in, labels=v_tg,
                                       chunk_size=1024).loss.item()
                    nb += 1
            val_loss = vloss / max(1, nb)
            t_avg = sum(losses[-args.val_every:]) / max(1, len(losses[-args.val_every:]))
            gap = t_avg - val_loss
            val_history.append((step + 1, t_avg, val_loss))
            print(f"[pretrain] step {step+1}  train={t_avg:.4f}  val={val_loss:.4f}  "
                  f"gap={gap:+.4f}  ppl_val={math.exp(min(20, val_loss)):.1f}", flush=True)
            if HAS_TRACKIO and os.environ.get("TRACKIO_SPACE_ID"):
                trackio.log({"step": step + 1, "val_loss": val_loss,
                             "train_val_gap": gap})

            if val_loss < best_val:
                best_val = val_loss
                overfit_strikes = 0
            else:
                overfit_strikes += 1

            # Overfitting: train loss falling while held-out loss does not.
            # 3 consecutive non-improvements past the halfway point is a
            # strong signal that the remaining budget would be wasted.
            if overfit_strikes >= 3 and (step + 1) > args.steps * 0.5:
                print(f"[pretrain] WARNING: val loss has not improved for "
                      f"{overfit_strikes} checks (best {best_val:.4f}, "
                      f"now {val_loss:.4f}) past 50%% of training.", flush=True)
                print(f"[pretrain]          The model is likely overfitting "
                      f"{args.steps * tok_per_step / 1e9:.2f}B tokens for a "
                      f"{n/1e6:.1f}M model.", flush=True)
                print(f"[pretrain]          Consider: fewer steps, more data, "
                      f"or a smaller model. Continuing to step {args.steps}.", flush=True)
            model.train()

        if (step + 1) % args.push_every == 0 or (step + 1) == args.steps:
            ckpt = save_checkpoint(model, optim, step + 1, args, losses,
                                   val_history, best_val)
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
    print(f"[pretrain] final train loss: {sum(losses[-100:])/max(1,len(losses[-100:])):.4f} "
          f"(avg last 100)", flush=True)
    print(f"[pretrain] tokens seen: {args.steps * tok_per_step / 1e9:.2f}B", flush=True)

    if val_history:
        print("\n" + "=" * 64)
        print("TRAIN / VALIDATION CURVE  (overfit check)")
        print("=" * 64)
        print(f"{'step':>8} {'train':>9} {'val':>9} {'gap':>8} {'val ppl':>9}")
        for st, tr, vl in val_history:
            print(f"{st:>8} {tr:>9.4f} {vl:>9.4f} {tr - vl:>+8.4f} "
                  f"{math.exp(min(20, vl)):>9.1f}")
        print("-" * 64)
        best_st, _, best_vl = min(val_history, key=lambda r: r[2])
        last_gap = val_history[-1][1] - val_history[-1][2]

        # How far into the run are we? A gap only means something once the
        # model has actually had time to fit anything.
        progress = args.steps and val_history[-1][0] / args.steps
        mature = progress is not None and progress >= 0.5

        # Require the gap to be BOTH meaningful in size AND persistent across
        # consecutive checks. A single negative reading is noise: dropout-free
        # training still has batch-to-batch variance, and early in a run the
        # held-out split is simply easier than the training draw.
        recent = val_history[-3:]
        n_negative = sum(1 for _, tr, vl in recent if tr - vl < -0.05)

        print(f"best val    {best_vl:.4f} at step {best_st}")
        print(f"final gap   {last_gap:+.4f}  (train - val)")
        print(f"progress    {(progress or 0)*100:.0f}% of run complete")

        if not mature:
            print("VERDICT     TOO EARLY TO TELL -- the run is under 50% complete, so")
            print("            train and val have not diverged yet. Re-read this")
            print("            table at the end of training.")
        elif last_gap < -0.05 and n_negative >= 2:
            print("VERDICT     OVERFITTING -- held-out loss is now persistently BELOW")
            print(f"            training loss (last {n_negative} of {len(recent)} checks).")
            print("            Either the val split leaked, or the model is memorizing.")
            print("            Fix: fewer steps, lower LR late in the schedule, or more data.")
        elif last_gap < -0.05:
            print("VERDICT     probably noise -- one negative reading, not a trend yet.")
        elif last_gap > 0.15:
            print("VERDICT     generalising, but a widening gap -- monitor; if it")
            print("            keeps growing the run will eventually memorize.")
        else:
            print("VERDICT     healthy -- train and val are tracking together.")
        print("=" * 64 + "\n")

    if HAS_TRACKIO and os.environ.get("TRACKIO_SPACE_ID"):
        trackio.finish()


if __name__ == "__main__":
    main()
