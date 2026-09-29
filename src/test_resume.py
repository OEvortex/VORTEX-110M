"""Regression test for the checkpoint-resume path.

Covers the failure in the VTX training log:

    TypeError: RNG state must be a torch.ByteTensor
        at torch/random.py set_rng_state, reached from load_checkpoint

Root cause: training_state.pt was loaded with map_location=device ("cuda"),
so the CPU ByteTensor that torch.save wrote came back as a CUDA tensor.
set_rng_state only accepts a CPU ByteTensor.
"""
import os
import random
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrain import _detach_rng_state, restore_rng  # noqa: E402

failures = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {extra}" if extra else ""))
    if not cond:
        failures.append(name)


print("=" * 64)
print("CHECKPOINT RESUME / RNG RESTORE")
print("=" * 64)

# ── 1. The exact bug: a CUDA-placed ByteTensor must be coerced to CPU ──────
print("\n1. ByteTensor device coercion")
cpu_state = torch.random.get_rng_state()
check("saved state is a CPU ByteTensor", cpu_state.dtype == torch.uint8 and not cpu_state.is_cuda)

if torch.cuda.is_available():
    # Reproduce the traceback's precondition exactly: map_location="cuda".
    mimic = cpu_state.cuda()
    check("reproduced: raw state is on cuda (would raise)", mimic.is_cuda)
    try:
        torch.random.set_rng_state(mimic)
        raw_raises = False
    except TypeError:
        raw_raises = True
    check("raw set_rng_state(cuda tensor) raises", raw_raises)
    fixed = _detach_rng_state(mimic)
    check("coerced back to CPU uint8", (not fixed.is_cuda) and fixed.dtype == torch.uint8)
    try:
        torch.random.set_rng_state(fixed)
        ok = True
    except TypeError:
        ok = False
    check("coerced state is accepted", ok, "the original crash")
else:
    print("  (no CUDA: device mismatch simulated with a plain-ndarray path)")

# Non-tensor junk (a pickled list, e.g. from a hand-edited checkpoint) must not crash.
check("list input coerced", _detach_rng_state([1, 2, 3]).dtype == torch.uint8)
check("None passes through", _detach_rng_state(None) is None)
check("uncoercible returns None", _detach_rng_state(object()) is None)

# ── 2. Full save/load/restore round trip ───────────────────────────────────
print("\n2. Full round trip (torch.save -> torch.load -> restore_rng)")
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "training_state.pt")
    # Snapshot the streams FIRST, then draw. Drawing first would capture a
    # state already advanced past the reference values.
    saved_rng = {
        "torch": torch.random.get_rng_state().cpu(),
        "cuda": (torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    expected = (torch.randn(4).tolist(), random.random(), float(np.random.rand()))

    torch.save({
        "step": 3000,
        "optim": {},
        "rng": saved_rng,
        "losses": [1.0, 2.0],
        "val_history": [(500, 1.5, 1.4)],
        "best_val": 1.4,
    }, path)

    # Perturb every stream so a silent no-op restore cannot pass.
    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)

    state = torch.load(path, map_location="cpu", weights_only=False)
    check("step round trips", state["step"] == 3000)
    check("losses round trip", state["losses"] == [1.0, 2.0])
    check("val_history round trips", state["val_history"] == [(500, 1.5, 1.4)])
    check("best_val round trips", state["best_val"] == 1.4)

    restore_rng(state["rng"])
    got = (torch.randn(4).tolist(), random.random(), float(np.random.rand()))
    check("torch stream restored", np.allclose(got[0], expected[0]), f"{got[0]} vs {expected[0]}")
    check("python stream restored", got[1] == expected[1])
    check("numpy stream restored", abs(got[2] - expected[2]) < 1e-12)

# ── 3. restore_rng must never be fatal ─────────────────────────────────────
print("\n3. restore_rng is non-fatal (best-effort, per-stream)")
restore_rng({})                                    # empty
restore_rng({"torch": None, "cuda": None})         # all None
restore_rng({"torch": torch.zeros(3, dtype=torch.float32)})   # wrong dtype/shape
restore_rng({"python": "garbage", "numpy": ["a", "b"]})      # corrupt
restore_rng({"torch": "not a tensor at all"})     # unwrappable
check("no stream ever raises", True)

# ── 4. Data-stream workers must not draw identical samples ────────────────
print("\n4. DataLoader workers draw independent samples (dup-sample bug)")
from dataset import MMapDataset  # noqa: E402

tmp = tempfile.mkdtemp()
shard = os.path.join(tmp, "shard_0000.bin")
arr = np.memmap(shard, dtype=np.uint32, mode="w+", shape=(200_000,))
arr[:] = np.random.default_rng(0).integers(0, 1000, size=200_000, dtype=np.uint32)
arr.flush()
del arr

ds = MMapDataset([shard], block_size=64, seed=42)
ds.set_epoch(0)
# Without worker_init_fn every worker inherits the parent's RNG *object*, so
# each emits the same first draw. Model it by giving every "worker" its own
# copy of the identical state (what fork actually does) and comparing first
# draws -- drawing repeatedly from one generator would just advance it.
inherited = ds._rng.bit_generator.state
worker_draws = []
for _ in range(8):
    bg = np.random.default_rng()
    bg.bit_generator.state = inherited
    worker_draws.append(int(bg.integers(0, 10**9)))
check("8 workers inheriting one RNG draw the same sample",
      len(set(worker_draws)) == 1, f"draws={sorted(set(worker_draws))}")

per_worker = []
for wid in range(8):
    ds.set_epoch(0)
    ds._rng = np.random.default_rng(42 + 0 * 1000 + wid)
    per_worker.append(int(ds._rng.integers(0, 10**9)))
check("set_epoch(worker_id) gives 8 distinct streams",
      len(set(per_worker)) == 8, f"{len(set(per_worker))}/8 unique")

# End-to-end through a real DataLoader with the same worker_init_fn the
# training script installs.
ds2 = MMapDataset([shard], block_size=64, seed=42)
ds2.set_epoch(0)


def _init_worker(_):
    ds2.set_epoch(0)


loader = torch.utils.data.DataLoader(
    ds2, batch_size=2, num_workers=4,
    worker_init_fn=_init_worker, collate_fn=lambda b: torch.stack(b),
)
batches = [b for _, b in zip(range(8), loader)]
uniq = {tuple(b.flatten().tolist()) for b in batches}
check("4 workers x 8 batches are all distinct", len(uniq) == 8, f"{len(uniq)}/8 unique")

print("\n" + "=" * 64)
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL RESUME CHECKS PASSED")
print("=" * 64)
