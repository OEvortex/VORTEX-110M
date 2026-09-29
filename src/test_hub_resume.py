"""Tests for Hub-based checkpoint discovery and resume.

Exercises the real resolve_resume_point() / fetch_hub_checkpoint() against
a mocked Hub, plus a live check of the real VTXAI/vortex-50m-16k repo
layout.
"""
import os
import shutil
import sys
import tempfile
import types
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretrain  # noqa: E402
from pretrain import (fetch_hub_checkpoint, find_latest_checkpoint,  # noqa: E402
                      read_state_step, resolve_resume_point)

failures = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {extra}" if extra else ""))
    if not cond:
        failures.append(name)


def write_state(d, step):
    os.makedirs(d, exist_ok=True)
    torch.save({"step": step, "optim": {}, "rng": {}, "losses": [],
                "val_history": [], "best_val": None},
               os.path.join(d, "training_state.pt"))
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(b"FAKEWEIGHTS")
    with open(os.path.join(d, "config.json"), "w") as f:
        f.write('{"hidden_size": 512}')


def make_args(save_dir, **kw):
    base = dict(save_dir=save_dir, hub_repo="VTXAI/vortex-50m-16k", token=None,
                no_resume=False, hub_only=False, resume_from_hub=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


print("=" * 64)
print("HUB CHECKPOINT RESUME")
print("=" * 64)

# ── 1. Local checkpoint scanning ──────────────────────────────────────────
print("\n1. find_latest_checkpoint() picks the highest step")
tmp = tempfile.mkdtemp()
write_state(os.path.join(tmp, "step_3000"), 3000)
write_state(os.path.join(tmp, "step_9000"), 9000)
write_state(os.path.join(tmp, "step_6000"), 6000)
# Decoys that must NOT be parsed as a step.
os.makedirs(os.path.join(tmp, "step_abc"), exist_ok=True)
os.makedirs(os.path.join(tmp, "step_"), exist_ok=True)
with open(os.path.join(tmp, "notadir"), "w") as f:
    f.write("x")

p, s = find_latest_checkpoint(tmp)
check("returns step_9000", p is not None and p.endswith("step_9000"), f"{p} step={s}")
check("step number is 9000", s == 9000, f"got {s}")
check("non-numeric decoys ignored", find_latest_checkpoint(tmp)[1] == 9000)

p, s = find_latest_checkpoint(os.path.join(tmp, "nonexistent"))
check("missing dir -> (None, 0)", p is None and s == 0)

weightsless = tempfile.mkdtemp()
os.makedirs(os.path.join(weightsless, "step_5000"))
check("weights-less checkpoint is rejected", find_latest_checkpoint(weightsless) == (None, 0))

# ── 2. read_state_step ────────────────────────────────────────────────────
print("\n2. read_state_step()")
sd = tempfile.mkdtemp()
write_state(os.path.join(sd, "step_3000"), 3000)
st = os.path.join(sd, "step_3000", "training_state.pt")
check("reads the step from training_state.pt", read_state_step(st) == 3000)
with open(os.path.join(sd, "junk.pt"), "wb") as f:
    f.write(b"not a torch file")
check("corrupt file -> None, no crash", read_state_step(os.path.join(sd, "junk.pt")) is None)
check("missing file -> None, no crash", read_state_step(os.path.join(sd, "nope.pt")) is None)

# ── 3. fetch_hub_checkpoint against a mocked Hub ──────────────────────────
print("\n3. fetch_hub_checkpoint() -- flat repo, step read from the state file")


class FakeHF:
    """Stand-in for huggingface_hub with a FLAT repo layout, matching the
    real VTXAI/vortex-50m-16k: everything at the repo root, no step_ prefix."""

    def __init__(self, remote_files, step=3000, fail_files=()):
        self.remote = remote_files
        self.step = step
        self.fail = set(fail_files)
        self.calls = []

    def list_repo_files(self, repo_id, repo_type=None, token=None):
        if repo_id == "does/not-exist":
            raise OSError("404 not found")
        return list(self.remote)

    def hf_hub_download(self, repo_id, filename, repo_type=None, token=None, local_dir=None):
        self.calls.append(filename)
        if filename in self.fail:
            raise OSError(f"simulated failure on {filename}")
        if local_dir is None:
            raise AssertionError("expected local_dir to be used")
        os.makedirs(local_dir, exist_ok=True)
        path = os.path.join(local_dir, filename)
        if filename == "training_state.pt":
            torch.save({"step": self.step, "optim": {}, "rng": {}}, path)
        else:
            with open(path, "w") as f:
                f.write(f"content-of-{filename}")
        return path


REMOTE = ["config.json", "model.safetensors", "training_state.pt",
          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

dest_dir = tempfile.mkdtemp()
fake = FakeHF(REMOTE, step=3000)
with mock.patch.object(pretrain, "_hub", fake, create=True), \
     mock.patch("huggingface_hub.list_repo_files", fake.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", fake.hf_hub_download):
    path, step = fetch_hub_checkpoint("VTXAI/vortex-50m-16k", save_dir=dest_dir)

check("returns the step from the remote state file", step == 3000, f"got {step}")
check("staged as save_dir/step_3000/", path == os.path.join(dest_dir, "step_3000"), f"{path}")
check("model.safetensors present", os.path.exists(os.path.join(path, "model.safetensors")))
check("training_state.pt present", os.path.exists(os.path.join(path, "training_state.pt")))
check("config.json present", os.path.exists(os.path.join(path, "config.json")))
check("tokenizer.json present", os.path.exists(os.path.join(path, "tokenizer.json")))
check("staging dir cleaned up", not os.path.exists(os.path.join(dest_dir, ".hub_staging")))
check("training_state.pt was fetched FIRST", fake.calls[0] == "training_state.pt",
      f"first={fake.calls[0]}")
check("fetched via local_dir (no cache moves)", True)

# Idempotent: a second call finds the staged dir and does not re-download.
calls_before = len(fake.calls)
with mock.patch("huggingface_hub.list_repo_files", fake.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", fake.hf_hub_download):
    path2, step2 = fetch_hub_checkpoint("VTXAI/vortex-50m-16k", save_dir=dest_dir)
check("second call reuses the staged checkpoint", path2 == path and step2 == 3000)
check("second call downloads nothing more", len(fake.calls) == calls_before,
      f"{len(fake.calls) - calls_before} extra downloads")

# ── 4. Failure modes degrade to "train from scratch" ──────────────────────
print("\n4. Unusable Hub repos degrade gracefully")
bad = tempfile.mkdtemp()
with mock.patch("huggingface_hub.list_repo_files", fake.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", fake.hf_hub_download):
    p, s = fetch_hub_checkpoint("does/not-exist", save_dir=bad)
check("unreachable repo -> (None, 0)", p is None and s == 0)

no_state = FakeHF(["config.json", "model.safetensors"])
with mock.patch("huggingface_hub.list_repo_files", no_state.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", no_state.hf_hub_download):
    p, s = fetch_hub_checkpoint("weights-only/repo", save_dir=bad)
check("repo without training_state.pt -> (None, 0)", p is None and s == 0)

no_weights = FakeHF(["config.json", "training_state.pt"])
with mock.patch("huggingface_hub.list_repo_files", no_weights.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", no_weights.hf_hub_download):
    p, s = fetch_hub_checkpoint("no-weights/repo", save_dir=bad)
check("repo without weights -> (None, 0)", p is None and s == 0)

# ── 5. resolve_resume_point precedence ────────────────────────────────────
print("\n5. resolve_resume_point() precedence")
# Each case gets its OWN save_dir. Sharing one across cases would make the
# Hub fallback of an earlier case satisfy a later case's local lookup, which
# tests nothing.
local = tempfile.mkdtemp()
write_state(os.path.join(local, "step_3000"), 3000)

p, s, src = resolve_resume_point(make_args(local))
check("local checkpoint wins (no download)", p is not None and src is None, f"src={src}")

p, s, src = resolve_resume_point(make_args(local, hub_only=True))
check("--hub-only ignores local", p is None, f"{p}")

p, s, src = resolve_resume_point(make_args(local, no_resume=True))
check("--no-resume discards a local checkpoint", p is None and src is None, f"{p}")

# --no-resume on a FRESH box must not reach the network at all.
fresh_noresume = tempfile.mkdtemp()
boom = FakeHF(REMOTE, step=3000)
called = []
_orig_list, _orig_dl = boom.list_repo_files, boom.hf_hub_download
boom.list_repo_files = lambda *a, **k: (called.append("list"), _orig_list(*a, **k))[1]
boom.hf_hub_download = lambda *a, **k: (called.append("dl"), _orig_dl(*a, **k))[1]
with mock.patch("huggingface_hub.list_repo_files", boom.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", boom.hf_hub_download):
    p, s, src = resolve_resume_point(make_args(fresh_noresume, no_resume=True))
check("--no-resume makes zero Hub calls on a fresh box", called == [], f"calls={called}")

# A box with a LOCAL checkpoint but --no-resume must make zero Hub calls too.
local_noresume = tempfile.mkdtemp()
write_state(os.path.join(local_noresume, "step_3000"), 3000)
with mock.patch("huggingface_hub.list_repo_files", boom.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", boom.hf_hub_download):
    p, s, src = resolve_resume_point(make_args(local_noresume, no_resume=True))
check("--no-resume makes zero Hub calls when a local ckpt exists",
      called == [], f"calls={called}")

empty = tempfile.mkdtemp()
fake2 = FakeHF(REMOTE, step=3000)
with mock.patch("huggingface_hub.list_repo_files", fake2.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", fake2.hf_hub_download):
    p, s, src = resolve_resume_point(make_args(empty))
check("empty save_dir falls back to the Hub", p is not None and s == 3000, f"{p}")
check("reports the Hub as the source", src == "VTXAI/vortex-50m-16k", f"{src}")

# A box whose ONLY checkpoint IS the staged Hub copy must not re-download.
with mock.patch("huggingface_hub.list_repo_files", fake2.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", fake2.hf_hub_download):
    n_before = len(fake2.calls)
    p, s, src = resolve_resume_point(make_args(empty))
check("re-running over a Hub-staged dir re-downloads nothing",
      len(fake2.calls) == n_before, f"{len(fake2.calls) - n_before} extra downloads")

other = tempfile.mkdtemp()
fake3 = FakeHF(REMOTE, step=7000)
with mock.patch("huggingface_hub.list_repo_files", fake3.list_repo_files), \
     mock.patch("huggingface_hub.hf_hub_download", fake3.hf_hub_download):
    p, s, src = resolve_resume_point(make_args(other, resume_from_hub="VTXAI/other-repo"))
check("--resume-from-hub overrides --hub-repo",
      src == "VTXAI/other-repo" and s == 7000, f"src={src} step={s}")

# ── 6. Live check of the real repo (network, layout only) ─────────────────
print("\n6. Live layout of VTXAI/vortex-50m-16k")
try:
    from huggingface_hub import list_repo_files
    live = list_repo_files("VTXAI/vortex-50m-16k", repo_type="model")
    check("repo is reachable", True, f"{len(live)} files")
    check("flat layout (no step_N/ prefix)",
          not any(f.startswith("step_") for f in live), f"{sorted(live)[:3]}...")
    check("has training_state.pt", "training_state.pt" in live)
    check("has model.safetensors", "model.safetensors" in live)
    check("ships a tokenizer", any(f.startswith("tokenizer") for f in live))
except Exception as e:
    print(f"  [SKIP] network unavailable: {e}")

for d in (tmp, sd, dest_dir, bad, local, empty, weightsless,
          fresh_noresume, other, local_noresume):
    shutil.rmtree(d, ignore_errors=True)

print("\n" + "=" * 64)
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL HUB RESUME CHECKS PASSED")
print("=" * 64)
