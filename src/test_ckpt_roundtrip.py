"""End-to-end save/resume test using the real model + real AdamW.

Exercises the actual save_checkpoint() / load_checkpoint() pair from
pretrain.py -- not reimplementations -- so a regression in the production
path fails here.
"""
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrain import load_checkpoint, save_checkpoint  # noqa: E402
from model import VortexConfig, VortexForCausalLM  # noqa: E402

failures = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {extra}" if extra else ""))
    if not cond:
        failures.append(name)


def one_step(m, o):
    o.zero_grad()
    loss = m(input_ids=ids, labels=labels, chunk_size=32).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    o.step()
    return loss.item()


print("=" * 64)
print("SAVE / RESUME ROUND TRIP  (real model + real AdamW)")
print("=" * 64)

cfg = VortexConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2,
                   intermediate_size=128, head_dim=16, max_position_embeddings=64)
torch.manual_seed(0)
model = VortexForCausalLM(cfg)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

ids = torch.randint(0, 256, (2, 64))
labels = torch.randint(0, 256, (2, 64))

# Take a few real steps so the AdamW moments are non-empty.
for _ in range(3):
    one_step(model, optim)

n_param = sum(p.numel() for p in model.parameters())
print(f"\nmodel: {n_param:,} params, {len(optim.state)} optimizer state entries")

args = types.SimpleNamespace(save_dir=tempfile.mkdtemp(), hub_repo=None)
losses = [3.1, 2.9, 2.7]
val_history = [(2, 2.9, 3.0), (3, 2.7, 2.95)]
best_val = 2.95

print("\n1. save_checkpoint()")
ckpt = save_checkpoint(model, optim, 3, args, losses, val_history, best_val)
check("checkpoint dir created", os.path.isdir(ckpt))
check("model weights written",
      any(f.endswith(".safetensors") for f in os.listdir(ckpt)))
check("training_state.pt written",
      os.path.exists(os.path.join(ckpt, "training_state.pt")))

print("\n2. from_pretrained + load_checkpoint() into a FRESH model + optimizer")
# Mirrors the production resume path exactly: main() does
#   VortexForCausalLM.from_pretrained(ckpt_path)
#   load_checkpoint(ckpt_path, model, optim, device)
# Weights are restored by from_pretrained; load_checkpoint restores ONLY
# optimizer/RNG/history, so it must be handed a model that already has them.
torch.manual_seed(777)          # different init on purpose
fresh = VortexForCausalLM.from_pretrained(ckpt)
fresh_optim = torch.optim.AdamW(fresh.parameters(), lr=1e-3)

ref = torch.cat([p.detach().flatten() for p in model.parameters()])
loaded = torch.cat([p.detach().flatten() for p in fresh.parameters()])
check("from_pretrained restored the saved weights",
      torch.equal(ref, loaded), f"max|d|={(ref - loaded).abs().max().item():.2e}")

before = loaded.clone()
res = load_checkpoint(ckpt, fresh, fresh_optim, "cpu")

check("returns next step (saved 3 -> resume 4)", res["step"] == 4, f"got {res['step']}")
check("losses restored", res["losses"] == losses, f"{res['losses']}")
check("val_history restored", [tuple(r) for r in res["val_history"]] == val_history)
check("best_val restored", res["best_val"] == best_val, f"{res['best_val']}")

after = torch.cat([p.detach().flatten() for p in fresh.parameters()])
check("load_checkpoint does not clobber weights",
      torch.equal(before, after), f"max|d|={(before - after).abs().max().item():.2e}")

print("\n3. optimizer state (AdamW moments) restored")
check("optimizer state non-empty", len(fresh_optim.state) > 0)
exp_avg = next((st["exp_avg"] for st in fresh_optim.state.values() if "exp_avg" in st), None)
check("exp_avg moment present", exp_avg is not None)
if exp_avg is not None:
    ref_exp = next((st["exp_avg"] for st in optim.state.values() if "exp_avg" in st), None)
    check("exp_avg matches saved exactly",
          ref_exp is not None and torch.equal(ref_exp, exp_avg))

print("\n4. resumed run continues on the same trajectory as an uninterrupted one")
# The reference: the ORIGINAL model, still holding its real optimizer state,
# simply takes another step with no save/load in between.
torch.manual_seed(31337)
ref_model = VortexForCausalLM.from_pretrained(ckpt)
ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-3)
ref_optim.load_state_dict(optim.state_dict())
l_ref = one_step(ref_model, ref_optim)
w_ref = torch.cat([p.detach().flatten() for p in ref_model.parameters()]).clone()

torch.manual_seed(31337)
again = VortexForCausalLM.from_pretrained(ckpt)
again_optim = torch.optim.AdamW(again.parameters(), lr=1e-3)
load_checkpoint(ckpt, again, again_optim, "cpu")
l_b = one_step(again, again_optim)
w_after_again = torch.cat([p.detach().flatten() for p in again.parameters()])

check("resumed loss == uninterrupted loss", abs(l_b - l_ref) < 1e-6,
      f"{l_b:.6f} vs {l_ref:.6f}")
check("resumed weights == uninterrupted weights",
      torch.equal(w_ref, w_after_again),
      f"max|d|={(w_ref - w_after_again).abs().max().item():.2e}")

print("\n5. edge cases")
empty = load_checkpoint(tempfile.mkdtemp(), fresh, fresh_optim, "cpu")
check("missing training_state.pt -> zero-step empty result",
      empty["step"] == 0 and empty["losses"] == [] and empty["val_history"] == [],
      str(empty))

print("\n" + "=" * 64)
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL ROUND-TRIP CHECKS PASSED")
print("=" * 64)
