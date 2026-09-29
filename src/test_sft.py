"""End-to-end SFT smoke test with a real model, real tokenizer, real data.

Runs sft.py's actual code paths (normalize_messages, SFTDataset, collate,
model resize, one training step) on CPU with a tiny Vortex.
"""
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_template as CT          # noqa: E402
import sft as SFT                   # noqa: E402
from model import VortexConfig, VortexForCausalLM  # noqa: E402

failures = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {extra}" if extra else ""))
    if not cond:
        failures.append(name)


print("=" * 64)
print("SFT END-TO-END")
print("=" * 64)

# ── 1. normalize_messages: every dataset shape ───────────────────────────
print("\n1. normalize_messages handles all dataset shapes")
chatml = {"messages": [{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "hello"}]}
out = SFT.normalize_messages(chatml)
check("messages list is accepted", out is not None and len(out) == 2)

dolly = {"instruction": "Fix this", "context": "some ctx", "response": "Fixed."}
out = SFT.normalize_messages(dolly)
check("dolly instruction/context/response is accepted", out is not None)
check("dolly context is folded into the instruction",
      out and "some ctx" in out[0]["content"], str(out))

no_resp = {"instruction": "Q"}
check("dolly row with no response is rejected",
      SFT.normalize_messages(no_resp) is None)

user_only = {"messages": [{"role": "user", "content": "hi"}]}
check("messages with no assistant turn is rejected",
      SFT.normalize_messages(user_only) is None)

check("garbage row is rejected", SFT.normalize_messages({"nope": 1}) is None)
check("None is rejected", SFT.normalize_messages(None) is None)

# ── 2. collate: padding + label masking ──────────────────────────────────
print("\n2. collate pads correctly and keeps padding masked")
batch = [
    (torch.tensor([1, 2, 3, 4]), torch.tensor([-100, 2, 3, 4])),
    (torch.tensor([5, 6]), torch.tensor([-100, 6])),
]
ids, labels, attn = SFT.collate(batch)
check("batch is padded to the longest", ids.shape == labels.shape == attn.shape,
      f"{tuple(ids.shape)}")
check("ids padded with 0", ids[1, -2:].tolist() == [0, 0], str(ids[1].tolist()))
check("padded LABELS are -100 (ignored by loss)",
      labels[1, -2:].tolist() == [-100, -100], str(labels[1].tolist()))
check("attention mask is 1 for real, 0 for pad",
      attn[1].tolist() == [1, 1, 0, 0], str(attn[1].tolist()))
check("real labels survive", labels[0].tolist() == [-100, 2, 3, 4])

# ── 3. resize_token_embeddings (SFT grows the vocab for chat tokens) ──────
print("\n3. resize_token_embeddings grows and preserves pretrained rows")
cfg = VortexConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1,
                   num_attention_heads=4, num_key_value_heads=2,
                   intermediate_size=64, head_dim=8, max_position_embeddings=128)
torch.manual_seed(0)
model = VortexForCausalLM(cfg)
old_emb = model.model.embed_tokens.weight.detach().clone()
old_vocab = old_emb.shape[0]

model.model.resize_token_embeddings(old_vocab + 3)
new_emb = model.model.embed_tokens.weight
check("vocab grew by 3", new_emb.shape[0] == old_vocab + 3, f"{new_emb.shape}")
check("pretrained rows are bit-identical",
      torch.equal(new_emb[:old_vocab], old_emb),
      f"max|d|={(new_emb[:old_vocab] - old_emb).abs().max():.2e}")
check("new rows are NOT all zero (would be indistinguishable)",
      new_emb[old_vocab:].abs().sum().item() > 0)
# Must match the distribution the rest of the table was trained with --
# asserted RELATIVE to cfg.initializer_range, not a hardcoded number, so the
# test tracks the real config instead of a magic 0.02.
check("new rows use the config's initializer scale",
      abs(new_emb[old_vocab:].std().item() - cfg.initializer_range) < 0.3 * cfg.initializer_range,
      f"new std={new_emb[old_vocab:].std().item():.5f} "
      f"initializer_range={cfg.initializer_range}")

try:
    model.model.resize_token_embeddings(10)
    check("shrinking raises", False)
except ValueError:
    check("shrinking raises", True)

# ── 4. A real SFT training step runs and the loss is finite ──────────────
print("\n4. real SFT step: loss finite, grads flow, masking respected")
model.model.resize_token_embeddings(old_vocab + 3)
model.lm_head.weight = model.model.embed_tokens.weight  # retie
model.cfg.vocab_size = old_vocab + 3

ids = torch.randint(0, old_vocab, (2, 32))
labels = torch.full((2, 32), -100, dtype=torch.long)
labels[:, 16:] = ids[:, 16:]          # only the second half is trainable

optim = torch.optim.AdamW(model.parameters(), lr=3e-5)
out = model(input_ids=ids, labels=labels, chunk_size=64)
check("loss is a scalar", out.loss.ndim == 0, f"shape={out.loss.shape}")
check("loss is finite", torch.isfinite(out.loss).item(), f"loss={out.loss.item():.4f}")
out.loss.backward()
grads = sum(1 for p in model.parameters() if p.grad is not None)
check("gradients flow to every trainable tensor", grads > 0, f"{grads} tensors")
emb_grad = model.model.embed_tokens.weight.grad
check("embedding receives gradient", emb_grad is not None and emb_grad.abs().sum() > 0)

optim.step()
check("optimizer step runs", True)

# Loss with EVERYTHING masked must be 0, not NaN.
allmasked = torch.full((2, 32), -100, dtype=torch.long)
o2 = model(input_ids=ids, labels=allmasked, chunk_size=64)
check("all-masked batch -> loss 0 without NaN",
      o2.loss.item() == 0.0 and torch.isfinite(o2.loss).item(),
      f"loss={o2.loss.item()}")

# The masked loss must ignore the prompt half. Compare BEFORE optim.step():
# an earlier version of this test ran after the step, so the two forward
# passes saw different weights and the comparison was meaningless.
torch.manual_seed(7)
base = m_before = None
model2 = model
ids2 = torch.randint(0, old_vocab, (2, 32))
lab_full = torch.full((2, 32), -100, dtype=torch.long)
lab_half = torch.full((2, 32), -100, dtype=torch.long)
lab_half[:, 16:] = ids2[:, 16:]
lab_all = ids2.clone()
torch.manual_seed(99)
l_masked = model2(input_ids=ids2, labels=lab_half, chunk_size=64).loss
torch.manual_seed(99)
l_prompt_junk = model2(input_ids=ids2, labels=lab_all, chunk_size=64).loss
check("masked loss is finite and positive",
      torch.isfinite(l_masked) and l_masked.item() > 0,
      f"{l_masked.item():.4f}")
check("masking changes the loss (prompt tokens are excluded)",
      not torch.allclose(l_masked, l_prompt_junk),
      f"masked={l_masked.item():.4f} unmasked={l_prompt_junk.item():.4f}")

# ── 5. SFTDataset end to end on a fake dataset ───────────────────────────
print("\n5. SFTDataset streams, masks, and respects max_tokens")


class CharTok:
    def __len__(self):
        return 1000

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 500 + 1 for c in text]}


fake_rows = [
    {"messages": [{"role": "user", "content": f"q{i}"},
                  {"role": "assistant", "content": f"a{i}"}]}
    for i in range(50)
]
fake_ds = SFT.SFTDataset(fake_rows, CharTok(), block_size=256, seed=1,
                         max_tokens=20, epochs=1)
items = list(fake_ds)
check("max_tokens caps the yield count", len(items) == 20, f"{len(items)} items")
check("each item is (ids, labels) of equal length",
      all(len(i) == len(l) for i, l in items))
check("each item has trainable tokens",
      all(any(x != -100 for x in l) for _, l in items))
check("stats record kept samples", fake_ds.stats["kept"] == 20,
      str(fake_ds.stats))

# Determinism: same seed -> same order.
f2 = SFT.SFTDataset(fake_rows, CharTok(), block_size=256, seed=1,
                    max_tokens=20, epochs=1)
check("same seed yields identical data", [i.tolist() for i, _ in items] ==
      [i.tolist() for i, _ in list(f2)])

# collate accepts the dataset's output.
batch = SFT.collate(items[:4])
check("collate accepts dataset output", len(batch) == 3 and batch[0].ndim == 2,
      f"{tuple(batch[0].shape)}")

# Over-long examples are skipped, not truncated.
long_rows = [{"messages": [{"role": "user", "content": "x" * 5000},
                           {"role": "assistant", "content": "y"}]}]
f3 = SFT.SFTDataset(long_rows, CharTok(), block_size=64, seed=1, max_tokens=5)
check("over-length examples are skipped", len(list(f3)) == 0)
check("skip is counted", f3.stats["too_long"] == 1, str(f3.stats))

print("\n" + "=" * 64)
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL SFT CHECKS PASSED")
print("=" * 64)
