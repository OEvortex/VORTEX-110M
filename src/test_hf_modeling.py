"""Parity + HF-integration checks for modeling_vortex.py.

Run: python src/test_hf_modeling.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HERE = os.path.dirname(os.path.abspath(__file__))
# Source of the two remote-code modules, read verbatim so the `auto_map` test
# exercises the exact files that ship.
REMOTE_CONFIG_SRC = open(os.path.join(_HERE, "configuration_vortex.py")).read()
REMOTE_MODELING_SRC = open(os.path.join(_HERE, "modeling_vortex.py")).read()

from configuration_vortex import VortexConfig as HFConfig
from modeling_vortex import VortexForCausalLM as HFCausalLM
from modeling_vortex import VortexConfig, VortexModel

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}  {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def small_cfg(**kw):
    base = dict(vocab_size=256, hidden_size=64, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2,
                intermediate_size=128, max_position_embeddings=128)
    base.update(kw)
    return VortexConfig(**base)


def randomize(m):
    """Break the zero-init residuals so the two impls must actually agree."""
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.05, generator=g)
    return m.eval()


# ──────────────────────────────────────────────────────────────────────
print("\n1. Numerical parity with the training module (src/model.py)")
# ──────────────────────────────────────────────────────────────────────
import model as legacy

torch.manual_seed(0)
old = randomize(legacy.VortexForCausalLM(legacy.VortexConfig(**{
    "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 3,
    "num_attention_heads": 4, "num_key_value_heads": 2,
    "intermediate_size": 128, "max_position_embeddings": 128,
})))
new = randomize(HFCausalLM(small_cfg()))

old_keys, new_keys = set(old.state_dict()), set(new.state_dict())
check("state_dict keys identical (checkpoints load unchanged)",
      old_keys == new_keys,
      f"only-old={sorted(old_keys - new_keys)[:3]} only-new={sorted(new_keys - old_keys)[:3]}")

# Transfer the exact same weights into the new implementation.
missing, unexpected = new.load_state_dict(old.state_dict(), strict=True), None
check("load_state_dict(strict=True) from legacy weights", True)

ids = torch.randint(0, 256, (2, 37))
with torch.no_grad():
    a = old(input_ids=ids, labels=None, chunk_size=0).logits
    b = new(input_ids=ids, labels=None).logits
d = (a - b).abs().max().item()
check("forward logits match legacy", d < 1e-5, f"max|diff| = {d:.2e}")

la = old(input_ids=ids, labels=ids, chunk_size=8).loss
lb = new(input_ids=ids, labels=ids, chunk_size=8).loss
d = (la - lb).abs().item()
check("chunked loss matches legacy", d < 1e-5, f"{la.item():.6f} vs {lb.item():.6f} (diff {d:.2e})")

# ──────────────────────────────────────────────────────────────────────
print("\n2. Zero-init residual identity (the initial-loss guarantee)")
# ──────────────────────────────────────────────────────────────────────
fresh = HFCausalLM(small_cfg())
with torch.no_grad():
    h = fresh.model.embed_tokens(ids)
    residual = h.clone()
    for blk in fresh.model.layers:
        h = blk(h, position_embeddings=fresh.model.rotary_emb(h, ids.shape[1]))
# Every block is an identity at init, so the stack must return the embedding
# untouched. This is what makes the initial loss ln(vocab_size) instead of the
# hundreds that saturated default init produces.
check("untrained stack is a bit-exact passthrough",
      torch.equal(h, residual), f"max|diff| = {(h - residual).abs().max().item():.2e}")
check("zero-init residual outputs are exactly zero",
      all(torch.count_nonzero(b.attn.o_proj.weight) == 0
          and torch.count_nonzero(b.mlp.down_proj.weight) == 0
          for b in fresh.model.layers))
# With blocks as identities, the initial loss is ln(vocab_size) up to the
# embedding init -- near-uniform but not exactly uniform, so the bound is
# "close to", not "equal to". Saturating default init lands in the hundreds.
with torch.no_grad():
    init_loss = HFCausalLM(small_cfg(vocab_size=16_384))(input_ids=ids, labels=ids).loss.item()
check("initial loss is near ln(vocab_size), not saturated",
      abs(init_loss - math.log(16_384)) < 0.5,
      f"loss={init_loss:.4f}, ln(16384)={math.log(16_384):.4f}")

# ──────────────────────────────────────────────────────────────────────
print("\n3. KV cache: incremental decode == full forward")
# ──────────────────────────────────────────────────────────────────────
m = HFCausalLM(small_cfg()).eval()
ids = torch.randint(0, 256, (2, 24))
with torch.no_grad():
    full = m(input_ids=ids, use_cache=False).logits

    # Feed the prompt, then decode one token at a time from the cache.
    prefill = m(input_ids=ids[:, :-1], use_cache=True)
    cache = prefill.past_key_values
    steps = [prefill.logits[:, -1]]
    for t in range(ids.shape[1] - 1, ids.shape[1]):
        nxt = ids[:, t:t + 1]
        cache_seen = cache.get_seq_length()
        out = m(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        steps.append(out.logits[:, -1])
    inc = torch.stack(steps, dim=1)

d = (full[:, -1 - (len(steps) - 1):] - inc).abs().max().item()
check("cached decode matches full forward", d < 1e-4, f"max|diff| = {d:.2e}")
check("cache grew by exactly one token per step",
      cache.get_seq_length() == ids.shape[1], f"len={cache.get_seq_length()}")

# ──────────────────────────────────────────────────────────────────────
print("\n4. Causality under a cache")
# ──────────────────────────────────────────────────────────────────────
with torch.no_grad():
    pre = m(input_ids=ids[:, :10], use_cache=True)
    logit_a = m(input_ids=ids[:, 10:11], past_key_values=pre.past_key_values, use_cache=True).logits
    # Same past, different next token -> different continuation. Each comparison
    # gets a fresh cache: `Cache.update` mutates in place, so reusing one would
    # grow it under the second call and silently test the wrong thing.
    pre2 = m(input_ids=ids[:, :10], use_cache=True)
    logit_b = m(input_ids=torch.zeros_like(ids[:, 10:11]), past_key_values=pre2.past_key_values,
                use_cache=True).logits
check("cached step responds to the new token",
      (logit_a - logit_b).abs().max().item() > 1e-6,
      f"max|diff| = {(logit_a - logit_b).abs().max().item():.2e}")

# A cached step must not see the future: extending the query block must not
# change the logits at a position already inside it. This is exactly the case
# top-left alignment gets wrong and the bottom-right-aligned mask fixes.
with torch.no_grad():
    c3 = m(input_ids=ids[:, :10], use_cache=True).past_key_values
    block = m(input_ids=ids[:, 10:14], past_key_values=c3, use_cache=True).logits
    c4 = m(input_ids=ids[:, :10], use_cache=True).past_key_values
    one = m(input_ids=ids[:, 10:11], past_key_values=c4, use_cache=True).logits
d = (block[:, 0] - one[:, 0]).abs().max().item()
check("prefix of a cached block is future-independent", d < 1e-4, f"max|diff| = {d:.2e}")

# A multi-token block fed against a cache must agree with the same positions
# computed in one shot.
with torch.no_grad():
    ref_block = m(input_ids=ids, use_cache=False).logits[:, 10:14]
d = (block - ref_block).abs().max().item()
check("multi-token cached block matches single-shot forward", d < 1e-4, f"max|diff| = {d:.2e}")

# ──────────────────────────────────────────────────────────────────────
print("\n5. Padding: left-padded batches give the same logits as unpadded")
# ──────────────────────────────────────────────────────────────────────
m2 = HFCausalLM(small_cfg()).eval()
real = torch.randint(0, 256, (1, 12))
pad = torch.zeros(1, 5, dtype=torch.long)
padded_ids = torch.cat([pad, real], dim=1)
padded_mask = torch.cat([torch.zeros(1, 5, dtype=torch.long), torch.ones(1, 12, dtype=torch.long)], dim=1)

with torch.no_grad():
    ref = m2(input_ids=real).logits
    # One prefill over the padded block, then decode. This is the path generate()
    # takes, and it exercises the bottom-right-aligned mask.
    p = m2(input_ids=padded_ids, attention_mask=padded_mask, use_cache=True)
    got = p.logits[:, -12:]
d = (ref - got).abs().max().item()
check("left-padded prefill matches unpadded", d < 1e-4, f"max|diff| = {d:.2e}")
check("no NaN in padded prefill", torch.isfinite(p.logits).all().item())

# NaN must not survive into the next cached step either.
with torch.no_grad():
    step = m2(input_ids=torch.full((1, 1), 7, dtype=torch.long),
              attention_mask=torch.cat([padded_mask, torch.ones(1, 1, dtype=torch.long)], dim=1),
              past_key_values=p.past_key_values, use_cache=True)
check("no NaN after a cached step on a padded batch", torch.isfinite(step.logits).all().item())

# ──────────────────────────────────────────────────────────────────────
print("\n6. generate()")
# ──────────────────────────────────────────────────────────────────────
m2.config.pad_token_id = 0
m2.config.bos_token_id = 1
m2.config.eos_token_id = 2
m2.generation_config.pad_token_id = 0
m2.generation_config.bos_token_id = 1
m2.generation_config.eos_token_id = 2

greedy = m2.generate(ids, max_new_tokens=8, do_sample=False)
check("greedy generate produces the requested length",
      greedy.shape == (2, 24 + 8), f"shape {tuple(greedy.shape)}")

# Greedy decode must agree with a manual argmax loop over the cache.
manual = ids.clone()
with torch.no_grad():
    c = m2(input_ids=manual, use_cache=True).past_key_values
    for _ in range(8):
        nxt = m2(input_ids=manual[:, -1:], past_key_values=c, use_cache=True)
        c = nxt.past_key_values
        manual = torch.cat([manual, nxt.logits[:, -1].argmax(-1, keepdim=True)], dim=1)
check("generate() == manual cached argmax loop",
      torch.equal(greedy, manual), "")

sampled = m2.generate(ids, max_new_tokens=8, do_sample=True, top_k=5, temperature=0.8)
check("sampled generate runs and returns right shape",
      sampled.shape == (2, 32), f"shape {tuple(sampled.shape)}")

batched = m2.generate(padded_ids, attention_mask=padded_mask, max_new_tokens=8, do_sample=False)
check("generate() on a left-padded batch",
      batched.shape == (1, 17 + 8), f"shape {tuple(batched.shape)}")

# ──────────────────────────────────────────────────────────────────────
print("\n7. save_pretrained / from_pretrained round trip")
# ──────────────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as d:
    m2.save_pretrained(d, safe_serialization=True)
    check("config.json written", os.path.exists(os.path.join(d, "config.json")))
    check("model.safetensors written", os.path.exists(os.path.join(d, "model.safetensors")))
    reloaded = HFCausalLM.from_pretrained(d).eval()
    with torch.no_grad():
        a = m2(input_ids=ids).logits
        b = reloaded(input_ids=ids).logits
    check("round trip is bit-exact", torch.equal(a, b),
          f"max|diff| = {(a - b).abs().max().item():.2e}")
    check("reloaded logits are finite (non-persistent buffers would be garbage)",
          torch.isfinite(b).all().item())
    # The RoPE table is derived state, not a checkpoint tensor. If it is ever
    # stored as a non-persistent buffer, transformers materialises it from
    # uninitialised memory on load and the model still "works" while producing
    # pure noise. Assert the frequency table itself round-trips.
    exp_freq = 1.0 / (10000.0 ** (torch.arange(0, reloaded.config.head_dim, 2).float() / reloaded.config.head_dim))
    cos, sin = reloaded.model.rotary_emb(b, 24)
    check("RoPE inv_freq is correct after reload",
          torch.allclose(reloaded.model.rotary_emb._get_inv_freq(torch.device("cpu")), exp_freq),
          f"first={reloaded.model.rotary_emb._get_inv_freq(torch.device('cpu'))[0]:.6f} (want 1.0)")
    check("RoPE cos/sin are finite after reload",
          torch.isfinite(cos).all().item() and torch.isfinite(sin).all().item())
    check("weights stay tied after reload",
          reloaded.lm_head.weight.data_ptr() == reloaded.model.embed_tokens.weight.data_ptr())
    check("no missing keys on reload",
          not reloaded.load_state_dict(reloaded.state_dict(), strict=True).missing_keys)

# ──────────────────────────────────────────────────────────────────────
print("\n8. Auto classes + logits_to_keep")
# ──────────────────────────────────────────────────────────────────────
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("vortex", VortexConfig)
AutoModelForCausalLM.register(VortexConfig, HFCausalLM)

with tempfile.TemporaryDirectory() as d:
    m2.save_pretrained(d, safe_serialization=True)
    ac = AutoConfig.from_pretrained(d)
    am = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True).eval()
    check("AutoConfig.from_pretrained returns VortexConfig", isinstance(ac, VortexConfig))
    check("AutoModelForCausalLM.from_pretrained returns VortexForCausalLM",
          isinstance(am, HFCausalLM), type(am).__name__)
    with torch.no_grad():
        diff = (am(input_ids=ids).logits - m2(input_ids=ids).logits).abs().max().item()
    check("Auto-loaded model reproduces logits", diff < 1e-6, f"max|diff| = {diff:.2e}")

# `trust_remote_code` is the path a Hub repo actually takes: transformers copies
# the two modules into its own dynamic-module directory and imports them there,
# which only works if they are self-contained.
with tempfile.TemporaryDirectory() as d:
    m2.save_pretrained(d, safe_serialization=True)
    with open(os.path.join(d, "configuration_vortex.py"), "w") as f:
        f.write(REMOTE_CONFIG_SRC)
    with open(os.path.join(d, "modeling_vortex.py"), "w") as f:
        f.write(REMOTE_MODELING_SRC)
    cfg_json = json.load(open(os.path.join(d, "config.json")))
    cfg_json["auto_map"] = {
        "AutoConfig": "configuration_vortex.VortexConfig",
        "AutoModelForCausalLM": "modeling_vortex.VortexForCausalLM",
    }
    json.dump(cfg_json, open(os.path.join(d, "config.json"), "w"))

    remote = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True).eval()
    check("loads via auto_map + trust_remote_code (self-contained modules)",
          remote.__class__.__name__ == "VortexForCausalLM", remote.__class__.__name__)
    with torch.no_grad():
        diff = (remote(input_ids=ids).logits - m2(input_ids=ids).logits).abs().max().item()
    check("remote-code model reproduces logits", diff < 1e-6, f"max|diff| = {diff:.2e}")
    remote.config.pad_token_id = 0
    remote.generation_config.pad_token_id = 0
    remote.generation_config.eos_token_id = 2
    out = remote.generate(ids, max_new_tokens=4, do_sample=False)
    check("remote-code model generates", out.shape == (2, 28), f"shape {tuple(out.shape)}")

with torch.no_grad():
    out = m2(input_ids=ids, logits_to_keep=1)
check("logits_to_keep=1 returns one position", out.logits.shape == (2, 1, 256),
      f"shape {tuple(out.logits.shape)}")
check("logits_to_keep=1 slice matches full logits",
      (out.logits[:, 0] - m2(input_ids=ids).logits[:, -1]).abs().max().item() < 1e-6)

with torch.no_grad():
    lab = m2(input_ids=ids, labels=ids)
check("labels= still withholds logits (memory win)", lab.logits is None)
check("labels= still returns a finite loss", torch.isfinite(lab.loss).item(), f"loss={lab.loss.item():.4f}")
with torch.no_grad():
    both = m2(input_ids=ids, labels=ids, logits_to_keep=1)
check("labels= + logits_to_keep gives both", both.logits is not None and torch.isfinite(both.loss).item())

# ──────────────────────────────────────────────────────────────────────
print("\n9. Gradient checkpointing value-neutrality")
# ──────────────────────────────────────────────────────────────────────
m3 = HFCausalLM(small_cfg())
m3.train()
m3.gradient_checkpointing_enable()
check("enable() flags VortexModel", m3.model.gradient_checkpointing is True)
with torch.no_grad():
    m3.eval()
    off = m3(input_ids=ids).logits
    m3.train()
    on = m3(input_ids=ids).logits
check("checkpointed forward matches uncheckpointed",
      (on - off).abs().max().item() < 1e-5, f"max|diff| = {(on - off).abs().max().item():.2e}")

m3.train()
out = m3(input_ids=ids, labels=ids)
check("checkpointed loss graph is connected", out.loss.requires_grad, "")
out.loss.backward()
check("gradients reach every parameter",
      all(p.grad is not None for p in m3.parameters() if p.requires_grad))
m3.gradient_checkpointing_disable()
check("disable() clears the flag", m3.model.gradient_checkpointing is False)

# Regression: the checkpoint recompute must not re-append to the KV cache.
# `Cache.update` mutates in place, so a cached forward under checkpointing
# doubles the cache during backward and corrupts every layer above it.
m5 = HFCausalLM(small_cfg())
m5.train()
m5.gradient_checkpointing_enable()
out = m5(input_ids=ids, labels=ids, use_cache=True)
check("checkpointed training forwards return no cache",
      out.past_key_values is None, "")
out.loss.backward()
check("checkpointed backward does not raise a recompute error", True)

# Gradients must still be right with checkpointing on, not merely present.
m6 = HFCausalLM(small_cfg()).eval()
m6.train()
plain = m6(input_ids=ids, labels=ids).loss
plain.backward()
g_plain = {n: p.grad.clone() for n, p in m6.named_parameters() if p.grad is not None}
m6.zero_grad()
m6.gradient_checkpointing_enable()
ckpt = m6(input_ids=ids, labels=ids).loss
ckpt.backward()
g_ckpt = {n: p.grad.clone() for n, p in m6.named_parameters() if p.grad is not None}
check("checkpointed gradients match uncheckpointed",
      set(g_plain) == set(g_ckpt)
      and max((g_plain[n] - g_ckpt[n]).abs().max().item() for n in g_plain) < 1e-5,
      f"max|diff| = {max((g_plain[n] - g_ckpt[n]).abs().max().item() for n in g_plain):.2e}")

# ──────────────────────────────────────────────────────────────────────
print("\n10. resize_token_embeddings")
# ──────────────────────────────────────────────────────────────────────
m4 = HFCausalLM(small_cfg())
before = m4.model.embed_tokens.weight[:256].clone()
m4.resize_token_embeddings(300)
check("table grew", m4.model.embed_tokens.weight.shape[0] == 300)
check("config vocab_size followed", m4.config.vocab_size == 300)
check("old rows preserved",
      torch.equal(m4.model.embed_tokens.weight[:256], before))
check("head re-tied after resize",
      m4.lm_head.weight.data_ptr() == m4.model.embed_tokens.weight.data_ptr())
try:
    m4.resize_token_embeddings(100)
    check("shrink is refused", False, "no error raised")
except ValueError as e:
    check("shrink is refused", True, f"ValueError: {str(e)[:48]}...")

# ──────────────────────────────────────────────────────────────────────
print("\n11. Config validation")
# ──────────────────────────────────────────────────────────────────────
for name, kw in [
    ("GQA divisibility", dict(num_attention_heads=8, num_key_value_heads=3, hidden_size=64)),
    ("odd head_dim", dict(hidden_size=60, num_attention_heads=4)),
    ("hidden not divisible", dict(hidden_size=100, num_attention_heads=8)),
    ("kv > q heads", dict(num_attention_heads=4, num_key_value_heads=8, hidden_size=64)),
]:
    bad = dict(vocab_size=256, hidden_size=64, num_hidden_layers=1,
               num_attention_heads=4, num_key_value_heads=2, intermediate_size=128)
    bad.update(kw)
    try:
        VortexConfig(**bad)
        check(f"rejects {name}", False, "no error raised")
    except ValueError as e:
        check(f"rejects {name}", True, f"ValueError: {str(e)[:52]}")

cfg = small_cfg()
check("head_dim derived", cfg.head_dim == 16, f"{cfg.head_dim}")
check("num_query_groups derived", cfg.num_query_groups == 2, f"{cfg.num_query_groups}")

# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print(f"  {PASS} passed, {FAIL} failed")
print("=" * 64)
sys.exit(1 if FAIL else 0)
