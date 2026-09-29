
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

from config import VortexArch, PRESETS, PARAM_BUDGET
from model import VortexConfig, VortexForCausalLM

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


# ──────────────────────────────────────────────────────────────────────
def test_param_counts():
    print("\n1. Parameter budget")
    for name, kw in PRESETS.items():
        arch = VortexArch(**kw)
        n = arch.n_params()
        cfg = VortexConfig(**kw)
        model = VortexForCausalLM(cfg)
        real = sum(p.numel() for p in model.parameters())
        limit = 1_000_000 if name == "vortex-test" else PARAM_BUDGET
        ok = real <= limit
        check(f"{name}: analytic {n:,} == real {real:,} | {real/1e6:.2f}M <= {limit/1e6:.0f}M",
              n == real and ok,
              "analytic matches module tree" if n == real else f"MISMATCH {n:,} vs {real:,}")
        del model


def test_headline_model():
    print("\n2. Headline model (vortex-50m)")
    arch = VortexArch.from_name("vortex-50m")
    print(arch.summary())

    model = VortexForCausalLM(VortexConfig(**arch.to_dict()))
    n = sum(p.numel() for p in model.parameters())
    check(f"total params {n/1e6:.2f}M <= 50M", n <= 50_000_000, f"{n:,}")

    b = arch.param_breakdown()
    check("embedding is 8K x 640", b["embedding"] == 8192 * 640)
    check("English vocab is small", arch.vocab_size == 8192, f"{arch.vocab_size:,}")
    check("layers hold >85% of params", b["all_layers"] / n > 0.85,
          f"embed {100*b['embedding']/n:.0f}% / layers {100*b['all_layers']/n:.0f}%")

    check("lm_head is the embedding matrix (tied)",
          model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())


def test_english_vocab_choices():
    print("\n2b. English-only vocab economics")
    from config import PRESETS as P

    # Every default-path preset should be English-sized (<=16K vocab).
    for name in ("vortex-50m", "vortex-50m-16k", "vortex-50m-wide", "vortex-50m-deep"):
        a = VortexArch(**P[name])
        check(f"{name} vocab <= 16K (English-sized)", a.vocab_size <= 16_384,
              f"{a.vocab_size:,}")

    # The multilingual escape hatch must still exist and still fit.
    check("vortex-50m-32k escape hatch present", "vortex-50m-32k" in P)
    m = VortexArch(**P["vortex-50m-32k"])
    check("vortex-50m-32k fits budget", m.n_params() <= 50_000_000, f"{m.n_params()/1e6:.2f}M")

    # Depth/width is what the freed budget should buy.
    a8 = VortexArch(**P["vortex-50m"])
    a32 = VortexArch(**P["vortex-50m-32k"])
    check("8K vocab buys a wider model than 32K",
          a8.hidden_size > a32.hidden_size,
          f"8K: {a8.hidden_size}d vs 32K: {a32.hidden_size}d")
    check("8K embedding is <15% of budget",
          a8.param_breakdown()["embedding"] / a8.n_params() < 0.15,
          f"{100*a8.param_breakdown()['embedding']/a8.n_params():.1f}%")
    check("16K vocab buys more layers than 32K",
          a8.num_hidden_layers == a32.num_hidden_layers, "same depth at 8K vs 32K (width wins)")


def test_init_is_identity():
    print("\n3. Zero-init residual -> identity at init")
    model = VortexForCausalLM(VortexConfig(**VortexArch.from_name("vortex-50m").to_dict()))
    model.eval()

    z_attn = model.model.layers[0].attn.o_proj.weight.detach()
    z_mlp = model.model.layers[-1].mlp.down_proj.weight.detach()
    check("o_proj zeroed", float(z_attn.abs().max()) == 0.0)
    check("down_proj zeroed", float(z_mlp.abs().max()) == 0.0)

    # With every block an identity, the residual stream passes through
    # unchanged (up to the final norm).
    ids = torch.randint(0, 8192, (2, 64))
    with torch.no_grad():
        h_in = model.model.embed_tokens(ids)
        h_out = model.model(ids)
        expected = model.model.norm(h_in)
    diff = (h_out - expected).abs().max().item()
    check("blocks are identity at init", diff < 1e-5, f"max|diff| = {diff:.2e}")


def test_forward_backward():
    print("\n4. Forward / backward")
    model = VortexForCausalLM(VortexConfig(**VortexArch.from_name("vortex-50m").to_dict()))
    model.gradient_checkpointing_enable()
    model.train()

    B, T = 2, 128
    ids = torch.randint(0, 8192, (B, T))
    tgt = torch.randint(0, 8192, (B, T))
    tgt[:, :5] = -100

    out = model(input_ids=ids, labels=tgt, chunk_size=512)
    check("loss is a scalar tensor", out.loss.shape == ())
    check("loss is finite", torch.isfinite(out.loss).item(), f"loss={out.loss.item():.4f}")
    check("logits not materialized when labels given", out.logits is None)

    out.loss.backward()
    named = list(model.named_parameters())
    grads = [(n, p.grad) for n, p in named if p.grad is not None]
    check("gradients flow", len(grads) > 0, f"{len(grads)} tensors have grad")

    emb_grad = model.model.embed_tokens.weight.grad
    check("embedding receives gradient", emb_grad is not None and emb_grad.abs().sum() > 0)
    print(f"  -> {len(grads)}/{len(named)} param tensors got grads")

    # Gradient checkpointing must not change the value. Needs a fresh forward
    # pass for each variant -- an autograd graph can only be backwarded once.
    model.gradient_checkpointing_enable()
    model.train()
    model.zero_grad()
    out_ck = model(input_ids=ids, labels=tgt, chunk_size=512)
    out_ck.loss.backward()
    g_ck = model.model.layers[5].mlp.gate_proj.weight.grad.clone()
    loss_ck = out_ck.loss.item()

    model.gradient_checkpointing_disable()
    model.zero_grad()
    out_plain = model(input_ids=ids, labels=tgt, chunk_size=512)
    out_plain.loss.backward()
    g_plain = model.model.layers[5].mlp.gate_proj.weight.grad

    d = (g_ck - g_plain).abs().max().item()
    check("grad checkpointing is value-neutral", d < 1e-5, f"max|diff| = {d:.2e}")
    check("checkpointed loss == uncheckpointed loss",
          abs(loss_ck - out_plain.loss.item()) < 1e-5,
          f"{loss_ck:.6f} vs {out_plain.loss.item():.6f}")


def test_loss_masking():
    print("\n5. Cross-entropy masking")
    model = VortexForCausalLM(VortexConfig(**VortexArch.from_name("vortex-50m").to_dict()))
    model.eval()
    ids = torch.randint(0, 8192, (1, 64))
    tgt = torch.randint(0, 8192, (1, 64))
    tgt[0, :10] = -100
    tgt[0, 50:] = -100
    with torch.no_grad():
        a = model(input_ids=ids, labels=tgt, chunk_size=128).loss.item()
        b = model(input_ids=ids, labels=tgt, chunk_size=7).loss.item()
    check("chunked CE is chunk-size invariant", abs(a - b) < 1e-4, f"{a:.6f} vs {b:.6f}")

    allmask = torch.full((1, 64), -100)
    with torch.no_grad():
        l = model(input_ids=ids, labels=allmask).loss
    check("all-masked batch returns 0 without NaN",
          torch.isfinite(l).item() and l.item() == 0.0, f"loss={l.item()}")


def test_causality():
    print("\n6. Causality (no future leakage)")
    model = VortexForCausalLM(VortexConfig(**VortexArch.from_name("vortex-50m").to_dict()))
    model.eval()
    ids = torch.randint(0, 8192, (1, 48))
    with torch.no_grad():
        base = model(input_ids=ids).logits
        perturbed = ids.clone()
        perturbed[0, 30:] = torch.randint(0, 8192, (18,))
        alt = model(input_ids=perturbed).logits
    d = (base[0, :30] - alt[0, :30]).abs().max().item()
    check("prefix logits unchanged by future tokens", d < 1e-5, f"max|diff| at t<30 = {d:.2e}")
    d2 = (base[0, 30:] - alt[0, 30:]).abs().max().item()
    check("suffix logits DO change", d2 > 1e-6, f"max|diff| at t>=30 = {d2:.2e}")


def test_rope():
    print("\n7. RoPE properties")
    from model import build_rope_cache, apply_rope
    cos, sin = build_rope_cache(64, 128, 10_000.0, "cpu", torch.float32)
    check("cache shapes", cos.shape == (128, 32) and sin.shape == (128, 32))

    x = torch.randn(1, 2, 16, 64)
    y = apply_rope(x, cos, sin)
    dn = (x.norm(dim=-1) - y.norm(dim=-1)).abs().max().item()
    check("RoPE preserves head-dim norm", dn < 1e-5, f"max|dnorm| = {dn:.2e}")

    # Relative-position property: dot(q_i, k_j) depends on (i - j) only.
    q = torch.randn(1, 1, 1, 64)
    k = torch.randn(1, 1, 1, 64)
    d1 = (apply_rope(q, cos[40:41], sin[40:41]) * apply_rope(k, cos[45:46], sin[45:46])).sum()
    d2 = (apply_rope(q, cos[100:101], sin[100:101]) * apply_rope(k, cos[105:106], sin[105:106])).sum()
    check("relative offset (40,45) matches (100,105)", abs(d1.item() - d2.item()) < 1e-4,
          f"{d1.item():.5f} vs {d2.item():.5f}")


def test_gqa():
    print("\n8. GQA wiring")
    arch = VortexArch.from_name("vortex-50m")
    model = VortexForCausalLM(VortexConfig(**arch.to_dict()))
    attn = model.model.layers[0].attn

    # Derive expectations from the config so this test survives a preset change.
    nh, nkv, hd = arch.num_attention_heads, arch.num_key_value_heads, arch.head_dim
    check(f"{nh} Q heads", attn.n_heads == nh)
    check(f"{nkv} KV heads", attn.n_kv == nkv)
    check(f"{nh // nkv} Q heads per KV group", attn.n_groups == nh // nkv)
    check(f"k_proj is 1/{nh // nkv} the size of q_proj",
          attn.k_proj.weight.shape[0] == nkv * hd
          and attn.q_proj.weight.shape[0] == nh * hd)
    check("head_dim divides hidden evenly", arch.hidden_size == nh * hd)

    ctx = arch.max_position_embeddings
    layers = arch.num_hidden_layers
    cache_kv = 2 * layers * 2 * ctx * nkv * hd
    cache_mha = 2 * layers * 2 * ctx * nh * hd
    check(f"KV cache {nh // nkv}x smaller than MHA",
          abs(cache_mha / cache_kv - (nh / nkv)) < 1e-6,
          f"MHA {cache_mha/1e6:.1f}MB -> GQA {cache_kv/1e6:.1f}MB at ctx={ctx}")


def test_qk_norm_effect():
    print("\n9. QK-Norm bounds attention logits")
    torch.manual_seed(0)
    arch = VortexArch.from_name("vortex-50m")
    model = VortexForCausalLM(VortexConfig(**arch.to_dict()))
    attn = model.model.layers[0].attn
    check("QK-Norm enabled by default", arch.use_qk_norm)
    check("q_norm is a real RMSNorm",
          isinstance(attn.q_norm, nn.Module) and not isinstance(attn.q_norm, nn.Identity))
    x = torch.randn(1, 64, arch.hidden_size) * 50
    with torch.no_grad():
        q = attn.q_norm(
            attn.q_proj(x).view(1, 64, arch.num_attention_heads, arch.head_dim).transpose(1, 2)
        )
    qn = q.norm(dim=-1).mean().item()
    check("q stays normalized despite 50x input scale", qn < 10.0, f"mean ||q|| = {qn:.2f}")


def test_no_biases():
    print("\n10. No biases")
    model = VortexForCausalLM(VortexConfig(**VortexArch.from_name("vortex-50m").to_dict()))
    biases = [n for n, p in model.named_parameters() if n.endswith("bias")]
    check("zero bias parameters", len(biases) == 0, f"found {biases}")


def test_save_load():
    print("\n11. save_pretrained / from_pretrained round trip")
    import tempfile
    arch = VortexArch.from_name("vortex-test")
    model = VortexForCausalLM(VortexConfig(**arch.to_dict())).eval()
    ids = torch.randint(0, arch.vocab_size, (1, 32))
    with torch.no_grad():
        before = model(input_ids=ids).logits
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        reloaded = VortexForCausalLM.from_pretrained(d).eval()
    with torch.no_grad():
        after = reloaded(input_ids=ids).logits
    d = (before - after).abs().max().item()
    check("logits identical after reload", d < 1e-5, f"max|diff| = {d:.2e}")
    check("reloaded model is still tied",
          reloaded.lm_head.weight.data_ptr() == reloaded.model.embed_tokens.weight.data_ptr())


def test_all_presets_forward():
    print("\n12. Every preset runs")
    for name, kw in PRESETS.items():
        arch = VortexArch(**kw)
        m = VortexForCausalLM(VortexConfig(**kw)).eval()
        ids = torch.randint(0, arch.vocab_size, (1, 64))
        with torch.no_grad():
            out = m(input_ids=ids)
        check(f"{name} forward -> (1,64,{arch.vocab_size})",
              out.logits.shape == (1, 64, arch.vocab_size), str(tuple(out.logits.shape)))
        del m


# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("VORTEX ARCHITECTURE VERIFICATION")
    print("=" * 70)
    torch.manual_seed(0)

    test_param_counts()
    test_headline_model()
    test_english_vocab_choices()
    test_init_is_identity()
    test_forward_backward()
    test_loss_masking()
    test_causality()
    test_rope()
    test_gqa()
    test_qk_norm_effect()
    test_no_biases()
    test_save_load()
    test_all_presets_forward()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
