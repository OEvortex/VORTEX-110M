
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional


# ──────────────────────────────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────────────────────────────
@dataclass
class VortexArch:

    # ── Shape ────────────────────────────────────────────────────────
    # Defaults mirror the `vortex-50m-16k` preset: 16,384 English tokens.
    vocab_size: int = 16_384
    hidden_size: int = 512
    num_hidden_layers: int = 18
    num_attention_heads: int = 8          # 8 x 64 = 512 = hidden_size
    num_key_value_heads: int = 2          # GQA: 2 KV heads serve 8 Q heads
    intermediate_size: int = 1_072        # ~2.09x hidden

    # ── Norms / positions ─────────────────────────────────────────────
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    max_position_embeddings: int = 2_048
    use_qk_norm: bool = True              # per-head RMSNorm on q, k
    tie_word_embeddings: bool = True
    zero_init_residual: bool = True       # zero-init o_proj & down_proj

    # ── Init / misc ──────────────────────────────────────────────────
    initializer_range: float = 0.02
    use_cache: bool = True
    # Residual output-projection scaling (GPT-2 style 1/sqrt(2L)). The
    # default below is the modern "unscaled" choice, which is safe when
    # paired with zero-init residuals.
    scale_residual: bool = False
    rope_interleaved: bool = True         # GPT-NeoX split-half vs interleaved

    # ── Identity ─────────────────────────────────────────────────────
    model_type: str = "vortex"
    name_or_path: str = "vortex-50m-16k"

    # ── Derived ──────────────────────────────────────────────────────
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_attention_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def n_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    # ── Param accounting ─────────────────────────────────────────────
    def n_params(self) -> int:
        hs = self.hidden_size
        nh = self.num_attention_heads
        nkv = self.num_key_value_heads
        hd = self.head_dim
        inter = self.intermediate_size

        # attention: q, k, v, o (no biases) + optional QK-Norm weights
        attn = hs * (nh * hd) + 2 * (hs * nkv * hd) + (nh * hd) * hs
        if self.use_qk_norm:
            attn += hd + hd                     # q_norm.weight, k_norm.weight

        # SwiGLU MLP: gate, up, down (no biases)
        mlp = 3 * hs * inter

        # two RMSNorm weight vectors per block
        block = attn + mlp + 2 * hs

        n = self.vocab_size * hs             # token embedding
        n += self.num_hidden_layers * block
        n += hs                              # final norm
        if not self.tie_word_embeddings:
            n += self.vocab_size * hs         # untied lm_head
        return n

    def param_breakdown(self) -> Dict[str, int]:
        hs = self.hidden_size
        nh = self.num_attention_heads
        nkv = self.num_key_value_heads
        hd = self.head_dim
        inter = self.intermediate_size
        attn = hs * (nh * hd) + 2 * (hs * nkv * hd) + (nh * hd) * hs
        if self.use_qk_norm:
            attn += 2 * hd
        mlp = 3 * hs * inter
        block = attn + mlp + 2 * hs
        embed = self.vocab_size * hs
        return {
            "embedding": embed,
            "per_layer_attn": attn,
            "per_layer_mlp": mlp,
            "per_layer_norms": 2 * hs,
            "all_layers": self.num_hidden_layers * block,
            "final_norm": hs,
            "lm_head": 0 if self.tie_word_embeddings else embed,
            "total": self.n_params(),
        }

    def summary(self) -> str:
        b = self.param_breakdown()
        return "\n".join([
            f"{self.name_or_path}  ({b['total']/1e6:.2f}M params, budget 50.00M)",
            f"  shape      hidden={self.hidden_size} layers={self.num_hidden_layers} "
            f"heads={self.num_attention_heads}q/{self.num_key_value_heads}kv "
            f"head_dim={self.head_dim} inter={self.intermediate_size}",
            f"  vocab      {self.vocab_size:,}  (tied={self.tie_word_embeddings})",
            f"  ctx        {self.max_position_embeddings:,}  rope_theta={self.rope_theta:,.0f}",
            f"  features   qk_norm={self.use_qk_norm} zero_init_residual={self.zero_init_residual} biases=none",
            f"  budget     embed {b['embedding']/1e6:.2f}M ({100*b['embedding']/b['total']:.0f}%)  "
            f"layers {b['all_layers']/1e6:.2f}M ({100*b['all_layers']/b['total']:.0f}%)",
        ])

    def __post_init__(self):
        # Fail loudly on an illegal shape at construction time rather than
        # crashing deep inside SDPA (e.g. "heads in key and value must divide
        # the number of heads in query") three layers into a forward pass.
        self.validate()

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} not divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        if self.num_key_value_heads < 1:
            raise ValueError("num_key_value_heads must be >= 1")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} not divisible by "
                f"num_key_value_heads {self.num_key_value_heads} "
                f"(GQA needs whole query groups; {self.num_attention_heads} heads "
                f"over {self.num_key_value_heads} kv heads)"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim {self.head_dim} must be even for RoPE")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads {self.num_key_value_heads} exceeds "
                f"num_attention_heads {self.num_attention_heads}"
            )

    @classmethod
    def from_name(cls, name: str) -> "VortexArch":
        key = name.lower().replace("_", "-")
        if key not in PRESETS:
            raise KeyError(f"Unknown preset {name!r}. Available: {sorted(PRESETS)}")
        return cls(**PRESETS[key])

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# Preset library  (ENGLISH-ONLY CORPUS)
# ──────────────────────────────────────────────────────────────────────
# With a tied embedding table, `vocab_size` is the single biggest lever on the
# parameter budget -- and the constraint FLIPS as the vocab shrinks:
#
#   * at 32K the table eats 34% of the budget, so hidden size is the limit
#   * at 8K  the table is only 13%, so LAYERS are what the budget buys
#
# Shrinking the vocab therefore does not just "save" parameters, it converts
# them into depth and width. The 8K preset spends them on a wider hidden
# (640d) and the 16K preset on more layers (18 vs 12).
#
# 8K-16K is also the proven band for English-only models at this scale
# (TinyStories trains 1M-33M English models on a 10K vocab).
PRESETS: Dict[str, dict] = {
    # ── English-only defaults ─────────────────────────────────────────
    # 49,993,248 params. Vocab 8,192 frees enough budget to widen to 640d.
    "vortex-50m": dict(
        vocab_size=8_192, hidden_size=640, num_hidden_layers=12,
        num_attention_heads=10, num_key_value_heads=2, intermediate_size=1_408,
        max_position_embeddings=2_048, name_or_path="vortex-50m",
    ),
    # 49,837,632 params. Deeper at 16K: 18 layers instead of 12. Better if you
    # want the bigger vocab for margin on rare words / names / numbers.
    # intermediate 1072 = 2.09x hidden; 1120 (2.19x) would overshoot 50M.
    "vortex-50m-16k": dict(
        vocab_size=16_384, hidden_size=512, num_hidden_layers=18,
        num_attention_heads=8, num_key_value_heads=2, intermediate_size=1_072,
        max_position_embeddings=2_048, name_or_path="vortex-50m-16k",
    ),
    # 48,527,232 params. Widest option -- 768d x 8L. Good if your data is
    # short documents where per-token width beats depth.
    "vortex-50m-wide": dict(
        vocab_size=8_192, hidden_size=768, num_hidden_layers=8,
        num_attention_heads=12, num_key_value_heads=4, intermediate_size=1_608,
        max_position_embeddings=2_048, name_or_path="vortex-50m-wide",
    ),
    # 49,837,632 params. Deepest -- 512d x 18L at 16K, with 4K context. Best
    # when you have lots of tokens and want strong multi-step reasoning.
    "vortex-50m-deep": dict(
        vocab_size=16_384, hidden_size=512, num_hidden_layers=18,
        num_attention_heads=8, num_key_value_heads=2, intermediate_size=1_072,
        max_position_embeddings=4_096, name_or_path="vortex-50m-deep",
    ),
    # 44,062,080 params. Deliberately under budget -- room to grow the vocab
    # to ~22K later without touching the architecture.
    "vortex-40m": dict(
        vocab_size=16_384, hidden_size=384, num_hidden_layers=24,
        num_attention_heads=6, num_key_value_heads=2, intermediate_size=1_024,
        max_position_embeddings=2_048, name_or_path="vortex-40m",
    ),

    # ── Multilingual / code fallback ───────────────────────────────────
    # Only needed if the corpus is EVER mixed-language or code. 32K was the
    # right default for Qwen-style multilingual text; for English-only it
    # wastes 34% of the budget on a lookup table.
    "vortex-50m-32k": dict(
        vocab_size=32_768, hidden_size=512, num_hidden_layers=12,
        num_attention_heads=8, num_key_value_heads=2, intermediate_size=1_344,
        max_position_embeddings=2_048, name_or_path="vortex-50m-32k",
    ),

    # Deliberately tiny, for pipeline / smoke tests.
    "vortex-test": dict(
        vocab_size=1_024, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=344,
        max_position_embeddings=512, name_or_path="vortex-test",
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Tokenizer profile
# ──────────────────────────────────────────────────────────────────────
DEFAULT_TOKENIZER_ID = "VTXAI/vortex-tok-16k"


@dataclass
class TokenizerProfile:
    tokenizer_id: str = DEFAULT_TOKENIZER_ID
    vocab_size: int = 16_384
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    chat_template: Optional[str] = None

    @classmethod
    def from_pretrained(cls, tokenizer_id: Optional[str] = None,
                        hub_token: Optional[str] = None) -> "TokenizerProfile":
        from transformers import AutoTokenizer

        tok_id = tokenizer_id or os.environ.get("VORTEX_TOKENIZER") or DEFAULT_TOKENIZER_ID
        tok = AutoTokenizer.from_pretrained(
            tok_id, token=os.environ.get("HF_TOKEN") or hub_token
        )
        # Effective vocab = the highest id the tokenizer can EMIT, plus one.
        # This must be a size, not an index, so no `+ 1` beyond this point.
        #
        # The old form was `max(len(tok), *ids) + 1`, which over-counted by
        # one: for a 16K vocab whose highest actual id is 16,383, it reported
        # 16,385. That is not cosmetic -- pretrain.py adopts this number
        # verbatim, so the embedding table was allocated 512 phantom rows
        # (16,385 x 512) that no token can ever reach, and the param count
        # was wrong. It also made a correct checkpoint look like a vocab
        # mismatch to the eval guard.
        #
        # `len(tok)` already counts the added special tokens
        # (<|pad|>..<|unk|>, ids 0-3), so it is the floor; the max over real
        # ids guards the case where a special token sits above len(tok).
        base = tok.get_vocab()
        ids = [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id]
        ids = [i for i in ids if i is not None]
        eff_vocab = max([len(tok), len(base)] + ids + [0])
        return cls(
            tokenizer_id=tok_id,
            vocab_size=eff_vocab,
            bos_token_id=tok.bos_token_id if tok.bos_token_id is not None else 1,
            eos_token_id=tok.eos_token_id if tok.eos_token_id is not None else 2,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
            chat_template=getattr(tok, "chat_template", None),
        )


# ──────────────────────────────────────────────────────────────────────
# Hub profile
# ──────────────────────────────────────────────────────────────────────
@dataclass
class HubConfig:
    model_repo: str = "VTXAI/vortex-50m"
    tokenizer_repo: str = DEFAULT_TOKENIZER_ID
    data_repo: str = "VTXAI/vortex-50m-data-16k"
    trackio_space_id: str = "VTXAI/vortex-50m-trackio"
    trackio_project: str = "vortex-50m"

    def __post_init__(self):
        for placeholder in ["<", "TODO", "todo"]:
            for s in (self.model_repo, self.tokenizer_repo, self.data_repo,
                      self.trackio_space_id, self.trackio_project):
                if placeholder in s:
                    raise ValueError(f"Placeholder in HubConfig: {s!r}")


# ──────────────────────────────────────────────────────────────────────
# Self-check on import: every preset must respect the budget
# ──────────────────────────────────────────────────────────────────────
PARAM_BUDGET = 50_000_000

if os.environ.get("VORTEX_SKIP_BUDGET_CHECK") != "1":
    for _name, _kw in PRESETS.items():
        _n = VortexArch(**_kw).n_params()
        if _n > PARAM_BUDGET and _name != "vortex-test":
            raise ValueError(
                f"Preset {_name!r} is {_n/1e6:.2f}M params, over the "
                f"{PARAM_BUDGET/1e6:.0f}M budget. Adjust PRESETS[{_name!r}]."
            )
