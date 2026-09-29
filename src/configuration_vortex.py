"""Vortex configuration — Hugging Face `PretrainedConfig` subclass.

Self-contained on purpose. When `trust_remote_code=True` is used,
`transformers` copies `configuration_vortex.py` and `modeling_vortex.py` into
`~/.cache/huggingface/modules/transformers_modules/<repo>/` and imports them as
*top-level* modules. Any import of a sibling file in this repository (e.g.
`from config import VortexArch`) would fail at that point, so this file may only
depend on the standard library and `transformers`.

Registering with the auto classes is what makes the checkpoint loadable with a
plain `AutoModelForCausalLM.from_pretrained(...)`:

    AutoConfig.register("vortex", VortexConfig)
    AutoModelForCausalLM.register(VortexConfig, VortexForCausalLM)

`src/export_hf.py` writes the equivalent `auto_map` block into `config.json`,
which is the serialised form of those two calls.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class VortexConfig(PretrainedConfig):
    """Configuration for the Vortex decoder-only Transformer.

    The defaults are the `vortex-50m-16k` preset: a 512d x 18L model with a
    16,384-token tied embedding table and 8Q/2KV grouped-query attention.

    Args:
        vocab_size (`int`, *optional*, defaults to 16384):
            Size of the token embedding table. With `tie_word_embeddings=True`
            this is also the size of the output head, and it is the single
            biggest lever on the parameter budget at this scale.
        hidden_size (`int`, *optional*, defaults to 512):
            Model dimension. Must be divisible by `num_attention_heads`.
        num_hidden_layers (`int`, *optional*, defaults to 18):
            Number of decoder blocks.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of query heads. `hidden_size // num_attention_heads` is the
            head dimension and must be even for RoPE.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            Number of key/value heads. Fewer than `num_attention_heads` selects
            grouped-query attention (GQA); must divide `num_attention_heads`.
        intermediate_size (`int`, *optional*, defaults to 1072):
            SwiGLU feed-forward width, ~2.09x `hidden_size`.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon inside every RMSNorm.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE base. Higher values stretch the wavelength of the
            high-frequency rotary components.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum context length. The RoPE tables are built to this size and
            grow on demand if a longer sequence is actually seen.
        use_qk_norm (`bool`, *optional*, defaults to `True`):
            Per-head RMSNorm on queries and keys before the attention matmul.
            The main defence against attention entropy collapse in small
            models; costs 2 * head_dim parameters per layer.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Share the `lm_head` weight with the input embedding. Halves the
            vocabulary-sized parameter cost.
        zero_init_residual (`bool`, *optional*, defaults to `True`):
            Initialise `o_proj` and `down_proj` to exactly zero so every block is
            an identity at step 0. Only affects fresh initialisation — it has no
            effect on loading trained weights.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the normal init for linear and embedding
            weights.
        use_cache (`bool`, *optional*, defaults to `True`):
            Return a key/value `Cache` from `forward` so `generate` runs in
            O(1) per token instead of re-running the full prefix.
        scale_residual (`bool`, *optional*, defaults to `False`):
            Scale residual branch outputs by `1/sqrt(2 * num_hidden_layers)`
            (GPT-2 style). Redundant next to zero-init residuals, so off.
        rope_interleaved (`bool`, *optional*, defaults to `True`):
            `True` uses the GPT-NeoX split-half pairing (`x1, x2 = x.chunk(2)`);
            `False` uses the interleaved-even/odd pairing. Recorded for
            provenance; the split-half layout is what the released weights were
            trained with.
    """

    model_type = "vortex"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Defaults mirror the `vortex-50m-16k` preset (src/config.py::VortexArch).
    # They are duplicated rather than imported so this file stays standalone.
    def __init__(
        self,
        vocab_size: int = 16_384,
        hidden_size: int = 512,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        intermediate_size: int = 1_072,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
        max_position_embeddings: int = 2_048,
        use_qk_norm: bool = True,
        tie_word_embeddings: bool = True,
        zero_init_residual: bool = True,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        scale_residual: bool = False,
        rope_interleaved: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.intermediate_size = int(intermediate_size)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.max_position_embeddings = int(max_position_embeddings)
        self.use_qk_norm = bool(use_qk_norm)
        self.zero_init_residual = bool(zero_init_residual)
        self.initializer_range = float(initializer_range)
        self.scale_residual = bool(scale_residual)
        self.rope_interleaved = bool(rope_interleaved)
        self.name_or_path = kwargs.pop("name_or_path", "")

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=bool(tie_word_embeddings),
            **kwargs,
        )

        # `use_cache` is a model-level flag, not a base `PretrainedConfig`
        # attribute — transformers 5 dropped it from the base class, so setting
        # it here is what makes `config.use_cache` readable on a loaded config.
        self.use_cache = bool(use_cache)

        self.validate()

    # ── derived ──────────────────────────────────────────────────────
    # `hidden_size` and `num_attention_heads` are also the names of the two
    # outermost `__init__` parameters, so these are read from the instance
    # rather than the caller's arguments. A `head_dim` passed in the config JSON
    # is a *derived* value: recomputing it keeps the model and its config from
    # disagreeing if someone edits one and not the other.

    @property
    def head_dim(self) -> int:
        """Query/key/value head dimension."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_query_groups(self) -> int:
        """Query heads served by each KV head under GQA."""
        return self.num_attention_heads // self.num_key_value_heads

    # ── validation ───────────────────────────────────────────────────
    def validate(self) -> None:
        """Reject an illegal shape at construction time.

        Without this, a bad GQA split surfaces as an opaque SDPA error
        ("heads in key and value must divide the number of heads in query")
        layers deep inside a forward pass.
        """
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads must be positive, got {self.num_attention_heads}"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        if self.num_key_value_heads < 1:
            raise ValueError(
                f"num_key_value_heads must be >= 1, got {self.num_key_value_heads}"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads {self.num_key_value_heads} exceeds "
                f"num_attention_heads {self.num_attention_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} is not divisible by "
                f"num_key_value_heads {self.num_key_value_heads}; GQA needs whole "
                f"query groups"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim {self.head_dim} must be even for RoPE; got "
                f"hidden_size {self.hidden_size} / {self.num_attention_heads} heads"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")


__all__ = ["VortexConfig"]
