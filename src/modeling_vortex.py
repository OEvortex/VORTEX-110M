"""Vortex modeling — Hugging Face `PreTrainedModel` implementation.

Self-contained on purpose. With `trust_remote_code=True`, `transformers` copies
`configuration_vortex.py` and `modeling_vortex.py` into
`~/.cache/huggingface/modules/transformers_modules/<repo>/` and imports them as
a package, so nothing here may import a sibling file from this repository.
`configuration_vortex` is the only dependency and it travels with this module, so
the pair is always copied together — see `_load_config_class` for why the import
is written the way it is.

What this adds over a bare `nn.Module` port, and why each piece is needed for
`AutoModelForCausalLM` / `generate` / `Trainer` to work:

* **Key/value cache.** `use_cache` was a config field with no implementation —
  every `forward` recomputed the whole prefix. `VortexAttention` now consumes a
  `transformers` `Cache`, which is what makes `model.generate()` viable.
* **Position offsets under a cache.** RoPE was sliced `cos[:T]`, i.e. positions
  were always 0-based. With a cache the query block starts at `past_len`; the
  rotary tables are now sliced `[offset : offset + T]`. RoPE is relative, so this
  leaves the pretraining fast path bit-identical.
* **A correct attention mask on the cached path.** SDPA's `is_causal=True`
  assumes top-left alignment and is only right when the cache is empty. Cached
  steps with left padding need an explicit bottom-right-aligned mask, which is
  what `VortexModel._build_causal_mask` builds. The empty-cache/no-padding case
  still takes the `is_causal=True` fast path, so training numerics and memory are
  unchanged.
* **Real `ModelOutput`s.** The previous `CausalLMOutput` was a plain object, so
  `output.logits` worked but nothing HF-side (generation, `Trainer`, tensor
  logging) recognised it.
* **Standard input plumbing** — `attention_mask`, `position_ids`,
  `inputs_embeds`, `num_items_in_batch`, `logits_to_keep`.

Two deliberate deviations from HF naming conventions:

* The decoder submodules keep their original names (`attn`, `ln_attn`, `ln_mlp`)
  rather than `self_attn`, `input_layernorm`, `post_attention_layernorm`. HF's
  `self_attn` means *cross*-attention, which this architecture does not have.
  More importantly, the released checkpoints on the Hub use the current names,
  and `from_pretrained` matches `state_dict` keys literally — renaming would
  break every one of them unless a key-remapping table were threaded through
  `from_pretrained`, which is a per-version API in transformers 5.x. The outer
  names (`model.*`, `embed_tokens`, `lm_head`, `norm`) already match HF.
* `logits_to_keep` is not decoration. Computing `(B, T, vocab_size)` logits for a
  full 2048-token batch is the largest single memory term in a training step, and
  the whole point of the chunked loss path is to never materialise it. That is
  why `labels=` returns `logits=None` unless logits are explicitly asked for.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging


def _load_config_class():
    """Import the sibling `configuration_vortex` module.

    Two different import mechanics have to be satisfied:

    * Under `trust_remote_code`, `transformers` copies both files into its
      dynamic-module package and imports them as a package, so a *relative*
      import is the one that resolves.
    * Running the repo's own tests imports this file as a top-level module from
      `src/`, where there is no package and no `__package__`.

    A plain top-level `from configuration_vortex import ...` is not an option:
    `dynamic_module_utils.check_imports` runs `importlib.import_module` on every
    statically-detected import *before* the sibling has been copied next to this
    file, so it fails with "No module named 'configuration_vortex'" and a
    misleading `pip install configuration_vortex`. Loading by file path keeps the
    statement out of the AST the checker inspects.
    """
    if __package__:
        from .configuration_vortex import VortexConfig

        return VortexConfig

    spec = importlib.util.spec_from_file_location(
        "configuration_vortex",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "configuration_vortex.py"),
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so the dataclass-free class object survives even if
    # something inside the module re-enters this lookup.
    sys.modules["configuration_vortex"] = module
    spec.loader.exec_module(module)
    return module.VortexConfig


VortexConfig = _load_config_class()

logger = logging.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Norm
# ──────────────────────────────────────────────────────────────────────
class VortexRMSNorm(nn.Module):
    """RMSNorm with the reduction and the norm-weight multiply in fp32.

    Upcasting is the point: with 18 pre-norm blocks in bf16 autocast, a bf16
    reduction over the residual stream loses enough precision to stall training.
    The output is cast back so the residual add stays in the activation dtype.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.normalized_shape = (hidden_size,)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ──────────────────────────────────────────────────────────────────────
# Rotary position embedding
# ──────────────────────────────────────────────────────────────────────
def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10_000.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the `(max_seq_len, head_dim / 2)` cos/sin tables for RoPE."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    position_ids = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(position_ids, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Rotate the last dim of `x` (GPT-NeoX split-half pairing).

    `x` is `(batch, heads, seq, head_dim)`. `cos`/`sin` are `(seq, head_dim / 2)`
    *absolute* position tables; `offset` selects the starting position, which is
    what puts a cached query block on the right rotary phase.

    Pre-sliced tables with the default `offset=0` are still accepted, so the
    direct-call form used by the verification suite keeps working.
    """
    if x.shape[-2] != cos.shape[0] or offset != 0:
        T = x.shape[-2]
        cos = cos[offset : offset + T]
        sin = sin[offset : offset + T]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class VortexRotaryEmbedding(nn.Module):
    """Per-model RoPE table, built once and shared by every attention layer.

    Held in non-persistent state so it never becomes a checkpoint tensor — it is
    fully determined by `head_dim`, `rope_theta` and the current device/dtype.
    """

    def __init__(self, config: VortexConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings

        # Plain attributes, deliberately not buffers. `from_pretrained` builds
        # the model on a meta device and materialises only the tensors it finds
        # in the checkpoint, so a *non-persistent* buffer is left as
        # uninitialised memory: the model loads without error and every RoPE
        # application is garbage. Keeping this out of `state_dict` also means the
        # key layout stays identical to the released training checkpoints, which
        # is what lets `load_state_dict(strict=True)` accept them.
        self._inv_freq: Optional[torch.Tensor] = None
        self._inv_freq_device: Optional[torch.device] = device
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._cached_len = 0
        self._cached_dtype: Optional[torch.dtype] = None

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        head_dim = self.config.head_dim
        if self._inv_freq is None or self._inv_freq_device != device:
            self._inv_freq = 1.0 / (
                self.config.rope_theta
                ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
            )
            self._inv_freq_device = device
            # Invalidate the cos/sin tables; they were built from the old one.
            self._cos = self._sin = None
        return self._inv_freq

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tables covering at least `seq_len` positions."""
        device, dtype = x.device, x.dtype
        if (
            self._cos is None
            or self._cached_len < seq_len
            or self._cos.device != device
            or self._cached_dtype != dtype
        ):
            inv_freq = self._get_inv_freq(device)
            self._cached_len = max(seq_len, self.config.max_position_embeddings)
            position_ids = torch.arange(self._cached_len, device=device, dtype=torch.float32)
            freqs = torch.outer(position_ids, inv_freq)
            self._cos = freqs.cos().to(dtype)
            self._sin = freqs.sin().to(dtype)
            self._cached_dtype = dtype
        return self._cos, self._sin


# ──────────────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────────────
class VortexAttention(nn.Module):
    """Causal grouped-query attention with optional QK-Norm.

    Goes through `F.scaled_dot_product_attention` with no hand-written softmax,
    which lets PyTorch dispatch to FlashAttention-2 on Ampere and later and to
    the math backend everywhere else. `enable_gqa` avoids materialising repeated
    KV heads; the `repeat_interleave` branch only runs on torch < 2.5.
    """

    def __init__(self, config: VortexConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_groups = self.n_heads // self.n_kv
        self.rope_theta = config.rope_theta
        self.use_qk_norm = bool(config.use_qk_norm)
        self.scale = self.head_dim**-0.5

        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hidden_size, bias=False)

        if self.use_qk_norm:
            # Per-head RMS over head_dim, applied before the attention matmul.
            # Without it, small models hit attention entropy collapse early: a
            # few heads saturate, their softmax goes one-hot, and those heads
            # are dead for the rest of the run. Costs 2 * head_dim params/layer.
            self.q_norm = VortexRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = VortexRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        # QK-Norm: bound the pre-softmax logits before RoPE mixes them.
        q = self.q_norm(q)
        k = self.k_norm(k)

        if position_embeddings is None:
            position_embeddings = build_rope_cache(
                self.head_dim, position_offset + T, self.rope_theta, x.device, x.dtype
            )
        cos, sin = position_embeddings
        q = apply_rope(q, cos, sin, offset=position_offset)
        k = apply_rope(k, cos, sin, offset=position_offset)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)

        # `is_causal=True` is only correct when the cache is empty: SDPA assumes
        # top-left alignment, and a cached block queries a suffix of the key
        # sequence. `VortexModel` hands over an explicit mask whenever that is
        # the case and leaves it `None` for the prefill fast path.
        is_causal = attention_mask is None and T > 1

        try:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
                enable_gqa=self.n_groups > 1,
            )
        except TypeError:  # torch < 2.5 has no `enable_gqa`
            if self.n_groups > 1:
                k = k.repeat_interleave(self.n_groups, dim=1)
                v = v.repeat_interleave(self.n_groups, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


# ──────────────────────────────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────────────────────────────
class VortexMLP(nn.Module):
    """SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`."""

    def __init__(self, config: VortexConfig):
        super().__init__()
        intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ──────────────────────────────────────────────────────────────────────
# Block
# ──────────────────────────────────────────────────────────────────────
class VortexBlock(nn.Module):
    """Pre-norm block: attention and MLP each add onto the residual stream."""

    def __init__(self, config: VortexConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = VortexAttention(config, layer_idx=layer_idx)
        self.mlp = VortexMLP(config)
        self.ln_attn = VortexRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ln_mlp = VortexRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # GPT-2 style 1/sqrt(2L) branch scaling. Off by default: it is redundant
        # next to zero-initialised residual outputs, which already make every
        # block an exact identity at init.
        self.resid_scale = (
            1.0 / math.sqrt(2.0 * config.num_hidden_layers) if config.scale_residual else 1.0
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ) -> torch.Tensor:
        x = x + self.resid_scale * self.attn(
            self.ln_attn(x),
            position_embeddings=position_embeddings,
            position_offset=position_offset,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        x = x + self.resid_scale * self.mlp(self.ln_mlp(x))
        return x


# ──────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────
class VortexPreTrainedModel(PreTrainedModel):
    """Weight init, tied-embedding bookkeeping and tokenizer plumbing."""

    config_class = VortexConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VortexBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True
    # SDPA already dispatches to FlashAttention-2 kernels on Ampere+, but the
    # `attn_implementation="flash_attention_2"` HF interface is not implemented
    # here. Claiming support would let `from_pretrained` pick a code path that
    # does not exist for this architecture.
    _supports_flash_attn = False
    _supports_attention_backend = False
    _supports_cache_class = True
    _supports_static_cache = True
    _can_record_outputs = {"hidden_states": VortexBlock, "attentions": VortexAttention}

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # A small vocab (16K) is far more tolerant than a 151K one, but
            # scaling down keeps initial logits O(1) rather than O(10).
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, VortexRMSNorm):
            nn.init.ones_(module.weight)
        # Dispatched per-submodule by `PreTrainedModel.post_init`, which applies
        # this over the whole tree. Overriding it here is what makes a freshly
        # constructed model an identity passthrough without a separate traversal.
        self._zero_init_residuals(module)

    def _zero_init_residuals(self, module: Optional[nn.Module] = None) -> None:
        """Zero `o_proj` and `down_proj` so every block starts as an identity.

        With 18 stacked pre-norm blocks, default init compounds the residual
        variance and saturates the stream before step 0. Zeroing the two branch
        outputs makes the untrained network a clean passthrough, so the initial
        loss is ln(vocab_size) = 9.7 rather than the hundreds default init gives.

        Dispatched per-module by `PreTrainedModel.post_init` via
        `_init_weights`; the recursion is over `self.modules()` so it also works
        when called with no argument.
        """
        if not getattr(self.config, "zero_init_residual", True):
            return
        if module is not None:
            if isinstance(module, VortexAttention):
                nn.init.zeros_(module.o_proj.weight)
            elif isinstance(module, VortexMLP):
                nn.init.zeros_(module.down_proj.weight)
            return
        for block in self.model.layers:
            nn.init.zeros_(block.attn.o_proj.weight)
            nn.init.zeros_(block.mlp.down_proj.weight)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """Grow the embedding table, never shrink it.

        Growing pads with fresh normal noise. Shrinking is refused rather than
        silently truncating: rows that have been trained keep meaning something,
        and a truncated table yields a model that evaluates fine and answers
        with the wrong tokens.
        """
        old_embeddings = self.get_input_embeddings()
        if old_embeddings is None:
            raise ValueError("cannot resize embeddings on a model with no input embeddings")

        old_num_tokens, embedding_dim = old_embeddings.weight.shape
        if new_num_tokens is None:
            new_num_tokens = old_num_tokens
        if pad_to_multiple_of is not None:
            new_num_tokens = math.ceil(new_num_tokens / pad_to_multiple_of) * pad_to_multiple_of
        new_num_tokens = int(new_num_tokens)

        if new_num_tokens < old_num_tokens:
            raise ValueError(
                f"cannot shrink token embeddings {old_num_tokens} -> {new_num_tokens}; "
                f"the vocabulary must only be extended"
            )
        if new_num_tokens == old_num_tokens:
            return old_embeddings

        new_embeddings = nn.Embedding(
            new_num_tokens, embedding_dim, device=old_embeddings.weight.device
        )
        with torch.no_grad():
            new_embeddings.weight.normal_(mean=0.0, std=self.config.initializer_range)
            new_embeddings.weight[:old_num_tokens].copy_(old_embeddings.weight)
        self.set_input_embeddings(new_embeddings)
        self.config.vocab_size = new_num_tokens

        # Keep the head in step with a tied table.
        if self.config.tie_word_embeddings and self.get_output_embeddings() is not None:
            self.tie_weights()
        return new_embeddings

    def tie_weights(self, recompute_mapping: bool = False, missing_keys=None):
        """Alias the `lm_head` weight onto the embedding table.

        Overridden rather than inherited because `missing_keys` has two
        incompatible shapes across transformers versions: a `set` in 4.x and a
        mapping in 4.56+. Only `lm_head` is tied here, so it is dropped from the
        "missing" report under either shape.
        """
        if getattr(self.config, "tie_word_embeddings", False):
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                output_embeddings.weight = input_embeddings.weight
        if missing_keys is None:
            return
        discard = getattr(missing_keys, "discard", None)
        if callable(discard):
            discard("lm_head.weight")
            return
        if hasattr(missing_keys, "pop"):
            try:
                missing_keys.pop("lm_head.weight")
            except TypeError:  # mapping-style pop(key, default)
                missing_keys.pop("lm_head.weight", None)


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────
class VortexModel(VortexPreTrainedModel):
    """Embedding + decoder stack + final norm."""

    def __init__(self, config: VortexConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [VortexBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = VortexRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VortexRotaryEmbedding(config)

        # Set by `PreTrainedModel.gradient_checkpointing_enable`, which targets
        # any submodule carrying this attribute.
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    # ── attention mask ───────────────────────────────────────────────
    @staticmethod
    def _build_causal_mask(
        q_len: int,
        kv_len: int,
        attention_mask_2d: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Bottom-right-aligned boolean mask, `True` = attend.

        Two things SDPA's `is_causal=True` cannot express:

        1. **Alignment.** With `past_len` cached keys, query `i` sits at absolute
           position `past_len + i`, so it may attend to keys `0 .. past_len + i`.
           Top-left alignment would bar the cached keys from every query.
        2. **Padding.** Left-padded batches need the pad columns removed.

        The self-diagonal is force-enabled on top of the mask so no query row is
        ever fully masked. A fully-masked row makes softmax return `NaN`, and
        those `NaN`s then ride in the padded key/value vectors into the next
        layer, where a `0 * NaN` in the weighted sum spreads them. Letting a
        padded query attend to itself is harmless — that position is masked out
        for every other query, so it cannot leak.
        """
        key_positions = torch.arange(kv_len, device=device)
        query_positions = torch.arange(q_len, device=device) + (kv_len - q_len)
        mask = (key_positions[None, :] <= query_positions[:, None])[None, None, :, :]

        if attention_mask_2d is not None:
            padding = attention_mask_2d.to(device=device)[:, None, None, :].bool()
            mask = mask & padding

        self_attends = (key_positions[None, :] == query_positions[:, None])[None, None, :, :]
        return (mask | self_attends).expand(-1, 1, -1, -1).contiguous()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states)
        return_dict = True if return_dict is None else bool(return_dict)

        checkpointing = bool(getattr(self, "gradient_checkpointing", False)) and self.training
        use_cache = self.config.use_cache if use_cache is None else bool(use_cache)

        # Checkpointing recomputes each block during the backward pass, and
        # `Cache.update` mutates in place -- so a cached forward would append the
        # same keys a second time and corrupt every downstream layer's mask
        # (observed: the cache silently doubling from 24 to 48 entries).
        # Training never needs the cache anyway, so it is dropped here. Inference
        # is unaffected because `self.training` is False.
        if checkpointing:
            use_cache = False
            past_key_values = None

        if output_attentions:
            raise NotImplementedError(
                "`output_attentions=True` is not supported: each block returns only "
                "hidden states, because attention runs fused inside SDPA."
            )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of `input_ids` or `inputs_embeds`")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if past_key_values is None and use_cache:
            past_key_values = DynamicCache(config=self.config)

        batch_size, seq_len, _ = inputs_embeds.shape
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = past_len + seq_len

        if position_ids is None:
            # RoPE is relative, so shifting every position by the same constant
            # leaves every attention score unchanged. Deriving absolute
            # positions from the cache length is therefore correct even for the
            # left-padded batches `generate` builds.
            position_ids = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape[-1] == kv_len and past_len > 0:
            position_ids = position_ids[:, past_len:]

        # A 2D `(batch, kv_len)` padding mask is what `generate` passes; a 4D
        # mask is taken as already built. Anything else is ignored rather than
        # guessed at.
        padding_mask_2d = None
        if attention_mask is not None and attention_mask.dim() == 2:
            padding_mask_2d = attention_mask
            attention_mask = None

        if attention_mask is None and (past_len > 0 or padding_mask_2d is not None):
            attention_mask = self._build_causal_mask(
                q_len=seq_len,
                kv_len=kv_len,
                attention_mask_2d=padding_mask_2d,
                device=inputs_embeds.device,
            )

        # One RoPE table for the whole stack rather than one per layer.
        position_embeddings = self.rotary_emb(inputs_embeds, kv_len)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        checkpoint_fn = getattr(self, "_gradient_checkpointing_func", None)
        if checkpointing and checkpoint_fn is None:
            checkpoint_fn = lambda fn, *args: checkpoint(fn, *args, use_reentrant=False)

        for block in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if checkpointing:
                hidden_states = checkpoint_fn(
                    block,
                    hidden_states,
                    position_embeddings,
                    past_len,
                    attention_mask,
                    past_key_values,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    position_offset=past_len,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values,
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return (hidden_states, past_key_values if use_cache else None, all_hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


# ──────────────────────────────────────────────────────────────────────
# Causal LM
# ──────────────────────────────────────────────────────────────────────
class VortexForCausalLM(VortexPreTrainedModel, GenerationMixin):
    """Vortex with a tied language-modelling head.

    `VortexModel` is the base model (`base_model_prefix = "model"`), so
    `save_pretrained` writes `model.embed_tokens.weight`, `model.layers.N.*` and
    `model.norm.weight` — the same key layout as the training checkpoints, which
    is what lets this class load them unchanged.

    `GenerationMixin` is inherited explicitly. From transformers v4.50 onward
    `PreTrainedModel` no longer provides it, so without this second base the
    model silently loses `generate`, `generate_from_model` and sampling helpers.
    It must come *after* `PreTrainedModel` in the MRO.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"embed_tokens": ["model.embed_tokens"], "layers": ["model.layers"]}

    def __init__(self, config: VortexConfig):
        super().__init__(config)
        self.model = VortexModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Without this a directly-constructed model keeps PyTorch's default init
        # — no `initializer_range`, no zeroed residual outputs. `from_pretrained`
        # calls it too, but construction has to be self-sufficient or
        # `VortexForCausalLM(config).to(device)` silently trains a broken model.
        self.post_init()
        if config.tie_word_embeddings:
            self.tie_weights()

    def post_init(self) -> None:
        """Initialise weights, then apply the zero-init residual scheme.

        `PreTrainedModel.post_init` is what registers `all_tied_weights_keys`,
        parallel plans and device-map hints, and it is also what drives
        `_init_weights` over every submodule. It must be delegated to rather than
        shadowed, but on its own it leaves `o_proj` and `down_proj` at their
        normal init, so the second pass below is what actually makes an untrained
        model a passthrough.
        """
        super().post_init()
        self._zero_init_residuals()
        if getattr(self.config, "tie_word_embeddings", False):
            self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def get_decoder(self) -> VortexModel:
        return self.model

    # ── loss ─────────────────────────────────────────────────────────
    def _chunked_cross_entropy(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-entropy without materialising `(N, vocab_size)` logits.

        This is the single largest avoidable memory term in a training step: at
        batch 32 x 2048 tokens x 16384 vocab in fp32 the logits alone are 4.3GB.
        Accumulating in time-chunks holds the peak at `chunk_size` rows instead.
        """
        n_valid = (labels != -100).sum()
        if n_valid.item() == 0:
            return hidden_states.sum() * 0.0  # keep the graph connected

        total = hidden_states.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden_states.shape[0], chunk_size):
            logits = self.lm_head(hidden_states[start : start + chunk_size]).float()
            total = total + F.cross_entropy(
                logits,
                labels[start : start + chunk_size],
                ignore_index=-100,
                reduction="sum",
            )
            del logits

        if num_items_in_batch is not None:
            # `Trainer` normalises by a token count accumulated across
            # gradient-accumulation steps. Matching it is what keeps the loss it
            # reports comparable to the standalone training loop's.
            return total / num_items_in_batch.to(total.device)
        return total / n_valid.clamp_min(1).float()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        chunk_size: int = 0,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""Causal language modelling.

        Args:
            labels (`torch.LongTensor`, *optional*):
                Targets for next-token prediction. When given, `logits` comes back
                `None` unless `logits_to_keep` asks for it — the loss accumulates
                in chunks precisely so the full `(batch, seq, vocab)` tensor never
                has to exist.
            logits_to_keep (`int`, *optional*, defaults to 0):
                Return logits for only the last `n` positions. 0 means all of
                them when no loss is being computed, and none when one is.
                `transformers` sets this to 1 during `generate`; passing any value
                alongside `labels` is how to ask for a loss *and* logits.
            chunk_size (`int`, *optional*, defaults to 0):
                Rows per cross-entropy chunk. 0 selects 1024.

        Returns:
            [`CausalLMOutputWithPast`]: `logits`, `loss`, and `past_key_values`
            when `use_cache` is set.
        """
        return_dict = True if return_dict is None else bool(return_dict)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        past_key_values = outputs.past_key_values

        # Keep only the tail when asked. During generation this is the single new
        # position, so the vocab-sized projection runs on one row instead of the
        # whole sequence.
        keep = int(logits_to_keep.item()) if isinstance(logits_to_keep, torch.Tensor) else int(logits_to_keep)

        loss = None
        if labels is not None:
            # Next-token alignment: predict token t+1 from position t.
            shift_hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.shape[-1])
            shift_labels = labels[..., 1:].reshape(-1)
            loss = self._chunked_cross_entropy(
                shift_hidden, shift_labels, chunk_size or 1024, num_items_in_batch
            )

        if labels is not None and keep == 0:
            # Keep the memory win. Ask with `logits_to_keep=1` if you need logits
            # alongside a loss.
            logits = None
        else:
            tail = hidden_states[:, -keep:, :] if keep > 0 else hidden_states
            logits = self.lm_head(tail)

        if not return_dict:
            return (logits, loss) if loss is None else (logits, loss, past_key_values)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ── generation ───────────────────────────────────────────────────
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        """Trim model inputs to the block `generate` is about to run.

        Handled entirely by `GenerationMixin`: it slices `input_ids` down to the
        tokens not yet in the cache. That slice is load-bearing rather than an
        optimisation — resending the full prefix would recompute it and corrupt
        the cache. Overridden only to keep the signature aligned with
        `transformers` 5.x and to forward `position_ids`, which the base
        implementation pops and re-slices.
        """
        return super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

    # ── gradient checkpointing ───────────────────────────────────────
    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Recompute decoder activations in the backward pass instead of storing them.

        Trades roughly 20-30% step time for most of the activation memory, which
        is what lets one 40GB card hold a large batch at 2K context. Only active
        in training mode — `VortexModel.forward` gates on `self.training`.
        """
        super().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs, **kwargs
        )
        self.model.gradient_checkpointing = True
        if not hasattr(self.model, "_gradient_checkpointing_func"):
            self.model._gradient_checkpointing_func = lambda fn, *args: checkpoint(
                fn, *args, use_reentrant=False
            )

    def gradient_checkpointing_disable(self, **kwargs) -> None:
        super().gradient_checkpointing_disable(**kwargs)
        self.model.gradient_checkpointing = False


__all__ = [
    "VortexConfig",
    "VortexPreTrainedModel",
    "VortexModel",
    "VortexForCausalLM",
    "VortexBlock",
    "VortexAttention",
    "VortexMLP",
    "VortexRMSNorm",
    "VortexRotaryEmbedding",
    "build_rope_cache",
    "apply_rope",
]
