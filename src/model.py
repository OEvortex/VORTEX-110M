"""
VTX-300M — a custom ~300M parameter decoder-only Transformer with GQA.

Features:
- Pre-LayerNorm (RMSNorm)
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA): 12 Q heads, 4 KV heads
- SwiGLU MLP
- Native SDPA attention (FlashAttention-2 on Blackwell/Ampere+)
- Strict weight tying between input embeddings and the LM head

This is a from-scratch architecture, not a port of an HF class — it is
Hub-compatible: `VortexForCausalLM` is a `PreTrainedModel` so it serializes
with `save_pretrained()` and loads with `from_pretrained()`.
"""
from __future__ import annotations
import math
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

from config import VortexArch


# ──────────────────────────────────────────────────────────────────────
# Hub-compatible config
# ──────────────────────────────────────────────────────────────────────
class VortexConfig(PretrainedConfig):
    model_type = "vortex"
    # Tells HF which weights to dedup at save time.
    _tied_weights_keys = ["model.embed_tokens.weight"]

    def __init__(self, **kwargs):
        # Pull defaults from VortexArch then override with kwargs
        defaults = asdict(VortexArch())
        defaults.update(kwargs)
        super().__init__(**defaults)
        for k, v in defaults.items():
            setattr(self, k, v)
        # Suppress the warning about weight sharing
        self.tie_word_embeddings = bool(self.tie_word_embeddings)


# ──────────────────────────────────────────────────────────────────────
# Components
# ──────────────────────────────────────────────────────────────────────
class VortexRMSNorm(nn.Module):
    """RMSNorm: y = x / rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(in_dtype) * self.weight


def _precompute_rope_cache(head_dim: int, max_seq_len: int, base: float, device, dtype):
    """Build (cos, sin) tables of shape (max_seq_len, head_dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to last dim of x (B, H, T, D).

    Standard interleaved RoPE:
        y[..., 2i]   = x[..., 2i]   * cos[i] - x[..., 2i+1] * sin[i]
        y[..., 2i+1] = x[..., 2i+1] * cos[i] + x[..., 2i]   * sin[i]
    """
    T = x.shape[-2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


class VortexAttention(nn.Module):
    """Grouped Query Self-Attention with RoPE and SDPA (FlashAttention-2).

    GQA: num_attention_heads Q heads, num_key_value_heads KV heads.
    When num_kv_heads < num_heads, KV heads are repeated to match Q heads.
    """
    def __init__(self, cfg: VortexConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Q projection: full heads.  KV projections: GQA reduced.
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

        # RoPE cache (built lazily on first forward)
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self.rope_theta = cfg.rope_theta

    def _ensure_rope(self, T: int, device, dtype):
        if self._cos is None or self._cos.shape[0] < T or self._cos.device != device or self._cos.dtype != dtype:
            cos, sin = _precompute_rope_cache(self.head_dim, T, self.rope_theta, device, dtype)
            self._cos, self._sin = cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)          # (B, T, num_heads * head_dim)
        k = self.k_proj(x)          # (B, T, num_kv_heads * head_dim)
        v = self.v_proj(x)          # (B, T, num_kv_heads * head_dim)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)       # (B, Hq, T, D)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)    # (B, Hkv, T, D)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)    # (B, Hkv, T, D)

        # RoPE on Q and K
        self._ensure_rope(T, x.device, x.dtype)
        q = _apply_rope(q, self._cos, self._sin)
        k = _apply_rope(k, self._cos, self._sin)

        # Expand KV heads to match Q heads for SDPA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)   # (B, Hq, T, D)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)   # (B, Hq, T, D)

        # Causal SDPA — FlashAttention-2 on Blackwell/Ampere+
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=0.0,
            scale=self.scaling,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class VortexMLP(nn.Module):
    """SwiGLU MLP:  y = down( silu(gate(x)) * up(x) )."""
    def __init__(self, cfg: VortexConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VortexBlock(nn.Module):
    """Pre-LN decoder block:  x = x + attn(rms(x));  x = x + mlp(rms(x))."""
    def __init__(self, cfg: VortexConfig):
        super().__init__()
        self.attn = VortexAttention(cfg)
        self.mlp = VortexMLP(cfg)
        self.attn_norm = VortexRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp_norm = VortexRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────
class VortexModel(PreTrainedModel):
    config_class = VortexConfig

    def __init__(self, cfg: VortexConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([VortexBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = VortexRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x)


class VortexForCausalLM(PreTrainedModel):
    config_class = VortexConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, cfg: VortexConfig | VortexArch):
        if isinstance(cfg, VortexArch):
            cfg = VortexConfig(**asdict(cfg))
        super().__init__(cfg)
        self.cfg = cfg
        self.model = VortexModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Strict weight tying: input embeddings == output projection
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize lm_head (and reset tied weight since it was just overwritten).
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=cfg.initializer_range)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        self.model.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        chunk_size: int = 0,
    ):
        hidden = self.model(input_ids)
        loss = None
        logits = None
        if labels is not None:
            shift_hidden = hidden[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            chunk = chunk_size if chunk_size > 0 else 256
            loss = self._chunked_ce(shift_hidden, shift_labels, chunk=chunk)
        else:
            logits = self.lm_head(hidden)
        return type("VortexOutput", (), {"loss": loss, "logits": logits})()

    def _chunked_ce(self, hidden: torch.Tensor, labels: torch.Tensor, chunk: int) -> torch.Tensor:
        """Cross-entropy in time chunks to avoid OOM on large vocab."""
        B, T, H = hidden.shape
        flat_h = hidden.view(B * T, H)
        flat_y = labels.reshape(B * T)
        n_valid = (flat_y != -100).sum()
        if n_valid.item() == 0:
            return torch.tensor(0.0, device=hidden.device, requires_grad=True)
        total = flat_h.new_zeros(())
        for i in range(0, flat_h.shape[0], chunk):
            h_chunk = flat_h[i:i + chunk]
            y_chunk = flat_y[i:i + chunk]
            logits = self.lm_head(h_chunk).float()
            loss = F.cross_entropy(logits, y_chunk, ignore_index=-100, reduction="sum")
            total = total + loss
            del logits
        return total / n_valid.clamp_min(1)

    def tie_weights(self, recompute_mapping: bool = False, missing_keys: dict | None = None):
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        if missing_keys is not None:
            missing_keys.discard("lm_head.weight")
