
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Optional, Tuple

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

    def __init__(self, **kwargs):
        # Start from the dataclass defaults, override with kwargs.
        defaults = asdict(VortexArch())
        defaults.update(kwargs)
        super().__init__(**{k: v for k, v in defaults.items() if v is not None})
        for k, v in defaults.items():
            setattr(self, k, v)
        self.tie_word_embeddings = bool(self.tie_word_embeddings)
        if hasattr(self, "torch_dtype") and self.torch_dtype is None:
            self.torch_dtype = None


# ──────────────────────────────────────────────────────────────────────
# Norm
# ──────────────────────────────────────────────────────────────────────
class VortexRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(in_dtype)

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"


# ──────────────────────────────────────────────────────────────────────
# Rotary position embedding
# ──────────────────────────────────────────────────────────────────────
def build_rope_cache(head_dim: int, max_seq_len: int, base: float,
                     device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)                    # (T, hd/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T = x.shape[-2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)               # (1, 1, T, D/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────────────
class VortexAttention(nn.Module):

    def __init__(self, cfg: VortexConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.n_groups = self.n_heads // self.n_kv
        self.rope_theta = cfg.rope_theta
        self.use_qk_norm = bool(getattr(cfg, "use_qk_norm", True))
        self.scale = self.hd ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.hidden_size, bias=False)

        if self.use_qk_norm:
            # Per-head RMS over head_dim. Cheap (2*hd params/layer) and the
            # main defence against attention entropy collapse.
            self.q_norm = VortexRMSNorm(self.hd, eps=cfg.rms_norm_eps)
            self.k_norm = VortexRMSNorm(self.hd, eps=cfg.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        # RoPE cache, built lazily on first forward.
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None

    def _rope(self, T: int, device, dtype):
        if (self._cos is None
                or self._cos.shape[0] < T
                or self._cos.device != device
                or self._cos.dtype != dtype):
            self._cos, self._sin = build_rope_cache(
                self.hd, T, self.rope_theta, device, dtype
            )
        return self._cos, self._sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)

        # QK-Norm: bound the pre-softmax logits before RoPE mixes them.
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self._rope(T, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Causal SDPA -> FlashAttention-2 on Ampere/Blackwell, math fallback
        # elsewhere. enable_gqa avoids materializing repeated KV heads.
        try:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0,
                scale=self.scale, enable_gqa=(self.n_groups > 1),
            )
        except TypeError:   # torch < 2.5 has no enable_gqa
            if self.n_groups > 1:
                k = k.repeat_interleave(self.n_groups, dim=1)
                v = v.repeat_interleave(self.n_groups, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0, scale=self.scale,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


# ──────────────────────────────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────────────────────────────
class VortexMLP(nn.Module):

    def __init__(self, cfg: VortexConfig):
        super().__init__()
        inter = cfg.intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ──────────────────────────────────────────────────────────────────────
# Block
# ──────────────────────────────────────────────────────────────────────
class VortexBlock(nn.Module):

    def __init__(self, cfg: VortexConfig):
        super().__init__()
        self.attn = VortexAttention(cfg)
        self.mlp = VortexMLP(cfg)
        self.ln_attn = VortexRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.ln_mlp = VortexRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        resid_scale = 1.0
        if getattr(cfg, "scale_residual", False):
            resid_scale = 1.0 / math.sqrt(2.0 * cfg.num_hidden_layers)
        self.resid_scale = resid_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_scale * self.attn(self.ln_attn(x))
        x = x + self.resid_scale * self.mlp(self.ln_mlp(x))
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
        self.layers = nn.ModuleList(
            [VortexBlock(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = VortexRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def resize_token_embeddings(self, new_num_tokens: int):
        old = self.embed_tokens.weight
        old_num, dim = old.shape
        if new_num_tokens == old_num:
            return old
        if new_num_tokens < old_num:
            raise ValueError(
                f"cannot shrink token embeddings {old_num} -> {new_num_tokens}; "
                f"the tokenizer must only be extended"
            )

        new = torch.nn.Embedding(new_num_tokens, dim).to(
            device=old.device, dtype=old.dtype
        )
        with torch.no_grad():
            new.weight.normal_(mean=0.0, std=self.cfg.initializer_range)
            new.weight[:old_num].copy_(old)
        self.embed_tokens = new
        return new

    def _init_weights(self, module: nn.Module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, VortexRMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x = layer(x)
        return self.norm(x)


class VortexForCausalLM(PreTrainedModel):
    config_class = VortexConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, cfg):
        if isinstance(cfg, VortexArch):
            cfg = VortexConfig(**cfg.to_dict())
        super().__init__(cfg)
        self.cfg = cfg
        self.model = VortexModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Critical: without this, a directly-constructed model keeps PyTorch's
        # default init (no `initializer_range`, no zeroed residual outputs).
        # `from_pretrained` calls it too, but construction must be self-sufficient
        # or `VortexForCausalLM(cfg).to(device)` silently trains a broken model.
        self.post_init()

    # ── gradient checkpointing ───────────────────────────────────────
    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        self.model.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    # ── forward ──────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        chunk_size: int = 0,
    ):
        hidden = self.model(input_ids)

        if labels is None:
            return CausalLMOutput(logits=self.lm_head(hidden), loss=None)

        # Next-token alignment: predict token t+1 from position t.
        shift_hidden = hidden[..., :-1, :].reshape(-1, hidden.shape[-1])
        shift_labels = labels[..., 1:].reshape(-1)
        loss = self._chunked_ce(shift_hidden, shift_labels, chunk_size or 1024)
        return CausalLMOutput(logits=None, loss=loss)

    def _chunked_ce(self, hidden: torch.Tensor, labels: torch.Tensor,
                    chunk: int) -> torch.Tensor:
        n_valid = (labels != -100).sum()
        if n_valid.item() == 0:
            return hidden.sum() * 0.0     # keeps the graph connected

        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, hidden.shape[0], chunk):
            logits = self.lm_head(hidden[i:i + chunk]).float()
            total = total + F.cross_entropy(
                logits, labels[i:i + chunk],
                ignore_index=-100, reduction="sum",
            )
            del logits
        return total / n_valid.clamp_min(1).float()

    # ── weight tying ─────────────────────────────────────────────────
    def tie_weights(self, recompute_mapping: bool = False, missing_keys=None):
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        if missing_keys is None:
            return
        # transformers passes a set in 4.x and a dict-like in 4.56+; only
        # lm_head is tied, so drop it from the "missing" report either way.
        discard = getattr(missing_keys, "discard", None)
        if callable(discard):
            discard("lm_head.weight")
            return
        if hasattr(missing_keys, "pop"):
            try:
                missing_keys.pop("lm_head.weight")
            except TypeError:      # mapping-style pop(key, default)
                missing_keys.pop("lm_head.weight", None)

    # ── init ─────────────────────────────────────────────────────────
    def _init_weights(self, module: nn.Module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Small vocab (32K) is far more tolerant than 151K, but scaling
            # down keeps initial logits O(1) rather than O(10).
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, VortexRMSNorm):
            nn.init.ones_(module.weight)

    def _zero_init_residuals(self):
        if not getattr(self.cfg, "zero_init_residual", True):
            return
        for layer in self.model.layers:
            nn.init.zeros_(layer.attn.o_proj.weight)
            nn.init.zeros_(layer.mlp.down_proj.weight)

    def post_init(self):
        # NOTE: must delegate to super() -- PreTrainedModel.post_init() is what
        # registers `all_tied_weights_keys` / parallel plans / device_map hints.
        # Shadowing it breaks `from_pretrained` in transformers >= 5.
        super().post_init()

        # Apply our own scheme on top of the base init.
        self.apply(self._init_weights)
        self._zero_init_residuals()

        # Re-tie after any re-init.
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


class CausalLMOutput:
    __slots__ = ("logits", "loss")

    def __init__(self, logits: Optional[torch.Tensor], loss: Optional[torch.Tensor]):
        self.logits = logits
        self.loss = loss

    def __iter__(self):
        # Allows `logits, loss = model(...)`
        return iter((self.logits, self.loss))

    def keys(self):
        return ["logits", "loss"]

    def __getitem__(self, k):
        return getattr(self, k)
