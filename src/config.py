"""
VTX-300M configuration and tokenizer profile.

Sources architecture hyperparams from the local dataclass and tokenizer
profile from `Qwen/Qwen3-4B` on the Hub.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Architecture (~300M active, weight-tied, GQA)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class VortexArch:
    """VTX-300M architecture hyperparameters (decoder-only Transformer).

    Sized to land at ~300M total params with the Qwen3 vocab (~151,670).
    With hidden=768, intermediate=2048, layers=18, GQA(12q/4kv):
      embed (151670 x 768)  = 116.5M
      18 x layer            = ~184M  (GQA attn + SwiGLU MLP)
      norms + head tied     =   +0
      ----------------------------------
      Total                 ≈ 300M
    """
    hidden_size: int = 768
    num_hidden_layers: int = 18
    num_attention_heads: int = 12
    num_key_value_heads: int = 4         # GQA: 4 KV heads shared across 12 Q heads
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    vocab_size: int = 151_670       # Qwen3 + EOS/PAD coverage
    type_vocab_size: int = 1
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True   # explicit: lm_head.weight = embed_tokens.weight

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_attention_kv_heads(self) -> int:
        return self.num_key_value_heads

    def n_params_estimate(self) -> int:
        """Crude estimate (no embeddings / no lm_head)."""
        hs = self.hidden_size
        nh = self.num_attention_heads
        nkv = self.num_key_value_heads
        hd = hs // nh
        # GQA attention: Q(12h*d) + K(4h*d) + V(4h*d) + O(h*h)
        attn = hs * (nh * hd + nkv * hd + nkv * hd + hs)
        # SwiGLU: gate + up + down
        mlp = 3 * hs * self.intermediate_size
        return self.num_hidden_layers * (attn + mlp)


# ──────────────────────────────────────────────────────────────────────
# Hub profile
# ──────────────────────────────────────────────────────────────────────
@dataclass
class HubConfig:
    model_repo: str = "VTXAI/vtx-300m"
    data_repo: str = "VTXAI/vortex-110m-data"
    trackio_space_id: str = "VTXAI/vtx-300m-trackio"
    trackio_project: str = "vtx-300m"

    def __post_init__(self):
        for placeholder in ["<", "TODO", "todo"]:
            for s in (self.model_repo, self.data_repo,
                      self.trackio_space_id, self.trackio_project):
                if placeholder in s:
                    raise ValueError(f"Placeholder in HubConfig: {s!r}")


# ──────────────────────────────────────────────────────────────────────
# Tokenizer profile
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TokenizerProfile:
    tokenizer_id: str = "Qwen/Qwen3-4B"
    vocab_size: int = 0
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0
    chat_template: Optional[str] = None

    @classmethod
    def from_hub(cls, tokenizer_id: str = "Qwen/Qwen3-4B", hub_token: Optional[str] = None) -> "TokenizerProfile":
        """Load the tokenizer and pull chat template / special tokens.

        vocab_size is set to len(tokenizer) but we expand it to cover all
        special tokens (eos/pad often sit *above* len(tokenizer) for some
        Qwen-family tokenizers — e.g. Qwen3's eos=151645 > len=151643).
        """
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            tokenizer_id,
            token=os.environ.get("HF_TOKEN") or hub_token,
            trust_remote_code=True,
        )
        eff_vocab = max(len(tok), tok.eos_token_id or 0, tok.pad_token_id or 0) + 1
        prof = cls(
            tokenizer_id=tokenizer_id,
            vocab_size=eff_vocab,
            bos_token_id=tok.bos_token_id or 0,
            eos_token_id=tok.eos_token_id or 0,
            pad_token_id=tok.pad_token_id or 0,
            chat_template=tok.chat_template,
        )
        return prof


# ──────────────────────────────────────────────────────────────────────
# Bootstrapping
# ──────────────────────────────────────────────────────────────────────
def bootstrap(arch: VortexArch, hub: HubConfig) -> tuple[VortexArch, HubConfig, TokenizerProfile]:
    """Load tokenizer from Hub and inject vocab/chat into the architecture."""
    # Token verification
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami(token=os.environ.get("HF_TOKEN"))
    print(f"[config] Authenticated as: {who.get('name', '?')}", flush=True)

    # Tokenizer
    profile = TokenizerProfile.from_hub(arch.tokenizer if hasattr(arch, "tokenizer") else "Qwen/Qwen3-4B")
    arch.vocab_size = profile.vocab_size
    print(f"[config] Tokenizer: {profile.tokenizer_id}  vocab={profile.vocab_size}  "
          f"bos={profile.bos_token_id}  eos={profile.eos_token_id}", flush=True)

    return arch, hub, profile
