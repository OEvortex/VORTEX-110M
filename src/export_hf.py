"""Convert a Vortex training checkpoint into a `transformers`-loadable Hub repo.

The training loop already writes a `PreTrainedModel` checkpoint, so the weights
need no conversion. What is missing for `AutoModelForCausalLM.from_pretrained`
is the *architecture* declaration: `config.json` must carry an `auto_map`
pointing at two files that ship alongside it, and those files must be
self-contained (see `modeling_vortex.py`).

    python src/export_hf.py --ckpt ./vortex_50m_ckpt/step_15258 \\
        --tokenizer ./vortex-tok-16k --out ./vortex-50m-16k-hf
    python src/export_hf.py --ckpt ... --out ... --push-to VTXAI/vortex-50m-16k

Then, anywhere:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("VTXAI/vortex-50m-16k",
                                                 trust_remote_code=True)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_vortex import VortexForCausalLM  # noqa: E402

CONFIG_MODULE = "configuration_vortex.py"
MODELING_MODULE = "modeling_vortex.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Training checkpoint dir, or a Hub repo id")
    p.add_argument("--out", required=True, help="Output directory for the HF repo")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer dir or Hub id. Falls back to the checkpoint's own, "
                        "then to the tokenizer referenced in config.json.")
    p.add_argument("--push-to", default=None, help="Push the result to this Hub repo id")
    p.add_argument("--private", action="store_true", help="Push as a private repo")
    p.add_argument("--max-length", type=int, default=None,
                   help="Override max_position_embeddings in the exported config")
    p.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True,
                   help="Write safetensors (default) or .bin (only for torch < 2.6)")
    return p.parse_args()


def resolve_tokenizer(ckpt: str, tok_arg: str | None) -> str | None:
    """Find the tokenizer this checkpoint was trained with.

    Order matters: an explicit `--tokenizer` wins, then a `tokenizer/` subdir
    written by `sft.py`, then whatever `config.json` recorded. Getting this
    wrong is a silent failure -- the model loads and generates fluent-looking
    nonsense from the wrong ids.
    """
    if tok_arg:
        return tok_arg
    local = Path(ckpt) / "tokenizer"
    if (local / "tokenizer.json").exists():
        return str(local)
    cfg_path = Path(ckpt) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for key in ("tokenizer_id", "name_or_path"):
            value = cfg.get(key)
            if value and not value.startswith("vortex-"):
                return value
    return None


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[export] loading checkpoint {args.ckpt}")
    model = VortexForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=None,  # keep fp32; casting is the consumer's call
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] {n_params:,} params, vocab {model.config.vocab_size:,}, "
          f"tied={model.config.tie_word_embeddings}")

    if args.max_length is not None:
        if args.max_length < model.config.max_position_embeddings:
            raise SystemExit(
                f"[export] refusing to shrink context "
                f"{model.config.max_position_embeddings} -> {args.max_length}: the "
                f"weights were trained at the larger length and RoPE tables "
                f"beyond it were never learned."
            )
        model.config.max_position_embeddings = args.max_length
        print(f"[export] max_position_embeddings -> {args.max_length}")

    # Re-save through the HF class. This writes config.json, model.safetensors
    # and generation_config.json in one shot, and -- because the state dict key
    # layout is unchanged -- reproduces the source checkpoint's tensors exactly.
    model.save_pretrained(out, safe_serialization=args.safe_serialization)

    # Ship the architecture. These two files are what `auto_map` points at, and
    # `transformers` copies them into its dynamic-module cache on load.
    for name in (CONFIG_MODULE, MODELING_MODULE):
        shutil.copy2(HERE / name, out / name)
    print(f"[export] copied {CONFIG_MODULE}, {MODELING_MODULE}")

    # Attach the tokenizer so the repo is self-contained.
    tok_src = resolve_tokenizer(args.ckpt, args.tokenizer)
    if tok_src:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tok_src)
        tok.save_pretrained(out)
        # The chat template is part of the tokenizer, but a `tokenizer_config.json`
        # written without it silently loses the ability to call apply_chat_template.
        template = getattr(tok, "chat_template", None)
        print(f"[export] tokenizer from {tok_src} "
              f"(vocab {len(tok):,}, chat_template={'yes' if template else 'no'})")

        # Reconcile the special-token ids the model will use at generation time.
        for attr, tok_attr in (("bos_token_id", "bos_token_id"),
                               ("eos_token_id", "eos_token_id"),
                               ("pad_token_id", "pad_token_id")):
            value = getattr(tok, tok_attr, None)
            if value is not None and getattr(model.config, attr, None) != value:
                print(f"[export] {attr}: {getattr(model.config, attr)} -> {value} (from tokenizer)")
                setattr(model.config, attr, value)
        model.save_pretrained(out, safe_serialization=args.safe_serialization)
    else:
        print("[export] WARNING no tokenizer found; the repo will not be usable "
              "for generation. Pass --tokenizer.")

    # `auto_map` is the serialised form of AutoConfig.register /
    # AutoModelForCausalLM.register. Without it, `from_pretrained` has no way to
    # resolve "vortex" to a class and fails with an unrecognised-model_type error.
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["auto_map"] = {
        "AutoConfig": "configuration_vortex.VortexConfig",
        "AutoModelForCausalLM": "modeling_vortex.VortexForCausalLM",
    }
    cfg["architectures"] = ["VortexForCausalLM"]
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print("[export] wrote auto_map into config.json")

    print("\n[export] contents:")
    for path in sorted(out.iterdir()):
        size = f"{path.stat().st_size / 1e6:.1f}MB" if path.is_file() else "dir"
        print(f"           {path.name:<34} {size}")

    if args.push_to:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.push_to, private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.push_to, folder_path=str(out),
                          commit_message="Add HF configuration + modeling modules")
        print(f"\n[export] pushed -> {args.push_to}")
        print(f"[export] load with:\n"
              f"           AutoModelForCausalLM.from_pretrained(\n"
              f"               '{args.push_to}', trust_remote_code=True)")
    else:
        print(f"\n[export] load with:\n"
              f"           AutoModelForCausalLM.from_pretrained(\n"
              f"               '{out}', trust_remote_code=True)")

    print(f"\n[export] total params {n_params:,}")


if __name__ == "__main__":
    main()
