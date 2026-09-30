from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The four tasks the submission is scored on, in the order they are reported.
SCORED_TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]

# WikiText-103 is ~540MB of text; a slice of this many lines is a few MB of
# tokens, which is enough to make the PPL stable to ~0.01 nats while keeping
# the eval to under a minute.
WIKITEXT_LINES = 20_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Checkpoint dir or Hub id to evaluate")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer used in training. Defaults to the one inside "
                        "the checkpoint, then to the arch default. MUST match.")
    p.add_argument("--tasks", nargs="+", default=SCORED_TASKS)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit documents per task (smoke tests only)")
    p.add_argument("--batch", type=str, default="auto",
                   help="lm-eval batch size, 'auto' or an integer")
    p.add_argument("--wikitext", type=int, default=WIKITEXT_LINES,
                   help="WikiText-103 lines to hold out for PPL")
    p.add_argument("--out", default="vortex_eval.json")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run to verify the plumbing end to end")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(ckpt, tok_arg, device):
    from model import VortexForCausalLM
    from config import DEFAULT_TOKENIZER_ID

    model = VortexForCausalLM.from_pretrained(ckpt)
    model = model.to(device).eval()
    cfg = model.cfg

    # Tokenizer resolution: explicit > one shipped in the checkpoint > default.
    if tok_arg:
        tok_src = tok_arg
    elif os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "tokenizer.json")):
        tok_src = ckpt
    else:
        tok_src = DEFAULT_TOKENIZER_ID

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    base_vocab = len(tok)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval] checkpoint : {ckpt}", flush=True)
    print(f"[eval] tokenizer  : {tok_src}", flush=True)
    print(f"[eval] params     : {n_params:,} ({n_params / 1e6:.2f}M)", flush=True)
    print(f"[eval] vocab      : {base_vocab:,} (model {cfg.vocab_size:,})", flush=True)

    if n_params > 50_000_000:
        raise SystemExit(
            f"[eval] FATAL: {n_params:,} params exceeds the 50,000,000 budget.")

    # Compare against what the MODEL can emit, not the raw tokenizer count.
    # A checkpoint whose table is a few rows larger than the tokenizer needs
    # is harmless padding (unreachable ids, no effect on any score); a table
    # SMALLER than the tokenizer is a hard error, because ids above the table
    # would index out of bounds or wrap.
    if cfg.vocab_size < base_vocab:
        raise SystemExit(
            f"[eval] FATAL: model vocab {cfg.vocab_size:,} is SMALLER than the "
            f"tokenizer's {base_vocab:,}. Token ids "
            f"{cfg.vocab_size:,}..{base_vocab - 1:,} are unreachable and the "
            f"scores would be wrong.\n"
            f"        Pass the tokenizer the model was trained with: "
            f"--tokenizer <path-or-hub-id>")
    if cfg.vocab_size > base_vocab:
        print(f"[eval] note: model vocab {cfg.vocab_size:,} exceeds tokenizer "
              f"{base_vocab:,} by {cfg.vocab_size - base_vocab} unreachable "
              f"row(s) -- harmless", flush=True)

    return model, tok, cfg, n_params


# ──────────────────────────────────────────────────────────────────────
# 1. lm-evaluation-harness
# ──────────────────────────────────────────────────────────────────────
def run_lm_eval(model, tok, tasks, limit, batch):
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        # `accelerate` is a hard dependency of HFLM but is NOT pulled in by a
        # bare `pip install lm-eval` in every environment, and the failure
        # surfaces as a missing module deep inside the harness -- which looks
        # like "harness not installed" if the message does not say which.
        print(f"[eval] lm-evaluation-harness unavailable: {e}", flush=True)
        print("       pip install 'lm-eval>=0.4.2' accelerate", flush=True)
        return None

    t0 = time.time()
    # HFLM wraps any HF-compatible model+tokenizer. `max_length` is taken from
    # the model config so a long ARC-Challenge passage is not silently split.
    lm = HFLM(pretrained=model, tokenizer=tok,
              batch_size=batch, max_length=model.cfg.max_position_embeddings)
    res = simple_evaluate(model=lm, tasks=tasks, limit=limit)
    print(f"[eval] lm-eval finished in {time.time() - t0:.0f}s", flush=True)
    return res


# ──────────────────────────────────────────────────────────────────────
# 2. WikiText-103 perplexity
# ──────────────────────────────────────────────────────────────────────
def wikitext_ppl(model, tok, n_lines, batch_size=8, device="cuda"):
    from datasets import load_dataset

    # `Salesforce/wikitext` is the canonical home of the corpus; the bare
    # `wikitext` alias now redirects to a repo whose script cannot be loaded
    # by current `datasets`. Both spellings have pointed at the same data.
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")

    # WikiText is one row per line with blank lines between articles; joining
    # the non-empty rows of the first `n_lines` reproduces the article stream.
    # The `test` split is used deliberately: it cannot overlap the training
    # shards, which were drawn from smollm-corpus.
    rows = ds["text"][:n_lines]
    text = "\n\n".join(t for t in rows if len(t.strip()) > 0)

    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    n_tok = ids.shape[1]
    print(f"[eval] WikiText-103: {n_tok:,} tokens from {n_lines:,} rows", flush=True)

    seq = model.cfg.max_position_embeddings
    stride = seq
    nll_sum, n_count = 0.0, 0

    with torch.no_grad():
        for start in range(0, n_tok - 1, stride):
            chunk = ids[:, start:start + stride]
            if chunk.shape[1] < 2:
                break
            chunk = chunk.to(device)
            # Labels = the chunk itself; the model shifts internally.
            out = model(input_ids=chunk, labels=chunk, chunk_size=1024)
            nll = out.loss.item() * (chunk.shape[1] - 1)
            nll_sum += nll
            n_count += chunk.shape[1] - 1
            if (start // stride) % 50 == 0 and start:
                print(f"  ...{start:,}/{n_tok:,} tokens", flush=True)

    mean_nll = nll_sum / max(1, n_count)
    return {
        "perplexity": math.exp(min(20, mean_nll)),
        "mean_nll_nats": mean_nll,
        "bits_per_byte": mean_nll / math.log(2),
        "n_tokens": n_tok,
        "n_scored": n_count,
    }


# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.smoke and args.limit is None:
        args.limit = 20
        args.wikitext = 200

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}\n", flush=True)

    model, tok, cfg, n_params = load_model_and_tokenizer(
        args.ckpt, args.tokenizer, device)

    results = {
        "checkpoint": args.ckpt,
        "architecture": {
            "name_or_path": cfg.name_or_path,
            "params_total": n_params,
            "params_budget": 50_000_000,
            "within_budget": n_params <= 50_000_000,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "vocab_size": cfg.vocab_size,
            "max_position_embeddings": cfg.max_position_embeddings,
        },
        "harness": "lm-evaluation-harness",
        "tasks": {},
    }

    # ── Harness tasks ────────────────────────────────────────────────
    lm = run_lm_eval(model, tok, args.tasks, args.limit, args.batch)
    if lm is not None:
        for task in args.tasks:
            r = lm.get("results", {}).get(task, {})
            # lm-eval reports several metrics per task. The competition score
            # for a multiple-choice task is the ONE the task defines as
            # primary: `acc_norm` where the harness computes it (HellaSwag,
            # ARC, PIQA are length-normalized by the harness standard), and
            # plain `acc` where it does not (WinoGrande has no acc_norm at
            # all). Averaging acc and acc_norm together would be meaningless,
            # so each task contributes exactly one number and it is recorded
            # which one.
            if r.get("acc_norm,none") is not None:
                primary, value = "acc_norm", r["acc_norm,none"]
            else:
                primary, value = "acc", r.get("acc,none")
            stderr = (r.get("acc_norm_stderr,none") if primary == "acc_norm"
                      else r.get("acc_stderr,none"))
            results["tasks"][task] = {
                "primary_metric": primary,
                "score": value,
                "acc": r.get("acc,none"),
                "acc_norm": r.get("acc_norm,none"),
                "stderr": stderr,
            }
            if value is not None:
                print(f"[eval] {task:14s} {primary:8s} = {value * 100:.2f}%"
                      + (f"  +-{stderr * 100:.2f}" if stderr else ""), flush=True)
    else:
        print("[eval] skipping harness tasks (harness unavailable)", flush=True)

    # ── WikiText-103 perplexity ─────────────────────────────────────
    print("", flush=True)
    results["wikitext103"] = wikitext_ppl(model, tok, args.wikitext, device=device)
    w = results["wikitext103"]
    print(f"\n[eval] WikiText-103 PPL = {w['perplexity']:.2f}  "
          f"({w['mean_nll_nats']:.4f} nats, {w['bits_per_byte']:.3f} bpb, "
          f"{w['n_scored']:,} scored tokens)", flush=True)

    # ── Summary ──────────────────────────────────────────────────────
    scored = [v["score"] for v in results["tasks"].values()
              if v.get("score") is not None]
    if scored:
        results["scored_avg"] = sum(scored) / len(scored)
        print(f"\n[eval] ===== SCORED AVERAGE ({len(scored)} tasks) = "
              f"{results['scored_avg'] * 100:.2f}% =====", flush=True)
    else:
        print("\n[eval] no harness scores collected -- README cannot cite them",
              flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
