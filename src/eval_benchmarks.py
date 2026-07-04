"""
VTX-300M benchmark evaluation.

Evaluates a trained VTX-300M checkpoint on standard NLP benchmarks:
- HellaSwag (commonsense)
- ARC-Easy, ARC-Challenge (reasoning)
- PIQA (physical reasoning)
- WinoGrande (coreference)

Uses lm-evaluation-harness via the simple "lm_eval" package if available,
or implements direct perplexity-based scoring as a fallback.

Run on a single GPU after pretraining is complete (or on any checkpoint).
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import time

import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to VTX-300M checkpoint dir (or Hub repo id)")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    p.add_argument("--tasks", nargs="+", default=["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"])
    p.add_argument("--limit", type=int, default=None, help="Limit examples per task (for smoke)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default="vtx_300m_eval.json")
    return p.parse_args()


@torch.no_grad()
def score_choices(model, tokenizer, context: str, choices: list[str], device: str) -> list[float]:
    """For each choice, compute log p(choice | context)."""
    scores = []
    max_pos = getattr(model.config, "max_position_embeddings", 2048)
    ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    for choice in choices:
        full = context + choice
        full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if full_ids.shape[1] > max_pos:
            continue
        n_ctx = ctx_ids.shape[1]
        out = model(input_ids=full_ids, labels=None, chunk_size=0)
        logits = out.logits
        shift_logits = logits[0, :-1, :].float()
        shift_labels = full_ids[0, 1:]
        T = shift_labels.shape[0]
        n_choice = T - n_ctx + 1
        if n_choice <= 0:
            scores.append(0.0)
            continue
        ce = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="none"
        )
        choice_loss = ce[n_ctx - 1:].sum().item()
        scores.append(-choice_loss)
    return scores


def eval_hellaswag(model, tokenizer, device: str, limit=None, batch_size=8) -> dict:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        ctx = ex["ctx_a"] + " " + ex["ctx_b"]
        endings = ex["endings"]
        label = int(ex["label"])
        scores = score_choices(model, tokenizer, ctx, endings, device)
        if not scores:
            continue
        pred = int(np.argmax(scores))
        if pred == label:
            correct += 1
        total += 1
    return {"name": "hellaswag", "acc": correct / max(1, total), "n": total}


def eval_arc(model, tokenizer, device: str, name: str, limit=None) -> dict:
    from datasets import load_dataset
    config = "ARC-Easy" if "easy" in name else "ARC-Challenge"
    ds = load_dataset("allenai/ai2_arc", config, split="test", trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        ctx = ex["question"]
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        ans = ex["answerKey"]
        try:
            gold = labels.index(ans)
        except ValueError:
            continue
        scores = score_choices(model, tokenizer, ctx, choices, device)
        if not scores:
            continue
        pred = int(np.argmax(scores))
        if pred == gold:
            correct += 1
        total += 1
    return {"name": name, "acc": correct / max(1, total), "n": total}


def eval_piqa(model, tokenizer, device: str, limit=None) -> dict:
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        ctx = ex["goal"]
        choices = [ex["sol1"], ex["sol2"]]
        gold = int(ex["label"])
        scores = score_choices(model, tokenizer, ctx, choices, device)
        if not scores:
            continue
        pred = int(np.argmax(scores))
        if pred == gold:
            correct += 1
        total += 1
    return {"name": "piqa", "acc": correct / max(1, total), "n": total}


def eval_winogrande(model, tokenizer, device: str, limit=None) -> dict:
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        ctx = ex["sentence"]
        opt1, opt2 = ex["option1"], ex["option2"]
        ans = int(ex["answer"]) - 1
        s1 = ctx.replace("_", opt1)
        s2 = ctx.replace("_", opt2)
        scores = score_choices(model, tokenizer, "", [s1, s2], device)
        if not scores:
            continue
        pred = int(np.argmax(scores))
        if pred == ans:
            correct += 1
        total += 1
    return {"name": "winogrande", "acc": correct / max(1, total), "n": total}


EVAL_FNS = {
    "hellaswag": eval_hellaswag,
    "arc_easy": lambda *a, **kw: eval_arc(*a, name="arc_easy", **kw),
    "arc_challenge": lambda *a, **kw: eval_arc(*a, name="arc_challenge", **kw),
    "piqa": eval_piqa,
    "winogrande": eval_winogrande,
}


def main():
    args = parse_args()
    print(f"[eval] loading model from {args.ckpt}", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import VortexForCausalLM, VortexConfig
    from config import VortexArch, TokenizerProfile

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cfg = VortexConfig()
    model = VortexForCausalLM.from_pretrained(args.ckpt, config=cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"[eval] model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {device}", flush=True)

    results = []
    for task in args.tasks:
        if task not in EVAL_FNS:
            print(f"[eval] unknown task: {task}, skipping", flush=True)
            continue
        print(f"[eval] running {task}...", flush=True)
        t0 = time.time()
        try:
            r = EVAL_FNS[task](model, tok, device, limit=args.limit, batch_size=args.batch)
            r["time_s"] = round(time.time() - t0, 1)
            results.append(r)
            print(f"  {task}: acc={r['acc']*100:.2f}%  n={r['n']}  ({r['time_s']}s)", flush=True)
        except Exception as e:
            print(f"  {task} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"name": task, "error": str(e)})

    avg = np.mean([r["acc"] for r in results if "acc" in r])
    print(f"\n[eval] avg acc: {avg*100:.2f}%  ({len(results)} tasks)", flush=True)
    out = {"model": args.ckpt, "avg_acc": float(avg), "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
