#!/usr/bin/env python
"""
Parameter budget report.

The competition rule is explicit: NO MORE THAN 50,000,000 total trainable
parameters, INCLUDING token embeddings and the output head. This script prints
that number from the real module tree (not an analytic formula) alongside the
config, and exits non-zero if the budget is blown -- so it doubles as a CI gate.

    python param_count.py                      # the submitted model
    python param_count.py --arch vortex-50m-16k
    python param_count.py --ckpt /root/vortex_50m_ckpt/step_15258
    python param_count.py --arch vortex-50m-16k --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PARAM_BUDGET = 50_000_000


def count(model):
    """Trainable-parameter total, split so the big consumers are visible.

    'Trainable' is the operative word: a frozen embedding would not count, and
    would not help you anyway at this budget.
    """
    total = trainable = 0
    by_group = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "other": 0}
    tied_lm_head = 0

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        if "embed_tokens" in name:
            by_group["embedding"] += n
        elif "q_proj" in name or "k_proj" in name or "v_proj" in name \
                or "o_proj" in name or "q_norm" in name or "k_norm" in name:
            by_group["attention"] += n
        elif "gate_proj" in name or "up_proj" in name or "down_proj" in name:
            by_group["mlp"] += n
        elif "norm" in name:
            by_group["norm"] += n
        elif "lm_head" in name:
            tied_lm_head += n
        else:
            by_group["other"] += n

    return total, trainable, by_group, tied_lm_head


def report(name, model, cfg, verbose=True):
    total, trainable, groups, tied = count(model)
    cfg_params = cfg.n_params() if hasattr(cfg, "n_params") else None

    if verbose:
        w = 66
        print("=" * w)
        print(f"  {name}")
        print("=" * w)
        print(f"  {'hidden':<22} {cfg.hidden_size}")
        print(f"  {'layers':<22} {cfg.num_hidden_layers}")
        print(f"  {'heads':<22} {cfg.num_attention_heads}Q / "
              f"{getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}KV")
        print(f"  {'head_dim':<22} {cfg.head_dim}")
        print(f"  {'intermediate':<22} {cfg.intermediate_size}")
        print(f"  {'context':<22} {cfg.max_position_embeddings}")
        print(f"  {'vocab':<22} {cfg.vocab_size:,} "
              f"(tied={getattr(cfg, 'tie_word_embeddings', True)})")
        print(f"  {'qk_norm':<22} {getattr(cfg, 'qk_norm', None)}")
        print(f"  {'zero_init_residual':<22} {getattr(cfg, 'zero_init_residual', None)}")
        print("-" * w)
        print(f"  {'TRAINABLE PARAMS':<22} {trainable:,}")
        print(f"  {'budget':<22} {PARAM_BUDGET:,}")
        over = trainable - PARAM_BUDGET
        verdict = "PASS" if over <= 0 else "FAIL"
        print(f"  {'verdict':<22} [{verdict}]  "
              f"{trainable / 1e6:.3f}M  "
              f"({trainable / PARAM_BUDGET * 100:.2f}% of budget)")
        if over > 0:
            print(f"  {'':22} OVER BY {over:,} ({over / 1e6:.3f}M)")
        print("-" * w)
        print(f"  {'embedding':<22} {groups['embedding']:>12,}  "
              f"{groups['embedding'] / trainable * 100:5.1f}%")
        print(f"  {'attention':<22} {groups['attention']:>12,}  "
              f"{groups['attention'] / trainable * 100:5.1f}%")
        print(f"  {'mlp':<22} {groups['mlp']:>12,}  "
              f"{groups['mlp'] / trainable * 100:5.1f}%")
        print(f"  {'norm':<22} {groups['norm']:>12,}  "
              f"{groups['norm'] / trainable * 100:5.1f}%")
        if tied:
            print(f"  {'(lm_head, tied)':<22} {tied:>12,}  shared with embedding")
        layers = trainable - groups["embedding"] - tied
        print(f"  {'---':<22} {'---':>12}")
        print(f"  {'in transformer blocks':<22} {layers:>12,}  "
              f"{layers / trainable * 100:5.1f}%")
        if cfg_params is not None and cfg_params != trainable:
            print(f"  {'NOTE: analytic != real':<22} {cfg_params:,} vs {trainable:,}")
        print("=" * w)
        print()

    return {
        "name": name,
        "trainable_params": trainable,
        "total_params": total,
        "budget": PARAM_BUDGET,
        "within_budget": trainable <= PARAM_BUDGET,
        "analytic_params": cfg_params,
        "groups": groups,
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "head_dim": cfg.head_dim,
            "intermediate_size": cfg.intermediate_size,
            "max_position_embeddings": cfg.max_position_embeddings,
            "vocab_size": cfg.vocab_size,
            "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", True),
            "qk_norm": getattr(cfg, "qk_norm", None),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default=None, help="Preset name, e.g. vortex-50m-16k")
    p.add_argument("--ckpt", default=None, help="Checkpoint dir or Hub id")
    p.add_argument("--json", action="store_true", help="Emit JSON only")
    args = p.parse_args()

    if not args.arch and not args.ckpt:
        args.arch = "vortex-50m-16k"   # the submitted configuration

    from config import VortexArch
    from model import VortexForCausalLM

    out = []
    if args.ckpt:
        m = VortexForCausalLM.from_pretrained(args.ckpt)
        out.append(report(args.ckpt, m, m.cfg, verbose=not args.json))
    else:
        arch = VortexArch.from_name(args.arch)
        m = VortexForCausalLM(arch)
        out.append(report(arch.name_or_path, m, arch, verbose=not args.json))

    if args.json:
        print(json.dumps(out, indent=2))

    if any(not r["within_budget"] for r in out):
        sys.exit(1)


if __name__ == "__main__":
    main()
