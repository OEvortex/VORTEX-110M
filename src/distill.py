"""
Vortex-110M post-distillation runner.

Calls Arcee AI's DistillKit with the prepared YAML config, using the trained
Vortex-110M as the student and arcee-ai/Qwen3-235B-Logits-Packed-8192 as the
teacher (pre-packed logit dataset with the exact compression spec the user
specified: d=151936 [vocab size, dynamically mirrored from student model],
k=128, exact_k=32, exact_dtype=float32, polynomial_terms=[0,1,2],
delta_encoding=True, error_diffusion=False).

Loss: 0.5*cross_entropy + 0.5*kl at temperature 1.0
"""
from __future__ import annotations
import os
import sys
import subprocess
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student-ckpt", default="VTXAI/vortex-110m",
                   help="HF repo id or local path of the student Vortex-110M")
    p.add_argument("--distill-config", default="distill_config.yml")
    p.add_argument("--out", default="VTXAI/vortex-110m-distilled")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=8192)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[distill] student ckpt: {args.student_ckpt}")
    print(f"[distill] distill config: {args.distill_config}")
    print(f"[distill] output: {args.out}")

    # We invoke Arcee DistillKit's CLI. DistillKit is `pip install distill-kit`
    # (arcee-ai/distill-kit on github). It ships `distillkit` or `distill` as
    # an entrypoint. We pass our YAML.
    cmd = [
        "distillkit", "run",
        "--config", args.distill_config,
        "--student.from_pretrained", args.student_ckpt,
        "--student.save_dir", args.out,
        "--training.steps", str(args.steps),
        "--training.lr", str(args.lr),
        "--training.per_device_train_batch_size", str(args.batch),
        "--training.gradient_accumulation_steps", str(args.grad_accum),
        "--training.max_seq_length", str(args.max_seq_len),
    ]
    print(f"[distill] running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
