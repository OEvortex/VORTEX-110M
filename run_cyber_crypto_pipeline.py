import os
import sys
import glob
import json
import time
import shutil
import subprocess
from pathlib import Path

# Setup environment
os.environ["HF_HUB_DISABLE_COLAB_SECRETS"] = "1"
if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
os.environ["PYTHONUNBUFFERED"] = "1"

print("=" * 70, flush=True)
print("VORTEX-50M: DOMAIN-SPECIFIC CYBERSECURITY & CRYPTOGRAPHY SFT PIPELINE", flush=True)
print("=" * 70, flush=True)

out_dir = "/content/sft_output"
export_dir = "/content/vortex-50m-hf"
eval_file = "/content/vortex_50m_eval.json"

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

# Phase 1: Run SFT Training on VTXAI/cyber-crypto-qa-50k
print("\n>>> Phase 1: SFT Training on VTXAI/cyber-crypto-qa-50k (Cybersecurity & Cryptography)", flush=True)
sft_cmd = [
    sys.executable, "-u", "/content/VORTEX-110M/src/sft.py",
    "--base", "VTXAI/vortex-50m-16k",
    "--tokenizer", "VTXAI/vortex-50m-16k",
    "--dataset", "VTXAI/cyber-crypto-qa-50k",
    "--max-tokens", "10000000",
    "--epochs", "1",
    "--batch", "8",
    "--grad-accum", "2",
    "--block", "1024",
    "--lr", "3e-5",
    "--min-lr", "3e-6",
    "--warmup", "30",
    "--num-workers", "0",
    "--out-dir", out_dir,
    "--val-every", "200",
    "--val-batches", "10",
    "--save-every", "500",
]

t_start = time.time()
p = subprocess.Popen(sft_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
p.wait()
if p.returncode != 0:
    raise RuntimeError(f"SFT failed with return code {p.returncode}")
print(f"\n[Phase 1 complete in {round(time.time() - t_start, 1)}s]", flush=True)

# Phase 2: Find checkpoint & export to HF format & Push to Hub
print("\n>>> Phase 2: Export to HF format & Push to Hub (VTXAI/vortex-50m)", flush=True)
ckpts = glob.glob(os.path.join(out_dir, "step_*"))
if not ckpts:
    raise RuntimeError(f"No checkpoints found in {out_dir}")

def get_step(path_str):
    try:
        return int(os.path.basename(path_str).split("_")[1])
    except Exception:
        return 0

latest_ckpt = max(ckpts, key=get_step)
print(f"[export] Using latest checkpoint: {latest_ckpt}", flush=True)
tok_dir = os.path.join(latest_ckpt, "tokenizer")

if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
os.makedirs(export_dir, exist_ok=True)

readme_content = """---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- vortex
- cybersecurity
- cryptography
- quantum-cryptography
- on-device
- sft
---

# Vortex-50M (Cybersecurity & Cryptography Specialist)

**Vortex-50M** is an on-device, sub-50M parameter decoder-only transformer with a 16,384-token vocabulary, fine-tuned (SFT) as a **domain-specific technical assistant for Cybersecurity, Post-Quantum Cryptography, and Security Protocols** on [`VTXAI/cyber-crypto-qa-50k`](https://huggingface.co/datasets/VTXAI/cyber-crypto-qa-50k) from the base pretrained model [`VTXAI/vortex-50m-16k`](https://huggingface.co/VTXAI/vortex-50m-16k).

## Model Architecture
- **Parameters**: 49,846,528 (~49.85M, within 50M budget)
- **Layers**: 18
- **Hidden dimension**: 512
- **Intermediate dimension**: 1072
- **Attention heads**: 8 query heads, 2 key/value heads (Grouped Query Attention)
- **Context length**: 2048
- **Vocabulary size**: 16,387 (tied embeddings, with ChatML control tokens)
- **Positional Embeddings**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMSNorm with QK-Norm

## Specialized Capabilities
Trained on verified technical Q&A across:
- **Network Security & Penetration Testing**: TLS/SSL, C2 detection, DDoS, authentication, vulnerability assessments.
- **Quantum & Classical Cryptography**: BB84 QKD, E91, Post-Quantum Cryptography (PQC), lattice cryptography, Harvest-Now-Decrypt-Later (HNDL), key distribution.
- **Application & Cloud Security**: Zero trust architectures, API security, input validation, encryption protocols.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "VTXAI/vortex-50m"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {"role": "user", "content": "What is two-factor authentication and why is it important?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.3, repetition_penalty=1.15)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
with open(os.path.join(export_dir, "README.md"), "w") as f:
    f.write(readme_content)

export_cmd = [
    sys.executable, "-u", "/content/VORTEX-110M/src/export_hf.py",
    "--ckpt", latest_ckpt,
    "--tokenizer", tok_dir,
    "--out", export_dir,
    "--push-to", "VTXAI/vortex-50m",
]

p = subprocess.Popen(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
p.wait()
if p.returncode != 0:
    raise RuntimeError(f"Export failed with return code {p.returncode}")

with open(os.path.join(export_dir, "README.md"), "w") as f:
    f.write(readme_content)
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=os.path.join(export_dir, "README.md"),
    path_in_repo="README.md",
    repo_id="VTXAI/vortex-50m"
)
print("[export] README.md uploaded to VTXAI/vortex-50m", flush=True)

# Phase 3: Run Benchmarks with eval_competition.py
print("\n>>> Phase 3: Benchmarking VTXAI/vortex-50m with eval_competition.py", flush=True)
eval_cmd = [
    sys.executable, "-u", "/content/VORTEX-110M/src/eval_competition.py",
    "--ckpt", export_dir,
    "--tokenizer", export_dir,
    "--tasks", "hellaswag", "arc_easy", "piqa", "winogrande",
    "--batch", "16",
    "--out", eval_file,
]

p = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
p.wait()
if p.returncode != 0:
    raise RuntimeError(f"Eval failed with return code {p.returncode}")

print("\n" + "=" * 70, flush=True)
print("EVALUATION RESULTS SUMMARY:", flush=True)
print("=" * 70, flush=True)
if os.path.exists(eval_file):
    with open(eval_file) as f:
        data = json.load(f)
        print(json.dumps(data, indent=2))

print("\nALL PHASES COMPLETED SUCCESSFULLY!", flush=True)
