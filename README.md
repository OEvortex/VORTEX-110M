---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- vortex
- sft
- cybersecurity
- cryptography
- pytorch
- transformers
---

# Vortex-50M (Cybersecurity & Cryptography Specialist)

A **49,846,528-parameter** instruction-tuned decoder-only Transformer, built from first principles on `torch.nn` and fine-tuned for specialized **cybersecurity, network defense, threat analysis, and cryptographic protocols**.

```
49,846,528 params  ·  512d × 18L  ·  8Q/2KV GQA  ·  vocab 16,387 (tied)
17% embedding  /  83% transformer blocks  ·  Strictly within 50M parameter budget
```

| Specification | Value |
|:---|:---|
| **Base Pretrained Model** | [`VTXAI/vortex-50m-16k`](https://huggingface.co/VTXAI/vortex-50m-16k) (trained from scratch on 1B tokens) |
| **Total Parameters** | **49,846,528** (99.69% of the 50,000,000 competition budget) |
| **Fine-Tuning Dataset** | [`VTXAI/cyber-crypto-balanced-qa`](https://huggingface.co/datasets/VTXAI/cyber-crypto-balanced-qa) |
| **Architecture Design** | Custom `VortexForCausalLM` (`trust_remote_code=True`), tied embeddings, QK-Norm, RoPE, ChatML |
| **Domain Focus** | Network security, CVE triage, symmetric/asymmetric cryptography, reverse engineering, web security |

---

## Benchmark Results

Evaluated directly via EleutherAI's `lm-evaluation-harness` across 4 standard multiple-choice and reasoning tasks using length normalization where applicable:

| Task / Benchmark | Primary Metric | Score | Stderr | Evaluation Grader |
|:---|:---:|:---:|:---:|:---|
| **PIQA** | `acc_norm` | **54.13%** | ±1.16% | Physical commonsense reasoning |
| **WinoGrande** | `acc` | **52.41%** | ±1.40% | Pronoun disambiguation & commonsense |
| **ARC-Easy** | `acc_norm` | **35.52%** | ±0.98% | Grade-school science question answering |
| **HellaSwag** | `acc_norm` | **27.23%** | ±0.44% | Hard commonsense NLI / continuation |
| **Scored Average** | - | **42.32%** | - | **Mean across all 4 competition tasks** |

> **Evaluation Methodology**: All benchmarks are scored using `src/eval_competition.py` with the official EleutherAI harness. No hosted inference APIs were touched during training or evaluation.

---

## Parameter Accounting & Budget

The parameter budget is verified from the live module tree using `src/param_count.py`:

```
==================================================================
  vortex-50m
==================================================================
  hidden                 512
  layers                 18
  heads                  8Q / 2KV
  head_dim               64
  intermediate           1072
  context                2048
  vocab                  16,387 (tied=True)
------------------------------------------------------------------
  TRAINABLE PARAMS       49,846,528
  budget                 50,000,000
  verdict                [PASS]  49.847M  (99.69% of budget)
------------------------------------------------------------------
  embedding                 8,390,144   16.8%
  attention                11,798,784   23.7%
  mlp                      29,638,656   59.5%
  norm                            512    0.0%
  ---                             ---
  in transformer blocks    41,456,384   83.2%
==================================================================
```

`lm_head` is tied to the input embedding table, ensuring maximum capacity is directed into the 18 transformer layers (83.2% of weights inside transformer blocks).

---

## Architectural Principles at 50M Parameters

- **Depth over Width via Compact Vocab**: A custom 16,384 BPE vocabulary costs only ~8.4M parameters in the embedding table (17% of budget). In contrast, standard 32K or 150K vocabularies consume 35% to 150%+ of a 50M budget before a single layer can be placed. This allows Vortex-50M to run **18 deep layers** at 512 hidden dimension.
- **Grouped-Query Attention (GQA 8/2)**: 8 query heads and 2 key/value heads provide a 4× KV cache reduction for fast on-device inference without sacrificing reasoning quality.
- **QK-Norm (RMSNorm on Q and K)**: Applied per head before attention computation to prevent attention entropy collapse and numerical instability at small scales.
- **SwiGLU MLP**: Intermediate dimension of 1,072 (2.09× hidden dimension) with SwiGLU activation for superior non-linear representation capacity.
- **ChatML Format & Special Control Tokens**: Native support for `<|im_start|>` and `<|im_end|>` sequence markers with assistant-only loss masking during instruction tuning.

---

## Quickstart & Inference

Vortex-50M runs natively with Hugging Face `transformers` using `trust_remote_code=True`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "VTXAI/vortex-50m"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# Format prompt using ChatML
messages = [
    {
        "role": "system",
        "content": "You are Vortex-50M, an expert cybersecurity and cryptography assistant."
    },
    {
        "role": "user",
        "content": "Explain why AES-GCM is preferred over AES-CBC with HMAC in modern network protocols."
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=True,
        temperature=0.3,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## Training & Fine-Tuning Pipeline

1. **Pretraining**: Pretrained from random initialization on 1 Billion tokens of curated high-quality web text (`cosmopedia-v2` + `fineweb-edu-dedup`) using cosine decay down to 10% peak LR.
2. **Supervised Instruction Tuning (SFT)**: Fine-tuned with ChatML masking on [`VTXAI/cyber-crypto-balanced-qa`](https://huggingface.co/datasets/VTXAI/cyber-crypto-balanced-qa) using AdamW (`lr=1.8e-5`, weight decay `0.01`, cosine scheduler).
3. **Loss Masking**: User prompts and system instructions are masked out of the loss calculation; gradients are computed strictly on assistant generation tokens.

---

## Verification & Integrity

The model passes all pre-flight and runtime consistency suites in `src/`:
- **Parity Check** (`src/test_hf_modeling.py`): Bit-exact agreement between custom `torch.nn` training code and Hugging Face `AutoModelForCausalLM`.
- **Roundtrip Check** (`src/test_export_hf.py`): Verified export, weight reload, and token generation consistency in clean environments.
- **Budget Compliance** (`src/param_count.py`): Verified 49,846,528 params ≤ 50,000,000 budget.
