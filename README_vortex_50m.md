---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- vortex
- sft
- vortex-50k
---

# Vortex-50M

**Vortex-50M** is a ~49.8M parameter decoder-only transformer with a 16,384-token vocabulary, fine-tuned (SFT) on [`OEvortex/Vortex-50k`](https://huggingface.co/datasets/OEvortex/Vortex-50k) from the base pretrained model [`VTXAI/vortex-50m-16k`](https://huggingface.co/VTXAI/vortex-50m-16k).

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

## Benchmark Results (`eval_competition.py`)

Evaluated with `lm-evaluation-harness` and WikiText-103:

| Benchmark | Primary Metric | Score | Stderr |
|:---|:---:|:---:|:---:|
| **PIQA** | `acc_norm` | **54.84%** | ±1.16% |
| **WinoGrande** | `acc` | **51.78%** | ±1.40% |
| **ARC-Easy** | `acc_norm` | **35.23%** | ±0.98% |
| **HellaSwag** | `acc_norm` | **27.81%** | ±0.45% |
| **Scored Average (4 tasks)** | - | **42.41%** | - |
| **WikiText-103** | Perplexity | **244.82** | 5.50 nats (7.94 bpb) |

## Quickstart & Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "VTXAI/vortex-50m"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {"role": "user", "content": "Which river runs through London?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.4, repetition_penalty=1.15)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
