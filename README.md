# VTX-300M

A from-scratch ~300M parameter decoder-only Transformer for pretraining and
post-training. Optimized for single-GPU training on **RTX 5090 Blackwell (32GB)**
with **bf16**, **torch.compile**, **GQA**, **FlashAttention-2**, and strict
**weight-tying**.

## Architecture

| Parameter | Value |
|---|---|
| Parameters | ~300M |
| Hidden size | 768 |
| Layers | 18 |
| Attention heads | 12 |
| KV heads (GQA) | 4 |
| Intermediate size | 2048 |
| Context length | 2048 |
| Tokenizer | Qwen3 (~151,670 vocab) |
| Norm | RMSNorm |
| Activation | SwiGLU |
| Position | RoPE |
| Attention | FlashAttention-2 (SDPA) |
| Weight tying | Yes (embed = lm_head) |

**GQA (Grouped Query Attention)**: 12 Q heads share 4 KV head groups, reducing
KV cache memory by 3x compared to MHA while maintaining quality.

## What you get out of the box

- **~300M params** with Qwen3-family tokenizer (~151,670 vocab)
- **Pretraining loop** with cosine LR, gradient accumulation, and checkpoint pushing
- **Distillation pipeline** via Arcee AI DistillKit (student -> 235B teacher logits)
- **Built-in evaluation** on HellaSwag, ARC, PIQA, WinoGrande
- **Hub-native** save/load via `transformers.PreTrainedModel`

## Requirements

- Python 3.10+
- NVIDIA GPU with **Blackwell (RTX 5090)** or **Ampere+** for SDPA + bf16
- CUDA 12.x

## Install

```bash
# Clone and cd
git clone https://github.com/OEvortex/VTX-300M.git && cd VTX-300M

# Create env
python -m venv .venv
source .venv/bin/activate

# Core deps
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers huggingface_hub datasets numpy scipy trackio

# Pretraining dataset (memmap shards) + FastAPI sandbox (optional)
pip install fastapi uvicorn pydantic

# Post-training distillation
pip install distill-kit
```

## Quick start: pretrain

The simplest run pulls data shards automatically from the Hub (`VTXAI/vtx-300m-data`).

```bash
cd src
python pretrain.py --config pretrain_config.json
```

### RTX 5090 (32GB VRAM) — recommended config

```bash
python pretrain.py \
  --steps 30000 \
  --warmup 1000 \
  --lr 6e-4 \
  --min-lr 6e-5 \
  --batch 8 \
  --grad-accum 4 \
  --block 2048 \
  --compile \
  --save-dir /tmp/vtx_300m_ckpt \
  --hub-repo VTXAI/vtx-300m \
  --push-every 3000 \
  --log-evr 25
```

Effective batch = 32, tokens/step = 65,536, total ~1.0B tokens over 30k steps.

If you OOM, drop `--batch` to 4 and raise `--grad-accum` to 8 (same effective
batch, halves activation memory).

### Larger context (4096+)

```bash
python pretrain.py \
  --steps 30000 \
  --warmup 1000 \
  --lr 4e-4 \
  --min-lr 4e-5 \
  --batch 4 \
  --grad-accum 8 \
  --block 4096 \
  --compile \
  --save-dir /tmp/vtx_300m_ckpt \
  --hub-repo VTXAI/vtx-300m \
  --push-every 3000 \
  --log-every 25
```

### Data setup

Two options:

1. **AUTO (default)** — downloads `.bin` shards from `VTXAI/vtx-300m-data`
   on the Hub. Just use `--shards AUTO` or omit the flag entirely.
2. **Local shards** — point to pre-tokenized `uint32` memmap files:

```bash
python pretrain.py --shards /data/shard_0000.bin /data/shard_0001.bin
```

### Config reference (`pretrain_config.json`)

```jsonc
{
  "steps": 30000,             // Total training steps
  "warmup": 1000,             // LR warmup steps
  "lr": 6e-4,                 // Peak learning rate
  "min_lr": 6e-5,             // Cosine LR floor
  "weight_decay": 0.1,
  "beta1": 0.9,
  "beta2": 0.95,
  "grad_clip": 1.0,           // Max gradient norm
  "batch": 8,                 // Per-device batch size
  "grad_accum": 4,            // Gradient accumulation steps
  "block": 2048,              // Sequence length
  "seed": 42,
  "shards": "AUTO",           // "AUTO" or list of .bin paths
  "hub_repo": "VTXAI/vtx-300m",
  "push_every": 3000,
  "log_every": 25,
  "save_dir": "/tmp/vtx_300m_ckpt",
  "compile": true
}
```

Effective batch size = `batch * grad_accum`.
Tokens per step = `effective_batch * block`.

## Monitor with Trackio

Set these env vars to log to a HF Space:

```bash
export TRACKIO_SPACE_ID="VTXAI/vtx-300m-trackio"
export TRACKIO_PROJECT="vtx-300m"
```

If the package or env vars are missing, Trackio is silently skipped.

## Post-training: distill

After pretraining, distill from the 235B teacher logits into your student.

```bash
cd src
python distill.py
```

This launches Arcee AI's **DistillKit** with `distill_config.yml`:

- **Teacher**: `arcee-ai/Qwen3-235B-Logits-Packed-8192` (prepacked logits)
- **Student**: `VTXAI/vtx-300m`
- **Loss**: `0.5 * cross_entropy + 0.5 * KL` at temperature=1.0
- **Output**: `VTXAI/vtx-300m-distilled`

Override defaults:

```bash
python distill.py \
  --student-ckpt VTXAI/vtx-300m \
  --distill-config distill_config.yml \
  --out VTXAI/vtx-300m-distilled \
  --steps 8000 \
  --lr 2e-5 \
  --batch 2 \
  --grad-accum 8 \
  --max-seq-len 2048
```

## Evaluate

```bash
python eval_benchmarks.py \
  --ckpt VTXAI/vtx-300m \
  --tokenizer Qwen/Qwen3-4B \
  --tasks hellaswag arc_easy arc_challenge piqa winogrande \
  --batch 8 \
  --out vtx_300m_eval.json
```

## GPU tips (RTX 5090 Blackwell, 32GB)

- **2048 context (recommended)**: use `batch=8, grad_accum=4` (effective=32).
  Model + optimizer ≈ 4.5GB, activations ≈ 10GB, total ≈ 15GB.
- **4096 context**: drop `batch` to 4, raise `grad_accum` to 8.
- **8192 context**: use `batch=2, grad_accum=16` if it fits; OOM is likely.
  Consider gradient checkpointing for very long sequences.
- **Data pipeline**: 8 DataLoader workers + prefetch=4 keeps Blackwell's
  CUDA cores fed. Increase `--num_workers` only if your CPU stalls.
- **Compile**: `--compile` cuts wall-time ~20-30% but increases peak memory
  slightly. Disable if OOM.
- **Precision**: bf16 via `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
  Blackwell/Ampere+ GPUs have native bf16 Tensor Core support.
- **GQA advantage**: 4 KV heads (vs 12 in MHA) means ~3x smaller KV cache
  during inference and lower memory during training attention.

## Parameter breakdown

| Component | Params |
|---|---|
| Embedding (151,670 x 768) | 116.5M |
| 18x Attention (GQA) | ~101M |
| 18x MLP (SwiGLU, 2048) | ~84M |
| RMSNorm + lm_head (tied) | ~0.2M |
| **Total** | **~302M** |

## Project structure

```
src/
  model.py              VTX-300M config, GQA attention, MLP, PreTrainedModel wrapper
  config.py             VortexArch, HubConfig, TokenizerProfile
  pretrain.py           Main training loop (single GPU, RTX 5090 optimized)
  dataset.py            MMapDataset — reads uint32 memmap shards
  pretrain_config.json  Default training hyperparameters
  distill.py            DistillKit runner (post-training)
  distill_config.yml    Distillation hyperparameters
  eval_benchmarks.py    HellaSwag / ARC / PIQA / WinoGrande evals
  sandbox_server.py     Optional FastAPI sandbox
```

## License

MIT (or your chosen license).
