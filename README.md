# Vortex-110M

A from-scratch ~110M parameter decoder-only Transformer for pretraining and
post-training. Optimized for single-GPU training with **bf16**, **torch.compile**,
**SDPA/FlashAttention-2**, and strict **weight-tying**.

## What you get out of the box

- **Architecture**: 512 hidden, 12 layers, 8 heads, SwiGLU, RoPE, RMSNorm
- **~111M params** with Qwen3-family tokenizer (~151,670 vocab)
- **Pretraining loop** with cosine LR, gradient accumulation, and checkpoint pushing
- **Distillation pipeline** via Arcee AI DistillKit (student -> 235B teacher logits)
- **Built-in evaluation** on HellaSwag, ARC, PIQA, WinoGrande
- **Hub-native** save/load via `transformers.PreTrainedModel`

## Requirements

- Python 3.10+
- NVIDIA GPU with **Ampere+ (A100 / RTX 30xx+)** for FlashAttention-2 + bf16
- CUDA 11.8+ / 12.x
- ~80GB VRAM for the default pretraining config (batch=8, block=2048)

## Install

```bash
# Clone and cd
git clone https://github.com/OEvortex/VORTEX-110M.git && cd VORTEX-110M

# Create env
python -m venv .venv
source .venv/bin/activate

# Core deps
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers huggingface_hub datasets numpy scipy trackio

# Pretraining dataset (memmap shards) + FastAPI sandbox (optional)
pip install fastapi uvicorn pydantic

# Post-training distillation
pip install distill-kit
```

## Quick start: pretrain

The simplest run pulls data shards automatically from the Hub (`VTXAI/vortex-110m-data`).

```bash
# Run from inside src/
cd src
python pretrain.py --config pretrain_config.json
```

Or override everything from the CLI:

```bash
python pretrain.py \
  --steps 30000 \
  --warmup 1000 \
  --lr 6e-4 \
  --min-lr 6e-5 \
  --batch 16 \
  --grad-accum 4 \
  --block 2048 \
  --compile \
  --save-dir /tmp/vortex_ckpt \
  --hub-repo VTXAI/vortex-110m \
  --push-every 1500 \
  --log-every 25
```

### Data setup

Two options:

1. **AUTO (default)** — downloads `.bin` shards from `VTXAI/vortex-110m-data`
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
  "batch": 16,                // Per-device batch size
  "grad_accum": 4,            // Gradient accumulation steps
  "block": 2048,              // Sequence length
  "seed": 42,
  "shards": "AUTO",           // "AUTO" or list of .bin paths
  "hub_repo": "VTXAI/vortex-110m",
  "push_every": 1500,         // Push checkpoint every N steps
  "log_every": 25,
  "save_dir": "/tmp/vortex_ckpt",
  "compile": true             // torch.compile (mode=default)
}
```

Effective batch size = `batch * grad_accum`.
Tokens per step = `effective_batch * block`.

## Monitor with Trackio

Set these env vars to log to a HF Space:

```bash
export TRACKIO_SPACE_ID="VTXAI/vortex-110m-trackio"
export TRACKIO_PROJECT="vortex-110m"
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
- **Student**: `VTXAI/vortex-110m`
- **Loss**: `0.5 * cross_entropy + 0.5 * KL` at temperature=1.0
- **Output**: `VTXAI/vortex-110m-distilled`

Override defaults:

```bash
python distill.py \
  --student-ckpt VTXAI/vortex-110m \
  --distill-config distill_config.yml \
  --out VTXAI/vortex-110m-distilled \
  --steps 8000 \
  --lr 2e-5 \
  --batch 2 \
  --grad-accum 8 \
  --max-seq-len 8192
```

## Evaluate

```bash
python eval_benchmarks.py \
  --ckpt VTXAI/vortex-110m \
  --tokenizer Qwen/Qwen3-4B \
  --tasks hellaswag arc_easy arc_challenge piqa winogrande \
  --batch 8 \
  --out vortex_eval.json
```

## GPU tips

- **Batch sizing**: start small and scale up. On A100-80GB, `batch=8, grad_accum=8`
  is the published starting point. Reduce `batch` if you OOM.
- **Sequence length**: longer `--block` means more VRAM per sample.
- **Compile**: `--compile` cuts wall-time but increases peak memory slightly.
- **Precision**: training runs in `bf16` via `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
  Make sure your GPU supports bf16 (Ampere+).

## Project structure

```
src/
  model.py              Vortex config, attention, MLP, PreTrainedModel wrapper
  config.py             VortexArch, HubConfig, TokenizerProfile
  pretrain.py           Main training loop (single GPU)
  dataset.py            MMapDataset — reads uint32 memmap shards
  pretrain_config.json  Default training hyperparameters
  distill.py            DistillKit runner (post-training)
  distill_config.yml    Distillation hyperparameters
  eval_benchmarks.py    HellaSwag / ARC / PIQA / WinoGrande evals
  sandbox_server.py     Optional FastAPI sandbox
```

## License

MIT (or your chosen license).
