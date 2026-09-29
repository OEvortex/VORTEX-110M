# Vortex-50M

A **from-scratch, ≤50M-parameter** decoder-only Transformer, written to own its
architecture and its vocabulary end to end. No HF architecture is ported — every
tensor is hand-defined on `torch.nn`.

```
49,497,216 params  ·  640d × 12L  ·  10Q/2KV GQA  ·  vocab 8,192 (tied)
11% embedding  /  89% transformer layers
```

## Why the vocab is 8K — and why that makes the model *wider*

**English-only corpus.** That single decision shapes the whole model.

With tied embeddings the token embedding table costs exactly `vocab × hidden`.
At this scale it is the biggest tensor in the model — and the biggest lever on
what the rest of the budget buys:

| vocab | hidden | embed table | % of 50M | what the rest buys you |
|---|---|---|---|---|
| 151,670 (Qwen3) | 512 | 77.7M | 155% | **impossible** |
| 32,768 | 512 | 16.8M | 34% | 512d × 12L — only 66% in layers |
| 16,384 | 512 | 8.4M | 17% | 512d × 18L — 83% in layers |
| **8,192** | **640** | **5.2M** | **11%** | **640d × 12L — 89% in layers** |

Note the constraint **flips** as the vocab shrinks. At 32K the table dominates
and hidden size is the limit. At 8K the table is nearly free, so the same 50M
budget buys a **wider** model. Shrinking the vocab doesn't just save
parameters — it converts embedding parameters into transformer capacity.

Most of Qwen3's 151K merges are multilingual scripts, emoji and CJK that will
never appear in English prose. 8K–16K is the proven band: TinyStories trains
1M–33M English models on a 10K vocabulary with excellent results.

**The trade-off:** a smaller vocab compresses English less efficiently, so the
same corpus yields *more* tokens. Roughly 4.0–4.5 chars/token at 32K vs
3.2–3.8 at 8K–16K. The parameter win is certain; the token cost is
corpus-specific. Measure it before committing:

```bash
python src/train_tokenizer.py --out ./vortex-tok-8k --stats
```

If you ever add code or non-English text, switch to the `vortex-50m-32k`
preset and retrain — rare tokens fragment badly in an 8K English vocab.

## Architecture

| | |
|---|---|
| **Parameters** | **49.50M** (budget: 50.00M) |
| Hidden size | 640 |
| Layers | 12 |
| Attention | 10 query / 2 KV heads (GQA, 5 groups) |
| Head dim | 64 |
| Intermediate | 1,408 (2.20× hidden, SwiGLU) |
| Context | 2,048 (RoPE θ=10,000) |
| Vocab | 8,192 (**tied** embeddings) |
| Norm | RMSNorm (pre-norm, fp32 compute) |
| Attention | SDPA → FlashAttention-2 on Ampere/Blackwell |
| Biases | none |

### Choices that matter at this scale

- **QK-Norm** (LLaMA-3 / Qwen3) — per-head RMSNorm on q and k before the
  attention matmul. Without it, small models hit *attention entropy collapse*
  early: a few heads saturate, their softmax goes one-hot, gradients vanish,
  and those heads are dead for the rest of the run. Costs 2×64 params per layer.
- **Zero-initialized residual outputs** — `o_proj` and `down_proj` start at
  exactly zero, so every block is an *identity* at init and the untrained
  network is a clean passthrough. Without this, 12 stacked blocks compound
  their variances and saturate the residual stream before step 0. Verified:
  initial loss is `ln(8192) = 9.0`, not the ~459 you get from default init.
- **GQA 10/2** — 5× smaller KV cache than MHA.
- **Tied embeddings** — saves an entire 5.2M-parameter matrix.
- **Chunked cross-entropy** — logits `(B, T, 8192)` are never materialized
  during training; the loss accumulates in time-chunks. This is the largest
  single memory term in the step and it is entirely avoidable.

## Presets

All exact and budget-checked on import.

| preset | params | shape | when to use |
|---|---|---|---|
| **`vortex-50m`** | **49.50M** | 640d × 12L, 10/2, 8K vocab | **default — English** |
| `vortex-50m-16k` | 49.84M | 512d × 18L, 8/2, 16K vocab | deeper; vocab margin for names/numbers |
| `vortex-50m-wide` | 48.53M | 768d × 8L, 12/4, 8K vocab | short docs; width over depth |
| `vortex-50m-deep` | 49.84M | 512d × 18L, 8/2, **4K ctx** | lots of tokens; multi-step reasoning |
| `vortex-40m` | 44.06M | 384d × 24L, 6/2, 16K vocab | headroom to grow the vocab later |
| `vortex-50m-32k` | 49.43M | 512d × 12L, 8/2, 32K vocab | **only** if adding code/multilingual |
| `vortex-test` | 0.86M | 128d × 4L | smoke tests |

## Pipeline

The order matters: **tokenizer → retokenize → train**. The old `.bin` shards
become invalid the moment the vocab changes.

```bash
# 1. Train the 32K tokenizer on your raw corpus
python src/train_tokenizer.py --out ./vortex-tok-8k --vocab-size 32768 \
    /path/to/corpus/*.jsonl

#    optional: sanity-check round-trip and compression
python src/train_tokenizer.py --out ./vortex-tok-8k --stats

# 2. Rebuild the .bin shards with the new vocab
python src/retokenize.py --tokenizer ./vortex-tok-8k --out ./data8k \
    /path/to/corpus/*.jsonl

# 3. Verify the architecture (gates a training launch; exits non-zero on failure)
python src/verify_arch.py

# 4. Train
python src/pretrain.py --config src/pretrain_config.json \
    --tokenizer ./vortex-tok-8k --shards ./data8k/shard_*.bin
```

### Shard / vocab safety check

`pretrain.py` reads the real maximum token id out of the first shards and
refuses to start if it exceeds the model's vocab. Training on data tokenized
with a *different* tokenizer is otherwise a silent failure — ids out of range
reach the embedding table and either crash or wrap into valid-looking tokens.

## Training

Defaults in `src/pretrain_config.json` are tuned for a 32GB card (RTX 5090 /
A100). The model is small enough that a full run is cheap:

```bash
python src/pretrain.py --config src/pretrain_config.json \
    --arch vortex-50m \
    --tokenizer ./vortex-tok-8k \
    --shards ./data8k/shard_*.bin \
    --steps 60000 --lr 6e-4 --batch 8 --grad-accum 4 --block 2048 \
    --compile --push-every 3000
```

- **Optimizer**: Lion (1 state, ~half the optimizer VRAM of AdamW). Use an LR
  ~3× smaller than you would with AdamW.
- **Precision**: bf16 autocast, fp32 RMSNorm and RoPE tables.
- **Gradient checkpointing**: on by default (the model is small; this buys
  context length, not a necessity).

Useful flags:

| flag | meaning |
|---|---|
| `--arch` | preset name (`vortex-50m`, `vortex-40m`, …) |
| `--tokenizer` | tokenizer dir or Hub id **matching the shards** |
| `--rope-theta` | raise to `1000000` when extending context past 8K |
| `--no-resume` | ignore existing checkpoints |

## Verification

`src/verify_arch.py` is a 38-check suite that must pass before training. It
covers the things that actually break small models:

- analytic param count **equals** the real module tree, and is ≤ 50M
- blocks are an exact identity at init (zero-init residuals)
- causality: prefix logits are bit-identical when future tokens change
- RoPE is a true rotation (norm-preserving) and satisfies the relative-offset property
- chunked cross-entropy is chunk-size invariant, and an all-masked batch returns
  0 rather than NaN
- gradient checkpointing is value-neutral
- gradient flows to 100% of parameter tensors
- GQA divisibility is enforced at construction (a 7-head/2-KV config crashes
  deep inside SDPA otherwise)
- `save_pretrained` → `from_pretrained` round trip is bit-exact and stays tied

```
ALL CHECKS PASSED
```

## Files

| file | role |
|---|---|
| `src/config.py` | `VortexArch`, presets, exact param accounting, budget guard |
| `src/model.py` | the architecture — attention, MLP, blocks, RoPE, QK-Norm |
| `src/train_tokenizer.py` | trains the 32K BPE, reports compression stats |
| `src/retokenize.py` | rebuilds `uint32` `.bin` shards with the new vocab |
| `src/dataset.py` | memory-mapped streaming dataset |
| `src/pretrain.py` | training loop, checkpointing, Hub push |
| `src/verify_arch.py` | the 38-check verification suite |
| `src/eval_benchmarks.py` | HellaSwag / ARC / PIQA / WinoGrande |
| `VTX_300M_cloud_train.ipynb` | cloud training notebook |

## Requirements

- Python 3.10+, PyTorch 2.5+ (for SDPA `enable_gqa`), CUDA 12.x
- `transformers>=4.45`, `tokenizers`, `safetensors`, `numpy`
