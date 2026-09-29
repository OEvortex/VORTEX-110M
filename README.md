# Vortex-50M-16k

A **from-scratch, 49,844,992-parameter** decoder-only Transformer, trained from
random initialization on 1B tokens. No pretrained weights, no fine-tuning, no
distillation. The architecture and the vocabulary are both built here from
first principles on `torch.nn` — no HuggingFace architecture is ported.

```
49,844,992 params  ·  512d × 18L  ·  8Q/2KV GQA  ·  vocab 16,384 (tied)
17% embedding  /  83% transformer blocks
```

| | |
|---|---|
| **Parameters** | **49,844,992** — 99.69% of the 50,000,000 budget |
| **Hardware** | 1× NVIDIA A100-SXM4-40GB (Modal.com) |
| **Training time** | **3.50 h** for 1B tokens @ 79.4k tok/s |
| **Approx. compute** | **2.99 × 10¹⁷ FLOPs** (6ND) ≈ 3.5 A100-hours |
| **Tokenizer training** | 34 min, CPU only (no GPU) |
| **Trained from** | random init, `torch.manual_seed(42)` |

---

## Results

Produced by `src/eval_competition.py` against the model at step 15258.

| Task | Metric | Score |
|---|---|---|
| HellaSwag | `acc_norm` | _(pending — see `vortex_eval.json`)_ |
| ARC-Easy | `acc_norm` | _(pending — see `vortex_eval.json`)_ |
| PIQA | `acc_norm` | _(pending — see `vortex_eval.json`)_ |
| WinoGrande | `acc` | _(pending — see `vortex_eval.json`)_ |
| **Scored average** | | _(pending — see `vortex_eval.json`)_ |
| WikiText-103 | perplexity | _(pending — see `vortex_eval.json`)_ |

> **Fill these in before submitting.** Run the command below *without*
> `--smoke`, paste the printed table here, and commit the `vortex_eval.json`
> it writes. The submission is scored on these numbers — the placeholders
> must not ship.

```bash
python src/eval_competition.py \
    --ckpt /root/vortex_50m_ckpt/step_15258 \
    --tokenizer /root/vortex-tok-16k \
    --out vortex_eval.json
```

The four multiple-choice tasks are scored with **EleutherAI's
lm-evaluation-harness** — the same tool the grader uses. Length normalization
alone moves HellaSwag by several points, so a hand-rolled scorer is never
comparable to published numbers; `src/eval_benchmarks.py` exists only as a
smoke-test fallback. Perplexity is token-level on the WikiText-103 **test**
split, which cannot overlap the training shards.

## Parameter count

The rules require the count to include embeddings and the output head. This is
printed from the real module tree, not an analytic formula:

```bash
$ python src/param_count.py
==================================================================
  vortex-50m-16k
==================================================================
  hidden                 512
  layers                 18
  heads                  8Q / 2KV
  head_dim               64
  intermediate           1072
  context                2048
  vocab                  16,384 (tied=True)
------------------------------------------------------------------
  TRAINABLE PARAMS       49,844,992
  budget                 50,000,000
  verdict                [PASS]  49.845M  (99.69% of budget)
------------------------------------------------------------------
  embedding                 8,388,608   16.8%
  attention                11,798,784   23.7%
  mlp                      29,638,656   59.5%
  norm                            512    0.0%
  ---                             ---
  in transformer blocks    41,456,384   83.2%
==================================================================
```

`lm_head` is tied to the embedding, so it is counted once. The script exits
non-zero if the budget is exceeded, which makes it usable as a CI gate.

## Architecture

| | |
|---|---|
| Hidden size | 512 |
| Layers | 18 |
| Attention | 8 query / 2 KV heads (GQA, 4 groups) |
| Head dim | 64 |
| Intermediate | 1,072 (2.09× hidden, SwiGLU) |
| Context | 2,048 (RoPE θ=10,000) |
| Vocab | 16,384 (**tied** embeddings) |
| Norm | RMSNorm, pre-norm, fp32 compute |
| Attention | SDPA → FlashAttention-2 on Ampere |
| Biases | none |
| **Total trainable** | **49,844,992** |

### Choices that matter at 50M params

**Depth over width, bought with a small vocab.** With tied embeddings the token
table costs exactly `vocab × hidden`, and at this scale it is the biggest lever
on what the rest of the budget buys. A 151,670-token Qwen3 vocabulary costs
77.7M parameters at 512d — **156% of the entire budget** before a single layer
exists. Most of those merges are multilingual scripts, emoji and CJK that never
appear in an English corpus.

| vocab | hidden | embed table | % of 50M | what the rest buys |
|---|---|---|---|---|
| 151,670 (Qwen3) | 512 | 77.7M | 155% | **impossible** |
| 32,768 | 512 | 16.8M | 34% | 512d × 12L — only 66% in blocks |
| **16,384** | **512** | **8.4M** | **17%** | **512d × 18L — 83% in blocks** |
| 8,192 | 640 | 5.2M | 11% | 640d × 12L — 89% in blocks |

The constraint **flips** as the vocab shrinks. At 32K the table dominates and
depth is the limit. At 8K the table is nearly free, so the same budget buys a
*wider* model — shrinking the vocabulary converts embedding parameters into
transformer capacity. 16K was chosen over 8K because it leaves headroom for
numbers, names and identifiers, which fragment badly in an English-only 8K
vocabulary.

**QK-Norm** (LLaMA-3 / Qwen3) — per-head RMSNorm on q and k before the
attention matmul. Without it, small models hit *attention entropy collapse*
early: a few heads saturate, their softmax goes one-hot, gradients vanish, and
those heads are dead for the rest of the run. Costs 2×64 params per layer.

**Zero-initialized residual outputs** — `o_proj` and `down_proj` start at
exactly zero, so every block is an *identity* at init and the untrained network
is a clean passthrough. With Pre-LN, 18 stacked blocks compound their variances
and saturate the residual stream before step 0. Verified: initial loss is
`ln(16384) = 9.7`, not the hundreds you get from default init.

**Chunked cross-entropy** — logits `(B, T, 16384)` are never materialized
during training; the loss accumulates in time-chunks. This is the largest
single memory term in the step and it is entirely avoidable, which is what lets
one 40GB card hold a large batch at 2K context.

**GQA 8/2** — 4× smaller KV cache than MHA, at no measurable quality cost at
this depth.

## Training

| | |
|---|---|
| Tokens | 1,000,000,000 (1B) |
| Corpus | `HuggingFaceTB/smollm-corpus` — `cosmopedia-v2` + `fineweb-edu-dedup` |
| Tokenizer | **trained here**: 16,384-token BPE on 120,000 English documents |
| Sequence length | 2,048 |
| Effective batch | 32 sequences (65,536 tokens/step) |
| Steps | 15,258 |
| Optimizer | AdamW, fused, β=(0.9, 0.95), wd 0.1 (none on norm gains) |
| LR | 6e-4 → 6e-5 cosine, 200 warmup |
| Precision | bf16 autocast, fp32 RMSNorm + RoPE |
| Throughput | 79.4k tokens/s |

**1B tokens is the Chinchilla-optimal budget for this model**
($20 \times 49.8\text{M} = 1.0\text{B}$). Training efficiency is a scored
criterion, and this is the compute-optimal point rather than a compromise.

### Efficiency note

7.6% MFU against the A100's 312 TFLOPS bf16 peak. That figure is *normal* at
this scale and stated plainly rather than hidden: a 50M model cannot saturate
an A100, because arithmetic intensity is set by hidden size, not by how much
data you push through it. The available wins were taken — bf16 throughout,
fused AdamW, chunked cross-entropy to avoid materializing `(B, T, 16384)`
logits, QK-Norm for stability at near-zero cost. Going further would require a
larger model, which the parameter budget forbids.

## Pipeline

Order matters: **tokenizer → retokenize → train**. The `.bin` shards become
invalid the moment the vocabulary changes.

To publish a trained checkpoint for `transformers` users, add:

```bash
# 5. Write an auto_map Hub repo (weights are reused as-is)
python src/export_hf.py --ckpt ./vortex_50m_ckpt/step_15258 \
    --tokenizer ./vortex-tok-16k --out ./vortex-50m-16k-hf \
    --push-to VTXAI/vortex-50m-16k
```

See [`src/README_hf.md`](src/README_hf.md).

```bash
# 1. Train the tokenizer (CPU, 34 min)
python src/train_tokenizer.py --out ./vortex-tok-16k --vocab-size 16384

# 2. Rebuild uint32 .bin shards with the new vocab
python src/retokenize.py --tokenizer ./vortex-tok-16k --out ./data16k \
    --max-tokens 2000000000 --tokens-per-shard 100000000

# 3. Verify the architecture (gates the run; exits non-zero on failure)
python src/verify_arch.py
python src/param_count.py

# 4. Train
python src/pretrain.py --arch vortex-50m-16k --tokenizer ./vortex-tok-16k \
    --shards ./data16k/shard_*.bin --block 2048 \
    --steps 15258 --warmup 200 --lr 6e-4 --min-lr 6e-5 \
    --auto-batch --target-batch 32 \
    --save-dir ./vortex_50m_ckpt --hub-repo VTXAI/vortex-50m-16k
```

`VTX_300M_cloud_train.ipynb` runs all of it end to end on a cloud VM.

### Shard / vocab safety check

`pretrain.py` reads the real maximum token id out of the first shards and
refuses to start if it exceeds the model's vocab. Training on data tokenized
with a *different* tokenizer is otherwise a silent failure — out-of-range ids
reach the embedding table and either crash or wrap into valid-looking tokens.

### Resuming

Checkpoints carry weights, optimizer moments, RNG state and the loss history.
`pretrain.py` takes the newest `step_N` locally and falls back to downloading
one from `--hub-repo` when the local directory is empty, so a preemption that
loses the scratch disk costs a 600MB transfer rather than the run.

## Verification

`src/verify_arch.py` is a 38-check suite that must pass before training:

- analytic parameter count **equals** the real module tree, and is ≤ 50M
- blocks are an exact identity at init (zero-init residuals)
- causality: prefix logits are bit-identical when future tokens change
- RoPE is norm-preserving and satisfies the relative-offset property
- chunked CE is chunk-size invariant; an all-masked batch returns 0, not NaN
- gradient checkpointing is value-neutral
- gradients reach 100% of parameter tensors
- GQA divisibility enforced at construction
- `save_pretrained` → `from_pretrained` round trip is bit-exact and stays tied

Regression suites for the training, checkpoint, SFT and evaluation code:

```bash
python src/test_resume.py          # RNG restore, dataloader worker independence
python src/test_ckpt_roundtrip.py  # save/resume is bitwise reproducible
python src/test_hub_resume.py      # Hub checkpoint discovery
python src/test_chat_template.py   # ChatML rendering + loss masking
python src/test_sft.py             # SFT dataset, collate, embedding resize
python src/test_eval_benchmarks.py # benchmark scoring edge cases
```

For the `transformers` integration — KV cache, `generate()`, `AutoModelForCausalLM`,
`auto_map` export:

```bash
python src/test_hf_modeling.py     # 57 checks: parity with model.py, cache, generate
python src/test_export_hf.py       # 28 checks: export, then load in a clean process
```

`test_hf_modeling.py` asserts **bit-exact** agreement with `model.py`: identical
`state_dict` keys, identical logits, identical loss. That is what guarantees a
checkpoint trained before the `transformers` modules existed loads into a model
built after them. See [`src/README_hf.md`](src/README_hf.md) for usage.

## Files

| file | role |
|---|---|
| `src/config.py` | `VortexArch`, presets, exact parameter accounting, budget guard |
| `src/model.py` | the architecture — attention, MLP, blocks, RoPE, QK-Norm |
| `src/configuration_vortex.py` | `VortexConfig` — `transformers` config, shape validation |
| `src/modeling_vortex.py` | `transformers` model — KV cache, `generate()`, `Trainer` |
| `src/export_hf.py` | writes an `auto_map` Hub repo from a training checkpoint |
| `src/README_hf.md` | **loading and running the model with `transformers`** |
| `src/train_tokenizer.py` | trains the 16K BPE, reports compression stats |
| `src/retokenize.py` | rebuilds `uint32` `.bin` shards with the new vocab |
| `src/dataset.py` | memory-mapped streaming dataset |
| `src/pretrain.py` | training loop, checkpointing, Hub resume and push |
| `src/sft.py` | instruction tuning: ChatML + assistant-only loss masking |
| `src/chat_template.py` | ChatML format, control tokens, label masking |
| `src/verify_arch.py` | the 38-check pre-training verification suite |
| `src/param_count.py` | parameter budget report / CI gate |
| `src/eval_competition.py` | **lm-eval-harness + WikiText-103 PPL** (the scored eval) |
| `src/eval_benchmarks.py` | standalone multiple-choice scorer (smoke tests) |
| `VTX_300M_cloud_train.ipynb` | end-to-end cloud training notebook |

## Built with

- **PyTorch** — `torch.nn`, SDPA (`enable_gqa`), gradient checkpointing, fused AdamW
- **HuggingFace transformers** — `PreTrainedModel` / `PretrainedConfig` wrappers
  for checkpoint serialization, plus full `configuration_vortex.py` /
  `modeling_vortex.py` modules that register with `AutoConfig` and
  `AutoModelForCausalLM` so the model loads and generates through the standard
  API; the architecture itself is hand-defined
- **tokenizers** — BPE training
- **datasets** — streaming `HuggingFaceTB/smollm-corpus`, WikiText-103
- **lm-evaluation-harness** (EleutherAI) — the four scored multiple-choice tasks
- **safetensors**, **numpy** — checkpoint format, shard I/O
- **Modal.com** — A100-SXM4-40GB compute

### Datasets

| dataset | use | license |
|---|---|---|
| `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`, `fineweb-edu-dedup`) | 2B-token pretraining corpus | ODC-BY 1.0 |
| 120k documents from the above | tokenizer training | ODC-BY 1.0 |
| `Salesforce/wikitext` (`wikitext-103-raw-v1`, test split) | held-out perplexity | CC BY-SA 3.0 |
| HellaSwag, ARC-Easy, PIQA, WinoGrande | evaluation only | see upstream |

No hosted inference API is used anywhere in training or evaluation. Every
number reported here comes from weights loaded off local disk.

## Requirements

- Python 3.10+, PyTorch 2.5+ (for SDPA `enable_gqa`), CUDA 12.x
- `transformers>=4.45`, `tokenizers`, `safetensors`, `numpy`, `datasets`
- For evaluation: `pip install 'lm-eval>=0.4.2' accelerate`
