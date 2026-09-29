# Using Vortex with `transformers`

Two files make the model loadable through the standard `transformers` API:

| file | role |
|---|---|
| `configuration_vortex.py` | `VortexConfig(PretrainedConfig)` — architecture + shape validation |
| `modeling_vortex.py` | `VortexPreTrainedModel` / `VortexModel` / `VortexForCausalLM` |

They are additive: `model.py` stays the training module and the checkpoints
already on the Hub load into either implementation unchanged.

## Export a checkpoint

The weights need no conversion. What is missing for the auto classes is the
*architecture declaration* — an `auto_map` in `config.json` pointing at these two
files, which then travel with the repo.

```bash
python src/export_hf.py \
    --ckpt ./vortex_50m_ckpt/step_15258 \
    --tokenizer ./vortex-tok-16k \
    --out ./vortex-50m-16k-hf
```

Add `--push-to VTXAI/vortex-50m-16k` to publish it. The exported directory is a
complete, self-contained Hub repo.

## Load it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "VTXAI/vortex-50m-16k", trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained("VTXAI/vortex-50m-16k")
```

`trust_remote_code=True` is required — it is the flag that authorises executing
the two files above. Read them if you want to know what you're agreeing to; they
are the model.

### Registering the classes yourself

If you would rather not use `trust_remote_code` on every load, register once at
import time and drop the flag:

```python
from transformers import AutoConfig, AutoModelForCausalLM
from configuration_vortex import VortexConfig
from modeling_vortex import VortexForCausalLM

AutoConfig.register("vortex", VortexConfig)
AutoModelForCausalLM.register(VortexConfig, VortexForCausalLM)

model = AutoModelForCausalLM.from_pretrained("VTXAI/vortex-50m-16k")  # no flag
```

## Generate

```python
messages = [{"role": "user", "content": "Explain gradient checkpointing."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                     eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

`add_special_tokens=False` matters: the ChatML template already emits the control
tokens, and letting the tokenizer add another set shifts every position.

## Training and fine-tuning

The `Trainer` path works, and the chunked-loss design is preserved:

```python
from transformers import Trainer, TrainingArguments

model.gradient_checkpointing_enable()   # ~25% slower, most of the memory back
trainer = Trainer(model=model, args=TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    bf16=True,
))
trainer.train()
```

Two behaviours worth knowing, both intentional:

- **`labels=` returns `logits=None`.** The chunked cross-entropy exists so the
  `(batch, seq, vocab)` tensor is never materialised — at batch 32 x 2048 x 16384
  in fp32 that tensor alone is 4.3GB. Pass `logits_to_keep=1` when you want logits
  alongside a loss.
- **Gradient checkpointing disables the cache during training.** The backward
  pass recomputes each block, and `Cache.update` mutates in place, so a cached
  forward would append the same keys twice and corrupt every layer above it.
  Inference is unaffected.

## API surface

`VortexForCausalLM.forward` accepts `input_ids`, `attention_mask`, `position_ids`,
`past_key_values`, `inputs_embeds`, `labels`, `use_cache`, `output_hidden_states`,
`return_dict`, `logits_to_keep`, `chunk_size`, `num_items_in_batch`, and returns
`CausalLMOutputWithPast`.

`output_attentions=True` raises `NotImplementedError` — attention runs fused
inside SDPA, so per-head weights are not materialised.

### Caching

```python
out = model(input_ids=ids, use_cache=True)
out.past_key_values.get_seq_length()

# feed a block of new tokens against the cache
out = model(input_ids=new_ids, past_key_values=out.past_key_values, use_cache=True)
```

The KV cache is what makes generation practical: measured on CPU at 1024 context,
generating 64 tokens is **25x faster** with `use_cache=True`. Without it every step
recomputes the whole prefix.

## Tests

```bash
python src/test_hf_modeling.py   # 57 checks: parity, cache, generate, round trip
python src/test_export_hf.py     # 28 checks: export + load in a clean process
```

Both exit non-zero on failure, so they work as CI gates.

`test_hf_modeling.py` asserts bit-exact agreement with `model.py` — same
`state_dict` keys, same logits, same loss — which is what guarantees a checkpoint
trained before this existed loads correctly into a model built after it.
