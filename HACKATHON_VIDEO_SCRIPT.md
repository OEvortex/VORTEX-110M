# Vortex-50M: 3–4 Minute Hackathon Video Script

**Target Duration:** 3:00 to 4:00 minutes (comfortable 130–150 words/min pace)  
**Presenter Tone:** Confident, technical, and engineering-focused.

---

### [0:00 – 0:35] Hook & Problem Statement
*(Visual: Show the Hugging Face model page `VTXAI/vortex-50m` or terminal running `python src/param_count.py`)*

> "Hi everyone! Today, I’m presenting **Vortex-50M**, an ultra-compact, decoder-only transformer built entirely from first principles on PyTorch, strictly under a hard 50-million parameter budget.
>
> While frontier models require massive data centers with hundreds of billions of parameters, deploying AI on edge hardware, air-gapped security infrastructure, or local devices demands extreme efficiency. But at under 50M parameters, standard LLM recipes break: off-the-shelf tokenizers eat up your entire parameter budget, attention heads collapse, and pre-LN causes residual saturation.
>
> Vortex-50M was engineered to solve this: maximum architectural efficiency from scratch, trained on 1 billion tokens, and instruction-tuned as an on-device cybersecurity and cryptography specialist."

---

### [0:35 – 1:30] Architecture: Depth Over Width & The 16K Vocab Strategy
*(Visual: Screen share `src/model.py` and show the parameter accounting table)*

> "Let’s look at the core design decisions.
>
> In a 50M model, the embedding table is your biggest trade-off. Standard models use 32K or 150K token vocabularies. At 512 dimensions, a 150K vocabulary alone takes 77 million parameters—which is **155% of our entire budget** before we even place a single transformer layer!
>
> Instead, we trained a custom 16,384-token BPE tokenizer from scratch on 120,000 documents. This costs just 8.4 million parameters—only 17% of our budget. Because we tied the output projection head to the embedding table, that saved 83% of the parameter budget for the actual transformer blocks: allowing us to fit **18 deep layers** at 512 hidden dimension.
>
> We also implemented key stability techniques:
> 1. **Grouped Query Attention (8 query heads, 2 KV heads)**: 4× smaller KV-cache footprint for fast edge inference.
> 2. **QK-Norm (RMSNorm on Queries and Keys)**: This prevents attention entropy collapse—a notorious problem where small model attention heads saturate into one-hot distributions early in training.
> 3. **Zero-initialized residual projections**: Ensuring the network starts as a clean mathematical identity at step zero, avoiding residual stream saturation."

---

### [1:30 – 2:20] Pretraining & Domain SFT Specialization
*(Visual: Show training loop `src/pretrain.py`, `src/sft.py`, or Hugging Face dataset)*

> "We pretrained the base model, `VTXAI/vortex-50m-16k`, on **1 Billion tokens** using high-quality curated data from Cosmopedia and FineWeb-Edu, which represents the exact Chinchilla-optimal ratio for 50M parameters.
>
> For our specialized application, we instruction-tuned the model into a **Cybersecurity & Cryptography Specialist** (`VTXAI/vortex-50m`).
>
> Small models can easily suffer from catastrophic forgetting when tuned on narrow technical corpora. To counter this, we designed a balanced training mix: 70% technical cybersecurity and cryptography instructions paired with 30% conversational anchor data, combined with assistant-only loss masking and weight decay. This allows the model to understand domain concepts like AES-GCM versus CBC, buffer overflows, and threat vectors, while retaining stable natural language comprehension."

---

### [2:20 – 3:00] Benchmarks & Official Evaluation
*(Visual: Show `vortex_50m_eval.json` and the README benchmark table)*

> "To evaluate fairly, we didn't use loose hand-rolled scripts. We evaluated using EleutherAI’s official **lm-evaluation-harness** across standard reasoning benchmarks:
>
> - **PIQA**: 54.13%
> - **WinoGrande**: 52.41%
> - **ARC-Easy**: 35.52%
> - **HellaSwag**: 27.23%
> - **Overall Scored Average**: **42.32%**
>
> Crucially, our parameter count is verified by our automated suite at **49,846,528 parameters**—99.69% of the budget—with zero external APIs used during training or evaluation."

---

### [3:00 – 3:30] Live Inference & Conclusion
*(Visual: Open terminal or Python REPL and run inference on `VTXAI/vortex-50m`)*

> "Vortex-50M is fully compatible with Hugging Face `transformers` using `trust_remote_code=True` and native ChatML templates. With a memory footprint under 100 megabytes in quantized formats, it can run directly in real time on mobile devices, IoT microcontrollers, or embedded security appliances.
>
> Both the base model, SFT model, and dataset are open-source and live on Hugging Face at `VTXAI/vortex-50m`.
>
> Thank you!"
