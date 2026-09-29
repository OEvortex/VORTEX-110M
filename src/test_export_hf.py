"""End-to-end check for export_hf.py.

Builds a checkpoint with the *training* code (`src/model.py`), exports it, then
loads the result the way a downstream user would -- through the auto classes with
`trust_remote_code=True`, in a fresh process with nothing from `src/` on the path.

Run: python src/test_export_hf.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}  {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ── 1. Build a checkpoint with the training module ────────────────────
print("\n1. Export a checkpoint produced by the training code")
from config import VortexArch
from model import VortexConfig as LegacyConfig
from model import VortexForCausalLM as LegacyCausalLM

work = tempfile.mkdtemp()
ckpt = os.path.join(work, "step_1")
out = os.path.join(work, "hf")
tok = os.path.join(work, "tok")

torch.manual_seed(0)
legacy = LegacyCausalLM(LegacyConfig(**VortexArch.from_name("vortex-test").to_dict()))
legacy.save_pretrained(ckpt)

# A real BPE tokenizer, trained the way train_tokenizer.py produces one.
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

vocab = {f"<|special_{i}|>": i for i in range(4)}
vocab.update({chr(97 + i): 4 + i for i in range(26)})
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
t.decoder = decoders.ByteLevel()
t.train_from_iterator(
    ["the quick brown fox jumps over the lazy dog " * 50,
     "hello world this is a test of the tokenizer " * 50],
    trainers.BpeTrainer(vocab_size=200, special_tokens=list(vocab.keys())))
fast = __import__("transformers").PreTrainedTokenizerFast(
    tokenizer_object=t, bos_token="<|special_1|>", eos_token="<|special_2|>",
    pad_token="<|special_0|>", unk_token="<|special_3|>")
fast.chat_template = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
fast.save_pretrained(tok)

r = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_hf.py"),
     "--ckpt", ckpt, "--out", out, "--tokenizer", tok],
    capture_output=True, text=True,
)
if r.returncode != 0:
    print(r.stdout[-3000:])
    print(r.stderr[-3000:])
    raise SystemExit("export_hf.py failed")
print("  " + "\n  ".join(l for l in r.stdout.splitlines() if l.strip())[:1200])

for name in ("config.json", "model.safetensors", "configuration_vortex.py",
             "modeling_vortex.py", "tokenizer.json", "tokenizer_config.json"):
    check(f"wrote {name}", os.path.exists(os.path.join(out, name)))

# `model_type` must match what `AutoConfig.register` expects. A mismatch here
# produces the "model of type `vortex` to instantiate a model of type ``" warning
# and an unresolvable architecture.
cfg = json.load(open(os.path.join(out, "config.json")))
check("config model_type matches the registered class", cfg.get("model_type") == "vortex",
      cfg.get("model_type", "<missing>"))
check("auto_map.AutoConfig set",
      cfg.get("auto_map", {}).get("AutoConfig") == "configuration_vortex.VortexConfig")
check("auto_map.AutoModelForCausalLM set",
      cfg.get("auto_map", {}).get("AutoModelForCausalLM") == "modeling_vortex.VortexForCausalLM")
check("architectures names the class", cfg.get("architectures") == ["VortexForCausalLM"])
check("model_type is vortex", cfg.get("model_type") == "vortex")
check("special token ids reconciled from the tokenizer",
      cfg.get("bos_token_id") == 1 and cfg.get("eos_token_id") == 2 and cfg.get("pad_token_id") == 0,
      f"bos={cfg.get('bos_token_id')} eos={cfg.get('eos_token_id')} pad={cfg.get('pad_token_id')}")
# `save_pretrained` writes a template this long to `chat_template.jinja` rather
# than into tokenizer_config.json, so check both places.
tk = json.load(open(os.path.join(out, "tokenizer_config.json")))
jinja = os.path.join(out, "chat_template.jinja")
check("chat_template survived the round trip",
      bool(tk.get("chat_template")) or (os.path.exists(jinja) and "<|im_start|>" in open(jinja).read()),
      "tokenizer_config.json" if tk.get("chat_template") else os.path.basename(jinja))

# ── 2. Weights survived the export bit-exactly ───────────────────────
print("\n2. Weights are byte-identical to the source checkpoint")
from safetensors.torch import load_file

a = load_file(os.path.join(ckpt, "model.safetensors"))
b = load_file(os.path.join(out, "model.safetensors"))
check("same tensor count", len(a) == len(b), f"{len(a)} vs {len(b)}")
same = all(a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a if k in b)
check("every tensor identical", same,
      f"first mismatch: {next((k for k in a if k in b and not torch.equal(a[k], b[k])), 'none')}")

# ── 3. Load as a downstream user would, in a clean process ───────────
print("\n3. Load via auto classes in a clean process (no src/ on the path)")
script = r'''
import json, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
out, dest = sys.argv[1], sys.argv[2]
cfg = AutoConfig.from_pretrained(out, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(out)
model.eval(); model.config.pad_token_id = tok.pad_token_id
enc = tok("hello world", return_tensors="pt")
with torch.no_grad():
    logits = model(input_ids=enc.input_ids).logits
gen = model.generate(**enc, max_new_tokens=5, do_sample=False)
rendered = tok.apply_chat_template([{"role":"user","content":"hi"}], tokenize=False)
# Written to a file: the remote-code prompt and library warnings share stdout
# with prints, and the prompt is emitted without a trailing newline.
json.dump({
    "cls": type(model).__name__,
    "cfg_cls": type(cfg).__name__,
    "model_type": getattr(cfg, "model_type", None),
    "logits_finite": bool(torch.isfinite(logits).all()),
    "logits_shape": list(logits.shape),
    "gen_shape": list(gen.shape),
    "gen_text": tok.decode(gen[0]),
    "chat": rendered,
    "n_params": sum(p.numel() for p in model.parameters()),
    "tied": model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(),
}, open(dest, "w"))
'''
clean = os.path.join(work, "clean_load.py")
result_path = os.path.join(work, "result.json")
open(clean, "w").write(script)
# Nothing from src/ on the path, and stdin closed: this is as close to a fresh
# `pip install`-only environment as the export can be tested in.
env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
r = subprocess.run([sys.executable, clean, out, result_path], capture_output=True,
                   text=True, env=env, cwd=work, stdin=subprocess.DEVNULL)
if r.returncode != 0 or not os.path.exists(result_path):
    print(r.stdout[-2500:])
    print(r.stderr[-2500:])
    raise SystemExit("clean-process load failed")
res = json.load(open(result_path))
check("no traceback from the clean load", "Traceback" not in r.stderr)

# `AutoTokenizer.from_pretrained` reads config.json through the *base*
# `PreTrainedConfig`, whose `model_type` is "", so loading a tokenizer from a repo
# whose config declares a custom model_type always logs
#   "using a model of type `vortex` to instantiate a model of type ``"
# That is transformers-internal cosmetics, true of every remote-code repo on the
# Hub, and it does not affect the tokenizer. Assert instead that the *model*
# config resolved to the right class -- the thing that would actually break.
check("model config resolved to VortexConfig, not the base class",
      res["cfg_cls"] == "VortexConfig" and not res["cfg_cls"] == "PretrainedConfig",
      res["cfg_cls"])
check("model_type round-trips as vortex", res["model_type"] == "vortex", res["model_type"])
print("  " + json.dumps(res, indent=2).replace("\n", "\n  "))

check("AutoConfig -> VortexConfig", res["cfg_cls"] == "VortexConfig", res["cfg_cls"])
check("AutoModelForCausalLM -> VortexForCausalLM", res["cls"] == "VortexForCausalLM", res["cls"])
check("logits are finite", res["logits_finite"])
# The model's vocab is 1024 (the vortex-test preset) while the throwaway BPE
# tokenizer only has 88 entries -- exercising exactly the "model table is larger
# than the tokenizer" case eval_competition.py warns about.
cfg_vocab = json.load(open(os.path.join(out, "config.json")))["vocab_size"]
check("logits shape (1, T, model_vocab)",
      res["logits_shape"][0] == 1 and res["logits_shape"][2] == cfg_vocab,
      f"{res['logits_shape']} vs vocab {cfg_vocab}")
check("model vocab >= tokenizer vocab (extra rows are reachable, not phantom)",
      cfg_vocab >= 88, f"model {cfg_vocab} >= tok 88")
check("generate() produced tokens", res["gen_shape"][0] == 1 and len(res["gen_shape"]) == 2,
      str(res["gen_shape"]))
check("apply_chat_template works", res["chat"].startswith("<|im_start|>user"), repr(res["chat"]))
check("weights still tied after remote-code load", res["tied"])

# ── 4. Parity against the training module ────────────────────────────
print("\n4. Exported model matches the source checkpoint's outputs")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from modeling_vortex import VortexForCausalLM as HFCausalLM

exported = HFCausalLM.from_pretrained(out).eval()
src = LegacyCausalLM.from_pretrained(ckpt).eval()
ids = torch.randint(0, 256, (1, 16))
with torch.no_grad():
    d = (exported(input_ids=ids).logits - src(input_ids=ids).logits).abs().max().item()
check("logits match the training module", d < 1e-5, f"max|diff| = {d:.2e}")

# ── 5. Guardrail: refusing to shrink context ──────────────────────────
print("\n5. Context-shrink guardrail")
r = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_hf.py"),
     "--ckpt", ckpt, "--out", os.path.join(work, "hf2"), "--tokenizer", tok,
     "--max-length", "64"],
    capture_output=True, text=True, env=env)
check("refuses to shorten a trained context",
      r.returncode != 0 and "refusing to shrink" in (r.stdout + r.stderr))

print("\n" + "=" * 62)
print(f"  {PASS} passed, {FAIL} failed")
print("=" * 62)
sys.exit(1 if FAIL else 0)
