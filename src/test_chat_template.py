"""Tests for chat_template.py -- the ChatML format and, more importantly,
the loss masking. A masking bug is silent: training runs, loss falls, and the
model still echoes prompts and never stops.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_template as CT  # noqa: E402

failures = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {extra}" if extra else ""))
    if not cond:
        failures.append(name)


class CharTok:
    """One token per character, so token positions map 1:1 to characters.

    This makes the masking assertions exact and readable: token i is
    character i. `ord` is stable across processes (hash() is not).
    """

    def __len__(self):
        return 1000

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(c) % 500 + 1 for c in text]
        return {"input_ids": ids}


print("=" * 64)
print("CHATML TEMPLATE")
print("=" * 64)

# ── 1. Rendering ─────────────────────────────────────────────────────────
print("\n1. render()")
msgs = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]
txt = CT.render(msgs)
expected = ("<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\nHello!<|im_end|>\n")
check("full conversation renders in ChatML form", txt == expected, repr(txt))
check("system turn present", "system" in txt)
check("assistant turn present", "assistant" in txt)
check("no trailing whitespace after final im_end",
      txt.endswith(f"{CT.IM_END}\n"), repr(txt[-20:]))

p = CT.render(msgs, add_generation_prompt=True)
check("generation prompt appends the assistant header",
      p.endswith(f"{CT.IM_START}assistant\n"), repr(p[-30:]))
check("generation prompt has no im_end at the end",
      not p.rstrip().endswith(CT.IM_END))

try:
    CT.render([{"role": "wizard", "content": "x"}])
    check("unknown role raises", False)
except ValueError:
    check("unknown role raises", True)

# Multi-turn assistant (tool-style) still renders.
multi = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
         {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
check("multi-turn renders 4 blocks", CT.render(multi).count(CT.IM_START) == 4)

# ── 2. Jinja template is valid and agrees with render() ──────────────────
print("\n2. build_chat_template() matches render()")
try:
    from jinja2 import Template
    tpl = Template(CT.build_chat_template(), keep_trailing_newline=False)
    jinja_out = tpl.render(messages=msgs, add_generation_prompt=False)
    check("jinja template renders the same text as render()",
          jinja_out == txt, f"jinja={jinja_out!r}")
    jinja_gen = tpl.render(messages=msgs, add_generation_prompt=True)
    check("jinja generation prompt matches render_prompt()",
          jinja_gen == CT.render_prompt(msgs))
except ImportError:
    check("jinja2 available for template check", False, "jinja2 not installed")

# ── 3. Masking: the assistant content is trained, the prompt is not ───────
print("\n3. encode_example() masking (THE critical property)")
tok = CharTok()
enc, labels = CT.encode_example(tok, msgs, max_len=1000)
check("encode_example returned a sample", enc is not None)
check("labels align with input_ids", len(enc) == len(labels),
      f"enc={len(enc)} labels={len(labels)}")

# The user content "Hi" must be masked.
hi_idx = "".join(chr(e - 1) for e in enc).find("Hi")
check("user content is present in the sequence", hi_idx != -1)
if hi_idx != -1:
    user_lab = labels[hi_idx:hi_idx + 2]
    check("user content tokens are masked to -100",
          all(l == -100 for l in user_lab), f"{user_lab}")

# The assistant content "Hello!" must be trained.
hello_idx = "".join(chr(e - 1) for e in enc).find("Hello!")
check("assistant content is present in the sequence", hello_idx != -1)
if hello_idx != -1:
    a_lab = labels[hello_idx:hello_idx + 6]
    check("assistant content tokens are TRAINED (not -100)",
          all(l != -100 for l in a_lab), f"{a_lab}")
    check("assistant content labels equal the input ids (teacher forcing)",
          all(labels[i] == enc[i] for i in range(hello_idx, hello_idx + 6)))

# The system content must be masked.
sys_idx = "".join(chr(e - 1) for e in enc).find("You are helpful.")
if sys_idx != -1:
    s_lab = labels[sys_idx:sys_idx + 16]
    check("system content is masked to -100", all(l == -100 for l in s_lab), f"{s_lab}")

# The <|im_end|> that closes the assistant turn must be TRAINED so the model
# learns to stop.
im_end_pos = [i for i, e in enumerate(enc) if e == (ord(CT.IM_END[0]) % 500 + 1)]
# Find the im_end that immediately follows "Hello!"
if hello_idx != -1:
    after = "".join(chr(e - 1) for e in enc[hello_idx + 6:hello_idx + 8])
    # locate the im_end right after the assistant content
    e_idx = hello_idx + 6
    while e_idx < len(enc) and enc[e_idx] != (ord(CT.IM_END[0]) % 500 + 1):
        e_idx += 1
    if e_idx < len(enc):
        check("assistant's closing im_end is TRAINED (model learns to stop)",
              labels[e_idx] != -100, f"label={labels[e_idx]}")

check("at least one token is trainable", any(l != -100 for l in labels))
n_trained = sum(1 for l in labels if l != -100)
check("majority of tokens are masked (prompt dominates)",
      n_trained < len(labels) // 2, f"trained={n_trained}/{len(labels)}")

# ── 4. Edge cases ────────────────────────────────────────────────────────
print("\n4. encode_example() edge cases")
# No assistant turn -> nothing to train on.
only_user = [{"role": "user", "content": "hi"}]
out = CT.encode_example(tok, only_user, max_len=1000)
check("user-only conversation is rejected (no target)", out is None)

# Too long -> None, not truncated.
out = CT.encode_example(tok, msgs, max_len=5)
check("over-length conversation returns None (skipped, not truncated)", out is None)

# A user-only PREFIX followed by assistant IS trainable.
prefix_and_reply = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
out = CT.encode_example(tok, prefix_and_reply, max_len=1000)
check("user->assistant conversation is trainable", out is not None)

# Multiple assistant turns: all of them trained, user turns between masked.
multi_asst = [
    {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
]
enc_m, lab_m = CT.encode_example(tok, multi_asst, max_len=1000)
s_m = "".join(chr(e - 1) for e in enc_m)
q2 = s_m.find("q2")
a1 = s_m.find("a1")
a2 = s_m.find("a2")
check("all assistant turns are trained",
      lab_m[a1] != -100 and lab_m[a2] != -100)
check("interleaved user turn is masked",
      all(l == -100 for l in lab_m[q2:q2 + 2]), f"{lab_m[q2:q2 + 2]}")

print("\n" + "=" * 64)
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL CHATML CHECKS PASSED")
print("=" * 64)
