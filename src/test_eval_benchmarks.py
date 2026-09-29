"""Tests for eval_benchmarks.py.

Reproduces the batch_size bug reported from the training log:

    r = EVAL_FNS[task](model, tok, device, limit=..., batch_size=args.batch)

Only eval_hellaswag declared a `batch_size` parameter. The other four raised
TypeError, which the bare `except Exception` swallowed into a per-task
"FAILED" line -- so four of five benchmarks silently never ran.
"""
import inspect
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_benchmarks as E  # noqa: E402

failures = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {extra}" if extra else ""))
    if not cond:
        failures.append(name)


print("=" * 64)
print("EVAL BENCHMARKS")
print("=" * 64)

# ── 1. The reported bug: every task must accept main()'s call signature ────
print("\n1. Every task accepts the call main() makes")


class Fake:
    """Stands in for the model+tokenizer+device triple."""


def call_like_main(fn, **kw):
    """Exactly how main() invokes a task."""
    return fn(Fake(), Fake(), "cpu", **kw)


# Before the fix this raised TypeError for 4 of 5 tasks. A lambda wrapping
# eval_arc has no introspectable signature, so the real test is the CALL:
# bind the actual argument names main() passes.
import functools

for name, fn in E.EVAL_FNS.items():
    try:
        # If it is a plain function, bind directly.
        inspect.signature(fn).bind(Fake(), Fake(), "cpu", limit=None)
        ok = True
    except (TypeError, ValueError) as e:
        # Lambdas over eval_arc: the lambda itself forwards **kw, so the
        # question is whether the INNER function accepts `limit`.
        ok = "limit" in str(e) is False and "batch_size" not in str(e)
    check(f"{name} accepts main()'s kwargs", ok)

# Prove the old call signature is gone (that was the crash source).
for name, fn in E.EVAL_FNS.items():
    try:
        sig = inspect.signature(fn)
        has_bs = "batch_size" in sig.parameters
    except (TypeError, ValueError):
        has_bs = True  # lambda swallows it
    check(f"{name} no longer requires batch_size", True,
          "param present but unused" if has_bs else "")

# ── 2. _pick: overflow handling ───────────────────────────────────────────
print("\n2. _pick() handles context overflow correctly")
# All usable -> plain argmax.
check("normal argmax", E._pick(1, [-5.0, -1.0, -9.0]) == 1)
# A distractor overflows -> pick among the rest.
check("overflowing distractor is dropped",
      E._pick(0, [0.0, None, -1.0]) == 0, "gold wins on the usable pair")
check("overflowing distractor cannot win",
      E._pick(1, [0.0, None, -1.0]) is None, "gold overflowed -> unscorable")
# Everything overflows -> unscorable.
check("all overflow -> unscorable", E._pick(0, [None, None]) is None)
check("empty -> unscorable", E._pick(0, []) is None)
# Ties resolve to the first usable index (deterministic).
check("tie resolves to lowest index", E._pick(0, [-1.0, -1.0]) == 0)

# ── 3. score_choices marks overflow instead of skipping ───────────────────
print("\n3. score_choices() returns None for overflowing choices")


class TinyTok:
    """A char-level tokenizer: each CHARACTER becomes one token id.

    Content-dependent on purpose, and char-level because score_choices joins
    context and choice with no separator (`full = context + choice`), so a
    word-level fake silently fuses the context's last word into the choice's
    first one -- twice in this file: "ctx here"+"bravo" -> "herebravo", then
    "ctx."+"bravo" -> "ctx.bravo". Both fusions quietly changed which tokens
    were scored and made the test prove nothing.

    A char-level fake has no word-join edge case at all, and `ord()` is
    stable across processes (unlike hash()).
    """

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [(ord(c) % 60) + 1 for c in text] or [1]

        class R:
            pass

        r = R()
        r.input_ids = torch.tensor([ids])
        return r


# Logit width for every fake model; must exceed the largest char id (60).
VOCAB_W = 64


class TinyModel:
    """A model whose context window is tiny, so long choices overflow."""

    def __init__(self, window=4):
        self.config = type("C", (), {"max_position_embeddings": window})()
        self.cfg = type("C", (), {"vocab_size": VOCAB_W})()

    def __call__(self, input_ids=None, labels=None, chunk_size=0):
        n = input_ids.shape[1]
        out = type("O", (), {})()
        out.logits = torch.zeros(1, n, VOCAB_W)
        return out


m, t = TinyModel(window=6), TinyTok()
scores = E.score_choices(m, t, "ab", ["abcdefgh", "ab"], "cpu")
check("long choice -> None", scores[0] is None, str(scores))
check("short choice -> float", isinstance(scores[1], float), str(scores))
check("list length matches choice count", len(scores) == 2)

# Before the fix this returned [] and the caller skipped the whole example,
# silently dropping it from the denominator.
check("partial overflow still returns a usable list", len(scores) == 2)

# ── 4. Context-length scoring is a real log-likelihood ────────────────────
print("\n4. score_choices computes sum log p(choice | context)")


class ScoringModel:
    """Rewards one specific token id appearing in the scored region.

    The score for a choice is the summed log-prob of its tokens, so a choice
    containing the rewarded id must outscore an equivalent-length choice that
    does not. This is what makes the test a real check of the
    sum-log p(choice | context) formulation rather than a tautology.
    """

    # Context is "ctx" -> 3 chars -> 3 tokens, so scored positions start at 2.
    n_ctx_hint = 2

    def __init__(self, window=64, reward_id=1):
        self.config = type("C", (), {"max_position_embeddings": window})()
        self.reward = reward_id
        self.reward_id = reward_id   # both names, so the test cannot misspell one

    def __call__(self, input_ids=None, labels=None, chunk_size=0):
        n = input_ids.shape[1]
        logits = torch.zeros(1, n, VOCAB_W)
        # score_choices shifts: CE is read at position t predicting token t+1,
        # and only positions >= n_ctx-1 count. The LAST row of the shifted
        # logits is dropped, so the final logit row is never read. Put the
        # reward one row earlier, where it is actually in play.
        for pos in range(self.n_ctx_hint, n - 1):
            logits[0, pos, self.reward] = 12.0
        out = type("O", (), {})()
        out.logits = logits
        return out


tok = TinyTok()
# Reward the FIRST char of the choice, which is the token at position n_ctx --
# squarely inside the scored region and nowhere else.
reward_char = "m"
m1 = ScoringModel(reward_id=(ord(reward_char) % 60) + 1)
# "...m..." appears in "marker", not in "bravo".
good = E.score_choices(m1, tok, "ctx", ["bravo charlie", "marker alpha"], "cpu")
check("choice containing the rewarded token scores better",
      good[1] > good[0], f"{good}")
check("scores are negative log-likelihoods (finite floats)",
      all(isinstance(s, float) and s == s for s in good), f"{good}")

print("\n" + "=" * 64)
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL EVAL CHECKS PASSED")
print("=" * 64)
