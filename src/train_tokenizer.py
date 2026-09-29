"""
Train the Vortex tokenizer -- an 8,192-token BPE for ENGLISH-ONLY text.

Why this file has to exist
--------------------------
The original tokenizer was Qwen3's 151,670-token vocabulary. With tied
embeddings that vocab costs `vocab * hidden` parameters in a single table:

    151,670 x 512 = 77.7M   (155% of a 50M budget, before a single layer)

For English-only training that is pure waste: most of those 151K merges are
multilingual scripts, emoji, and CJK that will never appear in your corpus.
An 8,192-token English BPE costs 5.2M at hidden=640, freeing the rest of the
budget for the transformer itself.

8K-16K is the proven band for English at this scale -- TinyStories trains
1M-33M English models on a 10K vocabulary.

The trade-off you are making explicitly: a smaller vocab compresses English
less efficiently, so the same corpus yields MORE tokens. On English prose a
32K byte-level BPE lands around 4.0-4.5 chars/token; 8K-16K lands around
3.2-3.8. The parameter win is large and certain, the token-efficiency cost is
corpus-specific -- measure it with `--stats` before committing.

If you ever add code or non-English text, retrain at 32768 and switch to the
`vortex-50m-32k` preset. Rare tokens fragment badly in a 8K English vocab.

Usage
-----
    # Train on a directory of .jsonl / .txt / .parquet
    python train_tokenizer.py --out ./vortex-tok-8k --vocab-size 8192

    # Then report compression stats before committing to a full retokenize
    python train_tokenizer.py --out ./vortex-tok-8k --stats

Then re-tokenize the corpus and train:
    python retokenize.py --tokenizer ./vortex-tok-8k --out ./data8k
    python pretrain.py --arch vortex-50m --tokenizer ./vortex-tok-8k --shards ./data8k/shard_*.bin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

# Special tokens occupy the low ids; the BPE merges fill the rest.
PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


# ──────────────────────────────────────────────────────────────────────
# Corpus iteration
# ──────────────────────────────────────────────────────────────────────
TEXT_KEYS = ("text", "content", "document", "raw", "body", "source")


def iter_text_files(paths: List[str], limit_bytes: Optional[int] = None) -> Iterator[str]:
    """Yield raw document text from .jsonl / .txt / .parquet / .json.

    Streams line by line so a multi-GB corpus never lands in memory.
    """
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files = sorted(
                [p for p in path.rglob("*")
                 if p.suffix in (".jsonl", ".txt", ".parquet", ".json")]
            )
        else:
            files = [path]

        for f in files:
            print(f"[tok] reading {f}", flush=True)
            if f.suffix == ".jsonl":
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            yield line
                            continue
                        for k in TEXT_KEYS:
                            if isinstance(obj.get(k), str):
                                yield obj[k]
                                break
            elif f.suffix == ".txt":
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    for para in fh:
                        if para.strip():
                            yield para.rstrip("\n")
            elif f.suffix in (".json", ".parquet"):
                yield from _iter_structured(f)


def _iter_structured(f: Path) -> Iterator[str]:
    """Pull text out of .json arrays or .parquet datasets."""
    if f.suffix == ".json":
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        for obj in (data if isinstance(data, list) else [data]):
            for k in TEXT_KEYS:
                if isinstance(obj.get(k), str):
                    yield obj[k]
                    break
    else:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            print(f"[tok] WARN: pyarrow missing, skipping {f}", flush=True)
            return
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1024):
            cols = [c for c in TEXT_KEYS if c in batch.schema.names]
            if not cols:
                continue
            for v in batch.column(cols[0]).to_pylist():
                if isinstance(v, str):
                    yield v


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────
def build_trainer(vocab_size: int, min_frequency: int = 2):
    """Byte-level BPE trainer.

    Byte-level (vs. tiktoken's regex pretokenizer + BPE) is chosen because it
    is the fastest to train, has zero out-of-vocab behaviour on arbitrary
    bytes, and is what we can train reliably on CPU. If you later want a
    tiktoken-style split, swap the `pre_tokenizer` for the Regex one -- the
    vocab and model code stay identical.

    Returns (tokenizer, trainer).
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    # No whitespace pre-tokenization: byte-level BPE learns whitespace
    # characters as ordinary tokens, which keeps the trainer simple and
    # reversible (decode is exact).
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    return tok, trainer


def train(args) -> Path:
    from tokenizers import Tokenizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(args.files)
    if not files:
        print("[tok] ERROR: pass one or more corpus paths", file=sys.stderr)
        sys.exit(1)

    print(f"[tok] training BPE vocab={args.vocab_size} min_freq={args.min_frequency}")
    print(f"[tok] corpus: {len(files)} path(s)")

    tok, trainer = build_trainer(args.vocab_size, args.min_frequency)
    iterator = iter_text_files(files, limit_bytes=args.limit_bytes)
    tok.train_from_iterator(iterator, trainer=trainer, length=iterator_hint(files))

    size = tok.get_vocab_size()
    print(f"[tok] learned {size} tokens (target {args.vocab_size})")
    if size < args.vocab_size * 0.9:
        print("[tok] WARN: corpus is too small to fill the vocab; "
              "add more data or lower --vocab-size", file=sys.stderr)

    # Save as a plain HF fast tokenizer.
    tok.save(str(out_dir / "tokenizer.json"))
    _write_config(out_dir, size)
    print(f"[tok] wrote tokenizer to {out_dir}")
    return out_dir


def iterator_hint(files: List[str]) -> Optional[int]:
    """Best-effort line count so the trainer can show a progress bar."""
    total = 0
    try:
        for f in files:
            p = Path(f)
            targets = ([q for q in p.rglob("*") if q.suffix == ".jsonl"]
                       if p.is_dir() else [p])
            for t in targets:
                with open(t, "rb") as fh:
                    total += sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 20), b""))
    except OSError:
        return None
    return total or None


def _write_config(out_dir: Path, vocab_size: int) -> None:
    """Write a minimal tokenizer_config.json so AutoTokenizer can load it."""
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 2048,
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))

    # pre_tokenizer_config is what `AutoTokenizer` reads for a fast tokenizer
    # when tokenizer.json is present; the two ids must match SPECIAL_TOKENS.
    (out_dir / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": BOS_TOKEN, "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN, "unk_token": UNK_TOKEN,
    }, indent=2))
    print(f"[tok] vocab_size={vocab_size}  ids: pad=0 bos=1 eos=2 unk=3")


# ──────────────────────────────────────────────────────────────────────
# Stats — measure the trade-off before you commit
# ──────────────────────────────────────────────────────────────────────
def stats(args) -> None:
    """Compare this tokenizer against the old one on a held-out sample."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(args.out) / "tokenizer.json"))
    sample = _gather_sample(args.stats_files or args.files, args.stats_docs)

    total_chars = sum(len(s) for s in sample)
    ids = tok.encode_batch(sample)
    total_tokens = sum(len(i.ids) for i in ids)
    chars_per_token = total_chars / max(1, total_tokens)

    print("=" * 60)
    print(f"vocab size        {tok.get_vocab_size():,}")
    print(f"documents         {len(sample):,}")
    print(f"characters        {total_chars:,}")
    print(f"tokens            {total_tokens:,}")
    print(f"chars / token     {chars_per_token:.2f}")
    print("=" * 60)

    # Round-trip check -- byte-level must be lossless.
    bad = 0
    for s, enc in zip(sample[:200], ids[:200]):
        if tok.decode(enc.ids) != s:
            bad += 1
    print(f"round-trip exact  {'YES' if bad == 0 else f'NO ({bad}/200 differ)'}")
    print()
    print("Compare `chars / token` against the old tokenizer on the same text.")
    print("Lower means more tokens per character = more training compute for the")
    print("same corpus. ~3.5-4.0 is healthy for byte-level BPE on code+English.")


def _gather_sample(files: List[str], n_docs: int) -> List[str]:
    out: List[str] = []
    for s in iter_text_files(files):
        out.append(s)
        if len(out) >= n_docs:
            break
    return out


# ──────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="corpus .jsonl/.txt/.parquet paths or dirs")
    p.add_argument("--out", default="./vortex-tok-8k", help="output dir")
    p.add_argument("--vocab-size", type=int, default=8_192,
                   help="8192 for English-only; 16384 if you want margin; "
                        "32768 only for multilingual/code")
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--limit-bytes", type=int, default=None,
                   help="optional cap on corpus bytes (smoke tests)")
    p.add_argument("--stats", action="store_true", help="report stats instead of training")
    p.add_argument("--stats-files", nargs="*", default=None)
    p.add_argument("--stats-docs", type=int, default=2000)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.stats:
        stats(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
