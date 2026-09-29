
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tokenizer import iter_text_files, iter_hf_text  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
class ShardWriter:

    def __init__(self, out_dir: Path, tokens_per_shard: int = 100_000_000,
                 prefix: str = "shard"):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_per_shard = tokens_per_shard
        self.prefix = prefix
        self.shard_idx = 0
        self._buf: List[int] = []
        self._written = 0
        self.total_tokens = 0
        self.total_docs = 0
        self._index: List[dict] = []

    def add(self, ids: List[int]) -> None:
        self._buf.extend(ids)
        self.total_tokens += len(ids)
        while len(self._buf) >= self.tokens_per_shard:
            self._flush(self.tokens_per_shard)

    def add_doc(self, ids: List[int], eos_id: int) -> None:
        self.total_docs += 1
        self.add(list(ids) + [eos_id])

    def _flush(self, n: Optional[int] = None) -> None:
        n = n if n is not None else len(self._buf)
        if n <= 0:
            return
        path = self.out_dir / f"{self.prefix}_{self.shard_idx:04d}.bin"
        arr = np.asarray(self._buf[:n], dtype=np.uint32)
        arr.tofile(path)
        self._index.append({"file": path.name, "tokens": int(n)})
        self._buf = self._buf[n:]
        self._written += n
        self.shard_idx += 1
        print(f"[retok] wrote {path.name}  ({n:,} tokens, "
              f"{self.total_tokens:,} total)", flush=True)

    def close(self) -> Path:
        self._flush()
        meta = {
            "total_tokens": self.total_tokens,
            "total_docs": self.total_docs,
            "dtype": "uint32",
            "shards": self._index,
        }
        meta_path = self.out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[retok] done: {self.total_tokens:,} tokens from "
              f"{self.total_docs:,} docs across {len(self._index)} shard(s)")
        print(f"[retok] manifest: {meta_path}")
        return meta_path


# ──────────────────────────────────────────────────────────────────────
def batch_encode(tok, docs: List[str], batch_size: int = 1000) -> Iterator[List[int]]:
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i + batch_size]
        for enc in tok.encode_batch(chunk):
            yield enc.ids


def retokenize(args) -> None:
    from tokenizers import Tokenizer

    tok_dir = Path(args.tokenizer)
    tok_file = tok_dir / "tokenizer.json" if tok_dir.is_dir() else tok_dir
    tok = Tokenizer.from_file(str(tok_file))
    eos_id = _eos_id(tok_dir, tok)

    print(f"[retok] tokenizer: {tok_file}  vocab={tok.get_vocab_size():,}  eos={eos_id}")

    # Corpus source: local files, or stream straight from the Hub.
    if args.files:
        print(f"[retok] corpus: {len(args.files)} local path(s)")
        source = iter_text_files(args.files)
    else:
        cfgs = args.hf_configs or None
        print(f"[retok] corpus: streaming {args.hf_repo} configs={cfgs or 'default'}")
        # max_docs is deliberately 0 here -- the token budget is what stops us,
        # not a document count, so we never cut mid-target on doc count.
        source = iter_hf_text(repo=args.hf_repo, configs=cfgs,
                              max_docs=None, seed=args.seed)

    out_dir = Path(args.out)
    writer = ShardWriter(out_dir, tokens_per_shard=args.tokens_per_shard,
                         prefix=args.prefix)

    budget = args.max_tokens or 0
    if budget:
        print(f"[retok] token budget: {budget:,} "
              f"(~{budget/1e9:.2f}B) -- stopping when reached", flush=True)

    buf: List[str] = []
    n_seen = 0
    stop = False
    for doc in source:
        if not doc.strip():
            continue
        buf.append(doc)
        n_seen += 1
        if len(buf) >= args.encode_batch:
            _flush_docs(tok, buf, eos_id, writer, n_seen, budget)
            buf = []
            if writer.total_tokens >= budget or (args.max_docs and n_seen >= args.max_docs):
                stop = True
                break
    if buf and not stop:
        _flush_docs(tok, buf, eos_id, writer, n_seen, budget)

    writer.close()
    if budget:
        pct = 100.0 * writer.total_tokens / budget
        print(f"[retok] token budget used: {writer.total_tokens:,} / {budget:,} "
              f"({pct:.1f}%)", flush=True)


def _flush_docs(tok, docs: List[str], eos_id: int, writer: ShardWriter,
                seen: int, budget: int = 0) -> None:
    for ids in batch_encode(tok, docs):
        writer.add_doc(ids, eos_id)      # one doc + its EOS boundary
    if seen % 20_000 < len(docs):
        print(f"[retok] {seen:,} docs -> {writer.total_tokens:,} tokens", flush=True)


def _eos_id(tok_dir: Path, tok: Tokenizer) -> int:
    cfg_path = tok_dir / "tokenizer_config.json" if tok_dir.is_dir() else None
    if cfg_path and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        eos = cfg.get("eos_token")
        if isinstance(eos, str):
            tid = tok.token_to_id(eos)
            if tid is not None:
                return tid
    eos_id = tok.token_to_id("<|eos|>")
    if eos_id is None:
        raise ValueError("Could not resolve an EOS token id from the tokenizer")
    return eos_id


# ──────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="raw corpus paths (jsonl/txt/parquet) or dirs")
    p.add_argument("--tokenizer", required=True, help="path to the trained tokenizer dir")
    p.add_argument("--out", required=True, help="output dir for .bin shards")
    p.add_argument("--tokens-per-shard", type=int, default=100_000_000)
    p.add_argument("--encode-batch", type=int, default=1000, help="docs per encode batch")
    p.add_argument("--prefix", default="shard")
    p.add_argument("--max-docs", type=int, default=0, help="0 = all")

    # -- token budget + Hub streaming ------------------------------------
    p.add_argument("--max-tokens", type=int, default=0,
                   help="stop once this many tokens are written (0 = no limit). "
                        "Use this to cut the corpus at exactly 2B tokens.")
    p.add_argument("--hf-repo", default=None,
                   help="stream a Hub dataset instead of local files "
                        "(e.g. HuggingFaceTB/smollm-corpus)")
    p.add_argument("--hf-configs", nargs="*", default=None,
                   help="configs to stream; default cosmopedia-v2 + fineweb-edu-dedup")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed for Hub streaming")
    return p.parse_args(argv)


def main(argv=None) -> None:
    retokenize(parse_args(argv))


if __name__ == "__main__":
    main()
