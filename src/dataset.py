"""
Memory-mapped pre-tokenized dataset loader.

Reads one or more `.bin` files (uint32 little-endian token stream with
EOS-separated documents) and yields fixed-length blocks for training.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from typing import List, Optional


class MMapDataset(IterableDataset):
    """Stream fixed-length blocks of tokens from a list of memmap shards.

    Shards are concatenated logically; samples start at random offsets.
    Each sample is a (block_size+1,) tensor of int64 token ids (for input
    and target).
    """
    def __init__(
        self,
        shard_paths: List[str],
        block_size: int = 2048,
        seed: int = 0,
        drop_last: bool = True,
    ):
        super().__init__()
        self.shard_paths = [Path(p) for p in shard_paths]
        self.block_size = block_size
        self.drop_last = drop_last

        # Open memmaps and compute total length
        self.mmaps: List[np.memmap] = []
        self.lengths: List[int] = []
        self.offsets: List[int] = [0]
        total = 0
        for p in self.shard_paths:
            if not p.exists():
                raise FileNotFoundError(f"Shard not found: {p}")
            mm = np.memmap(p, dtype=np.uint32, mode="r")
            self.mmaps.append(mm)
            self.lengths.append(len(mm))
            total += len(mm)
            self.offsets.append(total)
        self.total_tokens = total
        # Random generator (one per worker for IterableDataset)
        self._seed = seed
        # Eagerly set epoch so worker_init_fn can pick it up
        self.set_epoch(0)

    @property
    def num_blocks(self) -> int:
        return max(self.total_tokens // (self.block_size + 1), 1)

    def set_epoch(self, epoch: int):
        # For IterableDataset we use a per-worker generator with seed+epoch
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            wid = 0
        else:
            wid = worker_info.id
        self._rng = np.random.default_rng(self._seed + epoch * 1000 + wid)

    def _locate(self, global_idx: int) -> tuple[np.memmap, int]:
        """Map a global token index to (mmap, local_index)."""
        # offsets is cumulative; find shard via bisect
        import bisect
        shard_idx = bisect.bisect_right(self.offsets, global_idx) - 1
        shard_idx = max(0, min(shard_idx, len(self.mmaps) - 1))
        local = global_idx - self.offsets[shard_idx]
        return self.mmaps[shard_idx], local

    def __iter__(self):
        bs = self.block_size
        while True:
            # Random start index
            max_start = self.total_tokens - (bs + 1)
            if max_start <= 0:
                raise StopIteration
            start = int(self._rng.integers(0, max_start))
            mm, local = self._locate(start)
            end = local + bs + 1
            if end <= len(mm):
                buf = mm[local:end]
            else:
                # Spans boundary: stitch
                a = mm[local:]
                needed = (bs + 1) - len(a)
                mm2, _ = self._locate(start + len(a))
                b = mm2[:needed]
                buf = np.concatenate([a, b])
            x = torch.from_numpy(buf.astype(np.int64)).long()
            yield x  # caller slices input/target

    def __len__(self):
        return self.num_blocks
