"""Length-aware token-budgeted batch sampler.

Greedy-packs indices from an inner per-sample sampler so the summed sequence
length of each batch stays under ``max_tokens_per_batch``. This bounds peak
activation memory regardless of the protein length distribution in a batch.

The inner sampler is responsible for any distributed sharding (``ClusterSampler``
already does this). For non-distributed inner samplers running under DDP, this
sampler additionally shards by ``batch_idx % world_size == rank`` so each rank
sees a disjoint set of batches with matching step counts.
"""

from typing import Iterable, Iterator, List, Optional

import torch
from loguru import logger
from torch.utils.data import Sampler


class TokenBudgetBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        index_sampler: Iterable[int],
        lengths: torch.Tensor,
        max_tokens_per_batch: int,
        max_batch_size: Optional[int] = None,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
        shard_in_sampler: bool = False,
    ):
        """
        Args:
            index_sampler: An iterable yielding per-sample dataset indices.
            lengths: 1-D LongTensor of per-dataset-index residue counts.
            max_tokens_per_batch: Cap on summed length per batch.
            max_batch_size: Optional cap on number of samples per batch (guards
                against very-short-protein batches).
            drop_last: Whether to drop a trailing partial batch.
            rank: Distributed rank of this process.
            world_size: Total number of distributed processes.
            shard_in_sampler: When True, this sampler shards batches across
                ranks via ``batch_idx % world_size == rank``. Set True when
                ``index_sampler`` is rank-agnostic (e.g. RandomSampler);
                False when the inner sampler already shards (e.g.
                ClusterSampler in distributed mode).
        """
        if max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be > 0")
        self.index_sampler = index_sampler
        self.lengths = lengths
        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.max_batch_size = max_batch_size
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.shard_in_sampler = shard_in_sampler

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[int] = []
        batch_tokens = 0
        global_batch_idx = 0
        for idx in self.index_sampler:
            length = int(self.lengths[idx])
            if length > self.max_tokens_per_batch:
                logger.warning(
                    f"Skipping index {idx} with length {length} > "
                    f"max_tokens_per_batch={self.max_tokens_per_batch}"
                )
                continue

            would_exceed_tokens = batch_tokens + length > self.max_tokens_per_batch
            would_exceed_count = (
                self.max_batch_size is not None and len(batch) >= self.max_batch_size
            )
            if batch and (would_exceed_tokens or would_exceed_count):
                if self._owns(global_batch_idx):
                    yield batch
                global_batch_idx += 1
                batch = []
                batch_tokens = 0

            batch.append(int(idx))
            batch_tokens += length

        if batch and not self.drop_last:
            if self._owns(global_batch_idx):
                yield batch

    def _owns(self, global_batch_idx: int) -> bool:
        if not self.shard_in_sampler or self.world_size <= 1:
            return True
        return global_batch_idx % self.world_size == self.rank

    def __len__(self) -> int:
        # Heuristic for Lightning's progress bar; actual count depends on
        # which proteins land in which order each epoch.
        avg_len = max(1, int(self.lengths.float().mean().item()))
        n = len(self.index_sampler) if hasattr(self.index_sampler, "__len__") else len(self.lengths)
        approx_samples_per_batch = max(1, self.max_tokens_per_batch // avg_len)
        if self.max_batch_size is not None:
            approx_samples_per_batch = min(approx_samples_per_batch, self.max_batch_size)
        approx_batches = n // approx_samples_per_batch
        if self.shard_in_sampler and self.world_size > 1:
            approx_batches = approx_batches // self.world_size
        return max(1, approx_batches)
