"""Eviction chooses only from free blocks; BlockManager owns block safety."""
from collections import deque
from typing import Protocol


class CacheEvictionPolicy(Protocol):
    def select_victim(self, free_block_ids: deque[int]) -> int:
        """Read candidates without modifying the manager's queue."""
        ...

    def on_acquire(self, block_id: int) -> None:
        """A request obtained a new or cached block reference, not a probe."""
        ...

    def on_release(self, block_id: int) -> None:
        """The last reference was dropped and the block became free."""
        ...


class OriginalFreeQueuePolicy:
    """Preserve the upstream free-queue order, with no explicit access clock."""

    def select_victim(self, free_block_ids: deque[int]) -> int:
        return free_block_ids[0]

    def on_acquire(self, block_id: int) -> None:
        pass

    def on_release(self, block_id: int) -> None:
        pass


class LRUCachePolicy:
    """LRU by successful reference acquisition, restricted to free candidates.

    This clock does not represent attention reads, wall time or release time.
    Never-acquired candidates have time zero; ties retain free-queue order.
    """

    def __init__(self):
        self.clock = 0
        self.last_access: dict[int, int] = {}

    def select_victim(self, free_block_ids: deque[int]) -> int:
        return min(free_block_ids, key=lambda block_id: self.last_access.get(block_id, 0))

    def on_acquire(self, block_id: int) -> None:
        self.clock += 1
        self.last_access[block_id] = self.clock

    def on_release(self, block_id: int) -> None:
        # Becoming free is not a new acquisition and must not refresh recency.
        pass


def create_cache_policy(name: str) -> CacheEvictionPolicy:
    if name == 'original':
        return OriginalFreeQueuePolicy()
    if name == 'lru':
        return LRUCachePolicy()
    raise ValueError('cache_eviction_policy must be original or lru')
