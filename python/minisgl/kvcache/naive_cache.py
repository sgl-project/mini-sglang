import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache

    def __init__(self):
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NaiveCacheHandle())

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
