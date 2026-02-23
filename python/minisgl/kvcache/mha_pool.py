from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool

if TYPE_CHECKING:
    from minisgl.hicache import HiCacheCounter


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
            pin_memory=pin_memory,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        self.counter: HiCacheCounter | None = None

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def set_hicache_counter(self, counter) -> None:
        self.counter = counter

    def create_host_memory_pool(self, num_pages: int) -> MHAKVCache:
        _, num_layers, _, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        return MHAKVCache(
            num_kv_heads=local_kv_heads * get_tp_info().size,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=self._kv_buffer.dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )
        if self.counter is not None:
            self.counter.wait(layer_id)

    def get_per_token_bytes(self) -> int:
        _, num_layers, _, _, local_kv_heads, head_dim = self._kv_buffer.shape
        return 2 * num_layers * local_kv_heads * head_dim * self._kv_buffer.element_size()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
