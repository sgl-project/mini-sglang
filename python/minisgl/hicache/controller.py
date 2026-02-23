from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, cast

import torch
import torch.distributed as dist
from minisgl.core import get_global_ctx

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle, BasePrefixCache
    from minisgl.kvcache.hiradix_cache import HiRadixPrefixCache


@dataclass
class HiCacheCounter:
    events: List[torch.cuda.Event]

    def wait(self, layer_id: int) -> None:
        current_stream = torch.cuda.current_stream()
        self.events[layer_id].wait(current_stream)


class Transaction(NamedTuple):
    handle: BaseCacheHandle
    host_list: List[torch.Tensor]
    cuda_list: List[torch.Tensor]


class Acknowledgement(NamedTuple):
    handles: List[BaseCacheHandle]
    finish_event: torch.cuda.Event


RING_BUFFER_SIZE = 3
WRITE_LENGTH_THRESHOLD = 64


class HiCacheController:
    def __init__(self, cache: BasePrefixCache, num_tokens: int):
        self.hiradix_cache = cast("HiRadixPrefixCache", cache)
        self.load_queue: List[Transaction] = []
        self.write_queue: List[Transaction] = []
        self.ack_write_queue: List[Acknowledgement] = []
        self.load_stream = torch.cuda.Stream()
        self.write_stream = torch.cuda.Stream()
        self.load_stream_ctx = torch.cuda.stream(self.load_stream)
        self.write_stream_ctx = torch.cuda.stream(self.write_stream)
        self.kvcache = get_global_ctx().kv_cache
        self.counter_ring_buffer = [
            HiCacheCounter([torch.cuda.Event() for _ in range(self.kvcache.num_layers)])
            for _ in range(RING_BUFFER_SIZE)
        ]
        self.ring_index = 0
        self.device = self.hiradix_cache.device
        self.free_slots = torch.arange(num_tokens, dtype=torch.int32, device="cpu")

    def try_allocate_host(self, length: int) -> torch.Tensor | None:
        if length > len(self.free_slots):
            evicted = self.hiradix_cache.try_evict_host(length - len(self.free_slots))
            self.free_slots = torch.cat([self.free_slots] + evicted)
            if length > len(self.free_slots):  # give up if still not enough
                return None
        allocated, self.free_slots = self.free_slots[:length], self.free_slots[length:]
        return allocated

    def prepare_load(
        self,
        host_handle: BaseCacheHandle,
        cuda_handle: BaseCacheHandle,
        cuda_indices: torch.Tensor,
    ) -> None:
        host_list = self.hiradix_cache.set_cuda(host_handle, cuda_indices)
        self.hiradix_cache.lock_handle(host_handle, unlock=False)
        self.hiradix_cache.lock_handle(cuda_handle, unlock=True)
        self.load_queue.append(Transaction(host_handle, host_list, [cuda_indices]))

    def prepare_write(self, cuda_handle: BaseCacheHandle) -> None:
        needed_len = self.hiradix_cache.get_cuda_only_length(cuda_handle)
        if needed_len < WRITE_LENGTH_THRESHOLD:
            return
        host_indices = self.try_allocate_host(needed_len)
        if host_indices is None:
            return
        cuda_list = self.hiradix_cache.set_host(cuda_handle, host_indices)
        self.hiradix_cache.lock_handle(cuda_handle, unlock=False)
        self.write_queue.append(Transaction(cuda_handle, [host_indices], cuda_list))
        self.start_write()  # do not batch write for now

    def start_load(self) -> None:
        if not self.load_queue:
            return self.kvcache.set_hicache_counter(None)
        counter = self.counter_ring_buffer[self.ring_index]
        self.kvcache.set_hicache_counter(counter)
        self.ring_index = (self.ring_index + 1) % RING_BUFFER_SIZE
        host_indices, cuda_indices = self._merge_transactions(self.load_queue)
        current_stream = torch.cuda.current_stream()
        with self.load_stream_ctx:
            self.load_stream.wait_stream(current_stream)
            for i in range(self.kvcache.num_layers):
                # TODO: do the real layer-wise copy here
                counter.events[i].record(self.load_stream)
            raise NotImplementedError("Hicache Load is not implemented yet.")
        # NOTE: must record here to avoid use after free
        host_indices.record_stream(self.load_stream)
        cuda_indices.record_stream(self.load_stream)

    def start_write(self) -> None:
        if not self.write_queue:
            return
        handles = [tx.handle for tx in self.write_queue]
        host_indices, cuda_indices = self._merge_transactions(self.write_queue)
        current_stream = torch.cuda.current_stream()
        with self.write_stream_ctx:
            self.write_stream.wait_stream(current_stream)
            # TODO: do the batch all-layer write here
            raise NotImplementedError("Hicache Write is not implemented yet.")
        # NOTE: must record stream to avoid use after free
        host_indices.record_stream(self.write_stream)
        cuda_indices.record_stream(self.write_stream)
        self.ack_write_queue.append(Acknowledgement(handles, self.write_stream.record_event()))

    def refresh(self, tp_cpu_group: torch.distributed.ProcessGroup) -> None:
        finish_count = 0
        for ack in self.ack_write_queue:
            if not ack.finish_event.query():
                break
            finish_count += 1

        # NOTE: must synchronize to reach consensus on the finished count
        finish_count = torch.tensor(finish_count, dtype=torch.int32, device="cpu")
        dist.all_reduce(finish_count, op=dist.ReduceOp.MIN, group=tp_cpu_group)
        finish_count = int(finish_count)

        for ack in self.ack_write_queue[:finish_count]:
            for handle in ack.handles:
                self.hiradix_cache.lock_handle(handle, unlock=True)
        self.ack_write_queue = self.ack_write_queue[finish_count:]

    def _merge_transactions(self, txs: List[Transaction]):
        assert len(txs) > 0
        host_list: List[torch.Tensor] = []
        cuda_list: List[torch.Tensor] = []
        for _, host_values, cuda_values in txs:
            host_list.extend(host_values)
            cuda_list.extend(cuda_values)
        host_indices, cuda_indices = torch.cat(host_list), torch.cat(cuda_list)
        return host_indices.to(self.device, non_blocking=True), cuda_indices
