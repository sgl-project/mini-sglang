from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, NamedTuple, cast

import torch
import torch.distributed as dist
from minisgl.core import get_global_ctx
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle, BasePrefixCache
    from minisgl.kvcache.hiradix_cache import HiRadixPrefixCache
    from minisgl.scheduler import SchedulerConfig


logger = init_logger(__name__)


@dataclass
class HiCacheCounter:
    num_layers: int
    start_event: torch.Event = field(init=False)
    events: List[torch.Event] = field(init=False)

    def __post_init__(self):
        self.events = [_create_event() for _ in range(self.num_layers)]
        self.start_event = _create_event(enable_timing=True)
        self.events[-1] = _create_event(enable_timing=True)

    def wait(self, layer_id: int) -> None:
        current_stream = torch.cuda.current_stream()
        self.events[layer_id].wait(current_stream)


class Transaction(NamedTuple):
    handle: BaseCacheHandle
    host_list: List[torch.Tensor]
    cuda_list: List[torch.Tensor]


class Ack(NamedTuple):
    ack_id: int
    handles: List[BaseCacheHandle]
    num_tokens: int
    start_event: torch.Event
    finish_event: torch.Event


RING_BUFFER_SIZE = 3
WRITE_LENGTH_THRESHOLD = 64
RESET_ACK_THRESHOLD = 512


class HiCacheController:
    def __init__(self, prefix_cache: BasePrefixCache, num_pages: int, config: SchedulerConfig):
        self.hiradix_cache = cast("HiRadixPrefixCache", prefix_cache)
        self.load_queue: List[Transaction] = []
        self.write_queue: List[Transaction] = []
        self.ack_load_queue: List[Ack] = []
        self.ack_write_queue: List[Ack] = []
        self.ack_cnt = 0
        self.load_stream = torch.cuda.Stream()
        self.write_stream = torch.cuda.Stream()
        self.load_stream_ctx = torch.cuda.stream(self.load_stream)
        self.write_stream_ctx = torch.cuda.stream(self.write_stream)
        self.kvcache = get_global_ctx().kv_cache
        self.counter_ring_buffer = [
            HiCacheCounter(self.kvcache.num_layers) for _ in range(RING_BUFFER_SIZE)
        ]
        self.ring_index = 0
        self.device = self.hiradix_cache.device
        num_host_pages = int(num_pages * config.hicache_ratio)
        num_host_tokens = num_host_pages * config.page_size
        self.token_bytes = self.kvcache.get_per_token_bytes()
        total_bytes_gb = num_host_tokens * self.token_bytes / (1024**3)
        self.free_slots = torch.arange(num_host_tokens, dtype=torch.int32, device="cpu")
        logger.info(
            f"Allocating {num_host_tokens} tokens "
            f"({total_bytes_gb:.2f} GB) for host memory pool"
        )
        self.host_kvcache = self.kvcache.create_host_memory_pool(num_host_pages)
        # DATA plane kernel args
        self.k_cache_ptrs = _make_ptrs(
            [self.kvcache.k_cache(i) for i in range(self.kvcache.num_layers)],
            device=self.device,
        )
        self.v_cache_ptrs = _make_ptrs(
            [self.kvcache.v_cache(i) for i in range(self.kvcache.num_layers)],
            device=self.device,
        )
        self.host_k_cache_ptrs = _make_ptrs(
            [self.host_kvcache.k_cache(i) for i in range(self.kvcache.num_layers)],
            device=self.device,
        )
        self.host_v_cache_ptrs = _make_ptrs(
            [self.host_kvcache.v_cache(i) for i in range(self.kvcache.num_layers)],
            device=self.device,
        )
        local_kv_heads, head_dim = self.kvcache.k_cache(0).shape[-2:]
        self.storage_shape = (-1, local_kv_heads * head_dim)
        item_bytes = self.kvcache.dtype.itemsize
        self.element_size = local_kv_heads * head_dim * item_bytes
        self.stride_bytes = self.kvcache.k_cache(0).stride(-2) * item_bytes
        self.host_stride_bytes = self.host_kvcache.k_cache(0).stride(-2) * item_bytes

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
        needed_len = self.hiradix_cache.get_writable_length(cuda_handle)
        if needed_len < WRITE_LENGTH_THRESHOLD:
            return
        host_indices = self._try_allocate_host(needed_len)
        if host_indices is None:
            return
        assert len(host_indices) == needed_len
        cuda_list = self.hiradix_cache.set_host(cuda_handle, host_indices)
        self.hiradix_cache.lock_handle(cuda_handle, unlock=False)
        self.write_queue.append(Transaction(cuda_handle, [host_indices], cuda_list))
        self.start_write()  # do not batch write for now

    def start_load(self) -> None:
        if not self.load_queue:
            return self.kvcache.set_hicache_counter(None)
        from minisgl.kernel import transfer_hicache_one_layer

        self.ring_index = (self.ring_index + 1) % RING_BUFFER_SIZE
        counter = self.counter_ring_buffer[self.ring_index]
        self.kvcache.set_hicache_counter(counter)
        host_indices, cuda_indices = self._merge_transactions(self.load_queue)
        num_tokens = len(host_indices)
        current_stream = torch.cuda.current_stream()
        counter.start_event.record(self.load_stream)
        with self.load_stream_ctx:
            self.load_stream.wait_stream(current_stream)
            for i in range(self.kvcache.num_layers):
                transfer_hicache_one_layer(
                    k_cache_dst=self.kvcache.k_cache(i).view(self.storage_shape),
                    v_cache_dst=self.kvcache.v_cache(i).view(self.storage_shape),
                    indices_dst=cuda_indices,
                    k_cache_src=self.host_kvcache.k_cache(i).view(self.storage_shape),
                    v_cache_src=self.host_kvcache.v_cache(i).view(self.storage_shape),
                    indices_src=host_indices,
                )
                counter.events[i].record(self.load_stream)
            finish_event = counter.events[-1]

        # NOTE: must record here to avoid use after free
        host_indices.record_stream(self.load_stream)
        cuda_indices.record_stream(self.load_stream)
        self.load_queue.clear()
        ack_id = self._allocate_ack_id()
        self.ack_load_queue.append(Ack(ack_id, [], num_tokens, counter.start_event, finish_event))
        logger.info_rank0(f"HiCache Load  [{ack_id}]: {num_tokens:>5} tokens")

    def start_write(self) -> None:
        if not self.write_queue:
            return
        from minisgl.kernel import transfer_hicache_all_layer

        handles = [tx.handle for tx in self.write_queue]
        host_indices, cuda_indices = self._merge_transactions(self.write_queue)
        num_tokens = len(host_indices)
        current_stream = torch.cuda.current_stream()
        start_event = _create_event(enable_timing=True)
        finish_event = _create_event(enable_timing=True)
        start_event.record(self.write_stream)
        with self.write_stream_ctx:
            self.write_stream.wait_stream(current_stream)
            # TODO: refactor this kernel
            transfer_hicache_all_layer(
                k_ptr_dst=self.host_k_cache_ptrs,
                v_ptr_dst=self.host_v_cache_ptrs,
                indices_dst=host_indices,
                k_ptr_src=self.k_cache_ptrs,
                v_ptr_src=self.v_cache_ptrs,
                indices_src=cuda_indices,
                kv_cache_src_stride_bytes=self.stride_bytes,
                kv_cache_dst_stride_bytes=self.host_stride_bytes,
                element_size=self.element_size,
            )

        # NOTE: must record stream to avoid use after free
        finish_event.record(self.write_stream)
        host_indices.record_stream(self.write_stream)
        cuda_indices.record_stream(self.write_stream)
        self.write_queue.clear()
        ack_id = self._allocate_ack_id()
        self.ack_write_queue.append(Ack(ack_id, handles, num_tokens, start_event, finish_event))
        logger.info_rank0(f"HiCache Write [{ack_id}]: {num_tokens:>5} tokens")

    def refresh(self, tp_cpu_group: torch.distributed.ProcessGroup) -> None:
        # NOTE: load has no side-effect (only logging), so no need to sync
        finish_count = 0
        for ack in self.ack_load_queue:
            if not ack.finish_event.query():
                break
            finish_count += 1
            self._log_transaction(ack, "Load ")
        self.ack_load_queue = self.ack_load_queue[finish_count:]

        finish_count = 0
        for ack in self.ack_write_queue:
            if not ack.finish_event.query():
                break
            finish_count += 1

        # NOTE: write must synchronize to reach consensus on the finished count
        finish_count = torch.tensor(finish_count, dtype=torch.int32, device="cpu")
        dist.all_reduce(finish_count, op=dist.ReduceOp.MIN, group=tp_cpu_group)
        finish_count = int(finish_count)

        for ack in self.ack_write_queue[:finish_count]:
            self._log_transaction(ack, "Write")
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

    def _try_allocate_host(self, length: int) -> torch.Tensor | None:
        if length > len(self.free_slots):
            evicted = self.hiradix_cache.try_evict_host(length - len(self.free_slots))
            self.free_slots = torch.cat([self.free_slots] + evicted)
            if length > len(self.free_slots):  # give up if still not enough
                return None
        allocated, self.free_slots = self.free_slots[:length], self.free_slots[length:]
        return allocated

    def _allocate_counter(self) -> HiCacheCounter:
        self.ring_index = (self.ring_index + 1) % RING_BUFFER_SIZE
        return self.counter_ring_buffer[self.ring_index]

    def _allocate_ack_id(self) -> int:
        self.ack_cnt = (self.ack_cnt + 1) % RESET_ACK_THRESHOLD
        return self.ack_cnt

    def _log_transaction(self, ack: Ack, stage: str):
        dur = ack.start_event.elapsed_time(ack.finish_event)
        bandwidth = (self.token_bytes * ack.num_tokens / (1024**3)) / (dur / 1000)
        logger.info(
            f"HiCache {stage} [{ack.ack_id}]: {ack.num_tokens:>5} tokens: "
            f"duration = {dur:>5.2f} ms, bandwidth = {bandwidth:>5.2f} GB/s"
        )


# NOTE: skip the annoying type checking here...
def _create_event(enable_timing: bool = False) -> torch.Event:
    return torch.cuda.Event(enable_timing=enable_timing)  # type: ignore


def _make_ptrs(ts: List[torch.Tensor], device: torch.device):
    return torch.tensor([t.data_ptr() for t in ts], device=device, dtype=torch.uint64)
