from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, List, Tuple

import math
import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.utils import alloc_delta, init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    new_token_ratio: float
    clip_max_new_tokens_estimation: int
    cache_manager: CacheManager
    table_manager: TableManager

    @classmethod
    def create(
        cls,
        token_budget: int,
        new_token_ratio: float,
        clip_max_new_tokens_estimation: int,
        cache_manager: CacheManager,
        table_manager: TableManager,
        running_reqs: List[Req],
    ) -> PrefillAdder:
        adder = cls(
            token_budget=token_budget,
            reserved_size=0,
            new_token_ratio=new_token_ratio,
            clip_max_new_tokens_estimation=clip_max_new_tokens_estimation,
            cache_manager=cache_manager,
            table_manager=table_manager,
        )
        adder.reserved_size = sum(adder._get_running_reserve(req) for req in running_reqs)
        return adder

    def _get_running_reserve(self, req: Req) -> int:
        if req.sampling_params.ignore_eos:
            tail_est = req.remain_len
        else:
            tail_est = math.ceil(
                min(req.remain_len, self.clip_max_new_tokens_estimation) * self.new_token_ratio
            )
        return alloc_delta(req.device_len, tail_est, self.cache_manager.page_size)

    def _get_new_reserve(self, req: PendingReq, cached_len: int, chunk_size: int) -> int:
        # =======================================================================================
        # [0, cached_len)                                   prefix len already in cache
        # [cached_len, cached_len + chunk_size)             input len in this round
        # [cached_len + chunk_size, req.prompt_len)         prompt len left for later chunks
        # [req.prompt_len, req.prompt_len + req.output_len) decode budget len
        # =======================================================================================
        added_len = chunk_size
        if cached_len + chunk_size < req.input_len:
            return alloc_delta(cached_len, added_len, self.cache_manager.page_size)
        if req.sampling_params.ignore_eos:
            added_len += req.output_len
        else:
            added_len += min(req.output_len, self.clip_max_new_tokens_estimation)
        return alloc_delta(cached_len, added_len, self.cache_manager.page_size)

    def _can_add_one(self, req: PendingReq, cached_len: int) -> bool:
        extend_len = req.input_len - cached_len
        chunk_size = min(self.token_budget, extend_len)
        estimated_len = self._get_new_reserve(req, cached_len, chunk_size)
        return estimated_len + self.reserved_size <= self.cache_manager.available_size

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        if not self._can_add_one(req, cached_len):
            return None
        self.cache_manager.lock(handle)
        if not self._can_add_one(req, cached_len):
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += self._get_new_reserve(pending_req, cached_len, chunk_size)
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            prompt_len=pending_req.prompt_len,
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            if not self._can_add_one(pending_req, chunked_req.cached_len):
                return None
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(
            PendingReq(
                uid=req.uid,
                input_ids=req.input_ids,
                sampling_params=req.sampling_params,
                prompt_len=len(req.input_ids),
            )
        )

    def schedule_next_batch(
        self,
        prefill_budget: int,
        new_token_ratio: float,
        clip_max_new_tokens_estimation: int,
    ) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        adder = PrefillAdder.create(
            token_budget=prefill_budget,
            new_token_ratio=new_token_ratio,
            clip_max_new_tokens_estimation=clip_max_new_tokens_estimation,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            running_reqs=list(self.decode_manager.running_reqs),
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    def requeue_req(self, req: Req, idx: int) -> None:
        self.pending_list.insert(
            idx,
            PendingReq(
                uid=req.uid,
                input_ids=req.input_ids,  # NOTE: contains the prompt and the generated token
                sampling_params=replace(req.sampling_params, max_tokens=req.remain_len),
                prompt_len=req.prompt_len,
                chunked_req=None,
            ),
        )

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
