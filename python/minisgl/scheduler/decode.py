from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Set

from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.utils import alloc_delta

if TYPE_CHECKING:
    from .cache import CacheManager
    from .prefill import PrefillManager
    from .table import TableManager


@dataclass
class DecodeManager:
    page_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    new_token_ratio: float = field(init=False)
    running_reqs: Set[Req] = field(default_factory=set)
    did_retract: bool = False

    def __post_init__(self) -> None:
        self.init_new_token_ratio = min(
            ENV.INIT_NEW_TOKEN_RATIO.value * ENV.SCHEDULE_CONSERVATIVENESS.value, 1.0
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * ENV.MIN_NEW_TOKEN_RATIO_FACTOR.value, 1.0
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / ENV.NEW_TOKEN_RATIO_DECAY_STEPS.value
        self.new_token_ratio = self.init_new_token_ratio
        self.clip_max_new_tokens_estimation = ENV.CLIP_MAX_NEW_TOKENS_ESTIMATION.value
        self.retract_decode_steps = ENV.RETRACT_DECODE_STEPS.value

    def reset_new_token_ratio(self) -> None:
        self.new_token_ratio = self.init_new_token_ratio

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    def check_decode_mem(self, available_size: int, steps: int = 1) -> bool:
        need = sum(
            alloc_delta(
                req.cached_len,
                min(req.max_device_len - req.cached_len, steps),
                self.page_size,
            )
            for req in self.running_reqs
        )
        return need <= available_size

    def estimated_inflight_tokens(self) -> int:
        return sum(self._get_running_reserve(req) for req in self.running_reqs)

    def _get_running_reserve(self, req: Req) -> int:
        if req.sampling_params.ignore_eos:
            tail_est = req.remain_len
        else:
            tail_est = math.ceil(
                min(req.remain_len, self.clip_max_new_tokens_estimation) * self.new_token_ratio
            )
        return alloc_delta(req.device_len, tail_est, self.page_size)

    def schedule_next_batch(
        self,
        prefill_manager: PrefillManager,
    ) -> Batch | None:
        self.did_retract = False
        if not self.runnable:
            return None
        if not self.check_decode_mem(self.cache_manager.available_size):
            insert_idx = 0
            while (
                insert_idx < len(prefill_manager.pending_list)
                and prefill_manager.pending_list[insert_idx].chunked_req is not None
            ):
                insert_idx += 1
            while not self.check_decode_mem(
                self.cache_manager.available_size, steps=self.retract_decode_steps
            ):
                if len(self.running_reqs) == 1:
                    raise RuntimeError("decode OOM")
                req = min(
                    self.running_reqs,
                    key=lambda req: (len(req.input_ids) - req.prompt_len, -req.prompt_len, req.uid),
                )
                self.running_reqs.remove(req)
                self.did_retract = True
                req.is_retracted = True
                self.cache_manager.cache_req(req, finished=True)
                self.table_manager.free(req.table_idx)
                prefill_manager.requeue_req(req, insert_idx)
                insert_idx += 1
        batch = Batch(reqs=list(self.running_reqs), phase="decode")
        if self.did_retract:
            decoded_tokens = sum(len(req.input_ids) - req.prompt_len for req in batch.reqs)
            total_tokens = sum(req.max_device_len - req.prompt_len for req in batch.reqs)
            self.new_token_ratio = min(
                1.0,
                (decoded_tokens + self.retract_decode_steps * len(batch.reqs)) / total_tokens,
            )
        else:
            self.new_token_ratio = max(
                self.min_new_token_ratio,
                self.new_token_ratio - self.new_token_ratio_decay,
            )
        return batch

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
