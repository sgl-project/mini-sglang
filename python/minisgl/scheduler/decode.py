from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Set

from minisgl.core import Batch, Req
from minisgl.utils import alloc_delta

if TYPE_CHECKING:
    from .cache import CacheManager
    from .prefill import PrefillManager
    from .table import TableManager


@dataclass
class DecodeManager:
    page_size: int
    new_token_ratio: float = 0.0
    min_new_token_ratio: float = 0.0
    new_token_ratio_decay: float = 0.0
    retract_decode_steps: int = 0
    running_reqs: Set[Req] = field(default_factory=set)
    did_retract: bool = False

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

    def check_decode_mem(self, available_size: int) -> bool:
        need = sum(
            alloc_delta(req.cached_len, req.extend_len, self.page_size) for req in self.running_reqs
        )
        return need <= available_size

    def schedule_next_batch(
        self,
        cache_manager: CacheManager,
        table_manager: TableManager,
        prefill_manager: PrefillManager,
    ) -> Batch | None:
        self.did_retract = False
        if not self.runnable:
            return None
        if not self.check_decode_mem(cache_manager.available_size):
            insert_idx = 0
            while (
                insert_idx < len(prefill_manager.pending_list)
                and prefill_manager.pending_list[insert_idx].chunked_req is not None
            ):
                insert_idx += 1
            while not self.check_decode_mem(cache_manager.available_size):
                if len(self.running_reqs) == 1:
                    raise RuntimeError("decode OOM")
                req = min(
                    self.running_reqs,
                    key=lambda req: (len(req.input_ids) - req.prompt_len, -req.prompt_len, req.uid),
                )
                self.running_reqs.remove(req)
                self.did_retract = True
                cache_manager.cache_req(req, finished=True)
                table_manager.free(req.table_idx)
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
