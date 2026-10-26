"""Regression coverage for length termination when overlap advances a request ahead."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock

import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.scheduler import Scheduler


class TestSchedulerLengthTermination(TestCase):
    def _check_max_tokens(self, overlap: bool) -> None:
        req = Req(
            input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            table_idx=0,
            cached_len=0,
            output_len=2,
            uid=0,
            sampling_params=SamplingParams(max_tokens=2, ignore_eos=True),
            cache_handle=Mock(),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cache_manager = SimpleNamespace(lazy_free_region=nullcontext, cache_req=Mock())
        scheduler.decode_manager = SimpleNamespace(remove_req=Mock())
        scheduler.finished_reqs = set()
        scheduler.eos_token_id = -1
        scheduler.send_result = Mock()
        scheduler._free_req_resources = Mock()

        def result(phase: str, token: int):
            batch = Batch(reqs=[req], phase=phase)
            return (
                SimpleNamespace(batch=batch),
                (None, torch.tensor([token], dtype=torch.int32), Mock()),
            )

        req.complete_one()
        if overlap:
            # The next forward advances this same Req before the first result is processed.
            req.complete_one()
            self.assertFalse(req.can_decode)

        scheduler._process_last_data(result("prefill", 4))
        first_reply = scheduler.send_result.call_args.args[0][0]
        self.assertFalse(first_reply.finished)
        scheduler._free_req_resources.assert_not_called()

        if not overlap:
            req.complete_one()
        scheduler._process_last_data(result("decode", 5))

        replies = [call.args[0][0] for call in scheduler.send_result.call_args_list]
        self.assertEqual([reply.next_token for reply in replies], [4, 5])
        self.assertEqual([reply.finished for reply in replies], [False, True])
        self.assertEqual(req.input_ids.tolist(), [1, 2, 3, 4, 5])
        scheduler._free_req_resources.assert_called_once_with(req)

    def test_max_tokens_without_overlap(self) -> None:
        self._check_max_tokens(overlap=False)

    def test_max_tokens_with_overlap(self) -> None:
        self._check_max_tokens(overlap=True)


if __name__ == "__main__":
    main()
