from __future__ import annotations

import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from minisgl.core import SamplingParams
from minisgl.llm.llm import LLM
from minisgl.message import BatchBackendMsg, UserMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.table import TableManager


def make_llm(prefill_budget=16):
    """Run the real offline/scheduling path with CPU tables and controlled model output."""
    llm = LLM.__new__(LLM)
    llm.engine = SimpleNamespace(max_seq_len=8)
    llm.eos_token_id = -1
    llm.prefill_budget = prefill_budget
    llm.finished_reqs = set()
    llm.tokenizer = Mock()
    llm.tokenizer.decode.side_effect = lambda ids: ",".join(map(str, ids))
    page_table = torch.zeros((9, 8), dtype=torch.int32)
    llm.table_manager = TableManager(8, page_table)
    llm.cache_manager = CacheManager(128, 1, page_table, "naive")
    llm.decode_manager = DecodeManager(1)
    llm.prefill_manager = PrefillManager(llm.cache_manager, llm.table_manager, llm.decode_manager)
    llm.receive_msg = llm.offline_receive_msg
    llm.send_result = llm.offline_send_result
    # GPU attention/sampler preparation is outside this CPU regression's scope.
    llm._prepare_batch = lambda batch: SimpleNamespace(batch=batch)

    def forward(forward_input):
        batch = forward_input.batch
        llm.cache_manager.allocate_paged(batch.reqs)
        for req in batch.reqs:
            req.complete_one()
        llm.decode_manager.filter_reqs(batch.reqs)
        tokens = torch.tensor([100 + req.uid for req in batch.reqs], dtype=torch.int32)
        return tokens, tokens, SimpleNamespace(synchronize=lambda: None)

    llm._forward = forward

    def run():
        # Deliberately isolate non-overlap admission from the separate #176/#181 work.
        for _ in range(100):
            llm.normal_loop()
        raise AssertionError("Requests did not finish within 100 CPU scheduler steps")

    llm.run_forever = run
    return llm


class TestSamplingParamsIsolation(unittest.TestCase):
    def setUp(self):
        # CPU-only torch cannot pin memory; the tables/copies and allocations stay real.
        empty = torch.empty
        self.pin = patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor)
        self.alloc = patch.object(
            torch, "empty", side_effect=lambda *a, **k: empty(*a, **{**k, "pin_memory": False})
        )
        self.pin.start()
        self.alloc.start()
        self.addCleanup(self.pin.stop)
        self.addCleanup(self.alloc.stop)

    def assert_released(self, llm):
        self.assertFalse(llm.pending_requests)
        self.assertFalse(llm.prefill_manager.pending_list)
        self.assertFalse(llm.decode_manager.running_reqs)
        self.assertEqual(llm.table_manager.available_size, 8)
        self.assertEqual(len(llm.cache_manager.free_slots), 128)
        self.assertEqual(len(torch.unique(llm.cache_manager.free_slots)), 128)
        llm.cache_manager.check_integrity()

    def check_batch(self, lengths, expected, budget=16, explicit_list=False):
        params = SamplingParams(temperature=0.7, top_k=7, top_p=0.8, ignore_eos=True, max_tokens=4)
        original = asdict(params)
        llm = make_llm(budget)
        supplied = [params] * len(lengths) if explicit_list else params
        results = llm.generate([[i + 1] * n for i, n in enumerate(lengths)], supplied)
        self.assertEqual([len(r["token_ids"]) for r in results], expected)
        self.assertEqual(
            [r["token_ids"] for r in results], [[100 + i] * n for i, n in enumerate(expected)]
        )
        self.assertEqual(asdict(params), original)
        self.assert_released(llm)

    def test_long_then_short_broadcast(self):
        self.check_batch([7, 2], [1, 4])

    def test_short_then_long_broadcast(self):
        self.check_batch([2, 7], [4, 1])

    def test_shared_explicit_list(self):
        self.check_batch([7, 2], [1, 4], explicit_list=True)

    def test_multiple_different_clamps(self):
        self.check_batch([5, 7, 2], [3, 1, 4])

    def test_chunked_prefill(self):
        self.check_batch([2, 7], [4, 1], budget=3)

    def test_requests_admitted_on_separate_iterations(self):
        self.check_batch([7, 2], [1, 4], budget=3)

    def test_no_clamping_control(self):
        self.check_batch([1, 2, 4], [4, 4, 4])

    def test_empty_batch(self):
        self.check_batch([], [])

    def test_reuse_across_generate_calls(self):
        params = SamplingParams(ignore_eos=True, max_tokens=4)
        llm = make_llm()
        first = llm.generate([[1] * 7], params)
        self.assertEqual(len(first[0]["token_ids"]), 1)
        self.assert_released(llm)
        second = llm.generate([[2] * 2], params)
        self.assertEqual(len(second[0]["token_ids"]), 4)
        self.assertEqual(params.max_tokens, 4)
        self.assert_released(llm)

    def test_distinct_params_preserve_each_caller_object(self):
        params = [SamplingParams(ignore_eos=True, max_tokens=6), SamplingParams(max_tokens=2)]
        llm = make_llm()
        results = llm.generate([[1] * 7, [2]], params)
        self.assertEqual([len(r["token_ids"]) for r in results], [1, 2])
        self.assertEqual([p.max_tokens for p in params], [6, 2])
        self.assert_released(llm)

    def test_pending_requests_keep_independent_budgets(self):
        llm = make_llm()
        params = SamplingParams(temperature=0.7, top_k=7, top_p=0.8, ignore_eos=True, max_tokens=4)
        first = UserMsg(0, torch.tensor([1, 2], dtype=torch.int32), params)
        second = UserMsg(1, torch.tensor([3] * 7, dtype=torch.int32), params)
        llm._process_one_msg(BatchBackendMsg([first, second]))
        pending = llm.prefill_manager.pending_list
        self.assertEqual([p.output_len for p in pending], [4, 1])
        self.assertIs(pending[0].sampling_params, params)
        self.assertIsNot(pending[1].sampling_params, params)
        expected = asdict(params)
        expected["max_tokens"] = 1
        self.assertEqual(asdict(pending[1].sampling_params), expected)
        self.assertIs(first.sampling_params, params)
        self.assertIs(second.sampling_params, params)
        self.assertEqual(params.max_tokens, 4)

    def test_single_clamp_preserves_message(self):
        llm = make_llm()
        params = SamplingParams(max_tokens=4)
        msg = UserMsg(7, torch.tensor([1] * 7, dtype=torch.int32), params)
        llm._process_one_msg(msg)
        pending = llm.prefill_manager.pending_list[0]
        self.assertEqual(pending.output_len, 1)
        self.assertIs(pending.input_ids, msg.input_ids)
        self.assertEqual(pending.uid, msg.uid)
        self.assertEqual(params.max_tokens, 4)
        self.assertIs(msg.sampling_params, params)


if __name__ == "__main__":
    unittest.main()
