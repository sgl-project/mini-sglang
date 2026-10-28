from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from minisgl.core import SamplingParams


def load_offline_llm():
    scheduler = ModuleType("minisgl.scheduler")
    scheduler.Scheduler = object
    scheduler.SchedulerConfig = Mock()
    path = Path(__file__).resolve().parents[2] / "python/minisgl/llm/llm.py"
    spec = importlib.util.spec_from_file_location("offline_llm_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module, "minisgl.scheduler": scheduler}):
        spec.loader.exec_module(module)
    return module


class TestGenerateSamplingParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_offline_llm()

    def setUp(self):
        self.llm = self.module.LLM.__new__(self.module.LLM)
        self.llm.pending_requests = [([42], SamplingParams())]
        self.llm.status_map = {7: self.module.RequestStatus(7, [42], [43])}
        self.llm.counter = 8
        self.llm.tokenizer = Mock()
        self.llm.tokenizer.decode.side_effect = lambda ids: str(ids[0])
        self.admitted = []

        def finish_requests():
            self.admitted.extend(self.llm.pending_requests)
            for uid, (prompt, _) in enumerate(self.llm.pending_requests):
                self.llm.status_map[uid] = self.module.RequestStatus(uid, prompt, [uid + 10])
            raise self.module.RequestAllFinished()

        self.llm.run_forever = Mock(side_effect=finish_requests)

    def test_mismatched_lengths_rejected_before_state_changes(self):
        pending = self.llm.pending_requests
        statuses = self.llm.status_map
        for prompt_count, param_count in [(2, 1), (1, 2), (1, 0), (0, 1)]:
            with self.subTest(prompts=prompt_count, sampling_params=param_count):
                prompts = [[1]] * prompt_count
                params = [SamplingParams()] * param_count
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{param_count} sampling parameters for {prompt_count} prompts",
                ):
                    self.llm.generate(prompts, params)
                self.assertIs(self.llm.pending_requests, pending)
                self.assertEqual(self.llm.pending_requests[0][0], [42])
                self.assertIs(self.llm.status_map, statuses)
                self.assertEqual(self.llm.status_map[7].output_ids, [43])
                self.assertEqual(self.llm.counter, 8)
                self.llm.run_forever.assert_not_called()

    def test_matching_lists_preserve_request_order(self):
        prompts = [[1], [2]]
        params = [SamplingParams(max_tokens=1), SamplingParams(max_tokens=2)]
        result = self.llm.generate(prompts, params)
        self.assertEqual(self.admitted, list(zip(prompts, params)))
        self.assertEqual(
            result, [{"text": "10", "token_ids": [10]}, {"text": "11", "token_ids": [11]}]
        )
        self.llm.run_forever.assert_called_once_with()

    def test_single_sampling_params_broadcasts(self):
        params = SamplingParams(max_tokens=2)
        self.llm.generate(["first", "second"], params)
        self.assertEqual([prompt for prompt, _ in self.admitted], ["first", "second"])
        self.assertTrue(all(sp is params for _, sp in self.admitted))
        self.llm.run_forever.assert_called_once_with()

    def test_empty_batch(self):
        for params in ([], SamplingParams()):
            with self.subTest(sampling_params=params):
                self.assertEqual(self.llm.generate([], params), [])
                self.assertEqual(self.llm.pending_requests, [])
                self.assertEqual(self.llm.status_map, {})
                self.assertEqual(self.llm.counter, 0)
        self.assertEqual(self.admitted, [])


if __name__ == "__main__":
    unittest.main()
