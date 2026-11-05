"""Check overlap result ordering and CUDA completion without loading a model."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.scheduler import ForwardInput, Scheduler
from minisgl.scheduler.table import TableManager


def make_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.finished_reqs = set()
    scheduler.eos_token_id = 99
    scheduler.decode_manager = DecodeManager(page_size=1)
    scheduler.table_manager = TableManager(2, torch.zeros((2, 32), dtype=torch.int32))
    scheduler.cache_manager = Mock()
    scheduler.cache_manager.lazy_free_region.side_effect = nullcontext
    scheduler.prefill_manager = SimpleNamespace(runnable=False)
    scheduler.engine = SimpleNamespace(stream=Mock())
    scheduler.stream = Mock()
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.receive_msg = Mock(return_value=[])
    scheduler.send_result = Mock()
    scheduler._schedule_next_batch = Mock(return_value=None)
    return scheduler


def make_req(scheduler, uid=0, output_len=8, ignore_eos=False):
    req = Req(
        input_ids=torch.tensor([10, 11], dtype=torch.int32),
        table_idx=scheduler.table_manager.allocate(),
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=ignore_eos),
        cache_handle=Mock(),
    )
    scheduler.decode_manager.running_reqs.add(req)
    return req


def submit(scheduler, reqs, tokens, phase="decode"):
    for req in reqs:
        req.complete_one()
    scheduler.decode_manager.filter_reqs(reqs)
    batch = Batch(reqs=reqs, phase=phase)
    return SimpleNamespace(batch=batch), (None, torch.tensor(tokens), Mock())


@pytest.mark.parametrize("late_token", [42, 99])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_eos_keeps_resources_until_inflight_result_is_drained(late_token, phase):
    scheduler = make_scheduler()
    req = make_req(scheduler)
    eos_data = submit(scheduler, [req], [99], phase=phase)
    late_input = SimpleNamespace(batch=Batch(reqs=[req], phase="decode"))
    scheduler._schedule_next_batch.return_value = late_input
    scheduler._forward = Mock(side_effect=lambda _: submit(scheduler, [req], [late_token])[1])

    late_data = scheduler.overlap_loop(eos_data)

    assert req not in scheduler.decode_manager.running_reqs
    assert scheduler.table_manager.available_size == 1
    scheduler.cache_manager.cache_req.assert_not_called()
    assert req.input_ids.tolist() == [10, 11, 99]
    replies = scheduler.send_result.call_args.args[0]
    assert [(r.next_token, r.finished) for r in replies] == [(99, True)]

    # A new request cannot reuse the row still written by the speculative step.
    other = make_req(scheduler, uid=1)
    assert other.table_idx != req.table_idx
    scheduler._schedule_next_batch.return_value = None
    scheduler.send_result.reset_mock()

    def check_retirement(retired, *, finished):
        assert retired is req and finished
        late_data[1][2].synchronize.assert_called_once()

    scheduler.cache_manager.cache_req.side_effect = check_retirement
    scheduler.overlap_loop(late_data)

    late_data[1][2].synchronize.assert_called_once()
    assert req.input_ids.tolist() == [10, 11, 99]
    assert scheduler.send_result.call_args.args[0] == []
    scheduler.cache_manager.cache_req.assert_called_once_with(req, finished=True)
    assert scheduler.table_manager.allocate() == req.table_idx
    assert not scheduler.finished_reqs


def test_finished_request_does_not_hide_other_results():
    scheduler = make_scheduler()
    req = make_req(scheduler)
    other = make_req(scheduler, uid=1)
    first = submit(scheduler, [req, other], [99, 20])
    next_input = SimpleNamespace(batch=Batch(reqs=[req, other], phase="decode"))
    scheduler._schedule_next_batch.return_value = next_input
    scheduler._forward = Mock(side_effect=lambda _: submit(scheduler, [req, other], [42, 21])[1])
    second = scheduler.overlap_loop(first)
    scheduler._schedule_next_batch.return_value = None
    scheduler.overlap_loop(second)
    assert other.input_ids.tolist() == [10, 11, 20, 21]
    assert [(r.uid, r.next_token) for r in scheduler.send_result.call_args.args[0]] == [(1, 21)]


def test_retirement_preserves_completed_batch_order():
    scheduler = make_scheduler()
    first = make_req(scheduler, uid=1)
    second = make_req(scheduler, uid=0)
    eos_data = submit(scheduler, [first, second], [99, 99])
    next_input = SimpleNamespace(batch=Batch([second, first], "decode"))
    scheduler._schedule_next_batch.return_value = next_input
    scheduler._forward = Mock(side_effect=lambda _: submit(scheduler, [second, first], [42, 42])[1])
    late_data = scheduler.overlap_loop(eos_data)
    scheduler.cache_manager.cache_req.assert_not_called()
    scheduler._schedule_next_batch.return_value = None
    scheduler.overlap_loop(late_data)
    retired = [call.args[0].uid for call in scheduler.cache_manager.cache_req.call_args_list]
    assert retired == [0, 1]
    assert scheduler.table_manager.available_size == 2


@pytest.mark.parametrize("output_len", [1, 2, 3])
@pytest.mark.parametrize("overlap", [False, True])
def test_length_limit_keeps_every_output_token(output_len, overlap):
    scheduler = make_scheduler()
    req = make_req(scheduler, output_len=output_len, ignore_eos=True)
    data = submit(scheduler, [req], [20])
    for i in range(output_len):
        if overlap and i + 1 < output_len:
            next_input = SimpleNamespace(batch=Batch(reqs=[req], phase="decode"))
            scheduler._schedule_next_batch.return_value = next_input
            scheduler._forward = Mock(
                side_effect=lambda _, token=21 + i: submit(scheduler, [req], [token])[1]
            )
        else:
            scheduler._schedule_next_batch.return_value = None
        if overlap:
            data = scheduler.overlap_loop(data)
        else:
            scheduler._process_last_data(data)
            if i + 1 < output_len:
                data = submit(scheduler, [req], [21 + i])
        reply = scheduler.send_result.call_args.args[0]
        assert len(reply) == 1
        assert reply[0].finished == (i == output_len - 1)
        assert scheduler.table_manager.available_size == (2 if reply[0].finished else 1)
    assert req.input_ids.tolist() == [10, 11] + list(range(20, 20 + output_len))
    scheduler.cache_manager.cache_req.assert_called_once_with(req, finished=True)


def test_eos_without_inflight_use_frees_immediately():
    scheduler = make_scheduler()
    req = make_req(scheduler)
    scheduler.overlap_loop(submit(scheduler, [req], [99], phase="prefill"))
    assert scheduler.table_manager.available_size == 2
    scheduler.cache_manager.cache_req.assert_called_once_with(req, finished=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA streams")
def test_retirement_completion_event_covers_token_pool_write():
    from minisgl.engine import ForwardOutput

    scheduler = make_scheduler()
    req = make_req(scheduler)
    eos_data = submit(scheduler, [req], [99])
    stream = torch.cuda.Stream()
    scheduler.engine.stream = stream
    scheduler.stream = stream
    scheduler.engine_stream_ctx = torch.cuda.stream(stream)
    with torch.cuda.stream(stream):
        pool = torch.zeros((2, 32), dtype=torch.int32, device="cuda")
        indices = (torch.tensor([req.table_idx], device="cuda"), torch.tensor([2], device="cuda"))
        next_tokens = torch.tensor([42], dtype=torch.int32, device="cuda")
        written = torch.cuda.Event()

        class DelayedPool:
            def __getitem__(self, key):
                return pool[key]

            def __setitem__(self, key, value):
                # Make the gap between the engine's D2H event and the pool write observable.
                torch.cuda._sleep(50_000_000)
                pool[key] = value
                written.record(stream)

        scheduler.token_pool = DelayedPool()

        def forward_batch(batch, args):
            cpu_tokens = next_tokens.to("cpu", non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(stream)
            return ForwardOutput(next_tokens, cpu_tokens, copied)

        scheduler.engine.forward_batch = forward_batch
        scheduler._schedule_next_batch.return_value = ForwardInput(
            Batch([req], "decode"), None, indices, indices
        )
        _, data = scheduler.overlap_loop(eos_data)
        try:
            data.copy_done_event.synchronize()
            assert written.query(), "request retirement can race with the token_pool write"
        finally:
            stream.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA streams")
def test_eos_without_next_batch_orders_row_reuse_after_last_write():
    from minisgl.engine import ForwardOutput

    scheduler = make_scheduler()
    req = make_req(scheduler)
    engine_stream, scheduler_stream = torch.cuda.Stream(), torch.cuda.Stream()
    scheduler.engine.stream = engine_stream
    scheduler.stream = scheduler_stream
    with torch.cuda.stream(engine_stream):
        pool = torch.zeros((2, 32), dtype=torch.int32, device="cuda")
        indices = (torch.tensor([req.table_idx], device="cuda"), torch.tensor([2], device="cuda"))
        eos = torch.tensor([99], dtype=torch.int32, device="cuda")
        prompt_token = torch.tensor(7, dtype=torch.int32, device="cuda")
        pool[req.table_idx, 2].copy_(prompt_token)
        cpu_tokens = eos.to("cpu", non_blocking=True)
        copied = torch.cuda.Event()
        copied.record(engine_stream)
        torch.cuda._sleep(500_000_000)
        scheduler.token_pool = pool
        scheduler.engine.forward_batch = lambda batch, args: ForwardOutput(eos, cpu_tokens, copied)
        forward_input = ForwardInput(Batch([req], "decode"), None, indices, indices)
        output = scheduler._forward(forward_input)
    req.complete_one()
    data = (forward_input, output)
    try:
        with torch.cuda.stream(scheduler_stream):
            scheduler._process_last_data(data)
            row = scheduler.table_manager.allocate()
            assert row == req.table_idx
            pool[row, 2].copy_(prompt_token)  # the next request's prompt
        engine_stream.synchronize()
        scheduler_stream.synchronize()
        assert pool[row, 2].item() == 7
    finally:
        engine_stream.synchronize()
        scheduler_stream.synchronize()
