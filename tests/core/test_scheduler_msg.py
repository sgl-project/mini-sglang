"""
Test how the scheduler handles user messages, without a GPU engine.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import minisgl.distributed.info as dist_info
import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.message import DetokenizeMsg, UserMsg
from minisgl.scheduler.scheduler import Scheduler

EOS_TOKEN_ID = 2


@pytest.fixture(autouse=True)
def single_rank_tp_info():
    """The rank-0 logger needs TP info; restore the global afterwards."""
    old_info = dist_info._TP_INFO
    dist_info._TP_INFO = dist_info.DistributedInfo(0, 1)
    yield
    dist_info._TP_INFO = old_info


class _FakePrefillManager:
    def __init__(self) -> None:
        self.added: List[UserMsg] = []

    def add_one_req(self, msg: UserMsg) -> None:
        self.added.append(msg)


def _make_fake_scheduler(max_seq_len: int):
    sent: List[DetokenizeMsg] = []
    fake = SimpleNamespace(
        engine=SimpleNamespace(max_seq_len=max_seq_len),
        eos_token_id=EOS_TOKEN_ID,
        prefill_manager=_FakePrefillManager(),
        send_result=sent.extend,
    )
    return fake, sent


def _user_msg(uid: int, input_len: int, max_tokens: int) -> UserMsg:
    return UserMsg(
        uid=uid,
        input_ids=torch.zeros(input_len, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def test_too_long_request_is_finished_not_dropped_silently():
    fake, sent = _make_fake_scheduler(max_seq_len=8)
    Scheduler._process_one_msg(fake, _user_msg(uid=7, input_len=8, max_tokens=4))  # type: ignore

    assert fake.prefill_manager.added == []
    assert len(sent) == 1
    assert sent[0].uid == 7
    assert sent[0].finished
    assert sent[0].next_token == EOS_TOKEN_ID


def test_max_tokens_is_clamped_to_remaining_length():
    fake, sent = _make_fake_scheduler(max_seq_len=8)
    Scheduler._process_one_msg(fake, _user_msg(uid=1, input_len=5, max_tokens=100))  # type: ignore

    assert sent == []
    assert len(fake.prefill_manager.added) == 1
    assert fake.prefill_manager.added[0].sampling_params.max_tokens == 3
