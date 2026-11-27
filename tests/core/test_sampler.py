from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from minisgl.core import Batch, SamplingParams
from minisgl.engine import sample


@pytest.fixture
def sampler(monkeypatch):
    # Exercise row selection on CPU, without a CUDA runtime or FlashInfer kernel.
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _: nullcontext())
    monkeypatch.setattr(
        sample, "make_device_tensor", lambda data, dtype, device: torch.tensor(data, dtype=dtype)
    )
    return sample.Sampler(torch.device("cpu"), 4)


def batch(*params):
    return Batch(reqs=[SimpleNamespace(sampling_params=p) for p in params], phase="decode")


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_mixed_greedy_rows(sampler, monkeypatch, seed):
    logits = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [2.0, 1.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 3.0, 3.0, 0.0]]
    )
    params = batch(
        SamplingParams(temperature=0.8, top_k=3, top_p=0.9),
        SamplingParams(),
        SamplingParams(temperature=0.7),
        SamplingParams(temperature=0.8, top_k=1),
    )
    draws = torch.randint(4, (4,), generator=torch.Generator().manual_seed(seed), dtype=torch.int32)

    def draw(x, temperatures, top_k, top_p):
        torch.testing.assert_close(temperatures, torch.tensor([0.8, 1e-6, 0.7, 1e-6]))
        torch.testing.assert_close(top_k, torch.tensor([3, 4, 4, 1], dtype=torch.int32))
        torch.testing.assert_close(top_p, torch.tensor([0.9, 1.0, 1.0, 1.0]))
        x.zero_()  # A sampling backend may reuse the original float32 logits.
        return draws

    monkeypatch.setattr(sample, "sample_impl", draw)
    result = sampler.sample(logits, sampler.prepare(params))
    torch.testing.assert_close(result, torch.tensor([draws[0], 2, draws[2], 1], dtype=draws.dtype))


def test_homogeneous_fast_paths(sampler, monkeypatch):
    logits = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    draw = Mock(return_value=torch.tensor([3], dtype=torch.int32))
    monkeypatch.setattr(sample, "sample_impl", draw)
    greedy = sampler.prepare(batch(SamplingParams()))
    torch.testing.assert_close(sampler.sample(logits, greedy), torch.tensor([1]))
    draw.assert_not_called()

    transfer = Mock(wraps=sample.make_device_tensor)
    monkeypatch.setattr(sample, "make_device_tensor", transfer)
    monkeypatch.setattr(torch, "argmax", Mock(side_effect=AssertionError("unexpected argmax")))
    random = sampler.prepare(batch(SamplingParams(temperature=0.8)))
    assert [c.args[1] for c in transfer.call_args_list] == [torch.float32]
    assert sampler.sample(logits, random) is draw.return_value
    draw.assert_called_once_with(logits, random.temperatures, None, None)
