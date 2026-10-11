from __future__ import annotations

from minisgl.engine.graph import _determine_cuda_graph_bs

GiB = 1 << 30


def test_explicit_bs_list_is_kept():
    assert _determine_cuda_graph_bs([3, 5], None, 0) == [3, 5]


def test_zero_disables_cuda_graph():
    assert _determine_cuda_graph_bs(None, 0, 100 * GiB) == []


def test_small_max_bs_is_respected():
    # shell mode sets cuda_graph_max_bs=1: never capture graphs larger than that
    assert _determine_cuda_graph_bs(None, 1, 100 * GiB) == [1]
    assert _determine_cuda_graph_bs(None, 3, 100 * GiB) == [1, 2]


def test_large_max_bs():
    assert _determine_cuda_graph_bs(None, 24, 100 * GiB) == [1, 2, 4, 8, 16, 24]
    assert max(_determine_cuda_graph_bs(None, None, 100 * GiB)) == 256
    assert max(_determine_cuda_graph_bs(None, None, 40 * GiB)) == 160
