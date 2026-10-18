import pytest
import torch
import torch.nn.functional as F
import triton

from minisgl.kernel.triton.int8_lm_head import (
    _int8_lm_head_kernel,
    int8_lm_head,
    quantize_weight,
)


def _run_with_tile(x, int8_weight, scale, block_m, num_warps):
    m, n = x.shape[0], int8_weight.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(n, 128), triton.cdiv(m, block_m))
    _int8_lm_head_kernel[grid](
        x,
        int8_weight,
        scale,
        out,
        m,
        n,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        int8_weight.stride(0),
        int8_weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_K=64,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 4, 16, 32, 64])
def test_int8_lm_head_shapes_and_peak(m):
    torch.manual_seed(0)
    k, n = 256, 4096
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    x = weight[0:1].repeat(m, 1).contiguous()
    int8_weight, scale = quantize_weight(weight)
    out = int8_lm_head(x, weight, int8_weight, scale, refine_k=64)
    ref = F.linear(x, weight)
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.argmax(dim=-1), ref.argmax(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 16, 17, 32, 33, 64, 96])
def test_int8_lm_head_tile_boundaries(m):
    """Every tile height must still pick each row's own strongest logit."""
    torch.manual_seed(2)
    k, n = 512, 8192
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    rows = torch.randperm(n, device="cuda")[:m]
    x = weight[rows] * 2.0
    int8_weight, scale = quantize_weight(weight)
    out = int8_lm_head(x, weight, int8_weight, scale, refine_k=64)
    ref = F.linear(x, weight)
    assert out.shape == (m, n)
    assert torch.equal(out.argmax(dim=-1), ref.argmax(dim=-1))
    assert torch.equal(out.argmax(dim=-1), rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [32, 64, 96])
def test_int8_lm_head_tile_invariance(m):
    """A taller tile changes how often the weights are read, not the math."""
    torch.manual_seed(3)
    k, n = 256, 4096
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    int8_weight, scale = quantize_weight(weight)
    auto = int8_lm_head(x, weight, int8_weight, scale, refine_k=0)
    fixed = _run_with_tile(x, int8_weight, scale, block_m=16, num_warps=4)
    assert torch.equal(auto, fixed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_int8_lm_head_top5_overlap():
    torch.manual_seed(1)
    k, n = 256, 4096
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    int8_weight, scale = quantize_weight(weight)
    out = int8_lm_head(x, weight, int8_weight, scale, refine_k=64)
    ref = F.linear(x, weight)
    ref_top5 = ref.topk(5, dim=-1).indices
    out_top5 = out.topk(5, dim=-1).indices
    overlap = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ref_top5, out_top5))
    assert overlap >= int(0.95 * 5 * x.shape[0])
