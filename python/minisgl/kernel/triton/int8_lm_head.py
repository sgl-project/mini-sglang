from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_lm_head_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptrs, mask=offs_n[:, None] < N, other=0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    scale = tl.load(s_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc *= scale[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(
        y_ptrs,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8 quantization."""
    scale = (weight.abs().amax(dim=1, keepdim=True).float() / 127.0).clamp(min=1e-8)
    int8_weight = torch.round(weight.float() / scale).clamp(-127, 127).to(torch.int8).contiguous()
    return int8_weight, scale.reshape(-1).contiguous()


def _select_tile(m: int) -> Tuple[int, int]:
    """Pick a tile height that streams the weight matrix once per call.

    The kernel keeps bf16 tensor-core math and only halves the weight bytes, so
    its speed is decided by how many times the vocabulary matrix is read:
    ``ceil(m / BLOCK_M)`` passes. Decode batches up to 64 rows fit in one pass;
    past that the accumulator gets too large and a second pass is cheaper than
    the register pressure.
    """
    if m <= 16:
        return 16, 4
    if m <= 32:
        return 32, 8
    return 64, 8


def int8_lm_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    int8_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    refine_k: int = 64,
) -> torch.Tensor:
    """INT8 weight-only lm_head with optional exact BF16 top-k refinement.

    The GEMM reads half the weight bytes. When ``refine_k > 0``, the top-k
    logits are recomputed from the original BF16 weight rows; this keeps greedy
    decoding exact in practice while leaving the tail approximate.
    """
    m, n = x.shape[0], weight.shape[0]
    logits = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    block_m, num_warps = _select_tile(m)
    grid = (triton.cdiv(n, 128), triton.cdiv(m, block_m))
    _int8_lm_head_kernel[grid](
        x,
        int8_weight,
        weight_scale,
        logits,
        m,
        n,
        weight.shape[1],
        x.stride(0),
        x.stride(1),
        int8_weight.stride(0),
        int8_weight.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_K=64,
        num_warps=num_warps,
        num_stages=3,
    )
    if refine_k > 0:
        _, candidate_idx = torch.topk(logits, refine_k, dim=-1)
        candidate_rows = weight[candidate_idx]
        exact = (x.unsqueeze(1) * candidate_rows).sum(dim=-1)
        logits.scatter_(1, candidate_idx, exact)
    return logits
