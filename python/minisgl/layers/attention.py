from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
        has_rope: bool = True,
        attn_temperature_tuning: bool = False,
        floor_scale=8192,
        attn_scale=0.1,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.has_rope = has_rope
        if has_rope:
            self.rotary = get_rope(
                head_dim=head_dim,
                rotary_dim=rotary_config.rotary_dim,
                max_position=rotary_config.max_position,
                base=rotary_config.base,
                rope_scaling=(
                    tuple(rotary_config.scaling.items()) if rotary_config.scaling else None
                ),
            )
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.attn_temperature_tuning = attn_temperature_tuning
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return attn_scale.unsqueeze(-1).unsqueeze(-1)

    def _mul_attn_scale(self, positions, q: torch.Tensor) -> torch.Tensor:
        attn_scale = self._get_attn_scale(positions)
        return (q * attn_scale).to(q.dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        rope_first: bool = False,
        qk_norm_combined: bool = False,
        qk_norm: RMSNorm | None = None,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)

        if rope_first and self.has_rope:
            self.rotary.forward(ctx.batch.positions, q, k)

        if qk_norm_combined and qk_norm is not None:
            qk, _ = qkv.split([self.qo_attn_dim + self.kv_attn_dim, self.kv_attn_dim], dim=-1)
            qk_norm.forward_inplace(
                qk.view(-1, self.num_qo_heads + self.num_kv_heads, self.head_dim)
            )
            del qk
        else:
            if self.q_norm is not None:
                self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
            if self.k_norm is not None:
                self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        if not rope_first and self.has_rope:
            q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)

        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        # the inference-time temperature tuning function is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        if self.attn_temperature_tuning and not self.has_rope:
            q = self._mul_attn_scale(positions=ctx.batch.positions, q=q)

        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
