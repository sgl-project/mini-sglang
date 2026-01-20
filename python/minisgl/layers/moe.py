from typing import Optional

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers.base import BaseOP

# from minisgl.layers.moe.fused_moe.fused_moe_impl import fused_moe
from minisgl.utils import divide_even


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: Optional[int] = None,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        tp_size: Optional[int] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.params_dtype = params_dtype
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_info.size if tp_size is None else tp_size
        self.tp_rank = tp_info.rank if tp_size is None else 0

        assert self.intermediate_size % self.tp_size == 0, (
            f"Intermediate size ({self.intermediate_size}) must be divisible "
            f"by tp_size ({self.tp_size})"
        )

        self.intermediate_size_per_partition = self.intermediate_size // self.tp_size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.inplace = inplace
        self.no_combine = no_combine
        self.layer_id = layer_id

        intermediate_size_per_partition = divide_even(intermediate_size, self.tp_size)

        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        self.down_proj = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        ctx = get_global_ctx()

        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            inplace=self.inplace,
            activation=self.activation,
            no_combine=self.no_combine,
        )

        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)

        return final_hidden_states
