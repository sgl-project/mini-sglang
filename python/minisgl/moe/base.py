from abc import ABC, abstractmethod
from typing import Callable

import torch


class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
        no_softmax: bool = False,
        custom_routing_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor: ...
