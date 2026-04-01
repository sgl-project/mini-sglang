from typing import Dict, Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(
        self,
        size: int,
        eps: float,
        has_weight: bool = True,
    ) -> None:
        from flashinfer import rmsnorm

        self.has_weight = has_weight
        self.eps = eps
        self.weight = torch.ones(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)

    def state_dict(
        self, *, prefix: str = "", result: Dict[str, torch.Tensor] | None = None
    ) -> Dict[str, torch.Tensor]:
        if self.has_weight:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if self.has_weight:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

        if self.weight.device.type == "meta":
            ref = next(iter(state_dict.values()), None)
            if ref is None:
                raise RuntimeError(
                    f"Cannot materialize {prefix or 'RMSNorm'} weight without a reference tensor"
                )
            self.weight = torch.ones(self.weight.shape, dtype=self.weight.dtype, device=ref.device)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
