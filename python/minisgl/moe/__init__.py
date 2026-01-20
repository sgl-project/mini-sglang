from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend
logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(
        self
    ) -> BaseMoeBackend: ...

SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")

@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()

def resolve_auto_backend() -> str:
    return "fused"


def validate_backend(backend: str):
    if backend != "auto":
        supported = SUPPORTED_MOE_BACKENDS.supported_names()
        if backend not in supported:
            raise ValueError(f"Unsupported MoE backend: {backend}. Supported backends: {supported}")
    return True

def create_moe_backend(
    backend: str,
) -> BaseMoeBackend:
    if backend == "auto":
        backend = resolve_auto_backend()
        logger.info(f"Auto-selected MoE backend: {backend}")
    else:
        validate_backend(backend)
        logger.info(f"Selected MoE backend: {backend}")

    return SUPPORTED_MOE_BACKENDS[backend]()



__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
    "validate_backend",
]
