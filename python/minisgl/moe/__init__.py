from __future__ import annotations

from typing import Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()


def resolve_auto_backend() -> str:
    return "fused"


def validate_moe_backend(backend: str) -> bool:
    if backend != "auto":
        SUPPORTED_MOE_BACKENDS.assert_supported(backend)
    return True


def create_moe_backend(backend: str) -> BaseMoeBackend:
    if backend == "auto":
        backend = resolve_auto_backend()
        logger.info(f"Auto-selected MoE backend: {backend}")
    else:
        validate_moe_backend(backend)
        logger.info(f"Selected MoE backend: {backend}")

    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
    "validate_moe_backend",
]