from __future__ import annotations

import functools
import os
import pathlib
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


def _link_flags_for_nccl_lib_dir(lib_dir: pathlib.Path) -> list[str] | None:
    lib_dir = lib_dir.resolve()
    if not lib_dir.is_dir():
        return None
    rpath = f"-Wl,-rpath,{lib_dir}"
    if (lib_dir / "libnccl.so").exists():
        return [f"-L{lib_dir}", rpath, "-lnccl"]
    for name in ("libnccl.so.2", "libnccl.so.1"):
        if (lib_dir / name).exists():
            return [f"-L{lib_dir}", rpath, f"-l:{name}"]
    for so in sorted(lib_dir.glob("libnccl.so.*")):
        return [f"-L{lib_dir}", rpath, f"-l:{so.name}"]
    return None


def _discover_nccl_ldflags() -> list[str]:
    """Resolve NCCL for JIT link: system installs often lack libnccl; PyTorch wheels ship it under nvidia/nccl/lib."""
    for key in ("MINISGL_NCCL_LIB_DIR", "NCCL_LIB_DIR"):
        raw = os.environ.get(key)
        if raw:
            flags = _link_flags_for_nccl_lib_dir(pathlib.Path(raw))
            if flags:
                return flags
    try:
        import nvidia.nccl as nccl_pkg  # type: ignore[import-not-found]

        paths = getattr(nccl_pkg, "__path__", None)
        if paths:
            lib_dir = pathlib.Path(next(iter(paths))) / "lib"
            flags = _link_flags_for_nccl_lib_dir(lib_dir)
            if flags:
                return flags
    except ImportError:
        pass
    try:
        import torch

        flags = _link_flags_for_nccl_lib_dir(
            pathlib.Path(torch.__file__).resolve().parent / "lib"
        )
        if flags:
            return flags
    except ImportError:
        pass
    return ["-lnccl"]


@functools.cache
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=_discover_nccl_ldflags())


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
