from __future__ import annotations

import glob
import logging
import re
import time
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight, init_logger
from tqdm import tqdm

logger = init_logger(__name__)

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _shard_tensor_view(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Like _shard_tensor but returns zero-copy views. For dim-1 splits, returns
    .contiguous() since H2D requires contiguous memory. No .clone() on dim-0."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim]
        return value.chunk(n, dim=0)[r]
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].contiguous()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :]
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


class MergeAccumulator:
    """Accumulates merge groups (QKV, gate_up) and expert stacks.

    Decoupled from the loading loop so merge/stack can happen on GPU tensors
    after H2D, rather than on CPU mmap tensors (which would trigger page faults).
    """

    def __init__(self, is_moe: bool = False, num_experts: int = 0):
        self.is_moe = is_moe
        self.num_experts = num_experts
        self._merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
        self._expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}

    def process(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Feed a (name, tensor) pair. Returns a list of finalized (name, tensor) pairs.
        Returns [] if the tensor is buffered (waiting for merge/stack partners)."""

        # --- Step 1: merge groups (QKV / gate_up) ---
        merge_info = _get_merge_info(name)
        if merge_info is not None:
            merged_key, slot, all_slots = merge_info
            self._merge_buf.setdefault(merged_key, {})[slot] = tensor
            if not all(s in self._merge_buf[merged_key] for s in all_slots):
                return []
            parts = [self._merge_buf[merged_key][s] for s in all_slots]
            del self._merge_buf[merged_key]
            name, tensor = merged_key, torch.cat(parts, dim=0)

        # --- Step 2: expert stacking ---
        if self.is_moe:
            expert_info = _get_expert_stack_info(name)
            if expert_info is not None:
                packed_key, expert_idx = expert_info
                slots = self._expert_buf.setdefault(packed_key, {})
                slots[expert_idx] = tensor
                if len(slots) != self.num_experts:
                    return []
                experts = [slots[idx] for idx in range(self.num_experts)]
                del self._expert_buf[packed_key]
                return [(packed_key, torch.stack(experts, dim=0))]

        return [(name, tensor)]

    def assert_complete(self):
        """Call after all tensors are processed. Raises if any groups are incomplete."""
        assert not self._merge_buf, f"Incomplete merge groups: {list(self._merge_buf.keys())}"
        assert not self._expert_buf, f"Incomplete expert tensors: {list(self._expert_buf.keys())}"


def _load_sharded_by_file(
    model_path: str,
    tp_rank: int,
    tp_size: int,
    num_kv_heads: int,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Yield one batch of ``(name, cpu_view)`` per safetensors file.

    Tensors are loaded with ``device="cpu"`` and sharded via zero-copy views
    (``_shard_tensor_view``).  No merge or expert stacking is done here —
    that is the caller's responsibility (typically via ``MergeAccumulator``
    on GPU tensors after H2D).

    .. warning::

        Yielded tensors are **mmap-backed views**.  Each batch is yielded
        *inside* the ``safe_open`` context manager so the mmap stays alive,
        but the caller **must** fully consume a batch before advancing the
        generator to the next file.  Holding references across iterations
        leads to use-after-unmap.
    """
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files

    for file in tqdm(files, desc="Loading weights", disable=(tp_rank != 0)):
        batch: list[tuple[str, torch.Tensor]] = []
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                view = _shard_tensor_view(name, raw, tp_rank, tp_size, num_kv_heads)
                batch.append((name, view))
            yield batch  # IMPORTANT: yield INSIDE `with` block to keep mmap alive


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader with per-file batch H2D optimization.

    Pipeline: CPU zero-copy shard views → per-file flat GPU buffer → batch copy →
    GPU-side merge/stack → yield (name, tensor).

    Compared to the old per-tensor safetensors H2D path, this reduces cudaMalloc
    calls from ~N_tensors to ~N_files and avoids CPU page fault storms by keeping
    shard/merge on GPU.
    """
    from .config import ModelConfig

    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    tp_info = get_tp_info()
    accumulator = MergeAccumulator(is_moe=config.is_moe, num_experts=config.num_experts)
    is_gpu = device.type == "cuda"
    detailed_timing = logger.isEnabledFor(logging.DEBUG)

    t_total = time.perf_counter()
    t_alloc = 0.0
    t_h2d = 0.0
    t_merge = 0.0
    num_files = 0

    for cpu_batch in _load_sharded_by_file(
        model_path, tp_info.rank, tp_info.size, config.num_kv_heads
    ):
        if not cpu_batch:
            continue
        num_files += 1

        if is_gpu:
            # --- Per-file flat buffer allocation ---
            if detailed_timing:
                t0 = time.perf_counter()
            total_bytes = 0
            for _, t in cpu_batch:
                align = t.element_size()
                total_bytes = (total_bytes + align - 1) // align * align
                total_bytes += t.nelement() * t.element_size()
            flat_buf = torch.empty(total_bytes, dtype=torch.uint8, device=device)
            if detailed_timing:
                t_alloc += time.perf_counter() - t0

            # Slice flat_buf into per-tensor views and batch copy
            if detailed_timing:
                t0 = time.perf_counter()
            gpu_tensors = []
            offset = 0
            for name, cpu_view in cpu_batch:
                align = cpu_view.element_size()
                offset = (offset + align - 1) // align * align
                nbytes = cpu_view.nelement() * cpu_view.element_size()
                gpu_flat_view = flat_buf[offset : offset + nbytes]
                gpu_tensor = gpu_flat_view.view(cpu_view.dtype).reshape(cpu_view.shape)
                gpu_tensor.copy_(cpu_view)  # H2D copy
                gpu_tensors.append((name, gpu_tensor))
                offset += nbytes
            if detailed_timing:
                t_h2d += time.perf_counter() - t0

            # GPU-side merge/stack; clone passthrough tensors to cut flat_buf reference.
            # torch.cat/torch.stack (from MergeAccumulator) allocate new storage,
            # so only passthrough tensors remain views of flat_buf and need .clone().
            # We detect this by comparing untyped_storage().data_ptr(): all views
            # sliced from the same torch.empty() share a single underlying storage
            # object, so their storage base address is identical.
            if detailed_timing:
                t0 = time.perf_counter()
            for name, gpu_t in gpu_tensors:
                for final_name, final_t in accumulator.process(name, gpu_t):
                    if (
                        final_t.untyped_storage().data_ptr()
                        == flat_buf.untyped_storage().data_ptr()
                    ):
                        final_t = final_t.clone()
                    yield final_name, final_t
            if detailed_timing:
                t_merge += time.perf_counter() - t0

            del flat_buf  # free per-file buffer immediately
        else:
            # CPU path (for testing / non-GPU environments)
            for name, cpu_view in cpu_batch:
                tensor = cpu_view.to(device)
                for final_name, final_t in accumulator.process(name, tensor):
                    yield final_name, final_t

    accumulator.assert_complete()
    t_total = time.perf_counter() - t_total
    if is_gpu and tp_info.is_primary():
        logger.info(f"load_weight: {num_files} files, total={t_total:.2f}s")
        if detailed_timing:
            logger.debug(
                f"load_weight breakdown: alloc={t_alloc:.2f}s, "
                f"h2d={t_h2d:.2f}s, merge={t_merge:.2f}s, "
                f"mmap+shard={t_total - t_alloc - t_h2d - t_merge:.2f}s"
            )
