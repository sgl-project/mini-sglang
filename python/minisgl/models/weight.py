from __future__ import annotations

import glob
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from tqdm import tqdm
from minisgl.distributed import get_tp_info
from minisgl.utils import div_ceil, download_hf_weight

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
    ".q_proj": "q", ".k_proj": "k", ".v_proj": "v",
    ".gate_proj": "gate", ".up_proj": "up",
}


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int) -> torch.Tensor:
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
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


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    model_folder = download_hf_weight(model_path)
    files = sorted(glob.glob(f"{model_folder}/*.safetensors"))

    tp_info = get_tp_info()
    r, n = tp_info.rank, tp_info.size
    tp = n > 1
    disable_tqdm = (r != 0) if tp else False
    device_str = str(device)

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}

    for file in tqdm(files, desc="Loading weights", disable=disable_tqdm):
        load_device = "cpu" if tp else device_str
        with safetensors.safe_open(file, framework="pt", device=load_device) as f:
            for name in f.keys():
                raw = f.get_tensor(name)
                tensor = _shard_tensor(name, raw, r, n).to(device) if tp else raw
                del raw

                info = _get_merge_info(name)
                if info is None:
                    yield name, tensor
                    continue

                merged_key, slot, all_slots = info
                merge_buf.setdefault(merged_key, {})[slot] = tensor
                if all(s in merge_buf[merged_key] for s in all_slots):
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    yield merged_key, torch.cat(parts, dim=0)

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"