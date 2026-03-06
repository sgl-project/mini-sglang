from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Set, Tuple

import safetensors
import torch
from minisgl.distributed import get_ep_info, get_tp_info
from minisgl.utils import div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

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

_EXPERT_KEY_RE = re.compile(r"experts\.(\d+)\.")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, ep_size: int) -> torch.Tensor:
    if _EXPERT_KEY_RE.search(key) and ep_size > 1:
        return value
    elif any(key.count(sub) for sub in _SPLIT_DIM_0):
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


def _get_merge_info(key: str) -> Tuple[str, str, Tuple[str, ...]] | None:
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _build_ep_skip_set(files: list[str], ep_size: int) -> Tuple[Set[str], int]:
    ep_info = get_ep_info()
    max_idx = -1
    expert_keys: list[tuple[str, int]] = []
    for file in files:
        with safetensors.safe_open(file, framework="pt") as f:
            for name in f.keys():
                m = _EXPERT_KEY_RE.search(name)
                if m:
                    idx = int(m.group(1))
                    max_idx = max(max_idx, idx)
                    expert_keys.append((name, idx))
    if max_idx < 0:
        return set(), 0

    num_local = (max_idx + 1) // ep_size
    local_start = ep_info.rank * num_local
    skip = {name for name, idx in expert_keys if not (local_start <= idx < local_start + num_local)}
    return skip, local_start


def load_weight(
    model_path: str, device: torch.device, ep_size: int = 1
) -> Iterator[Tuple[str, torch.Tensor]]:
    model_folder = download_hf_weight(model_path)
    files = sorted(glob.glob(f"{model_folder}/*.safetensors"))

    tp_info = get_tp_info()
    r, n = tp_info.rank, tp_info.size
    tp = n > 1
    disable_tqdm = (r != 0) if tp else False
    device_str = str(device)

    skip_keys: Set[str] = set()
    ep_local_start = 0
    if ep_size > 1:
        skip_keys, ep_local_start = _build_ep_skip_set(files, ep_size)

    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}

    for file in tqdm(files, desc="Loading weights", disable=disable_tqdm):
        load_device = "cpu" if tp else device_str
        with safetensors.safe_open(file, framework="pt", device=load_device) as f:
            for name in f.keys():
                if name in skip_keys:
                    continue

                raw = f.get_tensor(name)
                tensor = _shard_tensor(name, raw, r, n, ep_size).to(device) if tp else raw
                del raw

                out_name = name
                if ep_size > 1:
                    m = _EXPERT_KEY_RE.search(name)
                    if m:
                        local_idx = int(m.group(1)) - ep_local_start
                        out_name = name[: m.start(1)] + str(local_idx) + name[m.end(1) :]

                info = _get_merge_info(out_name)
                if info is None:
                    yield out_name, tensor
                    continue

                merged_key, slot, all_slots = info
                merge_buf.setdefault(merged_key, {})[slot] = tensor
                if all(s in merge_buf[merged_key] for s in all_slots):
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    yield merged_key, torch.cat(parts, dim=0)

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
