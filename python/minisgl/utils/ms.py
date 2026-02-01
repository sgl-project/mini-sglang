from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import LlamaConfig


@lru_cache()
def _load_ms_config(model_path: str) -> Any:
    from modelscope import snapshot_download
    from transformers import AutoConfig

    local_path = snapshot_download(model_path, allow_file_pattern=["config.json"])
    return AutoConfig.from_pretrained(local_path)


def cached_load_ms_config(model_path: str) -> LlamaConfig:
    config = _load_ms_config(model_path)
    return type(config)(**config.to_dict())
