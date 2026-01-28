from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import LlamaConfig


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> LlamaConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())
