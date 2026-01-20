from __future__ import annotations

from .base import BaseLLMModel
from .config import ModelConfig
from .register import get_model_class
from .weight import load_hf_weight


def create_model(config: ModelConfig) -> BaseLLMModel:
    model_config = config.model_config
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_hf_weight"]
