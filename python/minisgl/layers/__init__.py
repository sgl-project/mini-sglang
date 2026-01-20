from __future__ import annotations

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .weight import load_hf_weight

from .register import get_model_class


def create_model(config: ModelConfig) -> BaseLLMModel:
    model_path = config.model_path
    model_config = config.model_config
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["BaseLLMModel", "load_hf_weight", "create_model", "ModelConfig", "RotaryConfig"]
