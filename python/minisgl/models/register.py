from .llama import LlamaForCausalLM
from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM
from .config import ModelConfig, RotaryConfig

model_map = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
}

def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in model_map:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    return model_map[model_architecture](model_config)

__all__ = [
    "get_model_class",
]