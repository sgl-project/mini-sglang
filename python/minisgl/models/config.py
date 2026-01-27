from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from transformers import LlamaConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, float] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    sliding_window: int | None

    @classmethod
    # def from_hf(cls, config: LlamaConfig) -> ModelConfig:
    # above assumes all HF config fields exist. but Mistral HF
    # config is not a LLamaConfig and may lack num_attention_heads 
    # or hidden_size, so getattr(config, "num_attention_heads", 
    # config.num_attention_heasd) can silently fial or return None. 
    # thus head_dim * num_kv_heads crashes via error
    #TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'
    # So force fallback values and ensure no None propagates:
    @classmethod
    def from_hf(cls, config) -> ModelConfig:  # don't type hint LlamaConfig, use generic config 
        # fallback for attention heads
        num_qo_heads = getattr(config, "num_attention_heads", None)
        if num_qo_heads is None:
            raise ValueError("HF config missing num_attention_heads")
        
        # fallback for key/value heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_qo_heads)
        
        # head_dim fallback
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("HF config missing hidden_size")
            head_dim = hidden_size // num_qo_heads

        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        sliding_window = getattr(config, "sliding_window", None)

        return cls(
            num_layers=getattr(config, "num_hidden_layers", 1),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=getattr(config, "hidden_size", head_dim * num_qo_heads),
            vocab_size=getattr(config, "vocab_size", 32000),
            intermediate_size=getattr(config, "intermediate_size", head_dim * 4),
            hidden_act=getattr(config, "hidden_act", "silu"),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            tie_word_embeddings=tie_word_embeddings,
            sliding_window=sliding_window,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=getattr(config, "max_position_embeddings", 2048),
                base=getattr(config, "rope_theta", 10000.0),
                scaling=getattr(config, "rope_scaling", None),
            ),
        )


    # def from_hf(cls, config) -> ModelConfig: #Dont type hint LLamaConfig
    #     num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    #     head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    #     tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
    #     sliding_window = getattr(config, "sliding_window", None)
    #     return cls(
    #         num_layers=config.num_hidden_layers,
    #         num_qo_heads=config.num_attention_heads,
    #         num_kv_heads=num_kv_heads,
    #         head_dim=head_dim,
    #         hidden_size=config.hidden_size,
    #         vocab_size=config.vocab_size,
    #         intermediate_size=config.intermediate_size,
    #         hidden_act=config.hidden_act,
    #         rms_norm_eps=config.rms_norm_eps,
    #         tie_word_embeddings=tie_word_embeddings,
    #         sliding_window=sliding_window,
    #         rotary_config=RotaryConfig(
    #             head_dim=head_dim,
    #             rotary_dim=head_dim,
    #             max_position=config.max_position_embeddings,
    #             base=config.rope_theta,
    #             scaling=getattr(config, "rope_scaling", None),
    #         ),
    #     )