from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


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
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]

    # ============================== Gemma3 ==============================
    layer_types: list[str] = field(default_factory=list)
    partial_rotary_factor: float = 1.0
    global_rope_theta: float | None = None
    local_rope_theta: float | None = None
    query_pre_attn_scalar: float | None = None
    attention_bias: bool = False
    sliding_window: int | None = None  # raw HF value (inclusive)
    # ============================== Gemma3 ==============================

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr;
        # Mistral: inside rope_scaling dict;
        # Gemma3: neither — falls back to 10000.0 (Gemma3 model uses global/local_rope_theta instead)
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = (rope_scaling or {}).get("rope_theta", 10000.0)

        # Gemma3 uses hidden_activation;
        # All other models use hidden_act
        hidden_act = getattr(config, "hidden_act", None) or getattr(
            config, "hidden_activation", "silu"
        )

        # ============================== Gemma3 ==============================
        layer_types = list(getattr(config, "layer_types", None) or [])
        partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
        qpas = getattr(config, "query_pre_attn_scalar", None)
        query_pre_attn_scalar = float(qpas) if qpas is not None else None
        attention_bias = bool(getattr(config, "attention_bias", False))
        # HF uses either sliding_window_size or sliding_window;
        sliding_window = getattr(config, "sliding_window_size", None) or getattr(
            config, "sliding_window", None
        )
        # Dual RoPE for Gemma3: nested v5 or flat v4 format
        rope_params = getattr(config, "rope_parameters", None) or {}
        if isinstance(rope_params, dict) and "full_attention" in rope_params:
            global_rope_theta = float(rope_params["full_attention"].get("rope_theta", 1000000.0))
            local_rope_theta = float(rope_params["sliding_attention"].get("rope_theta", 10000.0))
        elif rope_params or getattr(config, "rope_local_base_freq", None):
            global_rope_theta = float(
                rope_params.get("rope_theta", rope_theta) if rope_params else rope_theta
            )
            local_rope_theta = float(getattr(config, "rope_local_base_freq", 10000.0))
        else:
            global_rope_theta = None
            local_rope_theta = None
        # ============================== Gemma3 ==============================

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            layer_types=layer_types,
            partial_rotary_factor=partial_rotary_factor,
            global_rope_theta=global_rope_theta,
            local_rope_theta=local_rope_theta,
            query_pre_attn_scalar=query_pre_attn_scalar,
            attention_bias=attention_bias,
            sliding_window=sliding_window,
        )
