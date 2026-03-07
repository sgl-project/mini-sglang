from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import nvtx_annotate

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    OPList,
    ParallelLMHead,
    LinearReplicated,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
    gelu_and_mul,
    silu_and_mul,
)

from .base import BaseLLMModel
from .utils import MoEMLP as Qwen3MLP
from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as Llama4Attn

if TYPE_CHECKING:
    from .config import ModelConfig

class Llama4MoE(BaseOP):

    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        intermediate_size_moe = config.intermediate_size
        self.router = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

        def custom_routing_function(
            topk_weights: torch.Tensor,
        ) -> torch.Tensor:
            return torch.sigmoid(topk_weights)

        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            renormalize=False,
            apply_router_weight_on_input=True,
            topk_no_softmax=True,
            custom_routing_function=custom_routing_function,
        )

        self.shared_expert = LlamaMLP(config=config)

    def forward(
        self,
        hidden_states,
    ):
        shared_out, routed_out = self._forward_core(
            hidden_states
        )

        out_aD = routed_out + shared_out

        return out_aD

    def _forward_core(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.dim() == 2, "Expected hidden_states to be of shape [num_tokens, hidden_dim]"
        # router_scores: [num_tokens, num_experts]
        router_logits = self.router(hidden_states)
        shared_out = self.shared_expert(hidden_states)
        routed_out = self.experts(hidden_states, router_logits)
        return shared_out, routed_out

class Llama4DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.config = config
        self.is_moe_layer = self._is_moe_layer(layer_id)
        self.hidden_size = config.hidden_size
        self.use_rope = (layer_id + 1) % 4 != 0
        self.use_qk_norm = config.use_qk_norm and self.use_rope

        self.self_attn = Llama4Attn(
            config, 
            layer_id, 
            has_qk_norm=self.use_qk_norm,
            has_qk_norm_weight=False,
            has_rope=self.use_rope,
            attn_temperature_tuning=config.attn_temperature_tuning,
            floor_scale=config.floor_scale,
            attn_scale=config.attn_scale,
        )
        if self.is_moe_layer:
            self.feed_forward = Llama4MoE(config)
        else:
            self.feed_forward = LlamaMLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    def _is_moe_layer(self, layer_id: int) -> bool:
        if self.config.interleave_moe_layer_step == 0:
            return self.config.num_experts > 0
        return (layer_id + 1) % self.config.interleave_moe_layer_step == 0

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.feed_forward.forward(x)
        return x, residual
    
class Llama4Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Llama4DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]
    
class Llama4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Llama4Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Llama4ForCausalLM"]