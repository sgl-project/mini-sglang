from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Dict, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed.info import get_tp_info
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.utils import nvtx_annotate
from minisgl.utils.misc import div_even

from .base import BaseLLMModel
from .utils import GatedMLP as Llama4MLP

if TYPE_CHECKING:
    from .config import ModelConfig


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class Llama4Attn(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
        has_qk_norm_weight: bool = True,
        has_rope: bool = True,
        attn_temperature_tuning: bool = False,
        floor_scale=8192,
        attn_scale=0.1,
    ):
        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.qk_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, has_weight=has_qk_norm_weight)
        else:
            self.qk_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            has_rope=has_rope,
            attn_temperature_tuning=attn_temperature_tuning,
            floor_scale=floor_scale,
            attn_scale=attn_scale,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )
        self.layer_id = layer_id

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)

        del x
        o = self.attn.forward(qkv, rope_first=True, qk_norm_combined=True, qk_norm=self.qk_norm)

        return self.o_proj.forward(o)


class Llama4MoE(BaseOP):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
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

        def experts_load_state_dict(
            self,
            state_dict: Dict[str, torch.Tensor],
            *,
            prefix: str = "",
            _internal: bool = False,
        ) -> None:
            for name, param in self.__dict__.items():
                if name.startswith("_"):
                    continue

                if isinstance(param, torch.Tensor):
                    item = state_dict.pop(_concat_prefix(prefix, name))
                    assert isinstance(item, torch.Tensor)

                    if hasattr(param, "post_process"):
                        item = param.post_process(item)

                    assert param.shape == item.shape and param.dtype == item.dtype
                    setattr(self, name, item)
                elif isinstance(param, BaseOP):
                    param.load_state_dict(
                        state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                    )

            if not _internal and state_dict:
                raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

        self.experts.load_state_dict = MethodType(experts_load_state_dict, self.experts)

        tp_info = get_tp_info()
        intermediate_size_per_partition = div_even(intermediate_size_moe, tp_info.size)

        def gate_up_proj_post_process(weight: torch.Tensor) -> torch.Tensor:
            assert weight.shape == (
                self.experts.num_experts,
                self.experts.hidden_size,
                2 * intermediate_size_per_partition,
            )
            return weight.transpose(1, 2).contiguous()

        def down_proj_post_process(weight: torch.Tensor) -> torch.Tensor:
            assert weight.shape == (
                self.experts.num_experts,
                intermediate_size_per_partition,
                self.experts.hidden_size,
            )
            return weight.transpose(1, 2).contiguous()

        setattr(self.experts.gate_up_proj, "post_process", gate_up_proj_post_process)
        setattr(self.experts.down_proj, "post_process", down_proj_post_process)

        self.shared_expert = Llama4MLP(config=config)

    def forward(
        self,
        hidden_states,
    ):
        shared_out, routed_out = self._forward_core(hidden_states)

        out_aD = routed_out + shared_out

        return out_aD

    def _forward_core(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            hidden_states.dim() == 2
        ), "Expected hidden_states to be of shape [num_tokens, hidden_dim]"
        # router_scores: [num_tokens, num_experts]
        router_logits = self.router.forward(hidden_states)
        shared_out = self.shared_expert.forward(hidden_states)
        routed_out = self.experts.forward(hidden_states, router_logits)
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
            self.feed_forward = Llama4MLP(config, intermediate_size=config.intermediate_size_mlp)
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
