# tests/models/test_weight.py
"""Unit tests for weight loading utilities."""
import torch
import pytest

# We'll test the internal functions directly
from minisgl.models.weight import _shard_tensor, _shard_tensor_view
from minisgl.models.weight import MergeAccumulator
from safetensors.torch import save_file
from minisgl.models.weight import _load_sharded_by_file
from unittest.mock import patch, MagicMock
from minisgl.models.weight import load_weight
import minisgl.distributed.info as di


class TestShardTensorView:
    """Tests for _shard_tensor_view: zero-copy CPU view sharding."""

    def test_dim0_split_returns_view_not_copy(self):
        """dim-0 shard should be a view (shares storage), not a clone."""
        t = torch.randn(8, 4)
        result = _shard_tensor_view(".q_proj.weight", t, r=0, n=2, num_kv_heads=None)
        assert result.shape == (4, 4)
        assert result.data_ptr() == t.data_ptr()  # same storage = view

    def test_dim0_split_values_match_shard_tensor(self):
        """dim-0 view should have the same values as the clone-based _shard_tensor."""
        t = torch.randn(8, 4)
        view = _shard_tensor_view(".q_proj.weight", t, r=1, n=2, num_kv_heads=None)
        clone = _shard_tensor(".q_proj.weight", t, r=1, n=2, num_kv_heads=None)
        torch.testing.assert_close(view, clone)

    def test_dim1_split_is_contiguous(self):
        """dim-1 shard must be contiguous (required for H2D copy)."""
        t = torch.randn(4, 8)
        result = _shard_tensor_view(".o_proj.weight", t, r=0, n=2, num_kv_heads=None)
        assert result.shape == (4, 4)
        assert result.is_contiguous()

    def test_dim1_split_values_match_shard_tensor(self):
        """dim-1 view should have the same values as the clone-based _shard_tensor."""
        t = torch.randn(4, 8)
        view = _shard_tensor_view(".o_proj.weight", t, r=1, n=2, num_kv_heads=None)
        clone = _shard_tensor(".o_proj.weight", t, r=1, n=2, num_kv_heads=None)
        torch.testing.assert_close(view, clone)

    def test_kv_proj_with_fewer_heads_than_tp(self):
        """KV proj with num_kv_heads < tp_size should replicate correctly."""
        t = torch.randn(128, 64)  # 2 heads * head_dim=64
        for r in range(4):
            view = _shard_tensor_view(".k_proj.weight", t, r=r, n=4, num_kv_heads=2)
            clone = _shard_tensor(".k_proj.weight", t, r=r, n=4, num_kv_heads=2)
            torch.testing.assert_close(view, clone)

    def test_embedding_shard(self):
        """Embedding/lm_head shard should return a view."""
        t = torch.randn(32000, 128)
        result = _shard_tensor_view("lm_head.weight", t, r=0, n=4, num_kv_heads=None)
        assert result.shape[0] == 8000
        assert result.shape[1] == 128

    def test_unsplit_tensor_returns_same_object(self):
        """Tensors that don't match any split pattern should be returned as-is."""
        t = torch.randn(4, 4)
        result = _shard_tensor_view("model.norm.weight", t, r=0, n=2, num_kv_heads=None)
        assert result is t


class TestMergeAccumulator:
    """Tests for MergeAccumulator: GPU-side merge and expert stacking."""

    def _make_acc(self, is_moe=False, num_experts=0):
        return MergeAccumulator(is_moe=is_moe, num_experts=num_experts)

    def test_passthrough_normal_tensor(self):
        """Non-merge, non-expert tensor should pass through immediately."""
        acc = self._make_acc()
        t = torch.randn(4, 4)
        results = acc.process("model.layers.0.input_layernorm.weight", t)
        assert len(results) == 1
        assert results[0] == ("model.layers.0.input_layernorm.weight", t)

    def test_qkv_merge(self):
        """q/k/v tensors should merge into a single qkv_proj via torch.cat."""
        acc = self._make_acc()
        q = torch.randn(8, 4)
        k = torch.randn(4, 4)
        v = torch.randn(4, 4)
        r1 = acc.process("model.layers.0.self_attn.q_proj.weight", q)
        assert r1 == []  # not complete yet
        r2 = acc.process("model.layers.0.self_attn.k_proj.weight", k)
        assert r2 == []
        r3 = acc.process("model.layers.0.self_attn.v_proj.weight", v)
        assert len(r3) == 1
        name, merged = r3[0]
        assert name == "model.layers.0.self_attn.qkv_proj.weight"
        torch.testing.assert_close(merged, torch.cat([q, k, v], dim=0))

    def test_gate_up_merge(self):
        """gate/up tensors should merge into gate_up_proj."""
        acc = self._make_acc()
        gate = torch.randn(8, 4)
        up = torch.randn(8, 4)
        r1 = acc.process("model.layers.0.mlp.gate_proj.weight", gate)
        assert r1 == []
        r2 = acc.process("model.layers.0.mlp.up_proj.weight", up)
        assert len(r2) == 1
        name, merged = r2[0]
        assert name == "model.layers.0.mlp.gate_up_proj.weight"
        torch.testing.assert_close(merged, torch.cat([gate, up], dim=0))

    def test_expert_stack(self):
        """Expert tensors should be stacked when all experts are collected."""
        acc = self._make_acc(is_moe=True, num_experts=2)
        gate0, up0, gate1, up1 = [torch.randn(4, 4) for _ in range(4)]
        r = acc.process("model.layers.0.experts.0.gate_proj.weight", gate0)
        assert r == []
        r = acc.process("model.layers.0.experts.0.up_proj.weight", up0)
        assert r == []
        r = acc.process("model.layers.0.experts.1.gate_proj.weight", gate1)
        assert r == []
        r = acc.process("model.layers.0.experts.1.up_proj.weight", up1)
        assert len(r) == 1
        name, stacked = r[0]
        assert name == "model.layers.0.experts.gate_up_proj"
        assert stacked.shape[0] == 2
        expected = torch.stack(
            [torch.cat([gate0, up0], dim=0), torch.cat([gate1, up1], dim=0)], dim=0
        )
        torch.testing.assert_close(stacked, expected)

    def test_assert_on_incomplete_merge(self):
        """assert_complete should raise if merge groups are incomplete."""
        acc = self._make_acc()
        acc.process("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4))
        with pytest.raises(AssertionError, match="Incomplete merge"):
            acc.assert_complete()

    def test_assert_on_complete(self):
        """assert_complete should pass when all groups are flushed."""
        acc = self._make_acc()
        acc.process("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4))
        acc.process("model.layers.0.self_attn.k_proj.weight", torch.randn(4, 4))
        acc.process("model.layers.0.self_attn.v_proj.weight", torch.randn(4, 4))
        acc.assert_complete()  # should not raise


class TestLoadShardedByFile:
    """Tests for _load_sharded_by_file: per-file CPU view batching."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create a minimal 2-file safetensors model with config."""
        # File 1: embed + layer 0 attention
        save_file(
            {
                "model.embed_tokens.weight": torch.randn(32, 16),
                "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
                "model.layers.0.self_attn.k_proj.weight": torch.randn(8, 16),
                "model.layers.0.self_attn.v_proj.weight": torch.randn(8, 16),
            },
            tmp_path / "model-00001-of-00002.safetensors",
        )
        # File 2: layer 0 MLP + norm + lm_head
        save_file(
            {
                "model.layers.0.mlp.gate_proj.weight": torch.randn(32, 16),
                "model.layers.0.mlp.up_proj.weight": torch.randn(32, 16),
                "model.layers.0.mlp.down_proj.weight": torch.randn(16, 32),
                "model.layers.0.input_layernorm.weight": torch.randn(16),
                "lm_head.weight": torch.randn(32, 16),
            },
            tmp_path / "model-00002-of-00002.safetensors",
        )
        return tmp_path

    def test_yields_per_file_batches(self, model_dir):
        """Should yield one batch per safetensors file."""
        batches = list(_load_sharded_by_file(str(model_dir), tp_rank=0, tp_size=1, num_kv_heads=8))
        assert len(batches) == 2

    def test_tensors_are_cpu(self, model_dir):
        """All yielded tensors should be on CPU."""
        for batch in _load_sharded_by_file(str(model_dir), tp_rank=0, tp_size=1, num_kv_heads=8):
            for name, tensor in batch:
                assert tensor.device == torch.device("cpu"), f"{name} is not on CPU"

    def test_dim0_views_are_sharded(self, model_dir):
        """With tp_size=2, dim-0 tensors should be half the original size."""
        for batch in _load_sharded_by_file(str(model_dir), tp_rank=0, tp_size=2, num_kv_heads=8):
            for name, tensor in batch:
                if "q_proj" in name:
                    assert tensor.shape[0] == 8  # 16 / 2
                elif "gate_proj" in name:
                    assert tensor.shape[0] == 16  # 32 / 2

    def test_skips_vision_tower_keys(self, model_dir):
        """Keys starting with vision_tower.* should be skipped."""
        save_file(
            {"vision_tower.encoder.weight": torch.randn(4, 4)},
            model_dir / "vision.safetensors",
        )
        all_names = []
        for batch in _load_sharded_by_file(str(model_dir), tp_rank=0, tp_size=1, num_kv_heads=8):
            all_names.extend(name for name, _ in batch)
        assert not any("vision_tower" in n for n in all_names)


class TestLoadWeightIntegration:
    """Integration tests: load_weight pipeline produces correct results."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create minimal model with QKV merge scenario."""
        save_file(
            {
                "model.embed_tokens.weight": torch.randn(32, 16),
                "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
                "model.layers.0.self_attn.k_proj.weight": torch.randn(8, 16),
                "model.layers.0.self_attn.v_proj.weight": torch.randn(8, 16),
                "model.layers.0.self_attn.o_proj.weight": torch.randn(16, 16),
                "model.layers.0.mlp.gate_proj.weight": torch.randn(32, 16),
                "model.layers.0.mlp.up_proj.weight": torch.randn(32, 16),
                "model.layers.0.mlp.down_proj.weight": torch.randn(16, 32),
                "model.layers.0.input_layernorm.weight": torch.randn(16),
                "model.layers.0.post_attention_layernorm.weight": torch.randn(16),
                "model.norm.weight": torch.randn(16),
                "lm_head.weight": torch.randn(32, 16),
            },
            tmp_path / "model.safetensors",
        )
        return tmp_path

    @pytest.fixture
    def mock_tp_info(self):
        """Mock TP info for rank 0, size 1."""
        old = di._TP_INFO
        di._TP_INFO = di.DistributedInfo(rank=0, size=1)
        yield di._TP_INFO
        di._TP_INFO = old

    @pytest.fixture
    def mock_config(self):
        """Mock ModelConfig."""
        cfg = MagicMock()
        cfg.num_kv_heads = 8
        cfg.is_moe = False
        cfg.num_experts = 0
        return cfg

    def test_yields_merged_qkv(self, model_dir, mock_tp_info, mock_config):
        """QKV projections should be merged into qkv_proj."""
        with (
            patch("minisgl.models.weight.download_hf_weight", return_value=str(model_dir)),
            patch("minisgl.models.weight.cached_load_hf_config"),
            patch("minisgl.models.config.ModelConfig.from_hf", return_value=mock_config),
        ):
            results = dict(load_weight(str(model_dir), torch.device("cpu")))
        assert "model.layers.0.self_attn.qkv_proj.weight" in results
        assert "model.layers.0.self_attn.q_proj.weight" not in results

    def test_yields_merged_gate_up(self, model_dir, mock_tp_info, mock_config):
        """gate/up projections should be merged into gate_up_proj."""
        with (
            patch("minisgl.models.weight.download_hf_weight", return_value=str(model_dir)),
            patch("minisgl.models.weight.cached_load_hf_config"),
            patch("minisgl.models.config.ModelConfig.from_hf", return_value=mock_config),
        ):
            results = dict(load_weight(str(model_dir), torch.device("cpu")))
        assert "model.layers.0.mlp.gate_up_proj.weight" in results
        assert "model.layers.0.mlp.gate_proj.weight" not in results

    def test_all_tensors_on_target_device(self, model_dir, mock_tp_info, mock_config):
        """All yielded tensors should be on the target device."""
        with (
            patch("minisgl.models.weight.download_hf_weight", return_value=str(model_dir)),
            patch("minisgl.models.weight.cached_load_hf_config"),
            patch("minisgl.models.config.ModelConfig.from_hf", return_value=mock_config),
        ):
            for name, tensor in load_weight(str(model_dir), torch.device("cpu")):
                assert tensor.device == torch.device("cpu"), f"{name} wrong device"

    def test_no_incomplete_groups(self, model_dir, mock_tp_info, mock_config):
        """All merge groups should be complete (no assertion errors)."""
        with (
            patch("minisgl.models.weight.download_hf_weight", return_value=str(model_dir)),
            patch("minisgl.models.weight.cached_load_hf_config"),
            patch("minisgl.models.config.ModelConfig.from_hf", return_value=mock_config),
        ):
            list(load_weight(str(model_dir), torch.device("cpu")))
