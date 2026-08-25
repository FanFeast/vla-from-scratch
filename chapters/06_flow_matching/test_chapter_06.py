"""Tests for Chapter 06 -- Action Expert Transformer.

Covers sinusoidal timestep embedding, transformer blocks, and the full
action expert model (shape checks, parameter counts, gradient flow,
zero-init output projection).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from the chapter directory when run standalone
sys.path.insert(0, str(Path(__file__).parent))

import torch
import pytest

from action_expert import (
    ActionExpertBlock,
    ActionExpertTransformer,
    SinusoidalTimestepEmbedding,
    build_action_expert,
)

# Small config used across tests to keep them fast
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
FFN_DIM = 128
COND_DIM = 32
ACTION_DIM = 7
CHUNK_SIZE = 8
STATE_DIM = 2
BATCH = 2


# ---- SinusoidalTimestepEmbedding ------------------------------------------

class TestSinusoidalTimestepEmbedding:
    """Tests for the sinusoidal timestep embedding module."""

    def test_output_shape(self) -> None:
        emb = SinusoidalTimestepEmbedding(D_MODEL)
        t = torch.rand(BATCH)
        out = emb(t)
        assert out.shape == (BATCH, D_MODEL)

    def test_different_timesteps_give_different_embeddings(self) -> None:
        emb = SinusoidalTimestepEmbedding(D_MODEL)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])
        out1 = emb(t1)
        out2 = emb(t2)
        assert not torch.allclose(out1, out2), (
            "Embeddings for t=0 and t=1 should differ"
        )


# ---- ActionExpertBlock ----------------------------------------------------

class TestActionExpertBlock:
    """Tests for a single transformer block."""

    def test_output_shape(self) -> None:
        block = ActionExpertBlock(D_MODEL, NHEAD, FFN_DIM)
        x = torch.randn(BATCH, CHUNK_SIZE, D_MODEL)
        cond = torch.randn(BATCH, 5, D_MODEL)
        out = block(x, cond)
        assert out.shape == x.shape

    def test_residual_changes_input(self) -> None:
        block = ActionExpertBlock(D_MODEL, NHEAD, FFN_DIM)
        x = torch.randn(BATCH, CHUNK_SIZE, D_MODEL)
        cond = torch.randn(BATCH, 5, D_MODEL)
        out = block(x, cond)
        assert not torch.allclose(out, x), (
            "Block output should differ from input"
        )


# ---- ActionExpertTransformer ----------------------------------------------

class TestActionExpertTransformer:
    """Tests for the full action expert model."""

    @pytest.fixture()
    def small_model(self) -> ActionExpertTransformer:
        return build_action_expert(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_SIZE,
            state_dim=STATE_DIM,
            cond_dim=COND_DIM,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            ffn_dim=FFN_DIM,
        )

    def test_cls_conditioning_shape(
        self, small_model: ActionExpertTransformer
    ) -> None:
        """Vision embedding as a single CLS vector (2-D)."""
        out = small_model(
            noisy_actions=torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM),
            t=torch.rand(BATCH),
            vision_emb=torch.randn(BATCH, COND_DIM),
            state=torch.randn(BATCH, STATE_DIM),
        )
        assert out.shape == (BATCH, CHUNK_SIZE, ACTION_DIM)

    def test_patch_conditioning_shape(
        self, small_model: ActionExpertTransformer
    ) -> None:
        """Vision embedding as patch tokens (3-D)."""
        num_patches = 196
        out = small_model(
            noisy_actions=torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM),
            t=torch.rand(BATCH),
            vision_emb=torch.randn(BATCH, num_patches, COND_DIM),
            state=torch.randn(BATCH, STATE_DIM),
        )
        assert out.shape == (BATCH, CHUNK_SIZE, ACTION_DIM)

    def test_full_model_parameter_count(self) -> None:
        """Default-sized model should be in the 80M-130M range."""
        model = build_action_expert(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_SIZE,
            state_dim=STATE_DIM,
        )
        n_params = model.count_parameters()
        assert 80_000_000 <= n_params <= 130_000_000, (
            f"Expected 80M-130M params, got {n_params / 1e6:.1f}M"
        )

    def test_gradient_flow(
        self, small_model: ActionExpertTransformer
    ) -> None:
        """Backward pass should produce gradients for every parameter."""
        out = small_model(
            noisy_actions=torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM),
            t=torch.rand(BATCH),
            vision_emb=torch.randn(BATCH, COND_DIM),
            state=torch.randn(BATCH, STATE_DIM),
        )
        loss = out.sum()
        loss.backward()
        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_zero_initialized_output_projection(
        self, small_model: ActionExpertTransformer
    ) -> None:
        """Output projection weight and bias should be all zeros."""
        assert torch.all(small_model.output_proj.weight == 0)
        assert torch.all(small_model.output_proj.bias == 0)


# ---- Flow Matching Script Tests ------------------------------------------

from flow_matching import (
    NormStats,
    compute_norm_stats,
    RobotDataset,
    EMAModel,
    FlowMatchingTrainer,
    DDPMTransformerTrainer,
)

# Small config for trainer tests
_SMALL_COND = 32
_SMALL_D = 64
_SMALL_HEADS = 4
_SMALL_LAYERS = 1
_SMALL_FFN = 128


def _small_expert(action_dim: int = 2, chunk_size: int = 4, state_dim: int = 2):
    return build_action_expert(
        action_dim, chunk_size, state_dim,
        d_model=_SMALL_D, nhead=_SMALL_HEADS,
        num_layers=_SMALL_LAYERS, ffn_dim=_SMALL_FFN,
        cond_dim=_SMALL_COND,
    )


class TestNormStats:
    """Tests for NormStats normalize/denormalize roundtrip."""

    def test_roundtrip(self) -> None:
        data = torch.randn(50, 4) * 100
        stats = compute_norm_stats(data)
        norm = stats.normalize(data)
        assert norm.min() >= -1.0 - 1e-6
        assert norm.max() <= 1.0 + 1e-6
        recovered = stats.denormalize(norm)
        assert torch.allclose(data, recovered, atol=1e-3)


class TestRobotDataset:
    """Tests for RobotDataset with action chunking."""

    def test_getitem_shape(self) -> None:
        ds = RobotDataset(
            torch.randn(20, 768), torch.randn(20, 2),
            torch.randn(20, 2), torch.zeros(20, dtype=torch.long), 4,
        )
        item = ds[0]
        assert item["cls_emb"].shape == (768,)
        assert item["state"].shape == (2,)
        assert item["actions"].shape == (4, 2)

    def test_chunk_padding(self) -> None:
        eps = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        ds = RobotDataset(
            torch.randn(6, 32), torch.randn(6, 2),
            torch.arange(12, dtype=torch.float32).view(6, 2),
            eps, 4,
        )
        item = ds[1]
        assert item["actions"].shape == (4, 2)
        # idx=1, ep=0: valid at offsets 0,1 (idx 1,2), pad at 2,3
        assert torch.equal(item["actions"][2], item["actions"][3])


class TestEMAModel:
    """Tests for EMAModel tracking."""

    def test_tracks_model(self) -> None:
        model = _small_expert()
        ema = EMAModel(model, decay=0.0)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)
        for s, m in zip(ema.shadow.parameters(), model.parameters()):
            assert torch.allclose(s, m)


class TestFlowMatchingTrainer:
    """Tests for FlowMatchingTrainer loss and sampling."""

    def test_loss_scalar(self) -> None:
        model = _small_expert()
        trainer = FlowMatchingTrainer(model, 3)
        batch = {
            "cls_emb": torch.randn(4, _SMALL_COND),
            "state": torch.randn(4, 2),
            "actions": torch.randn(4, 4, 2),
        }
        loss = trainer.compute_loss(batch, "cpu")
        assert loss.dim() == 0 and loss.item() > 0

    def test_sample_shape(self) -> None:
        model = _small_expert()
        trainer = FlowMatchingTrainer(model, 3)
        out = trainer.sample(torch.randn(2, _SMALL_COND), torch.randn(2, 2))
        assert out.shape == (2, 4, 2)

    def test_sample_in_range(self) -> None:
        model = _small_expert()
        trainer = FlowMatchingTrainer(model, 5)
        out = trainer.sample(torch.randn(2, _SMALL_COND), torch.randn(2, 2))
        assert out.min() >= -1.0 and out.max() <= 1.0


class TestDDPMTransformerTrainer:
    """Tests for DDPMTransformerTrainer loss and sampling."""

    def test_loss_scalar(self) -> None:
        model = _small_expert()
        trainer = DDPMTransformerTrainer(model, 100, 5)
        batch = {
            "cls_emb": torch.randn(4, _SMALL_COND),
            "state": torch.randn(4, 2),
            "actions": torch.randn(4, 4, 2),
        }
        loss = trainer.compute_loss(batch, "cpu")
        assert loss.dim() == 0 and loss.item() > 0

    def test_sample_shape(self) -> None:
        model = _small_expert()
        trainer = DDPMTransformerTrainer(model, 100, 5)
        out = trainer.sample(torch.randn(2, _SMALL_COND), torch.randn(2, 2))
        assert out.shape == (2, 4, 2)
