"""Tests for Chapter 5: Scaling Up."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import (
    ProprioEncoder,
    DiscreteActionHead,
    RegressionActionHead,
    DiffusionActionHead,
    VLA,
    build_vla,
    RobotDataset,
    NormStats,
    compute_norm_stats,
    split_by_episode,
    FUSED_DIM,
)


class TestProprioEncoder:
    def test_output_shape_2d(self):
        enc = ProprioEncoder(state_dim=2)
        out = enc(torch.randn(4, 2))
        assert out.shape == (4, 64)

    def test_output_shape_14d(self):
        enc = ProprioEncoder(state_dim=14)
        out = enc(torch.randn(4, 14))
        assert out.shape == (4, 64)


class TestDiscreteHead:
    def test_loss_and_predict(self):
        head = DiscreteActionHead(FUSED_DIM, action_dim=2, chunk_size=4)
        x = torch.randn(4, FUSED_DIM)
        target = torch.rand(4, 4, 2) * 2 - 1
        loss = head.compute_loss(x, target)
        assert loss.ndim == 0 and loss.item() > 0
        pred = head.predict(x)
        assert pred.shape == (4, 4, 2)
        assert pred.min() >= -1 and pred.max() <= 1

    def test_14d_actions(self):
        head = DiscreteActionHead(FUSED_DIM, action_dim=14, chunk_size=1)
        pred = head.predict(torch.randn(2, FUSED_DIM))
        assert pred.shape == (2, 1, 14)


class TestRegressionHead:
    def test_loss_and_predict(self):
        head = RegressionActionHead(FUSED_DIM, action_dim=14, chunk_size=4)
        x = torch.randn(4, FUSED_DIM)
        target = torch.rand(4, 4, 14) * 2 - 1
        loss = head.compute_loss(x, target)
        assert loss.ndim == 0 and loss.item() > 0
        pred = head.predict(x)
        assert pred.shape == (4, 4, 14)
        assert pred.min() >= -1 and pred.max() <= 1


class TestDiffusionHead:
    def test_loss_and_predict(self):
        head = DiffusionActionHead(FUSED_DIM, action_dim=2, chunk_size=4)
        x = torch.randn(4, FUSED_DIM)
        target = torch.rand(4, 4, 2) * 2 - 1
        loss = head.compute_loss(x, target)
        assert loss.ndim == 0 and loss.item() > 0
        pred = head.predict(x)
        assert pred.shape == (4, 4, 2)

    def test_14d_actions(self):
        head = DiffusionActionHead(FUSED_DIM, action_dim=14, chunk_size=1)
        pred = head.predict(torch.randn(2, FUSED_DIM))
        assert pred.shape == (2, 1, 14)


class TestVLA:
    def test_pusht_regression(self):
        vla = build_vla("regression", state_dim=2, action_dim=2, chunk_size=4)
        pred = vla.predict(torch.randn(2, 768), torch.randn(2, 2))
        assert pred.shape == (2, 4, 2)

    def test_aloha_discrete(self):
        vla = build_vla("discrete", state_dim=14, action_dim=14, chunk_size=1)
        pred = vla.predict(torch.randn(2, 768), torch.randn(2, 14))
        assert pred.shape == (2, 1, 14)

    def test_loss_computation(self):
        vla = build_vla("diffusion", state_dim=2, action_dim=2, chunk_size=1)
        loss = vla(torch.randn(4, 768), torch.randn(4, 2), torch.rand(4, 1, 2) * 2 - 1)
        assert loss.ndim == 0 and loss.item() > 0

    def test_all_head_types(self):
        for ht in ["discrete", "regression", "diffusion"]:
            vla = build_vla(ht, state_dim=2, action_dim=2, chunk_size=4)
            pred = vla.predict(torch.randn(1, 768), torch.randn(1, 2))
            assert pred.shape == (1, 4, 2), f"Failed for {ht}"


class TestNormalization:
    def test_roundtrip(self):
        data = torch.tensor([[0.0, 100.0], [50.0, 500.0], [25.0, 300.0]])
        stats = compute_norm_stats(data)
        normed = stats.normalize(data)
        assert normed.min() >= -1.01 and normed.max() <= 1.01
        recovered = stats.denormalize(normed)
        assert torch.allclose(recovered, data, atol=1e-5)

    def test_single_value(self):
        data = torch.tensor([[5.0, 5.0], [5.0, 5.0]])
        stats = compute_norm_stats(data)
        normed = stats.normalize(data)
        # All same value -> normalizes to -1 (0 / epsilon - 1)
        assert normed.isfinite().all()


class TestSplitByEpisode:
    def test_no_overlap(self):
        eps = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
        train_m, val_m = split_by_episode(eps, val_ratio=0.25)
        assert (train_m & val_m).sum() == 0
        assert (train_m | val_m).all()
        assert val_m.sum() > 0

    def test_val_is_last_episodes(self):
        eps = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        _, val_m = split_by_episode(eps, val_ratio=0.2)
        # Last episode should be in val
        assert val_m[-1].item() is True
        assert val_m[-2].item() is True


class TestRobotDataset:
    def test_shape(self):
        ds = RobotDataset(
            embeddings=torch.randn(100, 768),
            states=torch.randn(100, 2),
            actions=torch.randn(100, 2),
            episode_indices=torch.zeros(100, dtype=torch.long),
            chunk_size=4,
        )
        emb, state, chunk = ds[50]
        assert emb.shape == (768,)
        assert state.shape == (2,)
        assert chunk.shape == (4, 2)

    def test_episode_boundary_padding(self):
        eps = torch.tensor([0, 0, 0, 1, 1, 1])
        actions = torch.arange(12).float().view(6, 2)
        ds = RobotDataset(
            embeddings=torch.randn(6, 768),
            states=torch.randn(6, 2),
            actions=actions,
            episode_indices=eps,
            chunk_size=4,
        )
        _, _, chunk = ds[2]  # Last frame of ep 0, chunk_size=4
        assert chunk.shape == (4, 2)
        # chunk[0] is action[2], chunk[1-3] should be padded with action[2]
        assert torch.equal(chunk[1], chunk[0])

    def test_cross_episode_no_leak(self):
        eps = torch.tensor([0, 0, 1, 1])
        actions = torch.tensor([[1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        ds = RobotDataset(
            embeddings=torch.randn(4, 768),
            states=torch.randn(4, 2),
            actions=actions,
            episode_indices=eps,
            chunk_size=4,
        )
        _, _, chunk = ds[1]  # Last of ep 0
        # Should NOT contain actions from ep 1
        assert chunk[0, 0].item() == 2.0  # action[1]
        assert chunk[1, 0].item() == 2.0  # padded with action[1]
