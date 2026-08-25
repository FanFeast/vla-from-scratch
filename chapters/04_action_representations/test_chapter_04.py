"""Tests for Chapter 4: Action Representations."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_pusht_continuous import ContinuousMiniPushT, ContinuousExpert


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

def test_env_obs_shape():
    env = ContinuousMiniPushT(size=64)
    obs, info = env.reset(seed=0)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8


def test_env_action_space_continuous():
    env = ContinuousMiniPushT(size=64)
    assert env.action_space.shape == (2,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_env_step_returns_correct_tuple():
    env = ContinuousMiniPushT(size=64)
    env.reset(seed=0)
    action = np.array([0.5, -0.3], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "agent_pos" in info


def test_env_clips_actions():
    env = ContinuousMiniPushT(size=64)
    env.reset(seed=0)
    # Out-of-range action should be clipped, not crash
    obs, _, _, _, _ = env.step(np.array([5.0, -5.0]))
    assert obs.shape == (64, 64, 3)


def test_expert_success_rate():
    env = ContinuousMiniPushT(size=64)
    expert = ContinuousExpert(size=64, noise_std=0.05)
    successes = 0
    for i in range(20):
        obs, info = env.reset(seed=i)
        done = False
        while not done:
            action = expert.act(info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if terminated:
            successes += 1
    assert successes >= 10, f"Expert only succeeded {successes}/20 times"


def test_expert_action_shape():
    env = ContinuousMiniPushT(size=64)
    expert = ContinuousExpert(size=64)
    _, info = env.reset(seed=0)
    action = expert.act(info)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

from action_heads import (
    VisionEncoder,
    DiscreteActionHead,
    RegressionActionHead,
    DiffusionActionHead,
    VLA,
    build_vla,
    continuous_to_bins,
    bins_to_continuous,
    N_BINS,
)


def test_bins_roundtrip():
    """Bin->continuous roundtrip should be approximately identity."""
    actions = torch.linspace(-1, 1, 100).unsqueeze(-1)
    bins = continuous_to_bins(actions)
    recovered = bins_to_continuous(bins)
    assert torch.allclose(actions, recovered, atol=0.01)


def test_bins_range():
    bins = continuous_to_bins(torch.tensor([-1.0, 0.0, 1.0]))
    assert bins[0] == 0
    assert bins[2] == N_BINS - 1


def test_discrete_head_loss_and_predict():
    head = DiscreteActionHead(input_dim=832, action_dim=2, chunk_size=1)
    features = torch.randn(4, 832)
    actions = torch.rand(4, 2) * 2 - 1
    loss = head.loss(features, actions)
    assert loss.shape == ()
    assert loss.item() > 0
    pred = head.predict(features)
    assert pred.shape == (4, 2)


def test_discrete_head_chunked():
    head = DiscreteActionHead(input_dim=832, action_dim=2, chunk_size=4)
    features = torch.randn(4, 832)
    actions = torch.rand(4, 4, 2) * 2 - 1
    loss = head.loss(features, actions)
    assert loss.shape == ()
    pred = head.predict(features)
    assert pred.shape == (4, 4, 2)


def test_regression_head_loss_and_predict():
    head = RegressionActionHead(input_dim=832, action_dim=2, chunk_size=1)
    features = torch.randn(4, 832)
    actions = torch.rand(4, 2) * 2 - 1
    loss = head.loss(features, actions)
    assert loss.shape == ()
    pred = head.predict(features)
    assert pred.shape == (4, 2)


def test_regression_head_chunked():
    head = RegressionActionHead(input_dim=832, action_dim=2, chunk_size=4)
    features = torch.randn(4, 832)
    actions = torch.rand(4, 4, 2) * 2 - 1
    loss = head.loss(features, actions)
    assert loss.shape == ()
    pred = head.predict(features)
    assert pred.shape == (4, 4, 2)


def test_diffusion_head_loss_and_predict():
    head = DiffusionActionHead(input_dim=832, action_dim=2, chunk_size=1)
    features = torch.randn(4, 832)
    actions = torch.rand(4, 2) * 2 - 1
    loss = head.loss(features, actions)
    assert loss.shape == ()
    pred = head.predict(features)
    assert pred.shape == (4, 2)
    assert pred.min() >= -1.0 and pred.max() <= 1.0


def test_diffusion_head_chunked():
    head = DiffusionActionHead(input_dim=832, action_dim=2, chunk_size=4)
    features = torch.randn(4, 832)
    actions = torch.rand(4, 4, 2) * 2 - 1
    loss = head.loss(features, actions)
    assert loss.shape == ()
    pred = head.predict(features)
    assert pred.shape == (4, 4, 2)


def test_vla_build_all_heads():
    """All three head types should build without error."""
    for head_type in ["discrete", "regression", "diffusion"]:
        model = build_vla(head_type, chunk_size=1)
        assert isinstance(model, VLA)


def test_vla_forward_with_embeddings():
    """VLA forward pass with precomputed embeddings (B, 768)."""
    for head_type in ["discrete", "regression", "diffusion"]:
        model = build_vla(head_type, chunk_size=1)
        model.eval()
        emb = torch.randn(2, 768)
        actions = torch.rand(2, 2) * 2 - 1
        loss = model(emb, actions=actions)
        assert loss.shape == ()
        with torch.no_grad():
            pred = model(emb)
        assert pred.shape == (2, 2)
