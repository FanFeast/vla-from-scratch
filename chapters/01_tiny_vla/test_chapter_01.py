"""Sanity-check tests for Chapter 1: Tiny VLA."""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mini_pusht import MiniPushT
from tiny_vla import train, evaluate
from tiny_vla import ScriptedExpert, VisionEncoder, LanguageEncoder, TinyVLA
from tiny_vla import collect_demos, PushTDataset


def test_env_reset_obs_shape():
    """Reset returns a 64x64x3 uint8 image."""
    env = MiniPushT()
    obs, info = env.reset(seed=42)
    assert obs.shape == (64, 64, 3), f"expected (64,64,3), got {obs.shape}"
    assert obs.dtype == np.uint8, f"expected uint8, got {obs.dtype}"


def test_env_reset_info_keys():
    """Reset info dict has agent_pos, block_pos, goal_pos, instruction."""
    env = MiniPushT()
    obs, info = env.reset(seed=42)
    for key in ("agent_pos", "block_pos", "goal_pos", "instruction"):
        assert key in info, f"missing key: {key}"


def test_env_step_output_shapes():
    """Step returns correct types and shapes."""
    env = MiniPushT()
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_truncates_after_max_steps():
    """Episode truncates after 200 steps."""
    env = MiniPushT()
    env.reset(seed=42)
    for _ in range(199):
        _, _, terminated, truncated, _ = env.step(4)  # stay
        if terminated:
            return  # ok if it terminates early
    _, _, _, truncated, _ = env.step(4)
    assert truncated


def test_env_instruction_passthrough():
    """Instruction passed to reset appears in info."""
    env = MiniPushT()
    _, info = env.reset(options={"instruction": "move left"})
    assert info["instruction"] == "move left"


def test_expert_returns_valid_action():
    """Expert always returns an action in [0, 4]."""
    env = MiniPushT()
    expert = ScriptedExpert()
    _, info = env.reset(seed=0)
    action = expert.act(info)
    assert 0 <= action <= 4, f"invalid action: {action}"


def test_expert_solves_env():
    """Expert reaches goal within 200 steps (at least 80% of episodes)."""
    env = MiniPushT()
    expert = ScriptedExpert()
    successes = 0
    n = 20
    for seed in range(n):
        obs, info = env.reset(seed=seed)
        done = False
        steps = 0
        while not done and steps < 200:
            action = expert.act(info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if terminated:
                successes += 1
    assert successes / n >= 0.8, f"expert success rate too low: {successes}/{n}"


def test_vision_encoder_output_shape():
    """VisionEncoder maps (B, 3, 64, 64) to (B, 128)."""
    enc = VisionEncoder()
    x = torch.zeros(4, 3, 64, 64)
    out = enc(x)
    assert out.shape == (4, 128), f"expected (4,128), got {out.shape}"


def test_language_encoder_single_string():
    """LanguageEncoder handles a single instruction string."""
    enc = LanguageEncoder()
    out = enc("push block to goal")
    assert out.shape == (1, 128), f"expected (1,128), got {out.shape}"


def test_language_encoder_batch():
    """LanguageEncoder handles a list of instructions."""
    enc = LanguageEncoder()
    out = enc(["push block to goal", "move left", "move right"])
    assert out.shape == (3, 128), f"expected (3,128), got {out.shape}"


def test_tiny_vla_forward():
    """TinyVLA forward pass produces (B, 5) logits."""
    model = TinyVLA(n_actions=5)
    images = torch.zeros(3, 3, 64, 64)
    logits = model(images, ["push block to goal", "move left", "move right"])
    assert logits.shape == (3, 5), f"expected (3,5), got {logits.shape}"


def test_collect_demos_structure():
    """collect_demos returns list of dicts with correct keys and shapes."""
    env = MiniPushT()
    expert = ScriptedExpert()
    demos = collect_demos(env, expert, n_episodes=3)
    assert len(demos) > 0, "no demos collected"
    for demo in demos:
        assert "image" in demo and "instruction" in demo and "action" in demo
        assert demo["image"].shape == (64, 64, 3)
        assert demo["image"].dtype == np.uint8
        assert isinstance(demo["instruction"], str)
        assert 0 <= demo["action"] <= 4


def test_pusht_dataset_getitem():
    """PushTDataset returns (image_tensor, instruction_str, action_int)."""
    env = MiniPushT()
    expert = ScriptedExpert()
    demos = collect_demos(env, expert, n_episodes=2)
    dataset = PushTDataset(demos)
    assert len(dataset) > 0
    image, instruction, action = dataset[0]
    assert image.shape == (3, 64, 64), f"expected (3,64,64), got {image.shape}"
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert isinstance(instruction, str)
    assert isinstance(action, int)


def test_training_step_runs():
    """One training run completes and returns a loss history."""
    env = MiniPushT()
    expert = ScriptedExpert()
    demos = collect_demos(env, expert, n_episodes=10)
    dataset = PushTDataset(demos)
    model = TinyVLA()
    losses = train(model, dataset, epochs=2, batch_size=16)
    assert len(losses) == 2
    assert all(loss > 0 for loss in losses)


def test_evaluate_returns_rate():
    """evaluate returns a float in [0, 1]."""
    env = MiniPushT()
    model = TinyVLA()
    rate = evaluate(model, env, n_episodes=5)
    assert 0.0 <= rate <= 1.0, f"unexpected rate: {rate}"
