"""Sanity tests for Chapter 7: Full VLA.

Tests the action expert independently (no pretrained model download needed)
and optionally tests the full SmolVLA model (requires network + GPU).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

# Ensure chapter directory is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import SmolVLAConfig
from model import (
    pixel_shuffle,
    SinusoidalTimestepEmbedding,
    ExpertSelfAttnBlock,
    ExpertCrossAttnBlock,
    ActionExpert,
)


def test_pixel_shuffle() -> None:
    """Pixel shuffle reduces 1024 tokens to 64 with dim 768 -> 12288."""
    B = 2
    x = torch.randn(B, 1024, 768)
    out = pixel_shuffle(x, scale_factor=4)
    assert out.shape == (B, 64, 12288), f"Expected (2, 64, 12288), got {out.shape}"
    print("[PASS] test_pixel_shuffle")


def test_timestep_embedding() -> None:
    """Timestep embedding maps (B,) -> (B, d_model)."""
    emb = SinusoidalTimestepEmbedding(720)
    t = torch.rand(4)
    out = emb(t)
    assert out.shape == (4, 720), f"Expected (4, 720), got {out.shape}"
    print("[PASS] test_timestep_embedding")


def test_expert_sa_block() -> None:
    """Self-attention block preserves shape."""
    block = ExpertSelfAttnBlock(720, nhead=12, ffn_dim=2880)
    x = torch.randn(2, 10, 720)
    out = block(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("[PASS] test_expert_sa_block")


def test_expert_ca_block() -> None:
    """Cross-attention block handles dim mismatch (720 query, 960 kv)."""
    block = ExpertCrossAttnBlock(720, kv_dim=960, nhead=12, ffn_dim=2880)
    x = torch.randn(2, 10, 720)
    kv = torch.randn(2, 75, 960)  # VLM prefix tokens
    out = block(x, kv)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("[PASS] test_expert_ca_block")


def test_action_expert() -> None:
    """Action expert forward pass with mock VLM features."""
    config = SmolVLAConfig()
    expert = ActionExpert(config)

    B, K = 2, config.chunk_size
    noisy_actions = torch.randn(B, K, config.action_dim)
    timesteps = torch.rand(B)
    # 3 CA layers need 3 VLM feature tensors
    vlm_features = [
        torch.randn(B, 75, config.vlm_hidden_dim) for _ in range(3)
    ]

    out = expert(noisy_actions, timesteps, vlm_features)
    assert out.shape == (B, K, config.action_dim), (
        f"Expected ({B}, {K}, {config.action_dim}), got {out.shape}"
    )

    # Check output starts near zero (zero-initialized output projection)
    assert out.abs().max() < 1.0, (
        f"Output too large for zero-init: max={out.abs().max():.3f}"
    )
    print("[PASS] test_action_expert")


def test_action_expert_param_count() -> None:
    """Verify action expert has ~20M trainable parameters."""
    config = SmolVLAConfig()
    expert = ActionExpert(config)
    n_params = sum(p.numel() for p in expert.parameters())
    print(f"  Action expert params: {n_params / 1e6:.1f}M")
    assert 10e6 < n_params < 40e6, (
        f"Expected 10-40M params, got {n_params / 1e6:.1f}M"
    )
    print("[PASS] test_action_expert_param_count")


def test_action_expert_gradient_flow() -> None:
    """Verify gradients flow through the action expert."""
    config = SmolVLAConfig()
    expert = ActionExpert(config)

    B, K = 2, config.chunk_size
    noisy_actions = torch.randn(B, K, config.action_dim)
    timesteps = torch.rand(B)
    vlm_features = [
        torch.randn(B, 75, config.vlm_hidden_dim) for _ in range(3)
    ]

    out = expert(noisy_actions, timesteps, vlm_features)
    loss = out.sum()
    loss.backward()

    # Check all parameters have gradients
    for name, p in expert.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

    print("[PASS] test_action_expert_gradient_flow")


def test_flow_matching_loss() -> None:
    """Verify flow matching loss computation (mock model)."""
    import torch.nn.functional as F

    config = SmolVLAConfig()
    expert = ActionExpert(config)

    B, K = 4, config.chunk_size
    x_1 = torch.randn(B, K, config.action_dim)
    x_0 = torch.randn_like(x_1)
    t = torch.rand(B) * 0.999 + 0.001

    t_expand = t[:, None, None]
    x_t = t_expand * x_0 + (1.0 - t_expand) * x_1
    v_target = x_0 - x_1

    vlm_features = [
        torch.randn(B, 75, config.vlm_hidden_dim) for _ in range(3)
    ]
    v_pred = expert(x_t, t, vlm_features)
    loss = F.mse_loss(v_pred, v_target)

    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"
    loss.backward()
    print(f"  Flow matching loss: {loss.item():.4f}")
    print("[PASS] test_flow_matching_loss")


def test_rerun_viz_normalizes_cached_raw_data() -> None:
    """Rerun viz should normalize raw cache tensors and preserve denormalization."""
    from rerun_viz import prepare_normalized_cache

    raw_states = torch.tensor(
        [
            [1.0, 10.0, -2.0],
            [3.0, 14.0, 0.0],
            [5.0, 18.0, 2.0],
        ]
    )
    raw_actions = torch.tensor(
        [
            [0.5, -1.0, 2.0],
            [1.5, 1.0, 4.0],
            [2.5, 3.0, 6.0],
        ]
    )
    cache = {"states": raw_states.clone(), "actions": raw_actions.clone()}

    normalized_cache, state_stats, action_stats = prepare_normalized_cache(cache)

    assert normalized_cache is not cache
    assert torch.allclose(cache["states"], raw_states)
    assert torch.allclose(cache["actions"], raw_actions)
    assert torch.allclose(
        normalized_cache["states"].mean(dim=0),
        torch.zeros(raw_states.shape[1]),
        atol=1e-6,
    )
    assert torch.allclose(
        normalized_cache["actions"].mean(dim=0),
        torch.zeros(raw_actions.shape[1]),
        atol=1e-6,
    )
    assert torch.allclose(
        state_stats.denormalize(normalized_cache["states"]),
        raw_states,
        atol=1e-6,
    )
    assert torch.allclose(
        action_stats.denormalize(normalized_cache["actions"]),
        raw_actions,
        atol=1e-6,
    )
    print("[PASS] test_rerun_viz_normalizes_cached_raw_data")


# ---------------------------------------------------------------------------
# Policy reality check (offline replay helpers; no model/download needed)
# ---------------------------------------------------------------------------


def test_policy_reality_check_selects_first_validation_episodes() -> None:
    """Episode selection should preserve validation episode order."""
    from policy_reality_check import select_episode_ids

    episode_indices = torch.tensor([0, 0, 1, 1, 2, 2, 5, 5, 9, 9])
    mask = torch.tensor(
        [False, False, False, False, True, True, True, True, True, True]
    )

    selected = select_episode_ids(episode_indices, mask, num_episodes=2)

    assert selected == [2, 5]
    print("[PASS] test_policy_reality_check_selects_first_validation_episodes")


def test_policy_reality_check_computes_action_metrics() -> None:
    """Action metrics should report mean and per-joint absolute error."""
    from policy_reality_check import compute_action_metrics

    pred = torch.tensor([[1.0, 3.0, -2.0], [2.0, 2.0, -1.0]])
    gt = torch.tensor([[2.0, 1.0, -2.0], [0.0, 2.0, 1.0]])

    metrics = compute_action_metrics(pred, gt)

    assert metrics["mae"] == 1.1666666269302368
    assert torch.allclose(metrics["per_joint_mae"], torch.tensor([1.5, 1.0, 1.0]))
    assert metrics["max_abs_error"] == 2.0
    print("[PASS] test_policy_reality_check_computes_action_metrics")


def test_policy_reality_check_samples_episode_frames_with_stride() -> None:
    """Frame sampling should include the first frame and obey stride."""
    from policy_reality_check import sample_episode_frame_indices

    episode_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])

    sampled = sample_episode_frame_indices(
        episode_indices=episode_indices, episode_id=0, stride=2
    )

    assert sampled.tolist() == [0, 2, 4]
    print("[PASS] test_policy_reality_check_samples_episode_frames_with_stride")


def test_policy_reality_check_large_vla_fallback_label() -> None:
    """Large-VLA comparison should have explicit status labels."""
    from policy_reality_check import LargeVLAResult

    result = LargeVLAResult(
        status="reference_only",
        model_id="lerobot/smolvla_base",
        message="Policy preprocessing is not compatible with cached Ch07 tensors.",
    )

    assert result.status == "reference_only"
    assert result.model_id == "lerobot/smolvla_base"
    assert "not compatible" in result.message
    print("[PASS] test_policy_reality_check_large_vla_fallback_label")


def test_policy_reality_check_large_vla_attempt_handles_missing_dependency(
    monkeypatch,
) -> None:
    """Large-VLA attempt should fail explicitly, not crash the whole report."""
    from policy_reality_check import attempt_large_vla_reference

    def fake_import_module(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    result = attempt_large_vla_reference("lerobot/smolvla_base", try_import=True)

    assert result.status == "reference_only"
    assert result.model_id == "lerobot/smolvla_base"
    assert "Could not import LeRobot policy loader" in result.message
    print("[PASS] test_policy_reality_check_large_vla_attempt_handles_missing_dependency")


if __name__ == "__main__":
    print("=" * 50)
    print("  Chapter 7: Full VLA -- Sanity Tests")
    print("=" * 50)

    test_pixel_shuffle()
    test_timestep_embedding()
    test_expert_sa_block()
    test_expert_ca_block()
    test_action_expert()
    test_action_expert_param_count()
    test_action_expert_gradient_flow()
    test_flow_matching_loss()
    test_rerun_viz_normalizes_cached_raw_data()
    test_policy_reality_check_selects_first_validation_episodes()
    test_policy_reality_check_computes_action_metrics()
    test_policy_reality_check_samples_episode_frames_with_stride()
    test_policy_reality_check_large_vla_fallback_label()

    print("\n" + "=" * 50)
    print("  All tests passed!")
    print("=" * 50)
