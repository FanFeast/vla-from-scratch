"""Tests for Chapter 2: Vision Backbone."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from compare_encoders import (
    PRESETS,
    MmapPushTDataset,
    PretrainedPatchEncoder,
    PretrainedVisionEncoder,
    PushTDataset,
    ScratchCNN,
    ScriptedExpert,
    VLA,
    _ood_reset,
    collect_demos,
    collect_demos_to_disk,
    evaluate_ood,
    load_demos,
    load_demos_as_dataset,
    load_demos_mmap,
    plot_tsne_comparison,
    save_demos,
    train,
)
from mini_pusht import MiniPushT


def test_env_obs_shape_and_dtype():
    env = MiniPushT()
    obs, info = env.reset(seed=0)
    assert obs.shape == (224, 224, 3), f"Expected (224,224,3), got {obs.shape}"
    assert obs.dtype == np.uint8


def test_env_info_keys():
    env = MiniPushT()
    _, info = env.reset(seed=0)
    for key in ("agent_pos", "block_pos", "goal_pos", "instruction", "step"):
        assert key in info, f"Missing key: {key}"


def test_scratch_cnn_output_shape():
    encoder = ScratchCNN()
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"


def test_vla_forward_scratch():
    encoder = ScratchCNN()
    model = VLA(vision_encoder=encoder, vision_dim=128)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x, ["push block to goal", "move left"])
    assert logits.shape == (2, 5), f"Expected (2, 5), got {logits.shape}"


def test_pretrained_clip_output_shape():
    encoder = PretrainedVisionEncoder("openai/clip-vit-base-patch16")
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"


def test_pretrained_siglip_output_shape():
    encoder = PretrainedVisionEncoder("google/siglip-base-patch16-224")
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"


def test_frozen_encoder_no_grad():
    encoder = PretrainedVisionEncoder("openai/clip-vit-base-patch16")
    for p in encoder.model.parameters():
        assert not p.requires_grad, "Pretrained params should be frozen"


def test_frozen_params_unchanged_after_step():
    encoder = PretrainedVisionEncoder("openai/clip-vit-base-patch16")
    model = VLA(vision_encoder=encoder, vision_dim=768)

    # Snapshot one frozen param
    frozen_param = next(encoder.model.parameters())
    before = frozen_param.clone()

    # One training step
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    x = torch.randn(1, 3, 224, 224)
    logits = model(x, "push block to goal")
    loss = logits.sum()
    loss.backward()
    optimizer.step()

    after = frozen_param
    assert torch.equal(before, after), "Frozen params changed after step"


def test_vla_forward_pretrained():
    encoder = PretrainedVisionEncoder("openai/clip-vit-base-patch16")
    model = VLA(vision_encoder=encoder, vision_dim=768)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x, ["push block to goal", "move left"])
    assert logits.shape == (2, 5), f"Expected (2, 5), got {logits.shape}"


def test_collect_demos_224():
    env = MiniPushT()
    expert = ScriptedExpert(size=224)
    demos = collect_demos(env, expert, n_episodes=3)
    assert len(demos) > 0
    assert demos[0]["image"].shape == (224, 224, 3)
    assert demos[0]["image"].dtype == np.uint8
    assert isinstance(demos[0]["instruction"], str)
    assert isinstance(demos[0]["action"], int)


def test_one_training_step():
    env = MiniPushT()
    expert = ScriptedExpert(size=224)
    demos = collect_demos(env, expert, n_episodes=2)
    dataset = PushTDataset(demos)
    model = VLA(vision_encoder=ScratchCNN(), vision_dim=128)
    losses = train(model, dataset, epochs=1, batch_size=4)
    assert len(losses) == 1
    assert losses[0] > 0


def test_tsne_visualization_runs():
    # Create dummy embeddings and labels
    embeddings_dict = {
        "ScratchCNN": np.random.randn(50, 128).astype(np.float32),
        "CLIP": np.random.randn(50, 768).astype(np.float32),
    }
    labels = np.random.randint(0, 5, size=50)
    # Should run without error, save to temp path
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tsne.png")
        plot_tsne_comparison(embeddings_dict, labels, save_path=save_path)
        assert os.path.exists(save_path)


def test_env_64x64():
    env = MiniPushT(size=64)
    obs, info = env.reset(seed=0)
    assert obs.shape == (64, 64, 3), f"Expected (64,64,3), got {obs.shape}"
    assert obs.dtype == np.uint8


def test_env_112x112():
    env = MiniPushT(size=112)
    obs, info = env.reset(seed=0)
    assert obs.shape == (112, 112, 3), f"Expected (112,112,3), got {obs.shape}"
    assert obs.dtype == np.uint8


def test_env_scaling_constants():
    env64 = MiniPushT(size=64)
    env224 = MiniPushT(size=224)
    # Movement speed should scale: 1 at 64, round(224/64)=4 at 224
    assert env64._speed == 1
    assert env224._speed == 4
    # Goal tolerance should scale: 4 at 64, 14 at 224
    assert env64.goal_tolerance == 4
    assert env224.goal_tolerance == 14


def test_expert_scales_with_resolution():
    # Expert should work at 64x64
    env64 = MiniPushT(size=64)
    expert64 = ScriptedExpert(size=64)
    demos = collect_demos(env64, expert64, n_episodes=3)
    assert len(demos) > 0
    assert demos[0]["image"].shape == (64, 64, 3)


def test_dataset_cache_save_load():
    env = MiniPushT(size=64)
    expert = ScriptedExpert(size=64)
    demos = collect_demos(env, expert, n_episodes=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_demos.npz")
        save_demos(demos, path)
        assert os.path.exists(path)

        loaded = load_demos(path)
        assert len(loaded) == len(demos)
        assert np.array_equal(loaded[0]["image"], demos[0]["image"])
        assert loaded[0]["instruction"] == demos[0]["instruction"]
        assert loaded[0]["action"] == demos[0]["action"]


def test_collect_demos_to_disk_streaming():
    """Streaming collector writes directly to disk and produces valid .npz."""
    env = MiniPushT(size=64)
    expert = ScriptedExpert(size=64)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "stream_demos.npz")
        n = collect_demos_to_disk(env, expert, path, n_episodes=3)
        assert n > 0
        assert os.path.exists(path)

        # Verify via load_demos_as_dataset
        dataset = load_demos_as_dataset(path)
        assert len(dataset) == n
        img, instr, action = dataset[0]
        assert img.shape == (3, 64, 64)
        assert img.dtype == torch.uint8
        assert isinstance(instr, str)

        # Also verify roundtrip consistency with load_demos
        demos = load_demos(path)
        assert len(demos) == n
        assert demos[0]["image"].shape == (64, 64, 3)


def test_preset_defaults():
    assert "tiny" in PRESETS
    assert "medium" in PRESETS
    assert "full" in PRESETS
    assert PRESETS["tiny"]["size"] == 64
    assert PRESETS["tiny"]["skip_pretrained"] is True
    assert PRESETS["full"]["size"] == 224
    assert PRESETS["full"]["skip_pretrained"] is False


def test_pretrained_resize_non224():
    encoder = PretrainedVisionEncoder("openai/clip-vit-base-patch16")
    # Feed 64x64 input -- should resize internally to 224x224
    x = torch.randn(2, 3, 64, 64)
    out = encoder(x)
    assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"


def test_attention_pool_encoder():
    """PretrainedPatchEncoder uses learned attention pooling over patch tokens."""
    encoder = PretrainedPatchEncoder("google/siglip-base-patch16-224")
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"

    # Attention pool params should be trainable, ViT params should be frozen
    trainable = [n for n, p in encoder.named_parameters() if p.requires_grad]
    frozen = [n for n, p in encoder.named_parameters() if not p.requires_grad]
    assert len(trainable) > 0, "AttentionPool should have trainable params"
    assert all("attn_pool" in n for n in trainable), (
        f"Only attn_pool should be trainable: {trainable}"
    )
    assert len(frozen) > 0, "ViT params should be frozen"


def test_attention_pool_gradients_flow():
    """Gradients flow through attention pool but not the frozen ViT."""
    encoder = PretrainedPatchEncoder("google/siglip-base-patch16-224")
    x = torch.randn(1, 3, 224, 224)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    assert encoder.attn_pool.query.weight.grad is not None, (
        "Attn pool should get gradients"
    )
    for param in encoder.model.parameters():
        assert param.grad is None, "Frozen ViT should NOT get gradients"


def test_ood_reset_places_starts_outside_training_band():
    """OOD reset must put both agent and block coords outside the training [lo, hi]."""
    env = MiniPushT(size=64)
    rng = np.random.default_rng(0)
    for _ in range(30):
        _, info = _ood_reset(env, rng)
        ax, ay = info["agent_pos"]
        bx, by = info["block_pos"]
        assert not (env._agent_lo <= ax <= env._agent_hi), (
            f"agent x={ax} inside training band [{env._agent_lo}, {env._agent_hi}]"
        )
        assert not (env._agent_lo <= ay <= env._agent_hi), (
            f"agent y={ay} inside training band [{env._agent_lo}, {env._agent_hi}]"
        )
        assert not (env._block_lo <= bx <= env._block_hi), (
            f"block x={bx} inside training band [{env._block_lo}, {env._block_hi}]"
        )
        assert not (env._block_lo <= by <= env._block_hi), (
            f"block y={by} inside training band [{env._block_lo}, {env._block_hi}]"
        )


def test_ood_reset_satisfies_min_distances():
    """OOD reset must still respect block-goal and agent-block min distances."""
    env = MiniPushT(size=64)
    rng = np.random.default_rng(1)
    for _ in range(30):
        _, info = _ood_reset(env, rng)
        block_goal = np.linalg.norm(info["block_pos"] - info["goal_pos"])
        agent_block = np.linalg.norm(info["agent_pos"] - info["block_pos"])
        assert block_goal > env._min_block_goal, (
            f"block-goal dist {block_goal} <= min {env._min_block_goal}"
        )
        assert agent_block > env._min_agent_block, (
            f"agent-block dist {agent_block} <= min {env._min_agent_block}"
        )


def test_ood_reset_is_deterministic_given_rng():
    """Same seed -> same OOD episodes (required for fair cross-encoder comparison)."""
    env_a = MiniPushT(size=64)
    env_b = MiniPushT(size=64)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    for _ in range(5):
        _, info_a = _ood_reset(env_a, rng_a)
        _, info_b = _ood_reset(env_b, rng_b)
        assert np.array_equal(info_a["agent_pos"], info_b["agent_pos"])
        assert np.array_equal(info_a["block_pos"], info_b["block_pos"])
        assert np.array_equal(info_a["goal_pos"], info_b["goal_pos"])


def test_evaluate_ood_runs_end_to_end():
    """Smoke test: evaluate_ood loop runs and returns a [0,1] success rate."""
    env = MiniPushT(size=64)
    encoder = ScratchCNN()
    model = VLA(vision_encoder=encoder, vision_dim=128)
    rate = evaluate_ood(model, env, n_episodes=3, device="cpu", seed=0)
    assert 0.0 <= rate <= 1.0


def test_mmap_dataset_loads_and_trains():
    """MmapPushTDataset streams from disk with near-zero RAM."""
    env = MiniPushT(size=64)
    expert = ScriptedExpert(size=64)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mmap_demos.npz")
        collect_demos_to_disk(env, expert, path, n_episodes=3)

        dataset = load_demos_mmap(path)
        assert isinstance(dataset, MmapPushTDataset)
        assert len(dataset) > 0

        img, instr, action = dataset[0]
        assert img.shape == (3, 64, 64)
        assert img.dtype == torch.uint8
        assert isinstance(instr, str)

        # One training step should work
        model = VLA(vision_encoder=ScratchCNN(), vision_dim=128)
        losses = train(model, dataset, epochs=1, batch_size=4)
        assert len(losses) == 1
        assert losses[0] > 0
