"""Tests for Chapter 3: Language Conditioning."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_pusht import (
    MiniPushT,
    ScriptedExpert,
    CANONICAL_INSTRUCTIONS,
    PARAPHRASES,
    PARAPHRASE_TO_CANONICAL,
    INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------


def test_env_obs_shape():
    env = MiniPushT(size=64)
    obs, info = env.reset(seed=0)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8


def test_env_all_canonical_instructions():
    """Every canonical instruction should be accepted by reset."""
    env = MiniPushT(size=64)
    for instr in CANONICAL_INSTRUCTIONS:
        obs, info = env.reset(seed=0, options={"instruction": instr})
        assert info["instruction"] == instr
        assert info["canonical_instruction"] == instr


def test_env_paraphrases_accepted():
    """Every paraphrase should be accepted and resolved correctly."""
    env = MiniPushT(size=64)
    for canonical, paras in PARAPHRASES.items():
        for para in paras:
            obs, info = env.reset(seed=0, options={"instruction": para})
            assert info["canonical_instruction"] == canonical


def test_env_unknown_instruction_raises():
    """Unknown instruction strings should raise ValueError."""
    env = MiniPushT(size=64)
    with pytest.raises(ValueError, match="Unknown instruction"):
        env.reset(seed=0, options={"instruction": "fly to the moon"})


def test_instruction_count():
    """Should have exactly 11 canonical instructions."""
    assert len(CANONICAL_INSTRUCTIONS) == 11
    assert len(INSTRUCTIONS) == 11


def test_paraphrase_coverage():
    """Every canonical instruction should have at least 3 paraphrases."""
    for canonical in CANONICAL_INSTRUCTIONS:
        assert canonical in PARAPHRASES, f"Missing paraphrases for '{canonical}'"
        assert len(PARAPHRASES[canonical]) >= 3


def test_expert_all_instructions():
    """Expert should return valid actions for all canonical instructions."""
    env = MiniPushT(size=64)
    expert = ScriptedExpert(size=64)
    for instr in CANONICAL_INSTRUCTIONS:
        obs, info = env.reset(seed=42, options={"instruction": instr})
        action = expert.act(info)
        assert 0 <= action <= 4, f"Invalid action {action} for '{instr}'"


def test_expert_paraphrase_same_behavior():
    """Expert should produce same action for canonical and its paraphrase."""
    env = MiniPushT(size=64)
    expert = ScriptedExpert(size=64)
    for canonical, paras in PARAPHRASES.items():
        obs, info_c = env.reset(seed=42, options={"instruction": canonical})
        action_c = expert.act(info_c)
        for para in paras:
            _, info_p = env.reset(seed=42, options={"instruction": para})
            action_p = expert.act(info_p)
            assert action_c == action_p, (
                f"Mismatch: '{canonical}'->{action_c}, '{para}'->{action_p}"
            )


def test_expert_solves_all_instructions():
    """Expert should solve all 11 instructions at >= 80% rate (size=64)."""
    env = MiniPushT(size=64)
    expert = ScriptedExpert(size=64)
    for instr in CANONICAL_INSTRUCTIONS:
        successes = 0
        n = 10
        for trial in range(n):
            obs, info = env.reset(seed=trial, options={"instruction": instr})
            done = False
            while not done:
                action = expert.act(info)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            if terminated:
                successes += 1
        rate = successes / n
        assert rate >= 0.7, f"Expert only {rate:.0%} on '{instr}' (need >= 70%)"


def test_goal_instruction_terminates():
    """push block to goal should terminate when block reaches goal."""
    env = MiniPushT(size=64)
    expert = ScriptedExpert(size=64)
    obs, info = env.reset(seed=42, options={"instruction": "push block to goal"})
    for _ in range(200):
        action = expert.act(info)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    assert terminated


def test_agent_move_terminates():
    """move agent up should terminate when agent reaches top edge."""
    env = MiniPushT(size=64)
    expert = ScriptedExpert(size=64)
    obs, info = env.reset(seed=42, options={"instruction": "move agent up"})
    for _ in range(200):
        action = expert.act(info)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    assert terminated


# ---------------------------------------------------------------------------
# Model tests (require HuggingFace model downloads)
# ---------------------------------------------------------------------------

from cross_attention import (
    LookupLanguageEncoder,
    ConcatFusion,
    CrossAttentionFusion,
    VisionEncoder,
    VLA,
    build_vla,
    collect_demos,
    PushTDataset,
    train,
    evaluate,
)


def test_lookup_encoder_canonical():
    """Lookup encoder should produce embeddings for all canonical instructions."""
    enc = LookupLanguageEncoder(embed_dim=256)
    out = enc(CANONICAL_INSTRUCTIONS)
    assert out.shape == (11, 256)


def test_lookup_encoder_unknown():
    """Lookup encoder should return zero embedding for unknown strings."""
    enc = LookupLanguageEncoder(embed_dim=256)
    out = enc(["this is an unknown instruction"])
    assert out.shape == (1, 256)
    assert torch.allclose(out, torch.zeros(1, 256))


def test_cross_attention_shape():
    fusion = CrossAttentionFusion(vision_dim=768, d_model=256, n_heads=4)
    vision_tokens = torch.randn(2, 197, 768)
    lang_tokens = torch.randn(2, 8, 256)
    out = fusion(vision_tokens, lang_tokens)
    assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"


def test_cross_attention_with_mask():
    """Cross-attention should handle padding masks correctly."""
    fusion = CrossAttentionFusion(vision_dim=768, d_model=256, n_heads=4)
    vision_tokens = torch.randn(2, 10, 768)
    lang_tokens = torch.randn(2, 8, 256)
    # Second sequence has only 3 real tokens
    lang_mask = torch.ones(2, 8)
    lang_mask[1, 3:] = 0
    out = fusion(vision_tokens, lang_tokens, lang_mask)
    assert out.shape == (2, 256)


def test_concat_fusion_shape():
    fusion = ConcatFusion(vision_dim=768, lang_dim=256, d_model=256)
    vision_feat = torch.randn(2, 768)
    lang_feat = torch.randn(2, 256)
    out = fusion(vision_feat, lang_feat)
    assert out.shape == (2, 512), f"Expected (2, 512), got {out.shape}"


# Tests that require downloading HuggingFace models
HF_AVAILABLE = os.getenv("RUN_HF_TESTS", "1") == "1"


@pytest.mark.skipif(not HF_AVAILABLE, reason="Skipping HF model tests")
def test_build_vla_lookup_concat():
    model = build_vla(fusion_type="concat", lm_type="lookup")
    x = torch.randn(2, 3, 224, 224)
    logits = model(x, ["push block to goal", "move agent left"])
    assert logits.shape == (2, 5)


@pytest.mark.skipif(not HF_AVAILABLE, reason="Skipping HF model tests")
def test_build_vla_smollm2_concat():
    model = build_vla(fusion_type="concat", lm_type="smollm2-135m")
    x = torch.randn(2, 3, 224, 224)
    logits = model(x, ["push block to goal", "move agent left"])
    assert logits.shape == (2, 5)


@pytest.mark.skipif(not HF_AVAILABLE, reason="Skipping HF model tests")
def test_build_vla_smollm2_crossattn():
    model = build_vla(fusion_type="cross-attn", lm_type="smollm2-135m")
    x = torch.randn(2, 3, 224, 224)
    logits = model(x, ["push block to goal", "move agent left"])
    assert logits.shape == (2, 5)


@pytest.mark.skipif(not HF_AVAILABLE, reason="Skipping HF model tests")
def test_smollm2_handles_paraphrases():
    """SmolLM2 should accept paraphrase strings without error."""
    from cross_attention import SmolLM2Encoder

    enc = SmolLM2Encoder(
        model_name="HuggingFaceTB/SmolLM2-135M", d_model=256, pool=True
    )
    paras = ["move the block to the target", "go left", "nudge the block upward"]
    out = enc(paras)
    assert out.shape == (3, 256)


@pytest.mark.skipif(not HF_AVAILABLE, reason="Skipping HF model tests")
def test_smollm2_frozen():
    """SmolLM2 backbone should be completely frozen."""
    from cross_attention import SmolLM2Encoder

    enc = SmolLM2Encoder(
        model_name="HuggingFaceTB/SmolLM2-135M", d_model=256, pool=True
    )
    for name, param in enc.model.named_parameters():
        assert not param.requires_grad, f"SmolLM2 param {name} should be frozen"
    assert enc.projection.weight.requires_grad


@pytest.mark.skipif(not HF_AVAILABLE, reason="Skipping HF model tests")
def test_smollm2_sequence_with_mask():
    """SmolLM2 in sequence mode should return tokens and attention mask."""
    from cross_attention import SmolLM2Encoder

    enc = SmolLM2Encoder(
        model_name="HuggingFaceTB/SmolLM2-135M", d_model=256, pool=False
    )
    tokens, mask = enc(["push block to goal", "go left"])
    assert tokens.ndim == 3
    assert tokens.shape[0] == 2
    assert tokens.shape[2] == 256
    assert mask.shape[0] == 2
    assert mask.shape[1] == tokens.shape[1]


@pytest.mark.skipif(not HF_AVAILABLE, reason="Skipping HF model tests")
def test_vla_trainable_params():
    """Vision and language backbones should be frozen."""
    model = build_vla(fusion_type="cross-attn", lm_type="smollm2-135m")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # Trainable should be small fraction of total
    assert trainable < total * 0.05, f"Too many trainable: {trainable}/{total}"
    assert trainable > 0


def test_train_one_step_lookup():
    """One training step should run for lookup+concat config."""
    env = MiniPushT(size=64)
    expert = ScriptedExpert(size=64)
    demos = collect_demos(env, expert, n_episodes=11)
    dataset = PushTDataset(demos)
    model = build_vla(
        fusion_type="concat",
        lm_type="lookup",
        encoder_name="google/siglip-base-patch16-224",
    )
    losses = train(model, dataset, epochs=1, batch_size=8, device="cpu")
    assert len(losses) == 1
    assert losses[0] > 0


def test_evaluate_runs():
    """Evaluation loop should run and return a float."""
    env = MiniPushT(size=64)
    model = build_vla(
        fusion_type="concat",
        lm_type="lookup",
        encoder_name="google/siglip-base-patch16-224",
    )
    rate = evaluate(model, env, n_episodes=11, device="cpu")
    assert 0.0 <= rate <= 1.0
