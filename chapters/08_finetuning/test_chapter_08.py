"""Sanity tests for Chapter 8: LoRA fine-tuning.

Covers the from-scratch LoRA implementation (injection, freezing, zero-init
equivalence, merge correctness, checkpointing) and the fine-tuning wiring
helpers. None of these tests download a pretrained model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).parent))

from lora import (
    LoRAConfig,
    LoRALinear,
    count_parameters,
    inject_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    mark_only_lora_as_trainable,
    merge_lora,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class TinyAttention(nn.Module):
    """Miniature attention block with Llama-style projection names."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5, -1)
        return self.o_proj(attn @ v)


def _build_model(dim: int = 32) -> nn.Module:
    return nn.Sequential(TinyAttention(dim), TinyAttention(dim))


# ---------------------------------------------------------------------------
# LoRALinear unit behavior
# ---------------------------------------------------------------------------


def test_lora_linear_zero_init_identity() -> None:
    """A freshly wrapped LoRALinear must equal its base layer (B is zero)."""
    base = nn.Linear(16, 24)
    wrapped = LoRALinear(base, LoRAConfig(r=4, alpha=8))
    x = torch.randn(5, 16)
    assert torch.allclose(wrapped(x), base(x), atol=1e-6)
    print("[PASS] test_lora_linear_zero_init_identity")


def test_lora_linear_freezes_base() -> None:
    """The base weight/bias must be frozen, the adapters trainable."""
    wrapped = LoRALinear(nn.Linear(16, 16), LoRAConfig(r=4))
    assert wrapped.base.weight.requires_grad is False
    assert wrapped.base.bias.requires_grad is False
    assert wrapped.lora_A.requires_grad is True
    assert wrapped.lora_B.requires_grad is True
    print("[PASS] test_lora_linear_freezes_base")


def test_lora_linear_merge_matches_unmerged() -> None:
    """Merged forward must match the unmerged forward after the adapter moves."""
    torch.manual_seed(0)
    wrapped = LoRALinear(nn.Linear(16, 16), LoRAConfig(r=4, alpha=8))
    # Move B away from zero so the adapter actually contributes.
    with torch.no_grad():
        wrapped.lora_B.normal_(0, 0.1)
    x = torch.randn(4, 16)
    before = wrapped(x)
    wrapped.merge()
    after = wrapped(x)
    assert wrapped.merged is True
    assert torch.allclose(before, after, atol=1e-5)
    # Unmerge restores the split representation.
    wrapped.unmerge()
    assert wrapped.merged is False
    assert torch.allclose(wrapped(x), before, atol=1e-5)
    print("[PASS] test_lora_linear_merge_matches_unmerged")


def test_lora_linear_rejects_non_linear() -> None:
    """Wrapping something that is not nn.Linear should raise TypeError."""
    try:
        LoRALinear(nn.ReLU(), LoRAConfig())  # type: ignore[arg-type]
    except TypeError:
        print("[PASS] test_lora_linear_rejects_non_linear")
        return
    raise AssertionError("Expected TypeError for non-Linear base")


# ---------------------------------------------------------------------------
# Injection / surgery
# ---------------------------------------------------------------------------


def test_inject_lora_wraps_only_targets() -> None:
    """Only q_proj and v_proj should be wrapped; k_proj and o_proj untouched."""
    model = _build_model()
    n = inject_lora(model, LoRAConfig(r=4, target_modules=("q_proj", "v_proj")))
    assert n == 4, f"Expected 4 wraps (2 blocks x 2 targets), got {n}"
    wrapped_names = [
        name for name, m in model.named_modules() if isinstance(m, LoRALinear)
    ]
    assert all("q_proj" in n or "v_proj" in n for n in wrapped_names)
    assert len(wrapped_names) == 4
    print("[PASS] test_inject_lora_wraps_only_targets")


def test_inject_lora_preserves_output_initially() -> None:
    """Injecting LoRA must not change the model output at initialization."""
    torch.manual_seed(0)
    model = _build_model()
    x = torch.randn(2, 6, 32)
    before = model(x)
    inject_lora(model, LoRAConfig(r=8, target_modules=("q_proj", "v_proj")))
    after = model(x)
    assert torch.allclose(before, after, atol=1e-6)
    print("[PASS] test_inject_lora_preserves_output_initially")


def test_mark_only_lora_as_trainable() -> None:
    """After marking, only lora_ params have requires_grad=True."""
    model = _build_model()
    inject_lora(model, LoRAConfig(r=4))
    mark_only_lora_as_trainable(model)
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad, f"{name} should be trainable"
        else:
            assert not p.requires_grad, f"{name} should be frozen"
    print("[PASS] test_mark_only_lora_as_trainable")


def test_count_parameters_reports_small_lora_fraction() -> None:
    """LoRA params should be a tiny fraction of the total."""
    model = _build_model(dim=128)
    inject_lora(model, LoRAConfig(r=8, target_modules=("q_proj", "v_proj")))
    mark_only_lora_as_trainable(model)
    counts = count_parameters(model)
    frac = counts["trainable"] / counts["total"]
    assert counts["lora"] == counts["trainable"]
    assert frac < 0.2, f"LoRA fraction unexpectedly large: {frac:.3f}"
    print(f"[PASS] test_count_parameters_reports_small_lora_fraction (frac={frac:.3f})")


# ---------------------------------------------------------------------------
# Training step + checkpoint round-trip
# ---------------------------------------------------------------------------


def test_lora_gradient_step_updates_only_adapters() -> None:
    """One optimizer step must move only LoRA params."""
    torch.manual_seed(0)
    model = _build_model()
    inject_lora(model, LoRAConfig(r=4, target_modules=("q_proj", "v_proj")))
    mark_only_lora_as_trainable(model)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.SGD(lora_parameters(model), lr=0.1)
    x = torch.randn(2, 6, 32)
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()

    moved = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert moved, "No parameters changed"
    assert all("lora_" in n for n in moved), f"Non-LoRA moved: {moved}"
    print("[PASS] test_lora_gradient_step_updates_only_adapters")


def test_lora_state_dict_roundtrip() -> None:
    """lora_state_dict + load_lora_state_dict should round-trip exactly.

    A LoRA checkpoint contains only the adapters, so the two models must share
    identical (frozen) base weights -- we seed both builds the same way.
    """
    torch.manual_seed(0)
    model = _build_model()
    inject_lora(model, LoRAConfig(r=4))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.1)

    sd = lora_state_dict(model)
    assert all("lora_" in k for k in sd)

    # Rebuild with the same seed so the frozen base weights match exactly.
    torch.manual_seed(0)
    fresh = _build_model()
    inject_lora(fresh, LoRAConfig(r=4))
    load_lora_state_dict(fresh, sd)

    x = torch.randn(2, 6, 32)
    assert torch.allclose(model(x), fresh(x), atol=1e-6)
    print("[PASS] test_lora_state_dict_roundtrip")


def test_merge_lora_counts() -> None:
    """merge_lora should merge every adapter and report the count."""
    model = _build_model()
    inject_lora(model, LoRAConfig(r=4, target_modules=("q_proj", "v_proj")))
    n = merge_lora(model)
    assert n == 4
    for m in model.modules():
        if isinstance(m, LoRALinear):
            assert m.merged
    print("[PASS] test_merge_lora_counts")


# ---------------------------------------------------------------------------
# Fine-tuning wiring helper
# ---------------------------------------------------------------------------


def test_collect_finetune_parameters_lora_only() -> None:
    """LoRA-only mode should freeze the action expert."""
    from finetune_smolvla import collect_finetune_parameters

    class Stand(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.action_expert = nn.Linear(8, 4)

    model = Stand()
    inject_lora(model, LoRAConfig(r=2, target_modules=("q_proj",)))
    params = collect_finetune_parameters(model, train_expert=False)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in n for n in names), names
    assert len(params) > 0
    print("[PASS] test_collect_finetune_parameters_lora_only")


def test_collect_finetune_parameters_with_expert() -> None:
    """train_expert=True should also unfreeze the action expert."""
    from finetune_smolvla import collect_finetune_parameters

    class Stand(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.action_expert = nn.Linear(8, 4)
            self.state_proj = nn.Linear(6, 8)

    model = Stand()
    inject_lora(model, LoRAConfig(r=2, target_modules=("q_proj",)))
    collect_finetune_parameters(model, train_expert=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("action_expert" in n for n in trainable)
    assert any("lora_" in n for n in trainable)
    print("[PASS] test_collect_finetune_parameters_with_expert")


# ---------------------------------------------------------------------------
# OpenVLA reference recipe (no download)
# ---------------------------------------------------------------------------


def test_openvla_recipe_describe() -> None:
    """The OpenVLA recipe should expose its canonical hyperparameters."""
    from lora_openvla import OpenVLALoRARecipe, missing_dependencies

    recipe = OpenVLALoRARecipe()
    text = recipe.describe()
    assert "openvla/openvla-7b" in text
    assert recipe.rank == 32
    # Should not raise even if deps are missing.
    assert isinstance(missing_dependencies(), list)
    print("[PASS] test_openvla_recipe_describe")


if __name__ == "__main__":
    print("=" * 50)
    print("  Chapter 8: LoRA Fine-tuning -- Sanity Tests")
    print("=" * 50)

    test_lora_linear_zero_init_identity()
    test_lora_linear_freezes_base()
    test_lora_linear_merge_matches_unmerged()
    test_lora_linear_rejects_non_linear()
    test_inject_lora_wraps_only_targets()
    test_inject_lora_preserves_output_initially()
    test_mark_only_lora_as_trainable()
    test_count_parameters_reports_small_lora_fraction()
    test_lora_gradient_step_updates_only_adapters()
    test_lora_state_dict_roundtrip()
    test_merge_lora_counts()
    test_collect_finetune_parameters_lora_only()
    test_collect_finetune_parameters_with_expert()
    test_openvla_recipe_describe()

    print("\n" + "=" * 50)
    print("  All tests passed!")
    print("=" * 50)
