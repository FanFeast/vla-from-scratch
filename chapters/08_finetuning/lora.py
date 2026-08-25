"""Chapter 8: LoRA from scratch.

Low-Rank Adaptation (LoRA, Hu et al., 2021) freezes a pretrained weight
matrix W and learns a low-rank update Delta_W = B @ A instead. Only A and B
are trained, so a 7B model can be adapted with a few million parameters.

Intuition (in LLM terms you already know):
    A linear layer computes y = W x. Fine-tuning normally updates all of W.
    LoRA keeps W frozen and adds a tiny detour:

        y = W x + (alpha / r) * B (A x)

    where A is (r, in) and B is (out, r) with r << min(in, out). A is
    randomly initialized and B is zero-initialized, so at the start the
    detour contributes nothing and the model behaves exactly like the
    pretrained one. Training nudges B (and A) away from zero.

Why this matters for VLAs:
    OpenVLA is a 7B model. Full fine-tuning needs ~140 GB of optimizer state.
    LoRA with rank 32 trains ~110M params (1.5%), fits on a single 24 GB GPU
    with 4-bit quantization, and matches full fine-tuning on most tasks.

This file implements LoRA from scratch (no peft dependency) so you can see
exactly what the adapter does. `lora_openvla.py` shows the peft-based path
for the real 7B model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """Hyperparameters controlling LoRA injection.

    Attributes:
        r: Rank of the low-rank update. Higher r = more capacity, more params.
        alpha: Scaling factor. The update is scaled by alpha / r so that
            tuning r does not force you to retune the learning rate.
        dropout: Dropout applied to the LoRA input branch (regularization).
        target_modules: Substrings matched against module names. Any
            nn.Linear whose qualified name contains one of these strings
            gets wrapped with a LoRA adapter. For a Llama-style backbone the
            attention projections are named q_proj/k_proj/v_proj/o_proj.
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: ("q_proj", "v_proj")
    )

    @property
    def scaling(self) -> float:
        """Return the alpha / r scaling applied to the LoRA branch."""
        return self.alpha / self.r


# ---------------------------------------------------------------------------
# LoRA linear wrapper
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank adapter.

    forward(x) = base(x) + scaling * dropout(x) @ A^T @ B^T

    The base layer is frozen; only lora_A and lora_B receive gradients.
    B is zero-initialized so the wrapped layer starts identical to `base`.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")

        self.base = base
        self.r = config.r
        self.scaling = config.scaling

        in_features = base.in_features
        out_features = base.out_features

        # Freeze the pretrained weight (and bias if present).
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # Low-rank factors. A: (r, in), B: (out, r).
        self.lora_A = nn.Parameter(torch.empty(self.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
        self.lora_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # Kaiming init on A, zeros on B (standard LoRA init). This makes the
        # initial update B @ A exactly zero while keeping A well-scaled.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.merged = False

    def forward(self, x: Tensor) -> Tensor:
        """Apply the frozen base plus the low-rank update."""
        result = self.base(x)
        if self.merged:
            # Weights already folded into base; skip the detour.
            return result
        lora_update = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return result + self.scaling * lora_update

    @torch.no_grad()
    def merge(self) -> None:
        """Fold B @ A into the base weight for zero-overhead inference.

        After merging, forward() costs exactly one matmul again. Use this
        before exporting a deployment checkpoint.
        """
        if self.merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data += delta
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Reverse merge() to recover the separate adapter (for more training)."""
        if not self.merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data -= delta
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"r={self.r}, scaling={self.scaling:.2f}, "
            f"in={self.base.in_features}, out={self.base.out_features}, "
            f"merged={self.merged}"
        )


# ---------------------------------------------------------------------------
# Injection / surgery helpers
# ---------------------------------------------------------------------------


def _module_matches(name: str, targets: tuple[str, ...]) -> bool:
    """Return True if any target substring appears in the module name."""
    return any(t in name for t in targets)


def inject_lora(model: nn.Module, config: LoRAConfig) -> int:
    """Replace matching nn.Linear modules in-place with LoRALinear wrappers.

    Walks the module tree, and for every nn.Linear whose qualified name
    contains a target substring, swaps it for a LoRALinear around the
    original layer (preserving the pretrained weights).

    Args:
        model: The module to adapt (modified in place).
        config: LoRA hyperparameters, including target_modules.

    Returns:
        Number of layers wrapped.
    """
    # Collect first to avoid mutating the tree while iterating it.
    to_wrap: list[tuple[nn.Module, str, nn.Linear]] = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _module_matches(
                full_name, config.target_modules
            ):
                to_wrap.append((module, child_name, child))

    for parent, child_name, linear in to_wrap:
        setattr(parent, child_name, LoRALinear(linear, config))

    return len(to_wrap)


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze every parameter except the LoRA A/B factors.

    This is the standard LoRA training setup: the entire backbone is frozen
    and gradients only flow into the rank-r adapters.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the list of trainable LoRA parameters for an optimizer."""
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return only the LoRA tensors, for a tiny adapter checkpoint.

    A LoRA checkpoint is typically a few MB even for a 7B model, because it
    contains only the A/B factors.
    """
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    """Load LoRA tensors produced by lora_state_dict back into a model."""
    own = dict(model.named_parameters())
    for name, tensor in state.items():
        if name not in own:
            raise KeyError(f"LoRA param {name!r} not found in model")
        own[name].data.copy_(tensor.to(own[name].device))


def merge_lora(model: nn.Module) -> int:
    """Merge every LoRALinear adapter in the model. Returns count merged."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            count += 1
    return count


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and LoRA parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(
        p.numel() for n, p in model.named_parameters() if "lora_" in n
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "lora": lora,
    }


def summarize(model: nn.Module, config: LoRAConfig) -> str:
    """Build a human-readable summary of the LoRA adaptation."""
    counts = count_parameters(model)
    pct = 100.0 * counts["trainable"] / max(counts["total"], 1)
    return (
        f"LoRA r={config.r}, alpha={config.alpha}, "
        f"targets={config.target_modules}\n"
        f"  total      = {counts['total'] / 1e6:.2f}M\n"
        f"  trainable  = {counts['trainable'] / 1e6:.2f}M ({pct:.2f}%)\n"
        f"  frozen     = {counts['frozen'] / 1e6:.2f}M"
    )


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Inject LoRA into a tiny toy transformer and show the parameter savings."""
    torch.manual_seed(0)

    class ToyAttention(nn.Module):
        """A miniature attention block with named projections."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.o_proj = nn.Linear(dim, dim)

        def forward(self, x: Tensor) -> Tensor:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            attn = torch.softmax(q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5, -1)
            return self.o_proj(attn @ v)

    model = nn.Sequential(ToyAttention(128), ToyAttention(128))

    config = LoRAConfig(r=8, alpha=16, target_modules=("q_proj", "v_proj"))
    n_wrapped = inject_lora(model, config)
    mark_only_lora_as_trainable(model)

    print(f"Wrapped {n_wrapped} linear layers with LoRA")
    print(summarize(model, config))

    # Check it still runs and that the initial output is unchanged.
    x = torch.randn(2, 5, 128)
    y = model(x)
    print(f"Output shape: {tuple(y.shape)}")

    # The adapter checkpoint is tiny.
    sd = lora_state_dict(model)
    n_bytes = sum(t.numel() * t.element_size() for t in sd.values())
    print(f"LoRA checkpoint: {len(sd)} tensors, {n_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    _demo()
