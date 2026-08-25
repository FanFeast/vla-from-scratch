"""Chapter 8: LoRA fine-tuning the real OpenVLA-7B (reference recipe).

`lora.py` and `finetune_smolvla.py` show LoRA from scratch on our own small
model. This file shows the production path: LoRA fine-tuning the actual
7-billion-parameter [OpenVLA](https://arxiv.org/abs/2406.09246) using the
`peft` library and 4-bit quantization so it fits on a single 24 GB GPU.

This script is a documented reference, not part of the test suite -- running
it downloads ~14 GB of weights and needs `peft`, `bitsandbytes`, and a LeRobot
dataset. It checks for those dependencies and prints guidance if they are
missing instead of crashing.

The OpenVLA LoRA recipe (from the official repo) in one place:

    base model    : openvla/openvla-7b   (Llama-2 7B + DINOv2 + SigLIP)
    quantization  : 4-bit NF4 (bitsandbytes) for the frozen backbone
    LoRA rank     : 32   (alpha 32, dropout 0.0)
    targets       : all linear layers ("all-linear")
    trainable     : ~110M params (~1.5% of 7B)
    optimizer     : AdamW, lr 5e-4, constant schedule
    batch / accum : 16 with gradient accumulation to taste
    action head   : 256-bin discrete tokens (RT-2 style, see Chapter 4)

Usage (outside the test suite):
    python lora_openvla.py --dataset lerobot/svla_so100_pickplace --steps 2000
"""
from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Dependency detection
# ---------------------------------------------------------------------------


REQUIRED = ("transformers", "peft", "bitsandbytes", "torch")


def missing_dependencies() -> list[str]:
    """Return the subset of REQUIRED packages that are not importable."""
    return [pkg for pkg in REQUIRED if importlib.util.find_spec(pkg) is None]


@dataclass
class OpenVLALoRARecipe:
    """The canonical OpenVLA LoRA hyperparameters."""

    base_model: str = "openvla/openvla-7b"
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    target_modules: str = "all-linear"
    learning_rate: float = 5e-4
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 2000
    use_4bit: bool = True

    def describe(self) -> str:
        """Return a readable description of the recipe."""
        return (
            "OpenVLA-7B LoRA recipe\n"
            f"  base_model     = {self.base_model}\n"
            f"  rank / alpha   = {self.rank} / {self.alpha}\n"
            f"  targets        = {self.target_modules}\n"
            f"  4-bit quant    = {self.use_4bit}\n"
            f"  lr             = {self.learning_rate}\n"
            f"  batch / accum  = {self.batch_size} / {self.grad_accum_steps}\n"
            f"  max_steps      = {self.max_steps}\n"
            "  est. trainable = ~110M params (~1.5% of 7B)"
        )


# ---------------------------------------------------------------------------
# Model construction (only runs if dependencies are present)
# ---------------------------------------------------------------------------


def build_openvla_lora(recipe: OpenVLALoRARecipe):  # type: ignore[no-untyped-def]
    """Load OpenVLA-7B in 4-bit and wrap it with PEFT LoRA adapters.

    Returns the PEFT model. Raises RuntimeError with guidance if any required
    dependency is missing.
    """
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(
            "Cannot build OpenVLA LoRA: missing packages "
            f"{missing}. Install with: pip install {' '.join(missing)}"
        )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForVision2Seq, BitsAndBytesConfig

    quant_config = None
    if recipe.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[openvla] Loading {recipe.base_model} (4-bit={recipe.use_4bit})...")
    model = AutoModelForVision2Seq.from_pretrained(
        recipe.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=recipe.rank,
        lora_alpha=recipe.alpha,
        lora_dropout=recipe.dropout,
        target_modules=recipe.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 8: LoRA fine-tune OpenVLA-7B (reference)"
    )
    parser.add_argument("--dataset", default="lerobot/svla_so100_pickplace")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the recipe and dependency status, then exit")
    args = parser.parse_args()

    recipe = OpenVLALoRARecipe(rank=args.rank, max_steps=args.steps)
    print(recipe.describe())

    missing = missing_dependencies()
    if missing:
        print(f"\n[openvla] Missing dependencies: {missing}")
        print("[openvla] Install: pip install peft bitsandbytes transformers")
        print("[openvla] This script is a reference recipe; the runnable, "
              "from-scratch LoRA path is finetune_smolvla.py.")
        return

    if args.dry_run:
        print("\n[openvla] Dependencies present. Re-run without --dry-run to train.")
        return

    model = build_openvla_lora(recipe)
    print("[openvla] Model ready. Plug in a LeRobot DataLoader + Trainer to "
          "fine-tune (see README for the full loop).")
    del model


if __name__ == "__main__":
    main()
