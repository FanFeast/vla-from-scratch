"""Chapter 8: paired verification of the LoRA fine-tuning effect.

The flow-matching validation loss is stochastic (random noise + timestep per
batch), so a single val pass is noisy and "best-of-50-epochs" selection biases
the number downward. This script measures the *real* effect of the r=8 LoRA
adapter with a paired comparison:

    for each seed s:
        seed(s); baseline_loss[s] = val(frozen backbone, LoRA B=0)
        seed(s); adapted_loss[s]  = val(same model + trained LoRA weights)

Because both models see identical noise/timestep draws for a given seed, the
per-seed difference isolates the adapter's contribution from sampling noise.

Usage:
    uv run python verify_baseline.py --seeds 8
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CH07_DIR = Path(__file__).resolve().parent.parent / "07_full_vla"
sys.path.insert(0, str(CH07_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune_smolvla import (  # noqa: E402
    CH07_CHECKPOINTS,
    CHECKPOINT_DIR,
    DEVICE,
    build_lora_vla,
    evaluate_val_loss,
)
from lora import LoRAConfig, load_lora_state_dict  # noqa: E402


def build_val_loader(model, config, batch_size: int) -> DataLoader:
    """Replicate the fine-tuning validation split and normalization."""
    from train import (  # noqa: E402
        VLADataset,
        compute_norm_stats,
        extract_and_cache,
        split_by_episode,
    )
    from config import PRESETS  # noqa: E402

    preset = PRESETS["so100"]
    model.vision_encoder.to(DEVICE)
    model.connector.to(DEVICE)
    data = extract_and_cache(
        model, preset["datasets"], config, CH07_CHECKPOINTS, skip=True
    )
    model.vision_encoder.cpu()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    state_stats = compute_norm_stats(data["states"])
    action_stats = compute_norm_stats(data["actions"])
    norm_states = state_stats.normalize(data["states"])
    norm_actions = action_stats.normalize(data["actions"])
    _train_mask, val_mask = split_by_episode(
        data["episode_indices"], config.val_ratio
    )

    val_ds = VLADataset(
        vision_tokens=data["vision_tokens"][val_mask],
        lang_token_ids=data["lang_token_ids"][val_mask],
        lang_attention_mask=data["lang_attention_mask"][val_mask],
        states=norm_states[val_mask],
        actions=norm_actions[val_mask],
        episode_indices=data["episode_indices"][val_mask],
        chunk_size=config.chunk_size,
    )
    return DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )


def run(seeds: int, rank: int) -> None:
    """Compute paired baseline vs LoRA-adapted validation losses."""
    from config import PRESETS, SmolVLAConfig  # noqa: E402
    from train import FlowMatchingVLA  # noqa: E402

    preset = PRESETS["so100"]
    config = SmolVLAConfig(
        action_dim=preset["action_dim"],
        state_dim=preset["state_dim"],
        chunk_size=preset["chunk_size"],
        batch_size=preset["batch_size"],
    )
    lora_config = LoRAConfig(r=rank, alpha=2 * rank, target_modules=("q_proj", "v_proj"))

    # build_lora_vla warm-starts the expert and injects zero-init LoRA (B=0),
    # so this model IS the Ch07 frozen-backbone policy at construction time.
    model = build_lora_vla(config, lora_config, warm_start=True)
    val_loader = build_val_loader(model, config, config.batch_size)

    model.vlm.to(DEVICE)
    model.connector.to(DEVICE)
    model.state_proj.to(DEVICE)
    model.action_expert.to(DEVICE)
    trainer = FlowMatchingVLA(model, config)

    seed_list = list(range(seeds))

    # Baseline: LoRA B=0 (no adaptation).
    baseline = []
    for s in seed_list:
        torch.manual_seed(s)
        baseline.append(evaluate_val_loss(trainer, val_loader))

    # Load the trained adapter and repeat with identical seeds.
    ckpt = torch.load(
        CHECKPOINT_DIR / f"lora_so100_r{rank}.pt",
        map_location="cpu",
        weights_only=False,
    )
    load_lora_state_dict(model, ckpt["lora"])
    adapted = []
    for s in seed_list:
        torch.manual_seed(s)
        adapted.append(evaluate_val_loss(trainer, val_loader))

    diffs = [b - a for b, a in zip(baseline, adapted)]  # positive = LoRA helps

    def fmt(xs: list[float]) -> str:
        mean = statistics.mean(xs)
        std = statistics.pstdev(xs)
        return f"{mean:.5f} +/- {std:.5f}"

    print("\n" + "=" * 60)
    print(f"  Paired verification (r={rank}, {seeds} seeds)")
    print("=" * 60)
    print(f"  baseline (frozen backbone): {fmt(baseline)}")
    print(f"  LoRA-adapted             : {fmt(adapted)}")
    print(f"  improvement (base-adapt) : {fmt(diffs)}")
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.pstdev(diffs)
    rel = 100.0 * mean_diff / statistics.mean(baseline)
    sig = abs(mean_diff) > 2 * std_diff if std_diff > 0 else False
    print(f"  relative improvement     : {rel:+.2f}%")
    print(
        f"  verdict                  : "
        f"{'gap exceeds 2x paired std -> real' if sig else 'within noise (<=2x paired std)'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired LoRA verification")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    run(args.seeds, args.rank)


if __name__ == "__main__":
    main()
