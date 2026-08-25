"""Chapter 8: plot LoRA fine-tuning training curves.

Reads the loss-history JSON files produced by `finetune_smolvla.py` and renders
a train/validation loss figure, one series per fine-tuning config. This is the
reproducible artifact that backs the results table in the README.

Usage:
    uv run python plot_results.py                     # auto-discover histories
    uv run python plot_results.py --history checkpoints/lora_so100_r8_history.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"


def load_histories(paths: list[Path]) -> list[dict[str, object]]:
    """Load history JSONs, skipping any that are missing."""
    runs: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            print(f"[plot] skip (not found): {path}")
            continue
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def label_for(run: dict[str, object]) -> str:
    """Build a short legend label from a run's config."""
    rank = run["rank"]
    expert = " + expert" if run.get("train_expert") else ""
    trainable = float(run["trainable_params"]) / 1e6
    return f"LoRA r={rank}{expert} ({trainable:.2f}M trainable)"


def plot(runs: list[dict[str, object]], out_path: Path) -> None:
    """Render train/val loss curves for each run into one figure."""
    if not runs:
        raise SystemExit("[plot] no histories to plot")

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis([i / max(len(runs), 1) for i in range(len(runs))])

    for run, color in zip(runs, colors):
        hist = run["history"]
        epochs = [h["epoch"] for h in hist]
        train = [h["train_loss"] for h in hist]
        val = [h["val_loss"] for h in hist]
        label = label_for(run)
        # Train loss has no epoch-0 entry (baseline is a val-only pass).
        train_epochs = [e for e, t in zip(epochs, train) if t == t]  # drop NaN
        train_vals = [t for t in train if t == t]
        ax_train.plot(train_epochs, train_vals, color=color, label=label)
        ax_val.plot(epochs, val, color=color, label=label)

        # Mark the epoch-0 baseline (frozen backbone) and the best val point.
        baseline_val = next((h["val_loss"] for h in hist if h["epoch"] == 0), None)
        if baseline_val is not None:
            ax_val.axhline(
                baseline_val, color=color, linestyle=":", alpha=0.7,
                label="Ch07 frozen-backbone baseline",
            )
        post = [h for h in hist if h["epoch"] >= 1]
        if post:
            best = min(post, key=lambda h: h["val_loss"])
            ax_val.scatter(
                [best["epoch"]], [best["val_loss"]], color=color, s=80,
                marker="*", zorder=5, label=f"best (epoch {best['epoch']})",
            )

    ax_train.set_title("Train loss (flow-matching MSE)")
    ax_val.set_title("Validation loss (flow-matching MSE)")
    for ax in (ax_train, ax_val):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(
        "Chapter 8: LoRA fine-tuning of the Ch07 SmolVLA backbone (SO100)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def summarize(runs: list[dict[str, object]]) -> None:
    """Print a markdown results table for pasting into the README."""
    print("\n| Config | Trainable | Best val loss | Epochs | s/epoch |")
    print("|--------|-----------|---------------|--------|---------|")
    for run in runs:
        hist = run["history"]
        s_per_epoch = sum(h.get("time_s", 0.0) for h in hist) / max(len(hist), 1)
        expert = " + expert" if run.get("train_expert") else ""
        trainable = float(run["trainable_params"]) / 1e6
        print(
            f"| LoRA r={run['rank']}{expert} | {trainable:.2f}M | "
            f"{float(run['best_val_loss']):.4f} | {run['epochs']} | "
            f"{s_per_epoch:.1f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Ch08 LoRA training curves")
    parser.add_argument(
        "--history",
        type=Path,
        nargs="*",
        default=None,
        help="History JSON files (default: auto-discover in checkpoints/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FIGURE_DIR / "ch08_lora_training_curves.png",
    )
    args = parser.parse_args()

    if args.history:
        paths = list(args.history)
    else:
        paths = sorted(CHECKPOINT_DIR.glob("lora_*_history.json"))
        print(f"[plot] discovered {len(paths)} history file(s)")

    runs = load_histories(paths)
    plot(runs, args.out)
    summarize(runs)


if __name__ == "__main__":
    main()
