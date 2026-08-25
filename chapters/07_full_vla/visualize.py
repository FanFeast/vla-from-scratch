"""Chapter 7: Full VLA — Visualization Script.

Generates publication-quality figures showing:
1. Training curves (train/val loss over epochs)
2. Predicted vs ground truth action trajectories for validation episodes
3. Per-joint prediction comparison across tasks
4. Flow matching denoising process visualization

Usage:
    python visualize.py
    python visualize.py --epoch 100    # Use specific checkpoint
    python visualize.py --num-episodes 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

from config import SmolVLAConfig
from model import build_smolvla
from train import (
    CHECKPOINT_DIR,
    FlowMatchingVLA,
    VLADataset,
    EMAModel,
    compute_norm_stats,
    split_by_episode,
)

JOINT_NAMES = [
    "Base rotate",
    "Shoulder",
    "Elbow",
    "Wrist pitch",
    "Wrist roll",
    "Gripper",
]

TASK_COLORS = {
    "Pick up the cube and place it in the box.": "#2196F3",
    "Put the red cube on top of the blue cube.": "#FF9800",
    "Put the red cube in the right box and the blue cube in the left box.": "#4CAF50",
}

TASK_SHORT = {
    "Pick up the cube and place it in the box.": "Pick & Place",
    "Put the red cube on top of the blue cube.": "Stacking",
    "Put the red cube in the right box and the blue cube in the left box.": "Sorting",
}

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def load_model_and_data(
    epoch: int | None = None,
) -> tuple[FlowMatchingVLA, EMAModel, VLADataset, dict, object, object, list[str]]:
    """Load model, checkpoint, and validation data."""
    config = SmolVLAConfig()
    model = build_smolvla(config)

    if epoch is not None:
        ckpt_path = CHECKPOINT_DIR / f"vla_epoch_{epoch}.pt"
    else:
        ckpt_path = CHECKPOINT_DIR / "vla_so100_best.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.state_proj.load_state_dict(ckpt["trainable"]["state_proj"])
    model.action_expert.load_state_dict(ckpt["trainable"]["action_expert"])
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}")

    ema = EMAModel(model, decay=config.ema_decay)
    ema.load_state_dict(ckpt["ema"])

    # Load data
    data = torch.load(CHECKPOINT_DIR / "vla_cache.pt", map_location="cpu", weights_only=True)
    tasks = torch.load(CHECKPOINT_DIR / "vla_tasks.pt", map_location="cpu", weights_only=False)

    state_stats = compute_norm_stats(data["states"])
    action_stats = compute_norm_stats(data["actions"])
    norm_states = state_stats.normalize(data["states"])
    norm_actions = action_stats.normalize(data["actions"])

    _, val_mask = split_by_episode(data["episode_indices"], config.val_ratio)

    val_ds = VLADataset(
        vision_tokens=data["vision_tokens"][val_mask],
        lang_token_ids=data["lang_token_ids"][val_mask],
        lang_attention_mask=data["lang_attention_mask"][val_mask],
        states=norm_states[val_mask],
        actions=norm_actions[val_mask],
        episode_indices=data["episode_indices"][val_mask],
        chunk_size=config.chunk_size,
    )

    val_tasks = [tasks[i] for i in range(len(tasks)) if val_mask[i]]
    trainer = FlowMatchingVLA(model, config)

    return trainer, ema, val_ds, data, state_stats, action_stats, val_tasks


def plot_training_curves() -> None:
    """Plot training curves from all periodic checkpoints."""
    epochs_list = []
    train_losses = []
    val_losses = []

    for ep in [25, 50, 75, 100, 125, 150, 175, 200]:
        path = CHECKPOINT_DIR / f"vla_epoch_{ep}.pt"
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            epochs_list.append(ep)
            val_losses.append(ckpt["val_loss"])

    # Also read the full log if available
    log_path = Path(__file__).parent / "training_output.log"
    all_epochs = []
    all_train = []
    all_val = []
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if "Epoch" in line and "train=" in line and "val=" in line:
                    parts = line.strip().split("|")
                    ep = int(parts[0].split("Epoch")[1].split("/")[0].strip())
                    train_loss = float(parts[1].split("=")[1].strip())
                    val_loss = float(parts[2].split("=")[1].strip())
                    all_epochs.append(ep)
                    all_train.append(train_loss)
                    all_val.append(val_loss)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if all_epochs:
        axes[0].plot(all_epochs, all_train, "b-", alpha=0.8, linewidth=1.5, label="Train")
        axes[0].plot(all_epochs, all_val, "r-", alpha=0.8, linewidth=1.5, label="Val")
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Flow Matching Loss (MSE)", fontsize=12)
        axes[0].set_title("Training Curves", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale("log")

        # Mark best epoch
        best_idx = np.argmin(all_val)
        axes[0].axvline(all_epochs[best_idx], color="green", linestyle="--", alpha=0.5)
        axes[0].annotate(
            f"Best val: {all_val[best_idx]:.3f}\n(epoch {all_epochs[best_idx]})",
            xy=(all_epochs[best_idx], all_val[best_idx]),
            xytext=(all_epochs[best_idx] + 15, all_val[best_idx] * 1.5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green",
        )

    # Learning rate schedule
    if all_epochs:
        import math

        config = SmolVLAConfig()
        total_steps = config.epochs * (65375 // config.batch_size + 1)
        steps = np.arange(total_steps)
        lrs = []
        for s in steps:
            if s < config.warmup_steps:
                lr = s / max(config.warmup_steps, 1) * config.lr
            else:
                progress = (s - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
                lr = 0.5 * (1.0 + math.cos(math.pi * progress)) * config.lr
            lrs.append(lr)

        steps_per_epoch = 65375 // config.batch_size + 1
        ep_x = np.arange(1, config.epochs + 1)
        ep_lrs = [lrs[min(int(e * steps_per_epoch), len(lrs) - 1)] for e in ep_x]

        axes[1].plot(ep_x, ep_lrs, "purple", linewidth=1.5)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Learning Rate", fontsize=12)
        axes[1].set_title("Cosine LR Schedule", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    plt.tight_layout()
    out_path = OUT_DIR / "ch07_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {out_path}")


def plot_action_trajectories(
    trainer: FlowMatchingVLA,
    ema: EMAModel,
    val_ds: VLADataset,
    action_stats: object,
    val_tasks: list[str],
    num_episodes: int = 6,
) -> None:
    """Plot predicted vs GT action trajectories for validation episodes."""
    config = trainer.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = trainer.model
    model.vision_encoder.cpu()
    model.vlm.to(device)
    model.connector.to(device)
    model.state_proj.to(device)
    model.action_expert.to(device)
    ema.to(device)

    # Find episode boundaries in val set
    ep_ids = val_ds.episode_indices.unique(sorted=True)
    selected_eps = ep_ids[:num_episodes].tolist()

    fig, axes = plt.subplots(
        num_episodes, 6, figsize=(24, 4 * num_episodes),
        sharex="row",
    )
    if num_episodes == 1:
        axes = axes[np.newaxis, :]

    for row, ep_id in enumerate(selected_eps):
        ep_mask = val_ds.episode_indices == ep_id
        ep_indices = torch.where(ep_mask)[0]

        # Find task for this episode
        task = val_tasks[ep_indices[0].item()] if ep_indices[0].item() < len(val_tasks) else "Unknown"
        task_short = TASK_SHORT.get(task, task[:30])
        task_color = TASK_COLORS.get(task, "#666666")

        # Sample predictions for every 5th frame
        stride = max(1, len(ep_indices) // 30)
        sample_indices = ep_indices[::stride]

        gt_actions_raw = []
        pred_actions_raw = []
        timesteps = []

        for idx in sample_indices:
            sample = val_ds[idx.item()]
            vis = sample["vision_tokens"].unsqueeze(0)
            lang_ids = sample["lang_token_ids"].unsqueeze(0)
            lang_mask = sample["lang_attention_mask"].unsqueeze(0)
            state = sample["state"].unsqueeze(0)
            gt = sample["actions"]  # (K, 6) normalized

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = trainer.sample(vis, lang_ids, lang_mask, state, ema=ema)

            # Denormalize
            gt_raw = action_stats.denormalize(gt)
            pred_raw = action_stats.denormalize(pred.squeeze(0).cpu())

            gt_actions_raw.append(gt_raw[0])  # First step of chunk
            pred_actions_raw.append(pred_raw[0])
            timesteps.append(idx.item() - ep_indices[0].item())

        gt_arr = torch.stack(gt_actions_raw).numpy()
        pred_arr = torch.stack(pred_actions_raw).numpy()
        t = np.array(timesteps)

        for col in range(6):
            ax = axes[row, col]
            ax.plot(t, gt_arr[:, col], "b-", linewidth=1.5, alpha=0.8, label="Ground Truth")
            ax.plot(t, pred_arr[:, col], "r--", linewidth=1.5, alpha=0.8, label="Predicted")
            ax.fill_between(
                t,
                gt_arr[:, col],
                pred_arr[:, col],
                alpha=0.15,
                color="red",
            )

            if row == 0:
                ax.set_title(JOINT_NAMES[col], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(
                    f"Ep {ep_id}\n({task_short})",
                    fontsize=10,
                    color=task_color,
                    fontweight="bold",
                )
            if row == num_episodes - 1:
                ax.set_xlabel("Frame", fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=8)

            if row == 0 and col == 5:
                ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(
        "Chapter 7: Full VLA — Predicted vs Ground Truth Actions\n"
        "(EMA model, 10 Euler steps, SO100 6-DOF joints)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "ch07_action_trajectories.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {out_path}")


def plot_per_joint_error(
    trainer: FlowMatchingVLA,
    ema: EMAModel,
    val_ds: VLADataset,
    action_stats: object,
    val_tasks: list[str],
    num_samples: int = 200,
) -> None:
    """Plot per-joint MAE across tasks."""
    config = trainer.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = trainer.model
    model.vlm.to(device)
    model.connector.to(device)
    model.state_proj.to(device)
    model.action_expert.to(device)
    ema.to(device)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(val_ds), size=min(num_samples, len(val_ds)), replace=False)

    per_task_errors: dict[str, list] = {}

    for idx in indices:
        sample = val_ds[idx]
        task = val_tasks[idx] if idx < len(val_tasks) else "Unknown"
        vis = sample["vision_tokens"].unsqueeze(0)
        lang_ids = sample["lang_token_ids"].unsqueeze(0)
        lang_mask = sample["lang_attention_mask"].unsqueeze(0)
        state = sample["state"].unsqueeze(0)
        gt = sample["actions"]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = trainer.sample(vis, lang_ids, lang_mask, state, ema=ema)

        gt_raw = action_stats.denormalize(gt)
        pred_raw = action_stats.denormalize(pred.squeeze(0).cpu())

        # Per-joint MAE for first action in chunk
        joint_mae = (gt_raw[0] - pred_raw[0]).abs().numpy()

        if task not in per_task_errors:
            per_task_errors[task] = []
        per_task_errors[task].append(joint_mae)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per-joint MAE bar chart grouped by task
    ax = axes[0]
    task_names = list(per_task_errors.keys())
    n_tasks = len(task_names)
    x = np.arange(6)
    width = 0.25

    for i, task in enumerate(task_names):
        errs = np.array(per_task_errors[task])
        means = errs.mean(axis=0)
        stds = errs.std(axis=0)
        short = TASK_SHORT.get(task, task[:20])
        color = TASK_COLORS.get(task, f"C{i}")
        offset = (i - n_tasks / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=short, color=color, alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("MAE (joint degrees)", fontsize=12)
    ax.set_title("Per-Joint Error by Task", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    # Overall error distribution
    ax = axes[1]
    all_errors = []
    all_labels = []
    for task, errs in per_task_errors.items():
        total_mae = np.array(errs).mean(axis=1)  # average across joints
        all_errors.append(total_mae)
        all_labels.append(TASK_SHORT.get(task, task[:20]))

    colors = [TASK_COLORS.get(t, f"C{i}") for i, t in enumerate(per_task_errors.keys())]
    bp = ax.boxplot(
        all_errors,
        labels=all_labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Mean Absolute Error (joint degrees)", fontsize=12)
    ax.set_title("Error Distribution by Task", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle(
        "Chapter 7: Full VLA — Per-Joint Prediction Accuracy",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "ch07_per_joint_error.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {out_path}")


def plot_denoising_process(
    trainer: FlowMatchingVLA,
    ema: EMAModel,
    val_ds: VLADataset,
    action_stats: object,
) -> None:
    """Visualize the flow matching denoising trajectory from noise to action."""
    config = trainer.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = trainer.model
    model.vlm.to(device)
    model.connector.to(device)
    model.state_proj.to(device)
    model.action_expert.to(device)
    ema.to(device)

    import copy

    # Temporarily apply EMA weights
    orig_sp = copy.deepcopy(model.state_proj.state_dict())
    orig_ae = copy.deepcopy(model.action_expert.state_dict())
    ema.apply_to(model)
    model.eval()

    # Pick a sample
    sample = val_ds[500]
    vis = sample["vision_tokens"].unsqueeze(0).to(device)
    lang_ids = sample["lang_token_ids"].unsqueeze(0).to(device)
    lang_mask = sample["lang_attention_mask"].unsqueeze(0).to(device)
    state = sample["state"].unsqueeze(0).to(device)
    gt = sample["actions"]  # (K, 6) normalized

    # Run denoising with intermediate steps captured
    B = 1
    K = config.chunk_size
    steps = 10
    dt = 1.0 / steps

    torch.manual_seed(42)
    x = torch.randn(B, K, config.action_dim, device=device)

    intermediates = [action_stats.denormalize(x.squeeze(0).cpu()).numpy()]
    t_values = [0.0]

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(steps):
            t = torch.full((B,), i * dt, device=device)
            v = model(vis, lang_ids, lang_mask, state, x, t)
            x = x + v * dt
            intermediates.append(action_stats.denormalize(x.squeeze(0).cpu()).numpy())
            t_values.append((i + 1) * dt)

    gt_raw = action_stats.denormalize(gt).numpy()

    # Restore original weights
    model.state_proj.load_state_dict(orig_sp)
    model.action_expert.load_state_dict(orig_ae)

    # Plot denoising trajectory for each joint
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    cmap = plt.cm.viridis
    for joint in range(6):
        ax = axes[joint]
        for step_i, (inter, t_val) in enumerate(zip(intermediates, t_values)):
            color = cmap(step_i / len(intermediates))
            alpha = 0.3 + 0.7 * (step_i / len(intermediates))
            chunk_x = np.arange(K)
            ax.plot(
                chunk_x, inter[:, joint],
                color=color, alpha=alpha, linewidth=1.0,
            )

        # Ground truth
        ax.plot(
            np.arange(K), gt_raw[:, joint],
            "k-", linewidth=2.5, label="Ground Truth", zorder=10,
        )
        # Final prediction
        ax.plot(
            np.arange(K), intermediates[-1][:, joint],
            "r--", linewidth=2.0, label="Final Prediction", zorder=9,
        )

        ax.set_title(JOINT_NAMES[joint], fontsize=12, fontweight="bold")
        ax.set_xlabel("Chunk step", fontsize=10)
        ax.set_ylabel("Joint value", fontsize=10)
        ax.grid(True, alpha=0.2)
        if joint == 0:
            ax.legend(fontsize=9)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Denoising progress (t=0: noise, t=1: prediction)", fontsize=11)

    plt.suptitle(
        "Chapter 7: Flow Matching Denoising Process\n"
        "10 Euler steps from noise to action chunk (10 steps x 6 DOF)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    out_path = OUT_DIR / "ch07_denoising_process.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {out_path}")


def plot_checkpoint_comparison() -> None:
    """Bar chart comparing different checkpoints."""
    # Hardcoded from our evaluation
    epochs = [25, 50, 75, 100, 125, 150, 200]
    mae_norm = [3.36, 2.98, 2.56, 2.22, 2.51, 2.43, 2.41]
    val_loss = [0.20, 0.24, 1.14, 0.80, 0.44, 0.48, 0.59]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#4CAF50" if e == 100 else "#2196F3" for e in epochs]

    ax = axes[0]
    bars = ax.bar(range(len(epochs)), mae_norm, color=colors, alpha=0.8)
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([f"Ep {e}" for e in epochs], fontsize=10)
    ax.set_ylabel("Normalized MAE", fontsize=12)
    ax.set_title("Action Prediction Error by Checkpoint", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    for bar, v in zip(bars, mae_norm):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{v:.2f}", ha="center", fontsize=9,
        )

    ax = axes[1]
    ax.plot(epochs, val_loss, "ro-", linewidth=2, markersize=8, alpha=0.8)
    ax.fill_between(epochs, val_loss, alpha=0.15, color="red")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Validation Loss Over Training", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axvline(100, color="green", linestyle="--", alpha=0.5, label="Best MAE (ep 100)")
    ax.legend(fontsize=10)

    plt.suptitle(
        "Chapter 7: Full VLA — Checkpoint Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "ch07_checkpoint_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 7 Visualizations")
    parser.add_argument("--epoch", type=int, default=100, help="Checkpoint epoch to use")
    parser.add_argument("--num-episodes", type=int, default=6, help="Episodes for trajectory plot")
    parser.add_argument("--num-samples", type=int, default=200, help="Samples for error analysis")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Chapter 7: Full VLA — Generating Visualizations")
    print(f"{'=' * 60}\n")

    # 1. Training curves (no model needed)
    print("[1/4] Training curves...")
    plot_training_curves()

    # 2. Checkpoint comparison (no model needed)
    print("\n[2/4] Checkpoint comparison...")
    plot_checkpoint_comparison()

    # 3-4. Model-based visualizations
    print(f"\n[3/4] Loading model (epoch {args.epoch})...")
    trainer, ema, val_ds, data, state_stats, action_stats, val_tasks = load_model_and_data(args.epoch)

    print(f"\n[3/4] Action trajectories ({args.num_episodes} episodes)...")
    plot_action_trajectories(trainer, ema, val_ds, action_stats, val_tasks, args.num_episodes)

    print(f"\n[4/4] Denoising process...")
    plot_denoising_process(trainer, ema, val_ds, action_stats)

    # Skip per-joint if fast mode
    if args.num_samples > 0:
        print(f"\n[bonus] Per-joint error analysis ({args.num_samples} samples)...")
        plot_per_joint_error(trainer, ema, val_ds, action_stats, val_tasks, args.num_samples)

    print(f"\n{'=' * 60}")
    print(f"  All visualizations saved to {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
