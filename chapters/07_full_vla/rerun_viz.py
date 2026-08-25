"""Chapter 7: Full VLA -- Interactive Rerun Visualization.

Loads trained model, runs inference on validation episodes, and displays
camera frames alongside predicted vs ground-truth joint actions in Rerun.

Usage:
    python rerun_viz.py                        # default: epoch 100, 5 episodes
    python rerun_viz.py --epoch 200            # specific checkpoint
    python rerun_viz.py --num-episodes 10      # more episodes
    python rerun_viz.py --save output.rrd      # save for offline viewing

Requires: rerun-sdk, pyav (for video decoding)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch
from torch import Tensor

from config import SmolVLAConfig, DATASETS_SO100
from model import build_smolvla
from train import (
    CHECKPOINT_DIR,
    DEVICE,
    DTYPE,
    FlowMatchingVLA,
    EMAModel,
    NormStats,
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


def prepare_normalized_cache(
    cache: dict[str, Any],
) -> tuple[dict[str, Any], NormStats, NormStats]:
    """Normalize raw cached state/action tensors for model inference."""
    state_stats = compute_norm_stats(cache["states"])
    action_stats = compute_norm_stats(cache["actions"])

    normalized_cache = dict(cache)
    normalized_cache["states"] = state_stats.normalize(cache["states"])
    normalized_cache["actions"] = action_stats.normalize(cache["actions"])

    return normalized_cache, state_stats, action_stats


def load_model_and_data(
    epoch: int,
) -> tuple[
    FlowMatchingVLA,
    EMAModel,
    dict[str, Any],
    NormStats,
    NormStats,
    Tensor,
    Tensor,
]:
    """Load trained model, EMA weights, normalized cache, and norm stats."""
    # Build model
    config = SmolVLAConfig()
    model = build_smolvla(config)

    # Load checkpoint
    ckpt_name = f"vla_epoch_{epoch}.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "vla_so100_best.pt"
        print(f"[warn] epoch {epoch} not found, using best checkpoint")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.state_proj.load_state_dict(ckpt["trainable"]["state_proj"])
    model.action_expert.load_state_dict(ckpt["trainable"]["action_expert"])
    model.to(DEVICE)
    print(f"[model] Loaded {ckpt_path.name} (epoch {ckpt.get('epoch', '?')})")

    # Build EMA
    ema = EMAModel(model, decay=config.ema_decay)
    if "ema" in ckpt:
        ema.shadow_state_proj.load_state_dict(ckpt["ema"]["state_proj"])
        ema.shadow_action_expert.load_state_dict(ckpt["ema"]["action_expert"])

    # Load cache
    cache_path = CHECKPOINT_DIR / "vla_cache.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)

    # Load tasks
    tasks_path = CHECKPOINT_DIR / "vla_tasks.pt"
    tasks = torch.load(tasks_path, map_location="cpu", weights_only=False) if tasks_path.exists() else None
    cache["tasks"] = tasks

    cache, state_stats, action_stats = prepare_normalized_cache(cache)

    # Split
    train_mask, val_mask = split_by_episode(cache["episode_indices"])

    fm = FlowMatchingVLA(model, config)

    return fm, ema, cache, state_stats, action_stats, train_mask, val_mask


def decode_episode_frames(
    episode_idx: int,
    episode_indices: Tensor,
    datasets_config: list[dict],
) -> list[np.ndarray]:
    """Decode raw camera frames for a given episode from the LeRobot dataset.

    Returns list of HWC uint8 numpy images.
    """
    from lerobot.datasets import LeRobotDataset

    # Figure out which dataset and episode this corresponds to
    mask = episode_indices == episode_idx
    frame_count = mask.sum().item()

    # Load all datasets and find the right one by counting episodes
    episode_offset = 0
    for ds_cfg in datasets_config:
        repo_id = ds_cfg["repo_id"]
        try:
            dataset = LeRobotDataset(repo_id, video_backend="pyav")
        except Exception:
            continue

        num_eps = dataset.num_episodes
        local_ep = episode_idx - episode_offset

        if 0 <= local_ep < num_eps:
            # This is the right dataset
            image_key = ds_cfg["image_key"]
            sample = dataset[0]
            if image_key not in sample:
                available = [k for k in sample.keys() if "image" in k.lower()]
                image_key = available[0] if available else image_key

            # Find frame indices for this episode
            frames = []
            for i in range(len(dataset)):
                item = dataset[i]
                if item["episode_index"].item() == local_ep:
                    img_t = item[image_key]  # (3, H, W) float [0,1]
                    img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    frames.append(img_np)
                elif len(frames) > 0:
                    break  # Past this episode

            return frames[:frame_count]

        episode_offset += num_eps

    return []


def run_rerun_viz(
    epoch: int = 100,
    num_episodes: int = 5,
    save_path: str | None = None,
) -> None:
    """Main visualization: camera + GT vs predicted actions in Rerun."""

    print("=" * 60)
    print("  Chapter 7: Full VLA -- Rerun Visualization")
    print("=" * 60)

    fm, ema, cache, state_stats, action_stats, train_mask, val_mask = (
        load_model_and_data(epoch)
    )
    config = fm.config

    # Get validation episodes
    val_indices = torch.where(val_mask)[0]
    val_episodes = cache["episode_indices"][val_indices].unique().tolist()
    episodes_to_viz = val_episodes[:num_episodes]
    print(f"[viz] Visualizing {len(episodes_to_viz)} val episodes: {episodes_to_viz}")

    # Init Rerun
    spawn_viewer = save_path is None
    rr.init("ch07_full_vla", spawn=spawn_viewer)

    if save_path:
        rr.save(save_path)
        print(f"[viz] Saving to {save_path}")

    # Set up the viewer layout with a blueprint
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Process each episode
    for ep_num, ep_idx in enumerate(episodes_to_viz):
        ep_mask = cache["episode_indices"] == ep_idx
        ep_frame_indices = torch.where(ep_mask)[0]
        n_frames = len(ep_frame_indices)

        # Get task name
        task_name = "unknown"
        if cache.get("tasks") and len(cache["tasks"]) > ep_frame_indices[0].item():
            task_name = cache["tasks"][ep_frame_indices[0].item()]

        print(f"\n[ep {ep_idx}] {n_frames} frames | task: {task_name}")

        # Decode camera frames from dataset
        print(f"  Decoding video frames...")
        camera_frames = decode_episode_frames(
            ep_idx, cache["episode_indices"], DATASETS_SO100
        )
        has_camera = len(camera_frames) > 0
        if has_camera:
            print(f"  Got {len(camera_frames)} camera frames")
        else:
            print(f"  [warn] No camera frames available")

        # Run model inference for each frame
        print(f"  Running inference...")
        fm.model.eval()

        for i, global_idx in enumerate(ep_frame_indices):
            idx = global_idx.item()

            # Set timeline
            rr.set_time("frame", sequence=i)
            rr.set_time("episode", sequence=ep_num)

            # Log camera image
            if has_camera and i < len(camera_frames):
                rr.log(
                    f"episode_{ep_idx}/camera",
                    rr.Image(camera_frames[i]),
                )

            # Log task as text
            rr.log(
                f"episode_{ep_idx}/task",
                rr.TextDocument(f"Episode {ep_idx} | Frame {i}/{n_frames}\nTask: {task_name}"),
            )

            # Get GT action in raw units for comparison
            gt_action_norm = cache["actions"][idx]
            gt_action = action_stats.denormalize(gt_action_norm)

            # Get predicted action via flow matching
            vision_tokens = cache["vision_tokens"][idx].unsqueeze(0).to(DEVICE)
            lang_ids = cache["lang_token_ids"][idx].unsqueeze(0).to(DEVICE)
            lang_mask = cache["lang_attention_mask"][idx].unsqueeze(0).to(DEVICE)
            state_norm = cache["states"][idx]
            state = state_norm.unsqueeze(0).to(DEVICE)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE, enabled=(DEVICE.type == "cuda")):
                pred_chunk_norm = fm.sample(
                    vision_tokens, lang_ids, lang_mask, state, ema=ema
                )  # (1, chunk_size, action_dim)

            pred_action_norm = pred_chunk_norm[0, 0].cpu()  # first step of chunk
            pred_action = action_stats.denormalize(pred_action_norm)

            # Log GT and predicted actions per joint
            for j in range(config.action_dim):
                joint = JOINT_NAMES[j]
                rr.log(
                    f"episode_{ep_idx}/joints/{joint}/ground_truth",
                    rr.Scalars(gt_action[j].item()),
                )
                rr.log(
                    f"episode_{ep_idx}/joints/{joint}/predicted",
                    rr.Scalars(pred_action[j].item()),
                )
                error = abs(gt_action[j].item() - pred_action[j].item())
                rr.log(
                    f"episode_{ep_idx}/joints/{joint}/error",
                    rr.Scalars(error),
                )

            # Log overall MAE for this frame
            mae = (gt_action - pred_action).abs().mean().item()
            rr.log(
                f"episode_{ep_idx}/metrics/MAE",
                rr.Scalars(mae),
            )

            # Log state (denormalized)
            state_raw = state_stats.denormalize(state_norm)
            for j in range(config.state_dim):
                rr.log(
                    f"episode_{ep_idx}/state/{JOINT_NAMES[j]}",
                    rr.Scalars(state_raw[j].item()),
                )

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{n_frames}] MAE={mae:.1f} deg")

    print(f"\n{'=' * 60}")
    print(f"  Done! {'Saved to ' + save_path if save_path else 'Rerun viewer should be open.'}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun visualization for Ch07 Full VLA")
    parser.add_argument("--epoch", type=int, default=100, help="Checkpoint epoch to load")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of val episodes to visualize")
    parser.add_argument("--save", type=str, default=None, help="Save .rrd file instead of spawning viewer")
    args = parser.parse_args()

    run_rerun_viz(epoch=args.epoch, num_episodes=args.num_episodes, save_path=args.save)


if __name__ == "__main__":
    main()
