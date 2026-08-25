"""Chapter 7 policy reality check.

Training loss says the action expert fit the data; it does not show whether the
policy would behave well on a robot. This script does an honest **offline
replay** on held-out validation episodes: for the same recorded camera frame,
language command, and robot state, it compares

  1. the ground-truth demonstration action, against
  2. what our trained Ch07 policy predicts (first action of each flow-matching
     chunk),

and reports per-joint error. It also has a gated slot for an official larger
SmolVLA checkpoint: if that policy cannot be run on our cached tensors, the
report labels the section a *reference*, not a measured comparison.

This is still offline replay, not closed-loop execution -- it shows action
prediction quality on recorded states, not whether errors compound over a real
rollout (that is Chapter 9's closed-loop story).

Usage:
    uv run python policy_reality_check.py --epoch 100 --num-episodes 1 --stride 50 \
        --save-figure figures/ch07_policy_reality_check.png
"""
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from train import CHECKPOINT_DIR  # noqa: E402

LargeVLAStatus = Literal["not_requested", "ran", "reference_only", "failed"]

JOINT_NAMES = [
    "Base rotate",
    "Shoulder",
    "Elbow",
    "Wrist pitch",
    "Wrist roll",
    "Gripper",
]


@dataclass(frozen=True)
class LargeVLAResult:
    """Outcome of the optional large SmolVLA comparison path."""

    status: LargeVLAStatus
    model_id: str
    message: str


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested, no model/download needed)
# ---------------------------------------------------------------------------


def select_episode_ids(
    episode_indices: Tensor,
    mask: Tensor,
    num_episodes: int,
) -> list[int]:
    """Return the first validation episode IDs in dataset order."""
    masked_episode_ids = episode_indices[mask]
    unique_ids = masked_episode_ids.unique(sorted=True)
    return [int(ep.item()) for ep in unique_ids[:num_episodes]]


def sample_episode_frame_indices(
    episode_indices: Tensor,
    episode_id: int,
    stride: int,
) -> Tensor:
    """Return global frame indices for one episode, sampled by stride."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    indices = torch.where(episode_indices == episode_id)[0]
    return indices[::stride]


def compute_action_metrics(pred_actions: Tensor, gt_actions: Tensor) -> dict[str, object]:
    """Compute action prediction errors in the units of the provided tensors."""
    abs_error = (pred_actions - gt_actions).abs()
    return {
        "mae": float(abs_error.mean().item()),
        "per_joint_mae": abs_error.mean(dim=0).cpu(),
        "max_abs_error": float(abs_error.max().item()),
    }


# ---------------------------------------------------------------------------
# Ch07 policy loading + replay prediction
# ---------------------------------------------------------------------------


def load_ch07_policy(epoch: int):  # type: ignore[no-untyped-def]
    """Load the Ch07 model, EMA weights, cached data, and normalization stats."""
    from config import SmolVLAConfig
    from model import build_smolvla
    from train import EMAModel, FlowMatchingVLA, compute_norm_stats

    config = SmolVLAConfig()
    model = build_smolvla(config)

    ckpt_path = CHECKPOINT_DIR / f"vla_epoch_{epoch}.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "vla_so100_best.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.state_proj.load_state_dict(ckpt["trainable"]["state_proj"])
    model.action_expert.load_state_dict(ckpt["trainable"]["action_expert"])

    ema = EMAModel(model, decay=config.ema_decay)
    ema.load_state_dict(ckpt["ema"])

    cache = torch.load(
        CHECKPOINT_DIR / "vla_cache.pt", map_location="cpu", weights_only=True
    )
    tasks_path = CHECKPOINT_DIR / "vla_tasks.pt"
    if tasks_path.exists():
        cache["tasks"] = torch.load(tasks_path, map_location="cpu", weights_only=False)

    state_stats = compute_norm_stats(cache["states"])
    action_stats = compute_norm_stats(cache["actions"])
    cache["states"] = state_stats.normalize(cache["states"])
    cache["actions"] = action_stats.normalize(cache["actions"])

    trainer = FlowMatchingVLA(model, config)
    return trainer, ema, state_stats, action_stats, cache, ckpt_path.name


@torch.no_grad()
def predict_ch07_first_actions(
    trainer,  # type: ignore[no-untyped-def]
    ema,  # type: ignore[no-untyped-def]
    cache: dict[str, object],
    frame_indices: Tensor,
    action_stats,  # type: ignore[no-untyped-def]
) -> tuple[Tensor, Tensor]:
    """Predict the first action in each action chunk for selected frames."""
    from train import DEVICE, DTYPE

    model = trainer.model
    model.vision_encoder.cpu()
    model.vlm.to(DEVICE)
    model.connector.to(DEVICE)
    model.state_proj.to(DEVICE)
    model.action_expert.to(DEVICE)
    ema.to(DEVICE)
    model.eval()

    pred_raw: list[Tensor] = []
    gt_raw: list[Tensor] = []

    for idx_tensor in frame_indices:
        idx = int(idx_tensor.item())
        vision_tokens = cache["vision_tokens"][idx].unsqueeze(0).to(DEVICE)
        lang_ids = cache["lang_token_ids"][idx].unsqueeze(0).to(DEVICE)
        lang_mask = cache["lang_attention_mask"][idx].unsqueeze(0).to(DEVICE)
        state = cache["states"][idx].unsqueeze(0).to(DEVICE)

        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(DEVICE.type == "cuda")):
            pred_chunk_norm = trainer.sample(
                vision_tokens, lang_ids, lang_mask, state, ema=ema
            )

        pred_action_norm = pred_chunk_norm[0, 0].float().cpu()
        gt_action_norm = cache["actions"][idx].cpu()
        pred_raw.append(action_stats.denormalize(pred_action_norm))
        gt_raw.append(action_stats.denormalize(gt_action_norm))

    return torch.stack(pred_raw), torch.stack(gt_raw)


# ---------------------------------------------------------------------------
# Large SmolVLA reference (gated)
# ---------------------------------------------------------------------------


def attempt_large_vla_reference(
    model_id: str,
    try_import: bool = True,
) -> LargeVLAResult:
    """Detect whether an official SmolVLA policy can be loaded on our tensors."""
    if not model_id:
        return LargeVLAResult(
            status="not_requested",
            model_id="",
            message="No official/full SmolVLA checkpoint was requested for this run.",
        )

    if not try_import:
        return LargeVLAResult(
            status="reference_only",
            model_id=model_id,
            message="Large SmolVLA loading was skipped by configuration.",
        )

    try:
        importlib.import_module("lerobot.policies.smolvla")
    except ImportError:
        return LargeVLAResult(
            status="reference_only",
            model_id=model_id,
            message=(
                "Could not import LeRobot policy loader for SmolVLA in this environment. "
                "Use this section as a reference comparison only."
            ),
        )

    return LargeVLAResult(
        status="reference_only",
        model_id=model_id,
        message=(
            "LeRobot SmolVLA policy code is importable, but direct inference has not "
            "been wired to Ch07 cached tensors (different observation format). Treat "
            "this as a reference slot, not a measured comparison."
        ),
    )


# ---------------------------------------------------------------------------
# Reporting: static figure + Rerun recording
# ---------------------------------------------------------------------------


def save_reality_check_figure(
    out_path: Path,
    task_name: str,
    frame_numbers: Tensor,
    pred_actions: Tensor,
    gt_actions: Tensor,
    metrics: dict[str, object],
    large_vla: LargeVLAResult,
) -> None:
    """Save a static summary of intended behavior vs current Ch07 predictions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    axes = axes.flatten()
    x = frame_numbers.cpu().numpy()

    for joint_idx, ax in enumerate(axes):
        ax.plot(x, gt_actions[:, joint_idx].numpy(), "b-", label="Ground truth")
        ax.plot(x, pred_actions[:, joint_idx].numpy(), "r--", label="Ch07 policy")
        ax.set_title(JOINT_NAMES[joint_idx])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Joint value")
        ax.grid(True, alpha=0.25)
        if joint_idx == 0:
            ax.legend()

    fig.suptitle(
        "Ch07 Policy Reality Check\n"
        f"Task: {task_name}\n"
        f"Ch07 MAE: {metrics['mae']:.2f} raw joint units | "
        f"Large SmolVLA: {large_vla.status} ({large_vla.model_id or 'none'})",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.5, 0.01, large_vla.message, ha="center", fontsize=10, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_rerun_reality_check(
    out_path: Path,
    episode_id: int,
    task_name: str,
    frame_indices: Tensor,
    pred_actions: Tensor,
    gt_actions: Tensor,
    large_vla: LargeVLAResult,
) -> None:
    """Save a Rerun recording with GT vs predicted joint actions."""
    import rerun as rr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("ch07_policy_reality_check", spawn=False)
    rr.save(str(out_path))

    rr.log(
        "summary/task",
        rr.TextDocument(
            f"Episode {episode_id}\n"
            f"Task: {task_name}\n"
            f"Large SmolVLA status: {large_vla.status}\n"
            f"{large_vla.message}"
        ),
        static=True,
    )

    for local_i, global_idx in enumerate(frame_indices.tolist()):
        rr.set_time("frame", sequence=local_i)
        rr.log("summary/global_frame", rr.Scalars(global_idx))
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            gt = float(gt_actions[local_i, joint_idx].item())
            pred = float(pred_actions[local_i, joint_idx].item())
            rr.log(f"joints/{joint_name}/ground_truth", rr.Scalars(gt))
            rr.log(f"joints/{joint_name}/ch07_policy", rr.Scalars(pred))
            rr.log(f"joints/{joint_name}/abs_error", rr.Scalars(abs(gt - pred)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Chapter 7 policy reality check")
    parser.add_argument("--epoch", type=int, default=100, help="Ch07 checkpoint epoch")
    parser.add_argument("--num-episodes", type=int, default=3, help="Val episodes to scan")
    parser.add_argument("--stride", type=int, default=5, help="Evaluate every Nth frame")
    parser.add_argument("--save-rrd", type=Path, default=None, help="Rerun .rrd output path")
    parser.add_argument("--save-figure", type=Path, default=None, help="Summary PNG path")
    parser.add_argument(
        "--large-vla-model",
        type=str,
        default="",
        help="Optional official SmolVLA model id, e.g. lerobot/smolvla_base",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    from train import split_by_episode

    args = parse_args()
    print("Chapter 7 policy reality check")
    print(f"checkpoint epoch: {args.epoch}")
    print(f"cache: {CHECKPOINT_DIR / 'vla_cache.pt'}")

    trainer, ema, _state_stats, action_stats, cache, ckpt_name = load_ch07_policy(
        args.epoch
    )
    print(f"loaded checkpoint: {ckpt_name}")

    _train_mask, val_mask = split_by_episode(cache["episode_indices"])
    episode_ids = select_episode_ids(cache["episode_indices"], val_mask, args.num_episodes)
    if not episode_ids:
        raise RuntimeError("No validation episodes found in Ch07 cache.")

    episode_id = episode_ids[0]
    frame_indices = sample_episode_frame_indices(
        cache["episode_indices"], episode_id=episode_id, stride=args.stride
    )
    pred_actions, gt_actions = predict_ch07_first_actions(
        trainer, ema, cache, frame_indices, action_stats
    )
    metrics = compute_action_metrics(pred_actions, gt_actions)

    task_name = "unknown"
    if cache.get("tasks"):
        task_name = cache["tasks"][int(frame_indices[0].item())]

    large_vla = attempt_large_vla_reference(args.large_vla_model)

    print(f"episode: {episode_id}")
    print(f"task: {task_name}")
    print(f"frames evaluated: {len(frame_indices)}")
    print(f"Ch07 MAE: {metrics['mae']:.2f} raw joint units")
    per_joint = metrics["per_joint_mae"]
    for name, err in zip(JOINT_NAMES, per_joint.tolist()):
        print(f"  {name:12s}: {err:.2f}")
    print(f"large VLA: {large_vla.status} | {large_vla.message}")

    if args.save_figure is not None:
        save_reality_check_figure(
            args.save_figure,
            task_name,
            frame_indices - frame_indices[0],
            pred_actions.cpu(),
            gt_actions.cpu(),
            metrics,
            large_vla,
        )
        print(f"saved figure: {args.save_figure}")

    if args.save_rrd is not None:
        save_rerun_reality_check(
            args.save_rrd,
            episode_id,
            task_name,
            frame_indices - frame_indices[0],
            pred_actions.cpu(),
            gt_actions.cpu(),
            large_vla,
        )
        print(f"saved Rerun recording: {args.save_rrd}")


if __name__ == "__main__":
    main()
