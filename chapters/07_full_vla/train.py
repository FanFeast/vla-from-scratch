"""Chapter 7: Full VLA -- Training Script.

End-to-end training of the SmolVLA-like architecture on community robot data.

Pipeline:
    1. Pre-extract: download LeRobot datasets, run SigLIP+connector, cache
       vision tokens + raw data to disk (one-time, ~30 min).
    2. Train: load cached features, run SmolLM2 per batch (frozen),
       train action expert with flow matching (overnight, ~8-12 hours).

Usage:
    python train.py --preset so100
    python train.py --preset so100 --skip-extract
    python train.py --preset pusht
    python train.py --preset so100 --eval-only
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from config import SmolVLAConfig, PRESETS
from model import SmolVLA, build_smolvla

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Data Extraction
# ===========================================================================


def extract_and_cache(
    model: SmolVLA,
    datasets: list[dict[str, Any]],
    config: SmolVLAConfig,
    cache_dir: Path,
    skip: bool = False,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Pre-extract vision tokens from all datasets using frozen SigLIP.

    Caches vision_tokens, states, actions, episode_indices, and tokenized
    language instructions to disk for fast training.

    Args:
        model: SmolVLA model (uses vision_encoder + connector).
        datasets: List of dataset configs with repo_id, keys, task.
        config: SmolVLAConfig.
        cache_dir: Directory to save cache files.
        skip: If True and cache exists, load from disk.
        batch_size: Batch size for SigLIP encoding.

    Returns:
        Dict with all cached tensors.
    """
    cache_path = cache_dir / "vla_cache.pt"
    if skip and cache_path.exists():
        print(f"[cache] Loading from {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        # Reconstruct task list from token IDs (we store it separately)
        tasks_path = cache_dir / "vla_tasks.pt"
        if tasks_path.exists():
            data["tasks"] = torch.load(
                tasks_path, map_location="cpu", weights_only=False
            )
        return data

    from PIL import Image

    device = next(model.vision_encoder.parameters()).device

    # Image preprocessing: resize to 512x512, normalize to [-1, 1]
    # (SigLIP in SmolVLM2 uses mean=0.5, std=0.5 normalization)
    image_transform = T.Compose([
        T.Resize((config.vision_image_size, config.vision_image_size)),
        T.ToTensor(),  # [0, 255] -> [0, 1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    all_vision_tokens: list[Tensor] = []
    all_states: list[Tensor] = []
    all_actions: list[Tensor] = []
    all_episodes: list[Tensor] = []
    all_lang_ids: list[Tensor] = []
    all_lang_masks: list[Tensor] = []
    all_tasks: list[str] = []

    episode_offset = 0

    for ds_cfg in datasets:
        repo_id = ds_cfg["repo_id"]
        print(f"\n[extract] Loading {repo_id}...")

        try:
            from lerobot.datasets import LeRobotDataset
            dataset = LeRobotDataset(repo_id, video_backend="pyav")
        except Exception as e:
            print(f"[extract] SKIP {repo_id}: {e}")
            continue

        n = len(dataset)
        num_eps = dataset.num_episodes
        print(f"[extract] {repo_id}: {n} frames, {num_eps} episodes")

        # Detect available keys
        sample = dataset[0]
        image_key = ds_cfg["image_key"]
        if image_key not in sample:
            # Try alternate keys
            available = [k for k in sample.keys() if "image" in k.lower()]
            if available:
                image_key = available[0]
                print(f"[extract] Using image key: {image_key}")
            else:
                print(f"[extract] SKIP {repo_id}: no image key found")
                continue

        state_key = ds_cfg["state_key"]
        if state_key not in sample:
            available = [k for k in sample.keys() if "state" in k.lower()]
            if available:
                state_key = available[0]
                print(f"[extract] Using state key: {state_key}")

        action_key = ds_cfg["action_key"]

        # Tokenize task instruction (use dataset's own task text if available)
        sample_task = sample.get("task", None)
        task_text = sample_task if isinstance(sample_task, str) else ds_cfg["task"]
        print(f"[extract] Task: {task_text!r}")
        tokens = model.tokenizer(
            task_text,
            return_tensors="pt",
            padding="max_length",
            max_length=config.max_lang_tokens,
            truncation=True,
        )
        lang_ids = tokens["input_ids"].squeeze(0)       # (T,)
        lang_mask = tokens["attention_mask"].squeeze(0)  # (T,)

        # Pre-allocate vision token storage
        vis_dim = config.vlm_hidden_dim
        n_vis = config.num_vision_tokens
        vision_tokens = torch.empty(n, n_vis, vis_dim, dtype=torch.float16)

        states_list: list[Tensor] = []
        actions_list: list[Tensor] = []
        episodes_list: list[int] = []

        # Process images in batches
        pil_batch: list = []
        batch_indices: list[int] = []
        t0 = time.time()

        for idx in range(n):
            frame = dataset[idx]

            states_list.append(frame[state_key])
            actions_list.append(frame[action_key])
            episodes_list.append(
                frame["episode_index"].item() + episode_offset
            )

            img_tensor = frame[image_key]  # (3, H, W) float [0, 1]
            img_np = (
                img_tensor.permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            pil_batch.append(Image.fromarray(img_np))
            batch_indices.append(idx)

            if len(pil_batch) == batch_size or idx == n - 1:
                # Run SigLIP + connector
                pixel_values = torch.stack(
                    [image_transform(img) for img in pil_batch]
                ).to(device)

                with torch.no_grad(), torch.amp.autocast(
                    "cuda", dtype=DTYPE, enabled=(device.type == "cuda")
                ):
                    vt = model.extract_vision_tokens(pixel_values)

                for i, emb_idx in enumerate(batch_indices):
                    vision_tokens[emb_idx] = vt[i].cpu().half()

                pil_batch = []
                batch_indices = []

            if (idx + 1) % 500 == 0 or idx == n - 1:
                elapsed = time.time() - t0
                fps = (idx + 1) / elapsed
                print(f"  [{idx + 1}/{n}] {fps:.0f} frames/s")

        # Stack raw data
        states = torch.stack(states_list)
        actions = torch.stack(actions_list)
        episode_indices = torch.tensor(episodes_list, dtype=torch.long)

        all_vision_tokens.append(vision_tokens)
        all_states.append(states)
        all_actions.append(actions)
        all_episodes.append(episode_indices)
        # Repeat lang tokens for every frame in this dataset
        all_lang_ids.append(lang_ids.unsqueeze(0).expand(n, -1))
        all_lang_masks.append(lang_mask.unsqueeze(0).expand(n, -1))
        all_tasks.extend([task_text] * n)

        episode_offset = episode_indices.max().item() + 1

        elapsed = time.time() - t0
        print(
            f"[extract] {repo_id} done in {elapsed:.1f}s | "
            f"states={states.shape}, actions={actions.shape}"
        )

    if not all_vision_tokens:
        raise RuntimeError("No datasets were loaded successfully!")

    # Concatenate all datasets
    result = {
        "vision_tokens": torch.cat(all_vision_tokens),
        "states": torch.cat(all_states),
        "actions": torch.cat(all_actions),
        "episode_indices": torch.cat(all_episodes),
        "lang_token_ids": torch.cat(all_lang_ids),
        "lang_attention_mask": torch.cat(all_lang_masks),
    }

    print(
        f"\n[extract] Total: {result['vision_tokens'].shape[0]} frames, "
        f"{result['episode_indices'].unique().shape[0]} episodes"
    )

    # Save to disk
    torch.save(result, cache_path)
    torch.save(all_tasks, cache_dir / "vla_tasks.pt")
    cache_mb = cache_path.stat().st_size / 1e6
    print(f"[cache] Saved to {cache_path} ({cache_mb:.0f} MB)")

    result["tasks"] = all_tasks
    return result


# ===========================================================================
# Normalization
# ===========================================================================


@dataclass
class NormStats:
    """Per-feature mean/std normalization."""

    mean: Tensor
    std: Tensor

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize to zero-mean, unit-variance."""
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x: Tensor) -> Tensor:
        """Reverse normalization."""
        return x * (self.std + 1e-8) + self.mean


def compute_norm_stats(data: Tensor) -> NormStats:
    """Compute per-feature mean/std from tensor (N, D)."""
    return NormStats(
        mean=data.mean(dim=0),
        std=data.std(dim=0),
    )


# ===========================================================================
# Train/Val Split
# ===========================================================================


def split_by_episode(
    episode_indices: Tensor,
    val_ratio: float = 0.13,
) -> tuple[Tensor, Tensor]:
    """Split by episode to prevent data leakage."""
    unique_eps = episode_indices.unique(sorted=True)
    n_val = max(1, int(len(unique_eps) * val_ratio))
    val_eps = unique_eps[-n_val:]
    val_mask = torch.isin(episode_indices, val_eps)
    train_mask = ~val_mask
    print(
        f"[split] {len(unique_eps) - n_val} train episodes "
        f"({train_mask.sum()} frames), "
        f"{n_val} val episodes ({val_mask.sum()} frames)"
    )
    return train_mask, val_mask


# ===========================================================================
# Dataset
# ===========================================================================


class VLADataset(Dataset):
    """PyTorch Dataset from pre-cached VLA features with action chunking."""

    def __init__(
        self,
        vision_tokens: Tensor,
        lang_token_ids: Tensor,
        lang_attention_mask: Tensor,
        states: Tensor,
        actions: Tensor,
        episode_indices: Tensor,
        chunk_size: int = 10,
    ) -> None:
        self.vision_tokens = vision_tokens
        self.lang_token_ids = lang_token_ids
        self.lang_attention_mask = lang_attention_mask
        self.states = states
        self.actions = actions
        self.episode_indices = episode_indices
        self.chunk_size = chunk_size
        self.n = len(vision_tokens)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ep = self.episode_indices[idx]

        # Build action chunk (pad at episode boundary)
        chunk: list[Tensor] = []
        for k in range(self.chunk_size):
            future_idx = idx + k
            if (
                future_idx < self.n
                and self.episode_indices[future_idx] == ep
            ):
                chunk.append(self.actions[future_idx])
            else:
                chunk.append(chunk[-1] if chunk else self.actions[idx])

        return {
            "vision_tokens": self.vision_tokens[idx],
            "lang_token_ids": self.lang_token_ids[idx],
            "lang_attention_mask": self.lang_attention_mask[idx],
            "state": self.states[idx],
            "actions": torch.stack(chunk),
        }


# ===========================================================================
# EMA
# ===========================================================================


class EMAModel:
    """Exponential moving average of trainable model weights only.

    Only tracks state_proj + action_expert to avoid duplicating the
    ~300M frozen VLM/SigLIP weights in memory.
    """

    def __init__(self, model: SmolVLA, decay: float = 0.999) -> None:
        self.decay = decay
        # Only deepcopy trainable components
        self.shadow_state_proj = copy.deepcopy(model.state_proj)
        self.shadow_action_expert = copy.deepcopy(model.action_expert)
        self.shadow_state_proj.eval()
        self.shadow_action_expert.eval()
        for p in self.shadow_state_proj.parameters():
            p.requires_grad = False
        for p in self.shadow_action_expert.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: SmolVLA) -> None:
        """Update shadow weights from model's trainable components."""
        for s, m in zip(
            self.shadow_state_proj.parameters(),
            model.state_proj.parameters(),
        ):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)
        for s, m in zip(
            self.shadow_action_expert.parameters(),
            model.action_expert.parameters(),
        ):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def apply_to(self, model: SmolVLA) -> None:
        """Swap trainable weights with EMA weights for inference."""
        model.state_proj.load_state_dict(self.shadow_state_proj.state_dict())
        model.action_expert.load_state_dict(
            self.shadow_action_expert.state_dict()
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_proj": self.shadow_state_proj.state_dict(),
            "action_expert": self.shadow_action_expert.state_dict(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.shadow_state_proj.load_state_dict(sd["state_proj"])
        self.shadow_action_expert.load_state_dict(sd["action_expert"])

    def to(self, device: torch.device) -> "EMAModel":
        self.shadow_state_proj.to(device)
        self.shadow_action_expert.to(device)
        return self


# ===========================================================================
# Flow Matching Trainer
# ===========================================================================


class FlowMatchingVLA:
    """Flow matching training and inference for SmolVLA.

    Uses rectified flow with Beta(a,b) timestep sampling as in SmolVLA/pi0.
    """

    def __init__(
        self,
        model: SmolVLA,
        config: SmolVLAConfig,
    ) -> None:
        self.model = model
        self.config = config

    def compute_loss(
        self, batch: dict[str, Tensor], device: torch.device
    ) -> Tensor:
        """Compute flow matching MSE loss.

        Interpolation: x_t = t * noise + (1 - t) * action
        Target: v = noise - action
        """
        x_1 = batch["actions"].to(device)       # (B, K, action_dim)
        vis = batch["vision_tokens"].to(device)  # (B, 64, 960)
        lang_ids = batch["lang_token_ids"].to(device)
        lang_mask = batch["lang_attention_mask"].to(device)
        state = batch["state"].to(device)

        B = x_1.shape[0]

        # Sample noise and Beta-distributed timestep
        x_0 = torch.randn_like(x_1)
        t = torch.distributions.Beta(
            self.config.beta_a, self.config.beta_b
        ).sample((B,)).to(device)
        t = t * 0.999 + 0.001  # clamp to (0.001, 1.0)

        # Interpolate
        t_expand = t[:, None, None]
        x_t = t_expand * x_0 + (1.0 - t_expand) * x_1

        # Target velocity
        v_target = x_0 - x_1

        # Predict velocity
        v_pred = self.model(vis, lang_ids, lang_mask, state, x_t, t)

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        vision_tokens: Tensor,
        lang_token_ids: Tensor,
        lang_attention_mask: Tensor,
        state: Tensor,
        ema: EMAModel | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Generate actions via Euler ODE integration."""
        model = self.model
        model.eval()

        # Temporarily apply EMA weights if provided
        if ema is not None:
            # Save original weights
            orig_state_proj = copy.deepcopy(model.state_proj.state_dict())
            orig_action_expert = copy.deepcopy(
                model.action_expert.state_dict()
            )
            ema.apply_to(model)

        device = next(
            p.device for p in model.action_expert.parameters()
        )

        vision_tokens = vision_tokens.to(device)
        lang_token_ids = lang_token_ids.to(device)
        lang_attention_mask = lang_attention_mask.to(device)
        state = state.to(device)

        B = vision_tokens.shape[0]
        K = self.config.chunk_size
        action_dim = self.config.action_dim
        steps = num_steps or self.config.inference_steps

        # Start from noise
        x = torch.randn(B, K, action_dim, device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((B,), i * dt, device=device)
            v = model(
                vision_tokens, lang_token_ids, lang_attention_mask,
                state, x, t,
            )
            x = x + v * dt

        # Restore original weights if EMA was applied
        if ema is not None:
            model.state_proj.load_state_dict(orig_state_proj)
            model.action_expert.load_state_dict(orig_action_expert)

        return x


# ===========================================================================
# Training Loop
# ===========================================================================


def train_vla(
    trainer: FlowMatchingVLA,
    train_ds: VLADataset,
    val_ds: VLADataset,
    config: SmolVLAConfig,
    ckpt_path: Path,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """Train SmolVLA with AdamW, cosine LR, bf16, grad clipping, EMA."""
    model = trainer.model

    # Move VLM + connector to GPU (vision_encoder already freed in caller)
    model.vlm.to(device)
    model.connector.to(device)
    model.state_proj.to(device)
    model.action_expert.to(device)

    # EMA of trainable components only (~20M params, not 300M)
    ema = EMAModel(model, decay=config.ema_decay)
    ema.to(device)

    # Only optimize trainable parameters
    trainable_params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    # Cosine schedule with warmup
    total_steps = config.epochs * (len(train_ds) // config.batch_size + 1)

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    global_step = 0

    print(
        f"\n[train] Starting training: {config.epochs} epochs, "
        f"batch_size={config.batch_size}, lr={config.lr}"
    )
    print(
        f"[train] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples"
    )
    print(f"[train] Total steps: {total_steps}, Warmup: {config.warmup_steps}")
    t0_total = time.time()

    for epoch in range(config.epochs):
        t0_epoch = time.time()

        # --- Train ---
        model.train()
        # Keep frozen components in eval mode
        if hasattr(model, "vision_encoder"):
            model.vision_encoder.eval()
        model.connector.eval()
        model.vlm.eval()

        total_loss, n_batch = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast(
                "cuda", dtype=DTYPE, enabled=(device.type == "cuda")
            ):
                loss = trainer.compute_loss(batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            total_loss += loss.item()
            n_batch += 1
            global_step += 1

        train_loss = total_loss / max(n_batch, 1)
        history["train_loss"].append(train_loss)

        # --- Validation ---
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast(
                    "cuda", dtype=DTYPE, enabled=(device.type == "cuda")
                ):
                    loss = trainer.compute_loss(batch, device)
                val_total += loss.item()
                val_n += 1
        val_loss = val_total / max(val_n, 1)
        history["val_loss"].append(val_loss)

        epoch_time = time.time() - t0_epoch

        if epoch % 5 == 0 or epoch == config.epochs - 1:
            print(
                f"  Epoch {epoch + 1:3d}/{config.epochs} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{epoch_time:.1f}s"
            )

        if val_loss < best_val:
            best_val = val_loss
            # Save only trainable weights to keep checkpoints small (~80MB)
            trainable_sd = {
                "state_proj": model.state_proj.state_dict(),
                "action_expert": model.action_expert.state_dict(),
            }
            torch.save(
                {
                    "trainable": trainable_sd,
                    "ema": ema.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )

        # Periodic checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            periodic_path = ckpt_path.parent / f"vla_epoch_{epoch + 1}.pt"
            trainable_sd = {
                "state_proj": model.state_proj.state_dict(),
                "action_expert": model.action_expert.state_dict(),
            }
            torch.save(
                {
                    "trainable": trainable_sd,
                    "ema": ema.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                periodic_path,
            )

    total_time = time.time() - t0_total
    print(
        f"\n[train] Done in {total_time / 3600:.1f}h | "
        f"best_val={best_val:.5f}"
    )

    # Load best checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.state_proj.load_state_dict(ckpt["trainable"]["state_proj"])
        model.action_expert.load_state_dict(
            ckpt["trainable"]["action_expert"]
        )
        ema.load_state_dict(ckpt["ema"])

    return {"history": history, "ema": ema, "best_val": best_val}


# ===========================================================================
# Visualization
# ===========================================================================


def plot_training_curves(
    history: dict[str, list[float]],
    out_dir: Path,
    preset_name: str,
) -> None:
    """Plot and save training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train", alpha=0.8)
    ax.plot(epochs, history["val_loss"], label="Val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow Matching Loss (MSE)")
    ax.set_title(f"Ch07 Full VLA ({preset_name}) -- Training Curves")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f"ch07_{preset_name}_training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved {out_path}")


# ===========================================================================
# Main
# ===========================================================================


def run_preset(
    preset_name: str,
    skip_extract: bool = False,
    eval_only: bool = False,
) -> None:
    """Run the full VLA training pipeline for a preset."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Choose from: {list(PRESETS)}"
        )

    preset = PRESETS[preset_name]
    config = SmolVLAConfig(
        action_dim=preset["action_dim"],
        state_dim=preset["state_dim"],
        chunk_size=preset["chunk_size"],
        epochs=preset["epochs"],
        batch_size=preset["batch_size"],
    )

    print(f"\n{'=' * 60}")
    print(f"  Chapter 7: Full VLA -- {preset_name}")
    print(f"{'=' * 60}")
    print(f"  Action dim: {config.action_dim}")
    print(f"  State dim:  {config.state_dim}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Epochs:     {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Device:     {DEVICE}")
    print(f"{'=' * 60}\n")

    # Build model
    model = build_smolvla(config)

    if eval_only:
        ckpt_path = CHECKPOINT_DIR / f"vla_{preset_name}_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.state_proj.load_state_dict(
                ckpt["trainable"]["state_proj"]
            )
            model.action_expert.load_state_dict(
                ckpt["trainable"]["action_expert"]
            )
            print(f"[eval] Loaded {ckpt_path}")
        else:
            print(f"[eval] No checkpoint found at {ckpt_path}")
        return

    # Move vision encoder + connector to GPU for fast extraction
    model.vision_encoder.to(DEVICE)
    model.connector.to(DEVICE)

    # Extract and cache
    data = extract_and_cache(
        model, preset["datasets"], config, CHECKPOINT_DIR, skip=skip_extract
    )

    # Free vision encoder from GPU (no longer needed)
    model.vision_encoder.cpu()
    torch.cuda.empty_cache()

    # Compute normalization stats
    state_stats = compute_norm_stats(data["states"])
    action_stats = compute_norm_stats(data["actions"])

    # Normalize
    norm_states = state_stats.normalize(data["states"])
    norm_actions = action_stats.normalize(data["actions"])

    # Train/val split
    train_mask, val_mask = split_by_episode(
        data["episode_indices"], config.val_ratio
    )

    train_ds = VLADataset(
        vision_tokens=data["vision_tokens"][train_mask],
        lang_token_ids=data["lang_token_ids"][train_mask],
        lang_attention_mask=data["lang_attention_mask"][train_mask],
        states=norm_states[train_mask],
        actions=norm_actions[train_mask],
        episode_indices=data["episode_indices"][train_mask],
        chunk_size=config.chunk_size,
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

    # Save norm stats
    torch.save(
        {"state": state_stats, "action": action_stats},
        CHECKPOINT_DIR / f"vla_{preset_name}_norm_stats.pt",
    )

    # Train
    trainer = FlowMatchingVLA(model, config)
    ckpt_path = CHECKPOINT_DIR / f"vla_{preset_name}_best.pt"

    results = train_vla(trainer, train_ds, val_ds, config, ckpt_path)

    # Plot
    plot_training_curves(results["history"], CHECKPOINT_DIR, preset_name)

    print(f"\n[done] Best val loss: {results['best_val']:.5f}")
    print(f"[done] Checkpoint: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 7: Full VLA Training"
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="so100",
        help="Dataset preset (default: so100)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip pre-extraction if cache exists",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load checkpoint and evaluate only",
    )
    args = parser.parse_args()

    run_preset(args.preset, args.skip_extract, args.eval_only)


if __name__ == "__main__":
    main()
