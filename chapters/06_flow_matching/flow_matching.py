"""Chapter 6: Flow Matching -- Transformer Action Expert.

Side-by-side comparison of rectified flow vs DDPM training on the same
~100M parameter transformer denoiser. Proves that transformer + flow matching
is the architectural combination that makes diffusion-style VLAs work --
where Chapter 5's MLP denoiser scored 0%.

Usage:
    python flow_matching.py --preset pusht
    python flow_matching.py --preset aloha
    python flow_matching.py --preset pusht --skip-precompute
    python flow_matching.py --preset pusht --eval-only
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

from action_expert import ActionExpertTransformer, build_action_expert

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
SIGLIP_DIM = 768
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Vision Encoder
# ===========================================================================


class VisionEncoder:
    """Frozen SigLIP ViT-B/16 -- extracts CLS and optionally patch tokens."""

    def __init__(self, device: torch.device = DEVICE) -> None:
        from transformers import AutoModel, AutoImageProcessor

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self.model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).vision_model
        self.model.eval().to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_pil(self, pil_image: Any) -> Tensor:
        """Single PIL -> (768,) CLS on CPU."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        outputs = self.model(pixel_values=inputs["pixel_values"].to(self.device))
        return outputs.pooler_output.squeeze(0).cpu()

    @torch.no_grad()
    def encode_batch(
        self, pil_images: list, return_patches: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Batch -> (B, 768) CLS. If return_patches: also (B, 196, 768)."""
        inputs = self.processor(images=pil_images, return_tensors="pt")
        outputs = self.model(pixel_values=inputs["pixel_values"].to(self.device))
        cls_emb = outputs.pooler_output.cpu()
        if return_patches:
            patches = outputs.last_hidden_state[:, 1:, :].cpu()
            return cls_emb, patches
        return cls_emb


# ===========================================================================
# Data Extraction (single-pass)
# ===========================================================================


def extract_and_cache(
    repo_id: str,
    image_key: str,
    state_key: str,
    action_key: str,
    cache_prefix: str,
    batch_size: int = 64,
    skip_embeddings: bool = False,
    extract_patches: bool = False,
) -> dict[str, Tensor]:
    """Single-pass extraction of embeddings, states, actions, episode indices.

    Downloads the LeRobot dataset, iterates once through all frames, extracts
    SigLIP CLS embeddings (and optionally patch tokens), states, actions, and
    episode indices. Caches everything to disk for fast reloading.

    Args:
        repo_id: HuggingFace dataset ID, e.g. 'lerobot/pusht'.
        image_key: Key for image tensor in dataset.
        state_key: Key for state tensor.
        action_key: Key for action tensor.
        cache_prefix: Prefix for cache files.
        batch_size: Batch size for SigLIP encoding.
        skip_embeddings: If True and cache exists, skip recomputation.
        extract_patches: If True, also cache patch tokens.

    Returns:
        Dict with keys: cls_embeddings, states, actions, episode_indices,
        and optionally patch_embeddings.
    """
    from PIL import Image

    cls_path = CHECKPOINT_DIR / f"{cache_prefix}_cls.pt"
    data_path = CHECKPOINT_DIR / f"{cache_prefix}_data.pt"
    patches_path = CHECKPOINT_DIR / f"{cache_prefix}_patches.pt"

    # Try loading from cache
    if skip_embeddings and cls_path.exists() and data_path.exists():
        print(f"[cache] Loading from {cls_path} and {data_path}")
        cls_embeddings = torch.load(cls_path, map_location="cpu", weights_only=True)
        data = torch.load(data_path, map_location="cpu", weights_only=True)
        result = {
            "cls_embeddings": cls_embeddings,
            "states": data["states"],
            "actions": data["actions"],
            "episode_indices": data["episode_indices"],
        }
        if extract_patches and patches_path.exists():
            result["patch_embeddings"] = torch.load(
                patches_path, map_location="cpu", weights_only=True
            )
        return result

    # Load dataset from Hub
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(repo_id)
    n = len(dataset)
    print(f"[data] Loaded {repo_id}: {n} frames, {dataset.num_episodes} episodes")

    # Pre-allocate tensors
    cls_embeddings = torch.empty(n, SIGLIP_DIM, dtype=torch.float32)
    patch_embeddings = (
        torch.empty(n, 196, SIGLIP_DIM, dtype=torch.float32)
        if extract_patches
        else None
    )
    states_list: list[Tensor] = []
    actions_list: list[Tensor] = []
    episodes_list: list[int] = []

    encoder = VisionEncoder(device=DEVICE)
    pil_batch: list = []
    batch_indices: list[int] = []

    print(f"[extract] Processing {n} frames (single pass)...")
    t0 = time.time()

    for idx in range(n):
        frame = dataset[idx]

        states_list.append(frame[state_key])
        actions_list.append(frame[action_key])
        episodes_list.append(frame["episode_index"].item())

        img_tensor = frame[image_key]  # (3, H, W) float [0, 1]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_batch.append(Image.fromarray(img_np))
        batch_indices.append(idx)

        if len(pil_batch) == batch_size or idx == n - 1:
            if extract_patches:
                cls_emb, patches = encoder.encode_batch(
                    pil_batch, return_patches=True
                )
                for i, emb_idx in enumerate(batch_indices):
                    cls_embeddings[emb_idx] = cls_emb[i]
                    patch_embeddings[emb_idx] = patches[i]
            else:
                cls_emb = encoder.encode_batch(pil_batch)
                for i, emb_idx in enumerate(batch_indices):
                    cls_embeddings[emb_idx] = cls_emb[i]
            pil_batch = []
            batch_indices = []

        if (idx + 1) % 5000 == 0 or idx == n - 1:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"  [{idx + 1}/{n}] {fps:.0f} frames/s")

    # Free encoder
    del encoder
    torch.cuda.empty_cache()

    states = torch.stack(states_list)
    actions = torch.stack(actions_list)
    episode_indices = torch.tensor(episodes_list, dtype=torch.long)

    print(f"[extract] Done in {time.time() - t0:.1f}s")
    print(f"  States: {states.shape}, Actions: {actions.shape}")
    print(f"  Episodes: {episode_indices.unique().shape[0]}")

    # Cache to disk
    torch.save(cls_embeddings, cls_path)
    torch.save(
        {
            "states": states,
            "actions": actions,
            "episode_indices": episode_indices,
        },
        data_path,
    )
    if extract_patches and patch_embeddings is not None:
        torch.save(patch_embeddings, patches_path)
    print(f"[cache] Saved to {cls_path} and {data_path}")

    result = {
        "cls_embeddings": cls_embeddings,
        "states": states,
        "actions": actions,
        "episode_indices": episode_indices,
    }
    if extract_patches and patch_embeddings is not None:
        result["patch_embeddings"] = patch_embeddings
    return result


# ===========================================================================
# Normalization
# ===========================================================================


@dataclass
class NormStats:
    """Per-feature min/max normalization statistics."""

    min_val: Tensor
    max_val: Tensor

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize to [-1, 1]."""
        return 2.0 * (x - self.min_val) / (self.max_val - self.min_val + 1e-8) - 1.0

    def denormalize(self, x: Tensor) -> Tensor:
        """Denormalize from [-1, 1] back to original range."""
        return (x + 1.0) / 2.0 * (self.max_val - self.min_val) + self.min_val


def compute_norm_stats(data: Tensor) -> NormStats:
    """Compute per-feature min/max from a tensor (N, D)."""
    return NormStats(
        min_val=data.min(dim=0).values,
        max_val=data.max(dim=0).values,
    )


# ===========================================================================
# Train/Val Split
# ===========================================================================


def split_by_episode(
    episode_indices: Tensor,
    val_ratio: float = 0.13,
) -> tuple[Tensor, Tensor]:
    """Split frame indices by episode to prevent data leakage.

    Uses the last val_ratio fraction of episodes as validation.

    Returns:
        (train_mask, val_mask) boolean tensors of shape (N,).
    """
    unique_episodes = episode_indices.unique(sorted=True)
    n_val = max(1, int(len(unique_episodes) * val_ratio))
    val_episodes = unique_episodes[-n_val:]
    val_mask = torch.isin(episode_indices, val_episodes)
    train_mask = ~val_mask
    n_train_eps = len(unique_episodes) - n_val
    print(
        f"[split] {n_train_eps} train episodes ({train_mask.sum()} frames), "
        f"{n_val} val episodes ({val_mask.sum()} frames)"
    )
    return train_mask, val_mask


# ===========================================================================
# Dataset
# ===========================================================================


class RobotDataset(Dataset):
    """PyTorch Dataset from pre-extracted tensors with action chunking.

    All inputs are in-memory tensors (fast, no I/O during training).
    Supports optional patch embeddings for patch-level vision conditioning.
    """

    def __init__(
        self,
        cls_emb: Tensor,
        states: Tensor,
        actions: Tensor,
        episode_indices: Tensor,
        chunk_size: int = 4,
        patch_emb: Tensor | None = None,
    ) -> None:
        self.cls_emb = cls_emb
        self.states = states
        self.actions = actions
        self.episode_indices = episode_indices
        self.chunk_size = chunk_size
        self.patch_emb = patch_emb
        self.n = len(cls_emb)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ep = self.episode_indices[idx]

        # Build action chunk: K consecutive actions, pad at episode boundary
        chunk: list[Tensor] = []
        for k in range(self.chunk_size):
            future_idx = idx + k
            if future_idx < self.n and self.episode_indices[future_idx] == ep:
                chunk.append(self.actions[future_idx])
            else:
                # Pad by repeating the last valid action
                chunk.append(chunk[-1] if chunk else self.actions[idx])

        item: dict[str, Tensor] = {
            "cls_emb": self.cls_emb[idx],
            "state": self.states[idx],
            "actions": torch.stack(chunk),
        }
        if self.patch_emb is not None:
            item["patch_emb"] = self.patch_emb[idx]
        return item


# ===========================================================================
# EMA
# ===========================================================================


class EMAModel:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights toward model weights."""
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        """Return shadow model state dict."""
        return self.shadow.state_dict()

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Load shadow model state dict."""
        self.shadow.load_state_dict(sd)


# ===========================================================================
# Flow Matching Trainer
# ===========================================================================


class FlowMatchingTrainer:
    """Rectified flow: velocity prediction along straight paths.

    Training: x_t = (1-t)*x_0 + t*x_1, target = x_1 - x_0
    Inference: Euler ODE from noise to actions
    """

    def __init__(
        self,
        model: ActionExpertTransformer,
        inference_steps: int = 10,
    ) -> None:
        self.model = model
        self.inference_steps = inference_steps

    def compute_loss(self, batch: dict[str, Tensor], device: str | torch.device) -> Tensor:
        """Compute flow matching loss for a batch.

        Args:
            batch: Dict with cls_emb, state, actions, and optionally patch_emb.
            device: Device to run on.

        Returns:
            Scalar MSE loss between predicted and target velocity.
        """
        x_1 = batch["actions"].to(device)  # (B, K, action_dim)
        state = batch["state"].to(device)  # (B, state_dim)

        # Use patch tokens if available, otherwise CLS
        if "patch_emb" in batch:
            vis = batch["patch_emb"].to(device)  # (B, N, cond_dim)
        else:
            vis = batch["cls_emb"].to(device)  # (B, cond_dim)

        B = x_1.shape[0]

        # Sample noise and timestep
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)

        # Interpolate: x_t = (1-t)*x_0 + t*x_1
        t_expand = t[:, None, None]  # (B, 1, 1) for broadcasting
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1

        # Target velocity
        v_target = x_1 - x_0

        # Predict velocity
        v_pred = self.model(x_t, t, vis, state)

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        vision_emb: Tensor,
        state: Tensor,
        ema_model: nn.Module | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Generate actions via Euler ODE integration.

        Args:
            vision_emb: (B, cond_dim) or (B, N, cond_dim) vision embedding.
            state: (B, state_dim) proprioceptive state.
            ema_model: If provided, use EMA weights for inference.
            num_steps: Override default inference steps.

        Returns:
            Predicted actions of shape (B, K, action_dim).
        """
        model = ema_model if ema_model is not None else self.model
        model.eval()
        device = next(model.parameters()).device

        vision_emb = vision_emb.to(device)
        state = state.to(device)

        B = vision_emb.shape[0]
        K = model.chunk_size
        action_dim = model.action_dim
        steps = num_steps if num_steps is not None else self.inference_steps

        # Start from noise
        x = torch.randn(B, K, action_dim, device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((B,), i * dt, device=device)
            v = model(x, t, vision_emb, state)
            x = x + v * dt

        return x.clamp(-1.0, 1.0)


# ===========================================================================
# DDPM Transformer Trainer
# ===========================================================================


class DDPMTransformerTrainer:
    """DDPM noise prediction with transformer denoiser.

    Uses cosine beta schedule (better than linear for large T).
    Normalizes discrete timesteps to [0,1] before passing to shared architecture.
    """

    def __init__(
        self,
        model: ActionExpertTransformer,
        train_timesteps: int = 1000,
        inference_steps: int = 20,
    ) -> None:
        self.model = model
        self.train_timesteps = train_timesteps
        self.inference_steps = inference_steps

        # Cosine schedule
        steps = torch.arange(train_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((steps / train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(max=0.999).float()

        alphas = (1.0 - betas).float()
        alphas_cumprod = torch.cumprod(alphas, dim=0).float()

        self._buffers: dict[str, Tensor] = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_ac": torch.sqrt(alphas_cumprod),
            "sqrt_1mac": torch.sqrt(1.0 - alphas_cumprod),
        }

    def _get_buf(self, name: str, device: str | torch.device) -> Tensor:
        """Get a schedule buffer on the correct device."""
        buf = self._buffers[name]
        if buf.device != torch.device(device):
            buf = buf.to(device)
            self._buffers[name] = buf
        return buf

    def compute_loss(self, batch: dict[str, Tensor], device: str | torch.device) -> Tensor:
        """Compute DDPM noise-prediction loss for a batch.

        Args:
            batch: Dict with cls_emb, state, actions, and optionally patch_emb.
            device: Device to run on.

        Returns:
            Scalar MSE loss between predicted and actual noise.
        """
        x_0 = batch["actions"].to(device)  # (B, K, action_dim)
        state = batch["state"].to(device)
        if "patch_emb" in batch:
            vis = batch["patch_emb"].to(device)
        else:
            vis = batch["cls_emb"].to(device)

        B = x_0.shape[0]
        T = self.train_timesteps

        # Sample timestep and noise
        t = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(x_0)

        sqrt_ac = self._get_buf("sqrt_ac", device)
        sqrt_1mac = self._get_buf("sqrt_1mac", device)

        # Forward diffusion
        x_t = sqrt_ac[t, None, None] * x_0 + sqrt_1mac[t, None, None] * noise

        # Normalize timestep to [0, 1] for shared architecture
        t_norm = t.float() / T

        # Predict noise
        pred_noise = self.model(x_t, t_norm, vis, state)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        vision_emb: Tensor,
        state: Tensor,
        ema_model: nn.Module | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Generate actions via reverse diffusion with strided timesteps.

        Args:
            vision_emb: (B, cond_dim) or (B, N, cond_dim) vision embedding.
            state: (B, state_dim) proprioceptive state.
            ema_model: If provided, use EMA weights for inference.
            num_steps: Override default inference steps.

        Returns:
            Predicted actions of shape (B, K, action_dim).
        """
        model = ema_model if ema_model is not None else self.model
        model.eval()
        device = next(model.parameters()).device

        vision_emb = vision_emb.to(device)
        state = state.to(device)

        B = vision_emb.shape[0]
        K = model.chunk_size
        action_dim = model.action_dim
        T = self.train_timesteps
        steps = num_steps if num_steps is not None else self.inference_steps

        alphas_cumprod = self._get_buf("alphas_cumprod", device)
        sqrt_ac = self._get_buf("sqrt_ac", device)
        sqrt_1mac = self._get_buf("sqrt_1mac", device)

        # Strided timestep schedule (descending)
        timesteps = torch.linspace(T - 1, 0, steps, device=device).long()

        x = torch.randn(B, K, action_dim, device=device)

        for t_val in timesteps:
            t_batch = t_val.expand(B)
            t_norm = t_batch.float() / T

            pred_noise = model(x, t_norm, vision_emb, state)

            # Predict x_0
            pred_x0 = (x - sqrt_1mac[t_val] * pred_noise) / sqrt_ac[t_val]
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

            if t_val > 0:
                alpha_bar_prev = alphas_cumprod[t_val - 1]
                x = (
                    torch.sqrt(alpha_bar_prev) * pred_x0
                    + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
                )
            else:
                x = pred_x0

        return x.clamp(-1.0, 1.0)


# ===========================================================================
# Training
# ===========================================================================


def train_model(
    trainer: FlowMatchingTrainer | DDPMTransformerTrainer,
    train_ds: RobotDataset,
    val_ds: RobotDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    ckpt_path: Path,
    device: torch.device = DEVICE,
    ema_decay: float = 0.999,
) -> dict[str, Any]:
    """Train with AdamW, cosine LR, bf16 AMP, grad clipping, EMA.

    Args:
        trainer: FlowMatchingTrainer or DDPMTransformerTrainer instance.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        ckpt_path: Path to save best checkpoint.
        device: Device to train on.
        ema_decay: EMA decay rate.

    Returns:
        Dict with 'history' (train/val loss lists) and 'ema' (EMAModel).
    """
    model = trainer.model
    model.to(device)
    ema = EMAModel(model, decay=ema_decay)
    ema.shadow.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss, n_batch = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(device.type == "cuda")):
                loss = trainer.compute_loss(batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()
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

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "ema": ema.state_dict()},
                ckpt_path,
            )

    # Load best checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])

    model.to("cpu")
    ema.shadow.to("cpu")
    return {"history": history, "ema": ema}


# ===========================================================================
# Evaluation -- Live (PushT)
# ===========================================================================


def evaluate_live_pusht(
    trainer: FlowMatchingTrainer | DDPMTransformerTrainer,
    ema: EMAModel,
    state_stats: NormStats,
    action_stats: NormStats,
    chunk_size: int,
    n_episodes: int = 10,
    max_steps: int = 300,
    device: torch.device = DEVICE,
    num_inference_steps: int | None = None,
) -> dict[str, float]:
    """Live rollout with chunked action execution in gym-pusht/PushT-v0."""
    import gymnasium as gym
    from PIL import Image

    try:
        import gym_pusht  # noqa: F401
    except ImportError:
        print("  [live-pusht] gym_pusht not installed, skipping")
        return {"success_rate": 0.0, "avg_length": 0.0, "avg_inference_ms": 0.0}

    encoder = VisionEncoder(device=device)
    ema.shadow.to(device)
    ema.shadow.eval()

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    successes = 0
    lengths: list[int] = []
    inference_times: list[float] = []
    coverages: list[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        action_buffer: list[Tensor] = []
        max_coverage = 0.0

        while not done and step < max_steps:
            if not action_buffer:
                pil_img = Image.fromarray(obs["pixels"])
                vision_emb = encoder.encode_pil(pil_img).unsqueeze(0).to(device)
                agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)
                state_norm = state_stats.normalize(agent_pos).unsqueeze(0).to(device)

                t0 = time.time()
                pred = trainer.sample(
                    vision_emb,
                    state_norm,
                    ema_model=ema.shadow,
                    num_steps=num_inference_steps,
                )
                inference_times.append((time.time() - t0) * 1000)

                action_buffer = [pred[0, k].cpu() for k in range(chunk_size)]

            action_norm = action_buffer.pop(0)
            action = action_stats.denormalize(action_norm).numpy().astype(np.float32)

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as exc:
                # A diverging policy can drive the simulator into an invalid
                # state (e.g. mujoco mjWARN_BADQACC). Score this episode a
                # failure and continue, rather than aborting the whole run --
                # a failing policy is a result, not a crash.
                print(f"  [live-pusht] ep {ep}: simulator diverged "
                      f"({type(exc).__name__}); scoring episode as failure")
                info = {}
                break
            done = terminated or truncated
            step += 1
            max_coverage = max(max_coverage, reward)

            if info.get("is_success", False):
                break

        successes += int(info.get("is_success", False))
        lengths.append(step)
        coverages.append(max_coverage)

    env.close()
    del encoder
    ema.shadow.to("cpu")
    torch.cuda.empty_cache()

    avg_ms = sum(inference_times) / max(len(inference_times), 1)
    avg_coverage = sum(coverages) / max(len(coverages), 1)
    metrics = {
        "success_rate": successes / n_episodes,
        "avg_length": sum(lengths) / len(lengths),
        "avg_inference_ms": avg_ms,
        "avg_coverage": avg_coverage,
    }
    print(
        f"  [live-pusht] success={metrics['success_rate']:.0%} | "
        f"coverage={avg_coverage:.1%} | "
        f"avg_len={metrics['avg_length']:.0f} | "
        f"inference={avg_ms:.1f}ms"
    )
    return metrics


# ===========================================================================
# Evaluation -- Live (ALOHA)
# ===========================================================================


def evaluate_live_aloha(
    trainer: FlowMatchingTrainer | DDPMTransformerTrainer,
    ema: EMAModel,
    state_stats: NormStats,
    action_stats: NormStats,
    chunk_size: int,
    n_episodes: int = 10,
    max_steps: int = 400,
    device: torch.device = DEVICE,
    num_inference_steps: int | None = None,
) -> dict[str, float]:
    """Live rollout with chunked action execution in gym-aloha."""
    import gymnasium as gym
    from PIL import Image

    try:
        import gym_aloha  # noqa: F401
    except ImportError:
        print("  [live-aloha] gym_aloha not installed, skipping")
        return {"success_rate": 0.0, "avg_length": 0.0, "avg_inference_ms": 0.0}

    encoder = VisionEncoder(device=device)
    ema.shadow.to(device)
    ema.shadow.eval()

    env = gym.make(
        "gym_aloha/AlohaInsertion-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    successes = 0
    lengths: list[int] = []
    inference_times: list[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        action_buffer: list[Tensor] = []

        while not done and step < max_steps:
            if not action_buffer:
                pil_img = Image.fromarray(obs["pixels"]["top"])
                vision_emb = encoder.encode_pil(pil_img).unsqueeze(0).to(device)
                agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)
                state_norm = state_stats.normalize(agent_pos).unsqueeze(0).to(device)

                t0 = time.time()
                pred = trainer.sample(
                    vision_emb,
                    state_norm,
                    ema_model=ema.shadow,
                    num_steps=num_inference_steps,
                )
                inference_times.append((time.time() - t0) * 1000)

                action_buffer = [pred[0, k].cpu() for k in range(chunk_size)]

            action_norm = action_buffer.pop(0)
            action = action_stats.denormalize(action_norm).numpy().astype(np.float32)
            action = np.clip(action, -1.0, 1.0)

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as exc:
                # A diverging policy can drive the simulator into an invalid
                # state (e.g. mujoco mjWARN_BADQACC). Score this episode a
                # failure and continue, rather than aborting the whole run --
                # a failing policy is a result, not a crash.
                print(f"  [live-aloha] ep {ep}: simulator diverged "
                      f"({type(exc).__name__}); scoring episode as failure")
                info = {}
                break
            done = terminated or truncated
            step += 1

            if info.get("is_success", False):
                break

        successes += int(info.get("is_success", False))
        lengths.append(step)

    env.close()
    del encoder
    ema.shadow.to("cpu")
    torch.cuda.empty_cache()

    avg_ms = sum(inference_times) / max(len(inference_times), 1)
    metrics = {
        "success_rate": successes / n_episodes,
        "avg_length": sum(lengths) / len(lengths),
        "avg_inference_ms": avg_ms,
    }
    print(
        f"  [live-aloha] success={metrics['success_rate']:.0%} | "
        f"avg_len={metrics['avg_length']:.0f} | "
        f"inference={avg_ms:.1f}ms"
    )
    return metrics


# ===========================================================================
# Plotting
# ===========================================================================


def save_comparison_plots(results: dict[str, dict], preset_name: str) -> None:
    """Save training curves and success rate bar chart."""
    out_dir = Path(__file__).parent / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {"flow_matching": "tab:blue", "ddpm": "tab:orange"}

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, res in results.items():
        h = res.get("history", {})
        method = "flow_matching" if "flow" in name else "ddpm"
        color = colors.get(method, "tab:gray")
        if "train_loss" in h and h["train_loss"]:
            ax1.plot(h["train_loss"], label=name, color=color,
                     linestyle="-" if "K4" in name or "K=4" in name else "--")
        if "val_loss" in h and h["val_loss"]:
            ax2.plot(h["val_loss"], label=name, color=color,
                     linestyle="-" if "K4" in name or "K=4" in name else "--")

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=7)
    ax1.set_yscale("log")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=7)
    ax2.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir / f"ch06_{preset_name}_training_curves.png", dpi=150)
    plt.close()

    # Success rates
    names = list(results.keys())
    rates = [results[n].get("live", {}).get("success_rate", 0) for n in names]
    if any(r > 0 for r in rates):
        fig, ax = plt.subplots(figsize=(9, 5))
        bar_colors = [
            colors.get("flow_matching" if "flow" in n else "ddpm", "tab:gray")
            for n in names
        ]
        bars = ax.bar(names, rates, color=bar_colors)
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Ch06 {preset_name} -- Flow Matching vs DDPM")
        ax.set_ylim(0, 1)
        for bar, r in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{r:.0%}",
                ha="center",
                fontsize=9,
            )
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"ch06_{preset_name}_success_rates.png", dpi=150)
        plt.close()


# ===========================================================================
# Presets and CLI
# ===========================================================================

METHODS = ["flow_matching", "ddpm"]
CHUNK_SIZES = [4, 16]

PRESETS: dict[str, dict[str, Any]] = {
    "pusht": {
        "repo_id": "lerobot/pusht",
        "image_key": "observation.image",
        "state_key": "observation.state",
        "action_key": "action",
        "state_dim": 2,
        "action_dim": 2,
        "epochs": 200,
        "batch_size": 64,
        "lr": 1e-4,
        "eval_fn": "pusht",
        "eval_episodes": 10,
        "fm_inference_steps": 10,
        "ddpm_inference_steps": 20,
        # Smaller model to avoid overfitting on ~22K frames (206 episodes).
        # Full 96.3M model memorizes the velocity field without generalizing;
        # 12.4M is comparable to Diffusion Policy's UNet (~16M).
        "d_model": 384,
        "nhead": 6,
        "num_layers": 6,
        "ffn_dim": 1024,
        "dropout": 0.1,
    },
    "aloha": {
        "repo_id": "lerobot/aloha_sim_insertion_human",
        "image_key": "observation.images.top",
        "state_key": "observation.state",
        "action_key": "action",
        "state_dim": 14,
        "action_dim": 14,
        "epochs": 200,
        "batch_size": 64,
        "lr": 1e-4,
        "eval_fn": "aloha",
        "eval_episodes": 10,
        "fm_inference_steps": 10,
        "ddpm_inference_steps": 20,
        # ALOHA has ~50 episodes (~8K frames), even smaller model needed
        "d_model": 384,
        "nhead": 6,
        "num_layers": 6,
        "ffn_dim": 1024,
        "dropout": 0.1,
    },
}


def run_preset(
    preset_name: str,
    skip_precompute: bool = False,
    eval_only: bool = False,
    force_retrain: bool = False,
) -> None:
    """Run the full pipeline for a given preset.

    1. Extract and cache data
    2. Split by episode
    3. Normalize (from training set only)
    4. For each method x chunk_size: train, evaluate
    5. Print summary and save plots
    """
    cfg = PRESETS[preset_name]
    cache_prefix = f"ch06_{preset_name}"

    # Step 1: Extract data
    print(f"\n{'='*60}")
    print(f"Chapter 6: {preset_name.upper()}")
    print(f"{'='*60}")

    data = extract_and_cache(
        repo_id=cfg["repo_id"],
        image_key=cfg["image_key"],
        state_key=cfg["state_key"],
        action_key=cfg["action_key"],
        cache_prefix=cache_prefix,
        skip_embeddings=skip_precompute,
    )

    # Step 2: Split by episode
    train_mask, val_mask = split_by_episode(data["episode_indices"])

    # Step 3: Normalize from training set only
    train_actions = data["actions"][train_mask]
    train_states = data["states"][train_mask]
    action_stats = compute_norm_stats(train_actions)
    state_stats = compute_norm_stats(train_states)

    norm_actions = action_stats.normalize(data["actions"])
    norm_states = state_stats.normalize(data["states"])

    # Save stats for eval-only mode
    stats_path = CHECKPOINT_DIR / f"ch06_{preset_name}_stats.pt"
    torch.save(
        {
            "action_min": action_stats.min_val,
            "action_max": action_stats.max_val,
            "state_min": state_stats.min_val,
            "state_max": state_stats.max_val,
        },
        stats_path,
    )

    # Step 4: Train/evaluate all configurations
    all_results: dict[str, dict] = {}

    for method in METHODS:
        for chunk_size in CHUNK_SIZES:
            config_name = f"{method}_K{chunk_size}"
            ckpt_path = CHECKPOINT_DIR / f"ch06_{preset_name}_{method}_K{chunk_size}.pt"
            print(f"\n--- {config_name} ---")

            # Build model
            model = build_action_expert(
                action_dim=cfg["action_dim"],
                chunk_size=chunk_size,
                state_dim=cfg["state_dim"],
                d_model=cfg.get("d_model", 768),
                nhead=cfg.get("nhead", 12),
                num_layers=cfg.get("num_layers", 10),
                ffn_dim=cfg.get("ffn_dim", 3072),
                dropout=cfg.get("dropout", 0.0),
            )
            print(f"  Model: {model.count_parameters() / 1e6:.1f}M params")

            # Build trainer
            if method == "flow_matching":
                trainer = FlowMatchingTrainer(
                    model, inference_steps=cfg["fm_inference_steps"]
                )
            else:
                trainer = DDPMTransformerTrainer(
                    model,
                    train_timesteps=1000,
                    inference_steps=cfg["ddpm_inference_steps"],
                )

            # Build datasets
            train_ds = RobotDataset(
                data["cls_embeddings"][train_mask],
                norm_states[train_mask],
                norm_actions[train_mask],
                data["episode_indices"][train_mask],
                chunk_size=chunk_size,
            )
            val_ds = RobotDataset(
                data["cls_embeddings"][val_mask],
                norm_states[val_mask],
                norm_actions[val_mask],
                data["episode_indices"][val_mask],
                chunk_size=chunk_size,
            )

            # Train or load
            if eval_only or (ckpt_path.exists() and not force_retrain):
                if ckpt_path.exists():
                    print(f"  Loading checkpoint: {ckpt_path}")
                    ckpt = torch.load(
                        ckpt_path, map_location="cpu", weights_only=True
                    )
                    model.load_state_dict(ckpt["model"])
                    ema = EMAModel(model)
                    ema.load_state_dict(ckpt["ema"])
                    result: dict[str, Any] = {"history": {}}
                else:
                    print("  No checkpoint found, skipping eval")
                    continue
            else:
                result = train_model(
                    trainer,
                    train_ds,
                    val_ds,
                    epochs=cfg["epochs"],
                    batch_size=cfg["batch_size"],
                    lr=cfg["lr"],
                    ckpt_path=ckpt_path,
                )
                ema = result["ema"]

            # Live eval
            infer_steps = (
                cfg["fm_inference_steps"]
                if method == "flow_matching"
                else cfg["ddpm_inference_steps"]
            )
            if cfg["eval_fn"] == "pusht":
                live = evaluate_live_pusht(
                    trainer,
                    ema,
                    state_stats,
                    action_stats,
                    chunk_size,
                    n_episodes=cfg["eval_episodes"],
                    num_inference_steps=infer_steps,
                )
            else:
                live = evaluate_live_aloha(
                    trainer,
                    ema,
                    state_stats,
                    action_stats,
                    chunk_size,
                    n_episodes=cfg["eval_episodes"],
                    num_inference_steps=infer_steps,
                )

            result["live"] = live
            all_results[config_name] = result

            # Free memory
            del model, trainer, ema, train_ds, val_ds
            torch.cuda.empty_cache()

    # Step 5: Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {preset_name}")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Success':>10} {'Coverage':>10} {'Avg Len':>10} {'Infer ms':>10}")
    print("-" * 65)
    for name, res in all_results.items():
        live = res.get("live", {})
        sr = live.get("success_rate", 0.0)
        cov = live.get("avg_coverage", 0.0)
        al = live.get("avg_length", 0.0)
        ms = live.get("avg_inference_ms", 0.0)
        print(f"{name:<25} {sr:>9.0%} {cov:>9.1%} {al:>10.0f} {ms:>10.1f}")

    # Step 6: Save plots
    save_comparison_plots(all_results, preset_name)
    print(f"\nPlots saved to assets/figures/ch06_{preset_name}_*.png")


def run_inference_ablation(
    preset_name: str = "pusht",
    chunk_size: int = 16,
) -> None:
    """Ablation: how many ODE steps does flow matching need?

    Loads a trained FM model and tests with varying inference steps,
    plotting success rate and inference time.
    """
    cfg = PRESETS[preset_name]
    ckpt_path = CHECKPOINT_DIR / f"ch06_{preset_name}_flow_matching_K{chunk_size}.pt"

    if not ckpt_path.exists():
        print(f"[ablation] No checkpoint found at {ckpt_path}")
        print("  Run training first: python flow_matching.py --preset", preset_name)
        return

    # Load stats
    stats_path = CHECKPOINT_DIR / f"ch06_{preset_name}_stats.pt"
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    action_stats = NormStats(stats["action_min"], stats["action_max"])
    state_stats = NormStats(stats["state_min"], stats["state_max"])

    # Load model
    model = build_action_expert(
        action_dim=cfg["action_dim"],
        chunk_size=chunk_size,
        state_dim=cfg["state_dim"],
        d_model=cfg.get("d_model", 768),
        nhead=cfg.get("nhead", 12),
        num_layers=cfg.get("num_layers", 10),
        ffn_dim=cfg.get("ffn_dim", 3072),
        dropout=cfg.get("dropout", 0.0),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    ema = EMAModel(model)
    ema.load_state_dict(ckpt["ema"])

    trainer = FlowMatchingTrainer(model, inference_steps=10)

    step_counts = [3, 5, 10, 20, 50]
    success_rates: list[float] = []
    avg_times: list[float] = []

    eval_fn = evaluate_live_pusht if cfg["eval_fn"] == "pusht" else evaluate_live_aloha

    for steps in step_counts:
        print(f"\n[ablation] Testing {steps} inference steps...")
        live = eval_fn(
            trainer,
            ema,
            state_stats,
            action_stats,
            chunk_size,
            n_episodes=cfg["eval_episodes"],
            num_inference_steps=steps,
        )
        success_rates.append(live["success_rate"])
        avg_times.append(live["avg_inference_ms"])

    # Plot
    out_dir = Path(__file__).parent / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(step_counts, success_rates, "bo-")
    ax1.set_xlabel("Inference Steps")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate vs Inference Steps")
    ax1.set_ylim(0, 1)
    ax1.set_xscale("log")

    ax2.plot(step_counts, avg_times, "ro-")
    ax2.set_xlabel("Inference Steps")
    ax2.set_ylabel("Inference Time (ms)")
    ax2.set_title("Latency vs Inference Steps")
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.savefig(
        out_dir / f"ch06_{preset_name}_inference_ablation.png", dpi=150
    )
    plt.close()
    print(f"\n[ablation] Plot saved to ch06_{preset_name}_inference_ablation.png")

    # Cleanup
    del model, ema, trainer
    torch.cuda.empty_cache()


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 6: Flow Matching")
    parser.add_argument(
        "--preset",
        choices=["pusht", "aloha"],
        required=True,
        help="Dataset/task preset to run",
    )
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Skip SigLIP embedding extraction (use cached)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, evaluate from checkpoints",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if checkpoint exists",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run inference step ablation on trained FM model",
    )
    args = parser.parse_args()

    if args.ablation:
        run_inference_ablation(args.preset)
    else:
        run_preset(args.preset, args.skip_precompute, args.eval_only, args.force_retrain)
