"""Chapter 5: Scaling Up -- Real Robot Data Pipeline.

Load real robot datasets from LeRobot Hub (PushT and ALOHA), precompute SigLIP
embeddings, train action heads (discrete/regression/diffusion) with chunking,
and evaluate via live rollout. Demonstrates that toy MLP action heads fail on
real robot complexity -- motivating Chapter 6's flow-matching expert.

Usage:
    python data_pipeline.py --preset pusht
    python data_pipeline.py --preset aloha
    python data_pipeline.py --preset pusht --skip-precompute
    python data_pipeline.py --preset pusht --eval-only
"""

from __future__ import annotations

import argparse
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
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
SIGLIP_EMBED_DIM = 768
PROPRIO_EMBED_DIM = 64
FUSED_DIM = SIGLIP_EMBED_DIM + PROPRIO_EMBED_DIM  # 832

NUM_BINS = 256
DDPM_TRAIN_STEPS = 100
DDPM_INFERENCE_STEPS = 10

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Vision Encoder
# ===========================================================================


class VisionEncoder:
    """Frozen SigLIP ViT-B/16 -- extracts 768-D CLS embeddings."""

    def __init__(self, device: torch.device = DEVICE) -> None:
        from transformers import AutoModel, AutoImageProcessor

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self.model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).vision_model
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_pil(self, pil_image: Any) -> Tensor:
        """Encode a single PIL image -> (768,) on CPU."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output.squeeze(0).cpu()

    @torch.no_grad()
    def encode_batch(self, pil_images: list) -> Tensor:
        """Encode a batch of PIL images -> (B, 768) on CPU."""
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output.cpu()


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
) -> dict[str, Tensor]:
    """Single-pass extraction of embeddings, states, actions, episode indices.

    Downloads the LeRobot dataset, iterates once through all frames, extracts
    vision embeddings (via SigLIP), states, actions, and episode indices.
    Caches everything to disk for fast reloading.

    Args:
        repo_id: HuggingFace dataset ID, e.g. 'lerobot/pusht'.
        image_key: Key for image tensor in dataset, e.g. 'observation.image'.
        state_key: Key for state tensor, e.g. 'observation.state'.
        action_key: Key for action tensor, e.g. 'action'.
        cache_prefix: Prefix for cache files (e.g. 'pusht').
        batch_size: Batch size for SigLIP encoding.
        skip_embeddings: If True and cache exists, skip recomputation.

    Returns:
        Dict with keys: 'embeddings', 'states', 'actions', 'episode_indices'.
    """
    from PIL import Image

    emb_path = CHECKPOINT_DIR / f"{cache_prefix}_embeddings.pt"
    data_path = CHECKPOINT_DIR / f"{cache_prefix}_data.pt"

    # Try loading from cache
    if skip_embeddings and emb_path.exists() and data_path.exists():
        print(f"[cache] Loading from {emb_path} and {data_path}")
        embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
        data = torch.load(data_path, map_location="cpu", weights_only=True)
        return {
            "embeddings": embeddings,
            "states": data["states"],
            "actions": data["actions"],
            "episode_indices": data["episode_indices"],
        }

    # Load dataset from Hub
    from lerobot.datasets import LeRobotDataset
    dataset = LeRobotDataset(repo_id)
    n = len(dataset)
    print(f"[data] Loaded {repo_id}: {n} frames, {dataset.num_episodes} episodes")

    # Single pass: extract everything
    states_list: list[Tensor] = []
    actions_list: list[Tensor] = []
    episodes_list: list[int] = []

    # For embeddings, we batch PIL images through SigLIP
    encoder = VisionEncoder(device=DEVICE) if not (skip_embeddings and emb_path.exists()) else None
    embeddings = torch.empty(n, SIGLIP_EMBED_DIM, dtype=torch.float32)
    pil_batch: list = []
    batch_indices: list[int] = []

    print(f"[extract] Processing {n} frames (single pass)...")
    t0 = time.time()

    for idx in range(n):
        frame = dataset[idx]

        # State and action (always extract)
        states_list.append(frame[state_key])
        actions_list.append(frame[action_key])
        episodes_list.append(frame["episode_index"].item())

        # Image for SigLIP
        if encoder is not None:
            img_tensor = frame[image_key]  # (3, H, W) float [0, 1]
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_batch.append(Image.fromarray(img_np))
            batch_indices.append(idx)

            if len(pil_batch) == batch_size or idx == n - 1:
                embs = encoder.encode_batch(pil_batch)
                for i, emb_idx in enumerate(batch_indices):
                    embeddings[emb_idx] = embs[i]
                pil_batch = []
                batch_indices = []

        if (idx + 1) % 5000 == 0 or idx == n - 1:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"  [{idx + 1}/{n}] {fps:.0f} frames/s")

    # Free encoder
    if encoder is not None:
        del encoder
        torch.cuda.empty_cache()

    states = torch.stack(states_list)
    actions = torch.stack(actions_list)
    episode_indices = torch.tensor(episodes_list, dtype=torch.long)

    print(f"[extract] Done in {time.time() - t0:.1f}s")
    print(f"  States: {states.shape}, Actions: {actions.shape}")
    print(f"  Episodes: {episode_indices.unique().shape[0]}")

    # Cache to disk
    torch.save(embeddings, emb_path)
    torch.save({
        "states": states,
        "actions": actions,
        "episode_indices": episode_indices,
    }, data_path)
    print(f"[cache] Saved to {emb_path} and {data_path}")

    return {
        "embeddings": embeddings,
        "states": states,
        "actions": actions,
        "episode_indices": episode_indices,
    }


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
    print(f"[split] {n_train_eps} train episodes ({train_mask.sum()} frames), "
          f"{n_val} val episodes ({val_mask.sum()} frames)")
    return train_mask, val_mask


# ===========================================================================
# Dataset
# ===========================================================================


class RobotDataset(Dataset):
    """PyTorch Dataset from pre-extracted tensors with action chunking.

    All inputs are in-memory tensors (fast, no I/O during training).
    """

    def __init__(
        self,
        embeddings: Tensor,
        states: Tensor,
        actions: Tensor,
        episode_indices: Tensor,
        chunk_size: int = 1,
    ) -> None:
        self.embeddings = embeddings
        self.states = states
        self.actions = actions
        self.episode_indices = episode_indices
        self.chunk_size = chunk_size
        self.n = len(embeddings)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        emb = self.embeddings[idx]
        state = self.states[idx]
        ep = self.episode_indices[idx]

        # Build action chunk: K consecutive actions, pad at episode boundary
        chunk = []
        for k in range(self.chunk_size):
            future_idx = idx + k
            if future_idx < self.n and self.episode_indices[future_idx] == ep:
                chunk.append(self.actions[future_idx])
            else:
                chunk.append(chunk[-1] if chunk else self.actions[idx])

        return emb, state, torch.stack(chunk)


# ===========================================================================
# Models
# ===========================================================================


class ProprioEncoder(nn.Module):
    """Projects proprioceptive state to a fixed-size embedding."""

    def __init__(self, state_dim: int, embed_dim: int = PROPRIO_EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


class DiscreteActionHead(nn.Module):
    """256-bin tokenization per action dimension (RT-2/OpenVLA style)."""

    def __init__(self, input_dim: int, action_dim: int, chunk_size: int = 1) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, chunk_size * action_dim * NUM_BINS),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Returns logits (B, K, D, NUM_BINS)."""
        B = x.shape[0]
        return self.net(x).view(B, self.chunk_size, self.action_dim, NUM_BINS)

    def compute_loss(self, x: Tensor, target: Tensor) -> Tensor:
        """Cross-entropy loss. target: (B, K, D) in [-1, 1]."""
        logits = self.forward(x)
        bins = ((target + 1.0) / 2.0 * (NUM_BINS - 1)).long().clamp(0, NUM_BINS - 1)
        logits_flat = logits.reshape(-1, NUM_BINS)
        bins_flat = bins.reshape(-1)
        return nn.functional.cross_entropy(logits_flat, bins_flat)

    def predict(self, x: Tensor) -> Tensor:
        """Predict actions (B, K, D) in [-1, 1]."""
        logits = self.forward(x)
        bins = logits.argmax(dim=-1)
        return bins.float() / (NUM_BINS - 1) * 2.0 - 1.0


class RegressionActionHead(nn.Module):
    """MSE regression with Tanh output."""

    def __init__(self, input_dim: int, action_dim: int, chunk_size: int = 1) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, chunk_size * action_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        return self.net(x).view(B, self.chunk_size, self.action_dim)

    def compute_loss(self, x: Tensor, target: Tensor) -> Tensor:
        pred = self.forward(x)
        return nn.functional.mse_loss(pred, target)

    def predict(self, x: Tensor) -> Tensor:
        return self.forward(x)


class DiffusionActionHead(nn.Module):
    """DDPM-based action head with 2-layer MLP denoiser."""

    def __init__(self, input_dim: int, action_dim: int, chunk_size: int = 1) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.action_size = action_dim * chunk_size

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, DDPM_TRAIN_STEPS)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_ac", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_1mac", torch.sqrt(1.0 - alphas_cumprod))

        # Time embedding (sinusoidal)
        time_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Denoiser
        denoiser_in = self.action_size + 64 + input_dim
        self.denoiser = nn.Sequential(
            nn.Linear(denoiser_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.action_size),
        )

    def _sinusoidal_embedding(self, t: Tensor) -> Tensor:
        """(B,) -> (B, 64)."""
        half = 32
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def compute_loss(self, x: Tensor, target: Tensor) -> Tensor:
        """Diffusion training loss (predict noise)."""
        B = x.shape[0]
        target_flat = target.reshape(B, -1)

        t = torch.randint(0, DDPM_TRAIN_STEPS, (B,), device=x.device)
        noise = torch.randn_like(target_flat)

        noisy = self.sqrt_ac[t].unsqueeze(1) * target_flat + self.sqrt_1mac[t].unsqueeze(1) * noise

        t_emb = self.time_mlp(self._sinusoidal_embedding(t.float()))
        pred_noise = self.denoiser(torch.cat([noisy, t_emb, x], dim=1))

        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """DDPM sampling."""
        B = x.shape[0]
        action = torch.randn(B, self.action_size, device=x.device)

        timesteps = torch.linspace(DDPM_TRAIN_STEPS - 1, 0, DDPM_INFERENCE_STEPS,
                                   device=x.device).long()

        for t_val in timesteps:
            t = t_val.expand(B)
            t_emb = self.time_mlp(self._sinusoidal_embedding(t.float()))
            pred_noise = self.denoiser(torch.cat([action, t_emb, x], dim=1))

            alpha_bar = self.alphas_cumprod[t_val]
            pred_x0 = (action - self.sqrt_1mac[t_val] * pred_noise) / self.sqrt_ac[t_val]
            pred_x0 = pred_x0.clamp(-1, 1)

            if t_val > 0:
                alpha_bar_prev = self.alphas_cumprod[t_val - 1]
                action = (torch.sqrt(alpha_bar_prev) * pred_x0
                          + torch.sqrt(1 - alpha_bar_prev) * pred_noise)
            else:
                action = pred_x0

        return action.view(B, self.chunk_size, self.action_dim)


class VLA(nn.Module):
    """Vision-Language-Action model (single-task, no language conditioning)."""

    def __init__(self, proprio_encoder: ProprioEncoder, action_head: nn.Module,
                 head_type: str) -> None:
        super().__init__()
        self.proprio_encoder = proprio_encoder
        self.action_head = action_head
        self.head_type = head_type

    def forward(self, vision_emb: Tensor, state: Tensor,
                target: Tensor | None = None) -> Tensor:
        """If target provided: return loss. Otherwise: return predictions."""
        proprio = self.proprio_encoder(state)
        fused = torch.cat([vision_emb, proprio], dim=1)

        if target is not None:
            return self.action_head.compute_loss(fused, target)
        return self.action_head.predict(fused)

    def predict(self, vision_emb: Tensor, state: Tensor) -> Tensor:
        return self.forward(vision_emb, state, target=None)


HEAD_TYPES = ["discrete", "regression", "diffusion"]
CHUNK_SIZES = [1, 4]


def build_vla(head_type: str, state_dim: int, action_dim: int,
              chunk_size: int = 1) -> VLA:
    """Factory: build a VLA with the specified action head."""
    proprio = ProprioEncoder(state_dim)
    if head_type == "discrete":
        head = DiscreteActionHead(FUSED_DIM, action_dim, chunk_size)
    elif head_type == "regression":
        head = RegressionActionHead(FUSED_DIM, action_dim, chunk_size)
    elif head_type == "diffusion":
        head = DiffusionActionHead(FUSED_DIM, action_dim, chunk_size)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
    return VLA(proprio, head, head_type)


# ===========================================================================
# Training
# ===========================================================================


def train(
    model: VLA,
    train_ds: RobotDataset,
    val_ds: RobotDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    ckpt_path: Path,
    device: torch.device = DEVICE,
) -> dict[str, list[float]]:
    """Train with cosine LR, bf16 AMP, gradient clipping, early stopping."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        total_loss, n_batch = 0.0, 0
        for emb, state, actions in train_loader:
            emb, state, actions = emb.to(device), state.to(device), actions.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(device.type == "cuda")):
                loss = model(emb, state, actions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()
        train_loss = total_loss / max(n_batch, 1)
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for emb, state, actions in val_loader:
                emb, state, actions = emb.to(device), state.to(device), actions.to(device)
                with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(device.type == "cuda")):
                    loss = model(emb, state, actions)
                val_total += loss.item()
                val_n += 1
        val_loss = val_total / max(val_n, 1)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Load best
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to("cpu")
    return history


# ===========================================================================
# Evaluation — Offline
# ===========================================================================


def evaluate_offline(model: VLA, val_ds: RobotDataset,
                     device: torch.device = DEVICE) -> dict[str, float]:
    """Offline metrics: val loss and mean absolute action error."""
    model.to(device)
    model.eval()
    loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    total_loss, total_mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for emb, state, actions in loader:
            emb, state, actions = emb.to(device), state.to(device), actions.to(device)
            with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(device.type == "cuda")):
                loss = model(emb, state, actions)
            pred = model.predict(emb, state)
            total_loss += loss.item()
            total_mae += (pred - actions).abs().mean().item()
            n += 1

    model.to("cpu")
    metrics = {"val_loss": total_loss / max(n, 1), "mae": total_mae / max(n, 1)}
    print(f"  [offline] loss={metrics['val_loss']:.4f} | MAE={metrics['mae']:.4f}")
    return metrics


# ===========================================================================
# Evaluation — Live (PushT)
# ===========================================================================


def evaluate_live_pusht(
    model: VLA,
    state_stats: NormStats,
    action_stats: NormStats,
    n_episodes: int = 50,
    max_steps: int = 300,
    device: torch.device = DEVICE,
) -> dict[str, float]:
    """Live rollout in gym-pusht/PushT-v0."""
    import gymnasium as gym
    from PIL import Image

    try:
        import gym_pusht  # noqa: F401
    except ImportError:
        print("  [live-pusht] gym_pusht not installed, skipping")
        return {"success_rate": 0.0, "avg_length": 0.0}

    encoder = VisionEncoder(device=device)
    model.to(device)
    model.eval()

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array")
    successes = 0
    lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            pil_img = Image.fromarray(obs["pixels"])
            vision_emb = encoder.encode_pil(pil_img).unsqueeze(0).to(device)

            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)
            state_norm = state_stats.normalize(agent_pos).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model.predict(vision_emb, state_norm)

            action_norm = pred[0, 0].cpu()
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

            if info.get("is_success", False):
                break

        successes += int(info.get("is_success", False))
        lengths.append(step)

    env.close()
    del encoder
    model.to("cpu")
    torch.cuda.empty_cache()

    metrics = {"success_rate": successes / n_episodes,
               "avg_length": sum(lengths) / len(lengths)}
    print(f"  [live-pusht] success={metrics['success_rate']:.0%} | "
          f"avg_len={metrics['avg_length']:.0f}")
    return metrics


# ===========================================================================
# Evaluation — Live (ALOHA)
# ===========================================================================


def evaluate_live_aloha(
    model: VLA,
    state_stats: NormStats,
    action_stats: NormStats,
    n_episodes: int = 50,
    max_steps: int = 400,
    device: torch.device = DEVICE,
) -> dict[str, float]:
    """Live rollout in gym-aloha/AlohaInsertion-v0."""
    import gymnasium as gym
    from PIL import Image

    try:
        import gym_aloha  # noqa: F401
    except ImportError:
        print("  [live-aloha] gym_aloha not installed, skipping")
        return {"success_rate": 0.0, "avg_length": 0.0}

    encoder = VisionEncoder(device=device)
    model.to(device)
    model.eval()

    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array")
    successes = 0
    lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            pil_img = Image.fromarray(obs["pixels"]["top"])
            vision_emb = encoder.encode_pil(pil_img).unsqueeze(0).to(device)

            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.float32)
            state_norm = state_stats.normalize(agent_pos).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model.predict(vision_emb, state_norm)

            action_norm = pred[0, 0].cpu()
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
    model.to("cpu")
    torch.cuda.empty_cache()

    metrics = {"success_rate": successes / n_episodes,
               "avg_length": sum(lengths) / len(lengths)}
    print(f"  [live-aloha] success={metrics['success_rate']:.0%} | "
          f"avg_len={metrics['avg_length']:.0f}")
    return metrics


# ===========================================================================
# Plotting
# ===========================================================================


def save_comparison_plots(results: dict[str, dict], preset_name: str) -> None:
    """Save training curves and success rate bar chart."""
    out_dir = Path("assets/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, res in results.items():
        h = res.get("history", {})
        if "train_loss" in h and h["train_loss"]:
            ax1.plot(h["train_loss"], label=name)
        if "val_loss" in h and h["val_loss"]:
            ax2.plot(h["val_loss"], label=name)

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
    plt.savefig(out_dir / f"ch05_{preset_name}_training_curves.png", dpi=150)
    plt.close()

    # Success rates
    names = list(results.keys())
    rates = [results[n].get("live", {}).get("success_rate", 0) for n in names]
    if any(r > 0 for r in rates):
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(names, rates, color="steelblue")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Ch05 {preset_name} — Live Eval")
        ax.set_ylim(0, 1)
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{r:.0%}", ha="center", fontsize=9)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"ch05_{preset_name}_success_rates.png", dpi=150)
        plt.close()


# ===========================================================================
# Presets and CLI
# ===========================================================================


PRESETS: dict[str, dict[str, Any]] = {
    "pusht": {
        "repo_id": "lerobot/pusht",
        "image_key": "observation.image",
        "state_key": "observation.state",
        "action_key": "action",
        "state_dim": 2,
        "action_dim": 2,
        "epochs": 200,
        "batch_size": 128,
        "lr": 1e-3,
        "patience": 20,
        "eval_fn": "pusht",
        "eval_episodes": 50,
    },
    "aloha": {
        "repo_id": "lerobot/aloha_sim_insertion_human",
        "image_key": "observation.images.top",
        "state_key": "observation.state",
        "action_key": "action",
        "state_dim": 14,
        "action_dim": 14,
        "epochs": 200,
        "batch_size": 128,
        "lr": 1e-3,
        "patience": 20,
        "eval_fn": "aloha",
        "eval_episodes": 50,
    },
}


def run_preset(preset_name: str, skip_precompute: bool = False,
               eval_only: bool = False) -> None:
    """Run the full pipeline for a preset."""
    cfg = PRESETS[preset_name]
    print(f"\n{'=' * 60}")
    print(f"  Chapter 5: Scaling Up -- {preset_name.upper()}")
    print(f"{'=' * 60}\n")

    # 1. Extract data (single pass, cached)
    data = extract_and_cache(
        repo_id=cfg["repo_id"],
        image_key=cfg["image_key"],
        state_key=cfg["state_key"],
        action_key=cfg["action_key"],
        cache_prefix=preset_name,
        skip_embeddings=skip_precompute,
    )

    embeddings = data["embeddings"]
    states = data["states"]
    actions = data["actions"]
    episode_indices = data["episode_indices"]

    # 2. Split by episode
    train_mask, val_mask = split_by_episode(episode_indices)

    # 3. Compute normalization (from training set only)
    action_stats = compute_norm_stats(actions[train_mask])
    state_stats = compute_norm_stats(states[train_mask])
    print(f"[norm] Action range: {action_stats.min_val.tolist()[:3]} to "
          f"{action_stats.max_val.tolist()[:3]}")

    # Normalize
    norm_actions = action_stats.normalize(actions)
    norm_states = state_stats.normalize(states)

    # Save stats for eval
    torch.save({"action_stats": action_stats, "state_stats": state_stats},
               CHECKPOINT_DIR / f"{preset_name}_stats.pt")

    # 4. Train all configs
    results: dict[str, dict] = {}

    for head_type in HEAD_TYPES:
        for chunk_size in CHUNK_SIZES:
            config_name = f"{head_type}_K{chunk_size}"
            ckpt_path = CHECKPOINT_DIR / f"{preset_name}_{config_name}.pt"
            print(f"\n--- {config_name} ---")

            model = build_vla(head_type, cfg["state_dim"], cfg["action_dim"], chunk_size)

            if eval_only and ckpt_path.exists():
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu",
                                                 weights_only=True))
                print(f"  Loaded from {ckpt_path}")
                history = {"train_loss": [], "val_loss": []}
            elif not eval_only:
                train_ds = RobotDataset(embeddings[train_mask], norm_states[train_mask],
                                        norm_actions[train_mask], episode_indices[train_mask],
                                        chunk_size)
                val_ds = RobotDataset(embeddings[val_mask], norm_states[val_mask],
                                      norm_actions[val_mask], episode_indices[val_mask],
                                      chunk_size)
                history = train(model, train_ds, val_ds,
                               epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                               lr=cfg["lr"], patience=cfg["patience"],
                               ckpt_path=ckpt_path, device=DEVICE)
            else:
                print(f"  WARNING: no checkpoint at {ckpt_path}")
                history = {"train_loss": [], "val_loss": []}

            # Offline eval
            val_ds_eval = RobotDataset(embeddings[val_mask], norm_states[val_mask],
                                       norm_actions[val_mask], episode_indices[val_mask],
                                       chunk_size)
            offline = evaluate_offline(model, val_ds_eval, DEVICE)

            # Live eval
            if cfg["eval_fn"] == "pusht":
                live = evaluate_live_pusht(model, state_stats, action_stats,
                                          n_episodes=cfg["eval_episodes"], device=DEVICE)
            else:
                live = evaluate_live_aloha(model, state_stats, action_stats,
                                          n_episodes=cfg["eval_episodes"], device=DEVICE)

            results[config_name] = {"history": history, "offline": offline, "live": live}
            del model
            torch.cuda.empty_cache()

    # 5. Summary
    print(f"\n{'=' * 60}")
    print(f"  {preset_name.upper()} Results")
    print(f"{'=' * 60}")
    print(f"{'Config':<20} {'Success':>8} {'MAE':>8} {'Val Loss':>10}")
    print("-" * 50)
    for name, res in results.items():
        sr = res["live"].get("success_rate", 0)
        mae = res["offline"].get("mae", 0)
        vl = res["offline"].get("val_loss", 0)
        print(f"{name:<20} {sr:>7.0%} {mae:>8.4f} {vl:>10.4f}")

    # 6. Plots
    save_comparison_plots(results, preset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chapter 5: Scaling Up")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), required=True)
    parser.add_argument("--skip-precompute", action="store_true",
                        help="Use cached embeddings (skip SigLIP extraction)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load checkpoints, evaluate only")
    args = parser.parse_args()

    run_preset(args.preset, skip_precompute=args.skip_precompute,
               eval_only=args.eval_only)
