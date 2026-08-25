"""Chapter 4: Action Representations -- comparing discrete, regression, and diffusion.

Compares three action head strategies on ContinuousMiniPushT:
  1. Discrete (256-bin tokenization, cross-entropy) -- RT-2/OpenVLA style
  2. Regression (MLP, MSE loss) -- simplest baseline
  3. Diffusion (DDPM denoising, noise prediction) -- pi0/SmolVLA style

Each head supports action chunking (K=1 single-step or K=4 trajectory).

Run standalone: python action_heads.py --head discrete --chunk-size 1 --preset quick
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor

sys.path.insert(0, os.path.dirname(__file__))
from mini_pusht_continuous import ContinuousMiniPushT, ContinuousExpert


# ---------------------------------------------------------------------------
# Vision encoder (frozen SigLIP, same as ch02/03)
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """Frozen SigLIP vision encoder -> pooled (B, 768) features."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224") -> None:
        super().__init__()
        processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_dim: int = self.model.config.vision_config.hidden_size
        self.register_buffer(
            "mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(processor.image_std).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float [0,1] -> (B, embed_dim)."""
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        pixel_values = (image - self.mean) / self.std
        outputs = self.model.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output


# ---------------------------------------------------------------------------
# Action Head 1: Discrete Tokenization (RT-2 / OpenVLA style)
# ---------------------------------------------------------------------------

N_BINS = 256


def continuous_to_bins(actions: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
    """Convert continuous actions in [-1, 1] to bin indices in [0, n_bins-1]."""
    clamped = actions.clamp(-1.0, 1.0)
    bins = ((clamped + 1.0) / 2.0 * (n_bins - 1)).long()
    return bins.clamp(0, n_bins - 1)


def bins_to_continuous(bins: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
    """Convert bin indices back to continuous values (bin centers)."""
    return (bins.float() / (n_bins - 1)) * 2.0 - 1.0


class DiscreteActionHead(nn.Module):
    """RT-2/OpenVLA style: bin each action dim into N_BINS tokens.

    For chunk_size K and action_dim D, predicts K*D sets of N_BINS logits.
    Loss: cross-entropy per dimension per timestep.
    """

    def __init__(
        self, input_dim: int, action_dim: int = 2, chunk_size: int = 1
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_bins = N_BINS
        total_tokens = action_dim * chunk_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, total_tokens * self.n_bins),
        )

    def loss(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            features: (B, input_dim)
            actions: (B, chunk_size, action_dim) or (B, action_dim) if chunk_size=1
        """
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)  # (B, 1, D)
        B = features.shape[0]
        logits = self.mlp(features)  # (B, K*D*N_BINS)
        logits = logits.view(B, self.chunk_size, self.action_dim, self.n_bins)
        targets = continuous_to_bins(actions)  # (B, K, D) long
        loss = F.cross_entropy(
            logits.reshape(-1, self.n_bins), targets.reshape(-1)
        )
        return loss

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict continuous actions from features.

        Returns: (B, chunk_size, action_dim) or (B, action_dim) if chunk_size=1
        """
        B = features.shape[0]
        logits = self.mlp(features)
        logits = logits.view(B, self.chunk_size, self.action_dim, self.n_bins)
        bin_indices = logits.argmax(dim=-1)  # (B, K, D)
        actions = bins_to_continuous(bin_indices)
        if self.chunk_size == 1:
            actions = actions.squeeze(1)  # (B, D)
        return actions


# ---------------------------------------------------------------------------
# Action Head 2: MSE Regression
# ---------------------------------------------------------------------------

class RegressionActionHead(nn.Module):
    """Direct regression with MSE loss. Simple MLP with Tanh output."""

    def __init__(
        self, input_dim: int, action_dim: int = 2, chunk_size: int = 1
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        output_dim = action_dim * chunk_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh(),
        )

    def loss(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """MSE loss between predicted and target actions."""
        if actions.dim() == 3:
            actions = actions.reshape(actions.shape[0], -1)  # (B, K*D)
        pred = self.mlp(features)  # (B, K*D)
        return F.mse_loss(pred, actions)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict continuous actions."""
        B = features.shape[0]
        pred = self.mlp(features)  # (B, K*D)
        if self.chunk_size > 1:
            pred = pred.view(B, self.chunk_size, self.action_dim)
        return pred


# ---------------------------------------------------------------------------
# Action Head 3: Diffusion Denoising (DDPM)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer or float timesteps -> (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionActionHead(nn.Module):
    """DDPM-style diffusion action head.

    Trains a small MLP to predict noise added to actions.
    At inference, iteratively denoises from Gaussian noise.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 2,
        chunk_size: int = 1,
        n_timesteps: int = 100,
        n_inference_steps: int = 10,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_timesteps = n_timesteps
        self.n_inference_steps = n_inference_steps
        self.action_flat_dim = action_dim * chunk_size

        time_dim = 64
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        net_input_dim = self.action_flat_dim + time_dim + input_dim
        self.net = nn.Sequential(
            nn.Linear(net_input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.action_flat_dim),
        )

        # Linear beta schedule
        betas = torch.linspace(1e-4, 0.02, n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _forward_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise given noised actions, timestep, and conditioning."""
        t_emb = self.time_embed(t)
        x_flat = x_t.view(x_t.shape[0], -1)
        net_input = torch.cat([x_flat, t_emb, features], dim=-1)
        return self.net(net_input)

    def loss(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """DDPM training loss: predict noise added to actions."""
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        B = actions.shape[0]
        x_0 = actions.reshape(B, -1)

        t = torch.randint(0, self.n_timesteps, (B,), device=features.device)
        noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise

        pred_noise = self._forward_noise(x_t, t, features)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """DDPM sampling: denoise from Gaussian noise."""
        B = features.shape[0]
        device = features.device

        x = torch.randn(B, self.action_flat_dim, device=device)
        step_size = max(1, self.n_timesteps // self.n_inference_steps)
        timesteps = list(range(self.n_timesteps - 1, -1, -step_size))

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            pred_noise = self._forward_noise(x, t, features)

            alpha = self.alphas[t_val]
            alpha_bar = self.alphas_cumprod[t_val]
            beta = self.betas[t_val]

            x = (1.0 / alpha.sqrt()) * (
                x - (beta / (1.0 - alpha_bar).sqrt()) * pred_noise
            )
            if t_val > 0:
                noise = torch.randn_like(x)
                x = x + beta.sqrt() * noise

        x = x.clamp(-1.0, 1.0)

        if self.chunk_size > 1:
            x = x.view(B, self.chunk_size, self.action_dim)
        else:
            x = x.view(B, self.action_dim)
        return x


# ---------------------------------------------------------------------------
# VLA backbone
# ---------------------------------------------------------------------------

class VLA(nn.Module):
    """Vision-Language-Action model with swappable action head.

    Vision: frozen SigLIP (precomputed at training time).
    Language: fixed instruction, no encoder (simple learned embedding).
    Action: one of DiscreteActionHead, RegressionActionHead, DiffusionActionHead.
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        action_head: nn.Module,
        vision_dim: int = 768,
        task_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.task_embedding = nn.Embedding(1, task_embed_dim)
        self.action_head = action_head

    def forward(
        self, image: torch.Tensor, actions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass. Supports precomputed embeddings (B, D) or raw images (B, 3, H, W)."""
        if image.dim() == 2:
            features = image
        else:
            features = self.vision_encoder(image)

        B = features.shape[0]
        task_emb = self.task_embedding(
            torch.zeros(B, dtype=torch.long, device=features.device)
        )
        features = torch.cat([features, task_emb], dim=-1)

        if actions is not None:
            return self.action_head.loss(features, actions)
        else:
            return self.action_head.predict(features)


def build_vla(
    head_type: str,
    chunk_size: int = 1,
    encoder_name: str = "google/siglip-base-patch16-224",
    task_embed_dim: int = 64,
) -> VLA:
    """Factory to build VLA with specified action head."""
    vision_encoder = VisionEncoder(encoder_name)
    vision_dim = vision_encoder.embed_dim
    input_dim = vision_dim + task_embed_dim  # 768 + 64 = 832

    if head_type == "discrete":
        action_head = DiscreteActionHead(input_dim, action_dim=2, chunk_size=chunk_size)
    elif head_type == "regression":
        action_head = RegressionActionHead(input_dim, action_dim=2, chunk_size=chunk_size)
    elif head_type == "diffusion":
        action_head = DiffusionActionHead(input_dim, action_dim=2, chunk_size=chunk_size)
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    return VLA(vision_encoder, action_head, vision_dim=vision_dim, task_embed_dim=task_embed_dim)


# ---------------------------------------------------------------------------
# Data collection (mmap, RAM-safe)
# ---------------------------------------------------------------------------

def collect_demos_to_disk(
    env: ContinuousMiniPushT,
    expert: ContinuousExpert,
    path: str,
    n_episodes: int = 1000,
) -> int:
    """Collect expert demos, stream to disk via mmap. RAM-safe.

    Returns the total number of transitions written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    size = env.size
    est_transitions = n_episodes * 200

    tmp_images = path + ".tmp_images.npy"
    images_mmap = np.lib.format.open_memmap(
        tmp_images, mode="w+", dtype=np.uint8,
        shape=(est_transitions, size, size, 3),
    )
    actions_list: list[np.ndarray] = []

    idx = 0
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = expert.act(info)
            if idx >= len(images_mmap):
                new_size = int(len(images_mmap) * 1.5)
                new_mmap = np.lib.format.open_memmap(
                    tmp_images + ".ext", mode="w+", dtype=np.uint8,
                    shape=(new_size, size, size, 3),
                )
                new_mmap[:idx] = images_mmap[:idx]
                del images_mmap
                os.replace(tmp_images + ".ext", tmp_images)
                images_mmap = np.lib.format.open_memmap(tmp_images, mode="r+")
            images_mmap[idx] = obs
            actions_list.append(action.copy())
            idx += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if (ep + 1) % max(1, n_episodes // 10) == 0:
            print(f"      {ep + 1}/{n_episodes} episodes ({idx} transitions)")

    total = idx
    actions_arr = np.array(actions_list[:total], dtype=np.float32)

    print(f"      Saving {total} transitions metadata to {path}")
    np.savez_compressed(path, actions=actions_arr)

    # Create properly-sized final mmap (chunk-copy to avoid RAM spike)
    final_images = path.replace(".npz", "_images.npy")
    del images_mmap

    tmp_mmap = np.lib.format.open_memmap(tmp_images, mode="r")
    final_mmap = np.lib.format.open_memmap(
        final_images, mode="w+", dtype=np.uint8,
        shape=(total, size, size, 3),
    )
    chunk = 10000
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        final_mmap[start:end] = tmp_mmap[start:end]
    del tmp_mmap, final_mmap
    os.remove(tmp_images)

    print(f"      Final: {total} transitions, images at {final_images}")
    return total


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class MmapPushTDataset(Dataset):
    """Memory-mapped dataset for continuous MiniPushT demos."""

    def __init__(self, npz_path: str, chunk_size: int = 1) -> None:
        data = np.load(npz_path, allow_pickle=False)
        self.actions = data["actions"]  # (N, 2) float32
        images_path = npz_path.replace(".npz", "_images.npy")
        self.images = np.load(images_path, mmap_mode="r")
        self.chunk_size = chunk_size
        self.n_total = len(self.actions) - (chunk_size - 1)

    def __len__(self) -> int:
        return self.n_total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[idx].copy())  # (H, W, 3) uint8
        if self.chunk_size == 1:
            action = torch.from_numpy(self.actions[idx].copy())  # (2,)
        else:
            action = torch.from_numpy(
                self.actions[idx : idx + self.chunk_size].copy()
            )  # (K, 2)
        return image, action


class EmbeddingDataset(Dataset):
    """Dataset of precomputed vision embeddings + actions."""

    def __init__(self, embeddings: torch.Tensor, actions: torch.Tensor) -> None:
        self.embeddings = embeddings
        self.actions = actions

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# Precompute embeddings
# ---------------------------------------------------------------------------

def precompute_embeddings(
    vision_encoder: VisionEncoder,
    dataset: MmapPushTDataset,
    device: str = "cpu",
    batch_size: int = 64,
) -> EmbeddingDataset:
    """Run frozen SigLIP once over dataset, return cached embeddings.

    Uses 0 DataLoader workers to avoid RAM explosion on large mmap files.
    """
    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device != "cpu"))

    all_embeds = []
    all_actions = []

    print("      Pre-computing SigLIP embeddings...")
    with torch.no_grad():
        for batch_idx, (images, actions) in enumerate(loader):
            images = images.to(device).float().permute(0, 3, 1, 2) / 255.0
            embeds = vision_encoder(images)
            all_embeds.append(embeds.cpu())
            all_actions.append(actions)
            if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                print(f"        batch {batch_idx + 1}/{len(loader)}")

    print(f"      Cached {sum(e.shape[0] for e in all_embeds)} embeddings.")
    return EmbeddingDataset(
        embeddings=torch.cat(all_embeds, dim=0),
        actions=torch.cat(all_actions, dim=0),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: VLA,
    dataset: EmbeddingDataset,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 0,
) -> list[float]:
    """Train VLA on precomputed embeddings. Returns per-epoch loss history.

    Uses bf16 AMP on GPU, cosine LR, gradient clipping.
    """
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = device != "cpu"
    use_workers = device != "cpu"
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2 if use_workers else 0,
        pin_memory=use_workers,
        persistent_workers=True if use_workers else False,
    )

    loss_history: list[float] = []
    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for embeddings, actions in loader:
            embeddings = embeddings.to(device)
            actions = actions.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(embeddings, actions=actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"        Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if patience > 0:
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"        Early stopping at epoch {epoch+1}")
                    break

    return loss_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: VLA,
    env: ContinuousMiniPushT,
    vision_encoder: VisionEncoder,
    n_episodes: int = 100,
    device: str = "cpu",
) -> dict:
    """Evaluate VLA on live environment episodes.

    Returns dict with success_rate, mean_length, mean_smoothness.
    """
    model = model.to(device)
    model.eval()
    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()

    successes = 0
    lengths = []
    smoothness_scores = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        steps = 0
        prev_action = None
        action_diffs = []

        while not done and steps < 200:
            img_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_t = img_t.to(device)

            with torch.no_grad():
                features = vision_encoder(img_t)
                action = model(features)

            if action.dim() == 3:
                action_chunk = action[0].cpu().numpy()  # (K, 2)
                for a in action_chunk:
                    if done:
                        break
                    obs, reward, terminated, truncated, info = env.step(a)
                    done = terminated or truncated
                    steps += 1
                    if prev_action is not None:
                        action_diffs.append(np.linalg.norm(a - prev_action))
                    prev_action = a.copy()
            else:
                a = action[0].cpu().numpy()  # (2,)
                obs, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                steps += 1
                if prev_action is not None:
                    action_diffs.append(np.linalg.norm(a - prev_action))
                prev_action = a.copy()

        if terminated:
            successes += 1
        lengths.append(steps)
        if action_diffs:
            smoothness_scores.append(np.mean(action_diffs))

    return {
        "success_rate": successes / n_episodes,
        "mean_length": np.mean(lengths),
        "mean_smoothness": np.mean(smoothness_scores) if smoothness_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_comparison_plots(
    results: dict[str, dict],
    losses: dict[str, list[float]],
    save_dir: str = "../../assets/figures",
) -> None:
    """Save comparison bar charts and training curves."""
    save_dir_abs = save_dir
    if not os.path.isabs(save_dir):
        save_dir_abs = os.path.join(os.path.dirname(__file__) or ".", save_dir)
    os.makedirs(save_dir_abs, exist_ok=True)

    names = list(results.keys())

    # Training curves
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, loss_hist in losses.items():
        ax.plot(range(1, len(loss_hist) + 1), loss_hist, marker="o", markersize=2, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Chapter 4: Training loss by action head")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir_abs, "ch04_training_curves.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Comparison bar charts
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["success_rate", "mean_smoothness", "mean_length"]
    titles = ["Success Rate", "Trajectory Smoothness\n(lower = smoother)", "Mean Episode Length\n(lower = faster)"]
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [results[n][metric] for n in names]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
        ax.bar(range(len(names)), vals, color=colors[:len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir_abs, "ch04_comparison.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Presets and CLI
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "quick": {"n_demos": 1000, "epochs": 15, "batch_size": 128},
    "full": {"n_demos": 3000, "epochs": 40, "batch_size": 256},
}

HEAD_TYPES = ["discrete", "regression", "diffusion"]
CHUNK_SIZES = [1, 4]


def config_name(head: str, chunk: int) -> str:
    """Human-readable config name."""
    return f"{head.capitalize()} K={chunk}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare action representations on ContinuousMiniPushT")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--head", choices=HEAD_TYPES, default=None,
                        help="Train only this head type (default: all)")
    parser.add_argument("--chunk-size", type=int, choices=CHUNK_SIZES, default=None,
                        help="Train only this chunk size (default: all)")
    parser.add_argument("--n-demos", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    n_demos = args.n_demos or preset["n_demos"]
    epochs = args.epochs or preset["epochs"]
    batch_size = args.batch_size or preset["batch_size"]
    ckpt_dir = args.checkpoint_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(os.path.dirname(__file__) or ".", ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    heads = [args.head] if args.head else HEAD_TYPES
    chunks = [args.chunk_size] if args.chunk_size else CHUNK_SIZES

    print(f"\n{'='*60}")
    print(f"Chapter 4: Action Representations ({args.preset} preset)")
    print(f"  demos={n_demos}  epochs={epochs}  batch={batch_size}  device={args.device}")
    print(f"  heads={heads}  chunks={chunks}")
    print(f"{'='*60}\n")

    # --- Step 1: Collect demos ---
    demo_path = os.path.join(ckpt_dir, f"demos_224x224_{n_demos}ep.npz")
    if not os.path.exists(demo_path):
        print("[1/4] Collecting expert demos...")
        env = ContinuousMiniPushT(size=224)
        expert = ContinuousExpert(size=224, noise_std=0.1)
        total = collect_demos_to_disk(env, expert, demo_path, n_episodes=n_demos)
        print(f"      Collected {total} transitions.\n")
    else:
        print(f"[1/4] Demos exist: {demo_path}\n")

    # --- Step 2: Precompute embeddings (per chunk size) ---
    embed_datasets: dict[int, EmbeddingDataset] = {}
    if not args.eval_only:
        print("[2/4] Precomputing SigLIP embeddings...")
        vision_encoder = VisionEncoder()
        for cs in chunks:
            raw_dataset = MmapPushTDataset(demo_path, chunk_size=cs)
            embed_ds = precompute_embeddings(vision_encoder, raw_dataset, device=args.device, batch_size=64)
            embed_datasets[cs] = embed_ds
            print(f"      chunk_size={cs}: {len(embed_ds)} samples")
        vision_encoder = vision_encoder.cpu()
        if args.device != "cpu":
            torch.cuda.empty_cache()
        print()

    # --- Step 3: Train all configs ---
    all_losses: dict[str, list[float]] = {}
    all_results: dict[str, dict] = {}

    if not args.eval_only:
        print("[3/4] Training...")
        for head in heads:
            for cs in chunks:
                name = config_name(head, cs)
                ckpt_path = os.path.join(ckpt_dir, f"{head}_K{cs}_{n_demos}d_{epochs}ep.pt")

                if os.path.exists(ckpt_path) and not args.force_retrain:
                    print(f"  {name}: checkpoint exists, skipping (use --force-retrain)")
                    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if "losses" in saved:
                        all_losses[name] = saved["losses"]
                    continue

                print(f"\n  --- {name} ---")
                t0 = time.time()
                model = build_vla(head, chunk_size=cs)
                losses = train(
                    model, embed_datasets[cs],
                    epochs=epochs, batch_size=batch_size,
                    lr=args.lr, device=args.device, patience=args.patience,
                )
                elapsed = time.time() - t0
                all_losses[name] = losses
                print(f"        Done in {elapsed:.0f}s, final loss={losses[-1]:.4f}")

                torch.save({
                    "state_dict": model.state_dict(),
                    "head_type": head,
                    "chunk_size": cs,
                    "losses": losses,
                    "epochs": len(losses),
                    "n_demos": n_demos,
                }, ckpt_path)
                print(f"        Saved: {ckpt_path}")

                model = model.cpu()
                if args.device != "cpu":
                    torch.cuda.empty_cache()

    # --- Step 4: Evaluate all configs ---
    if not args.train_only:
        print("\n[4/4] Evaluating...")
        env = ContinuousMiniPushT(size=224)
        vision_encoder = VisionEncoder()
        vision_encoder = vision_encoder.to(args.device)

        for head in heads:
            for cs in chunks:
                name = config_name(head, cs)
                ckpt_path = os.path.join(ckpt_dir, f"{head}_K{cs}_{n_demos}d_{epochs}ep.pt")
                if not os.path.exists(ckpt_path):
                    print(f"  {name}: no checkpoint, skipping eval")
                    continue

                print(f"  Evaluating {name}...")
                model = build_vla(head, chunk_size=cs)
                saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(saved["state_dict"])
                if name not in all_losses:
                    all_losses[name] = saved.get("losses", [])

                results = evaluate(model, env, vision_encoder, n_episodes=100, device=args.device)
                all_results[name] = results
                print(f"    success={results['success_rate']:.1%}  smoothness={results['mean_smoothness']:.3f}  length={results['mean_length']:.0f}")

                model = model.cpu()
                if args.device != "cpu":
                    torch.cuda.empty_cache()

    # --- Save visualizations ---
    if all_results and not args.no_viz:
        print("\nSaving comparison plots...")
        save_comparison_plots(all_results, all_losses)

    # --- Print summary table ---
    if all_results:
        print(f"\n{'='*70}")
        print(f"{'Config':<25} {'Success':>8} {'Smooth':>8} {'Length':>8}")
        print(f"{'-'*70}")
        for name, res in all_results.items():
            print(f"{name:<25} {res['success_rate']:>7.1%} {res['mean_smoothness']:>8.3f} {res['mean_length']:>8.0f}")
        print(f"{'='*70}")
