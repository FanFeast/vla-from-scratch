"""Chapter 2: Compare scratch CNN vs CLIP vs SigLIP vision encoders.

Run standalone: python compare_encoders.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from typing import Union

sys.path.insert(0, os.path.dirname(__file__))
from mini_pusht import MiniPushT


# ---------------------------------------------------------------------------
# Expert policy (same as Chapter 1)
# ---------------------------------------------------------------------------


class ScriptedExpert:
    """Deterministic two-phase oracle for MiniPushT.

    Phase 1: navigate agent toward the block.
    Phase 2: push block toward goal.
    Uses ground-truth state from env info dict (not the image).

    Args:
        size: Environment resolution. Scales the phase-switch threshold.
    """

    def __init__(self, size: int = 224) -> None:
        # At 64x64 base, threshold is 4px; scales linearly with resolution
        self._threshold = max(1, round(4 * size / 64))

    def act(self, info: dict) -> int:
        """Return optimal discrete action given environment state."""
        agent = info["agent_pos"].astype(float)
        block = info["block_pos"].astype(float)
        goal = info["goal_pos"].astype(float)

        if np.linalg.norm(agent - block) > self._threshold:
            return self._move_toward(agent, block)
        return self._move_toward(agent, goal)

    def _move_toward(self, src: np.ndarray, dst: np.ndarray) -> int:
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 2  # right or left
        else:
            return 1 if dy > 0 else 0  # down or up


# ---------------------------------------------------------------------------
# Vision encoders
# ---------------------------------------------------------------------------


class ScratchCNN(nn.Module):
    """4-layer CNN encoder, outputs 128-dim embedding.

    Handles any input resolution via AdaptiveAvgPool2d(1).
    At 224x224 the intermediate shapes are:
        Conv(3->32, 3x3, s=2) -> ReLU     # (B, 32, 111, 111)
        Conv(32->64, 3x3, s=2) -> ReLU    # (B, 64, 55, 55)
        Conv(64->128, 3x3, s=2) -> ReLU   # (B, 128, 27, 27)
        Conv(128->128, 3x3, s=2) -> ReLU  # (B, 128, 13, 13)
        AdaptiveAvgPool2d(1)               # (B, 128, 1, 1)
        Flatten -> output dim 128
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Args: image (B, 3, 224, 224) float in [0,1]. Returns: (B, 128)."""
        return self.net(image)


# ---------------------------------------------------------------------------
# Pretrained vision encoders (CLIP, SigLIP)
# ---------------------------------------------------------------------------


class PretrainedVisionEncoder(nn.Module):
    """Wraps a frozen CLIP or SigLIP vision model from HuggingFace.

    Supports:
      - "openai/clip-vit-base-patch16" (CLIPVisionModel, CLS token)
      - "google/siglip-base-patch16-224" (SiglipVisionModel, pooler output)

    The image processor handles model-specific normalization.
    All pretrained parameters are frozen (requires_grad=False).
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze all pretrained parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Both CLIPModel and SiglipModel expose vision_config.hidden_size
        self.embed_dim = self.model.config.vision_config.hidden_size

        # Extract normalization constants from processor so we can apply
        # them as tensor ops on GPU (avoids CPU round-trip via processor)
        self.register_buffer(
            "mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        )
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) float in [0, 1]. Any resolution accepted;
                   non-224 input is resized to 224x224 for the ViT.
        Returns:
            (B, embed_dim) vision embedding
        """
        # Resize to 224x224 if needed (ViT expects fixed input size)
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            image = F.interpolate(
                image, size=(224, 224), mode="bilinear", align_corners=False
            )

        # Normalize on-device (same math the HF processor does, but no CPU trip)
        pixel_values = (image - self.mean) / self.std

        # Both CLIPModel and SiglipModel expose .vision_model; use it directly
        # so we only run the image encoder (not the text tower).
        outputs = self.model.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output


class AttentionPool(nn.Module):
    """Learned single-head attention pooling over a sequence of tokens.

    Instead of mean-pooling (which treats every patch equally) or using a
    pre-trained pooler (which was optimised for text-image matching), this
    learns a task-specific weighting over patches so the model can focus on
    the few patches that contain the block, agent, and goal.

    Only 769 trainable parameters (one Linear(768, 1)).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args: tokens (B, N, D). Returns: (B, D) weighted sum."""
        weights = F.softmax(self.query(tokens).squeeze(-1), dim=1)  # (B, N)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1)  # (B, D)


class PretrainedPatchEncoder(nn.Module):
    """Frozen ViT backbone + learned attention pool over patch tokens.

    Unlike PretrainedVisionEncoder (which uses the model's built-in pooler),
    this extracts raw patch tokens from the ViT and applies a small learned
    attention pooler. The key idea: SigLIP's built-in pooler (mean + tanh)
    was trained for contrastive text-image matching and dilutes spatial info.
    A task-specific attention pooler can learn to focus on the ~6 patches
    that contain the block/agent/goal out of 196 total patches.

    Architecture:
        Frozen ViT → last_hidden_state (B, 197, 768)
        → AttentionPool (769 trainable params) → (B, 768)
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim = self.model.config.vision_config.hidden_size
        self.attn_pool = AttentionPool(self.embed_dim)  # trainable!

        self.register_buffer(
            "mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        )
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) float in [0, 1]
        Returns:
            (B, embed_dim) attention-pooled patch features
        """
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            image = F.interpolate(
                image, size=(224, 224), mode="bilinear", align_corners=False
            )

        pixel_values = (image - self.mean) / self.std

        # Run frozen ViT — no gradients through the backbone
        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state  # (B, N, D)

        # Learned attention pool — gradients flow here
        return self.attn_pool(patch_tokens)


# ---------------------------------------------------------------------------
# Language encoder (same as Chapter 1)
# ---------------------------------------------------------------------------


class LanguageEncoder(nn.Module):
    """Lookup-table language encoder for 3 fixed instructions."""

    VOCAB: dict[str, int] = {
        "push block to goal": 0,
        "move left": 1,
        "move right": 2,
    }

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(self.VOCAB), embed_dim)

    def forward(self, instruction: Union[str, list[str]]) -> torch.Tensor:
        """Args: instruction str or list[str]. Returns: (B, 128)."""
        if isinstance(instruction, str):
            instruction = [instruction]
        indices = torch.tensor(
            [self.VOCAB[i] for i in instruction],
            dtype=torch.long,
            device=self.embedding.weight.device,
        )
        return self.embedding(indices)


# ---------------------------------------------------------------------------
# VLA model (parameterized vision dim)
# ---------------------------------------------------------------------------


class VLA(nn.Module):
    """VLA with swappable vision encoder.

    forward(image, instruction) -> action logits
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_dim: int,
        n_actions: int = 5,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_encoder = LanguageEncoder(embed_dim=128)
        self.action_head = nn.Sequential(
            nn.Linear(vision_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(
        self,
        image: torch.Tensor,
        instruction: Union[str, list[str]],
    ) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 224, 224) float in [0, 1]
            instruction: str or list[str]
        Returns:
            (B, n_actions) logits
        """
        vision_feat = self.vision_encoder(image)  # (B, vision_dim)
        lang_feat = self.language_encoder(instruction)  # (B, 128)
        fused = torch.cat([vision_feat, lang_feat], dim=-1)
        return self.action_head(fused)


# ---------------------------------------------------------------------------
# Data collection and dataset
# ---------------------------------------------------------------------------


def collect_demos(
    env: MiniPushT,
    expert: ScriptedExpert,
    n_episodes: int = 1000,
    instruction: str = "push block to goal",
) -> list[dict]:
    """Collect expert demonstrations for behavior cloning.

    Returns:
        list of {"image": np.ndarray (H,W,3), "instruction": str, "action": int}
    """
    demos = []
    for _ in range(n_episodes):
        obs, info = env.reset(options={"instruction": instruction})
        done = False
        while not done:
            action = expert.act(info)
            demos.append(
                {
                    "image": obs.copy(),
                    "instruction": instruction,
                    "action": action,
                }
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    return demos


def collect_demos_to_disk(
    env: MiniPushT,
    expert: ScriptedExpert,
    path: str,
    n_episodes: int = 1000,
    instruction: str = "push block to goal",
) -> int:
    """Collect expert demos and stream them to disk incrementally.

    Instead of storing all images in RAM, we pre-allocate a memory-mapped
    numpy array on disk and write each transition as it's collected.
    Returns the total number of transitions written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    size = env.observation_space.shape[0]

    # First pass: count total transitions so we can pre-allocate.
    # Episodes average ~100-150 steps; we allocate generously then truncate.
    est_transitions = n_episodes * 200
    tmp_images = path + ".tmp_images.npy"
    tmp_actions = path + ".tmp_actions.npy"

    images_mmap = np.lib.format.open_memmap(
        tmp_images,
        mode="w+",
        dtype=np.uint8,
        shape=(est_transitions, size, size, 3),
    )
    actions_arr = np.empty(est_transitions, dtype=np.int32)

    idx = 0
    for ep in range(n_episodes):
        obs, info = env.reset(options={"instruction": instruction})
        done = False
        while not done:
            action = expert.act(info)
            if idx >= len(images_mmap):
                # Rare: grew beyond estimate — extend by 50%
                new_size = int(len(images_mmap) * 1.5)
                new_mmap = np.lib.format.open_memmap(
                    tmp_images + ".ext",
                    mode="w+",
                    dtype=np.uint8,
                    shape=(new_size, size, size, 3),
                )
                new_mmap[:idx] = images_mmap[:idx]
                del images_mmap
                os.replace(tmp_images + ".ext", tmp_images)
                images_mmap = np.lib.format.open_memmap(tmp_images, mode="r+")
                actions_arr = np.resize(actions_arr, new_size)
            images_mmap[idx] = obs
            actions_arr[idx] = action
            idx += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if (ep + 1) % 500 == 0:
            print(f"        {ep + 1}/{n_episodes} episodes ({idx} transitions)...")

    # Truncate to actual size and save as compressed npz
    total = idx
    instructions_arr = np.array([instruction] * total)
    actions_final = actions_arr[:total]

    # Write final compressed file in chunks to avoid loading full mmap into RAM
    print(f"      Compressing {total} demos to {path}...")
    np.savez_compressed(
        path,
        images=np.array(images_mmap[:total]),
        instructions=instructions_arr,
        actions=actions_final,
    )
    disk_mb = os.path.getsize(path) / 1e6
    ram_mb = total * size * size * 3 / 1e6
    print(
        f"      Saved {total} demos ({ram_mb:.0f}MB raw, {disk_mb:.0f}MB compressed) to {path}"
    )

    # Cleanup temp files
    del images_mmap
    if os.path.exists(tmp_images):
        os.remove(tmp_images)
    if os.path.exists(tmp_actions):
        os.remove(tmp_actions)

    return total


def save_demos(demos: list[dict], path: str) -> None:
    """Save collected demos to a .npz file on disk."""
    images = np.stack([d["image"] for d in demos])
    instructions = np.array([d["instruction"] for d in demos])
    actions = np.array([d["action"] for d in demos], dtype=np.int32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, images=images, instructions=instructions, actions=actions)
    print(f"      Saved {len(demos)} demos ({images.nbytes / 1e6:.0f}MB) to {path}")


def load_demos(path: str) -> list[dict]:
    """Load demos from a .npz file saved by save_demos().

    For backward compatibility, returns list of dicts.
    For memory-efficient loading, use load_demos_as_dataset() instead.
    """
    data = np.load(path, allow_pickle=False)
    images = data["images"]
    instructions = data["instructions"]
    actions = data["actions"]
    demos = []
    for i in range(len(images)):
        demos.append(
            {
                "image": images[i],
                "instruction": str(instructions[i]),
                "action": int(actions[i]),
            }
        )
    print(f"      Loaded {len(demos)} cached demos from {path}")
    return demos


def load_demos_as_dataset(path: str) -> "PushTDataset":
    """Load demos from .npz directly into a PushTDataset without intermediate dicts.

    This avoids the ~2x RAM overhead of load_demos() -> PushTDataset().
    """
    data = np.load(path, allow_pickle=False)
    images = data["images"]  # (N, H, W, 3) uint8
    instructions = [str(s) for s in data["instructions"]]
    actions = data["actions"].astype(np.int64)

    print(f"      Loaded {len(images)} cached demos from {path}")
    print("      Converting demos to tensors...", end=" ")

    dataset = PushTDataset.__new__(PushTDataset)
    dataset.images = torch.from_numpy(images).permute(0, 3, 1, 2)  # (N, C, H, W)
    dataset.instructions = instructions
    dataset.actions = torch.from_numpy(actions)
    print(f"done ({len(images)} samples, {dataset.images.nbytes / 1e9:.1f}GB)")
    return dataset


def load_demos_mmap(path: str) -> "MmapPushTDataset":
    """Load demos using memory-mapped .npy files -- near-zero RAM usage.

    Compressed .npz files cannot be memory-mapped (numpy silently loads them
    into RAM). This function extracts the images array to an uncompressed .npy
    file on first call, then memory-maps that file on subsequent calls.
    Only the batches actively being read reside in physical memory.
    """
    # Derive path for the uncompressed mmap-able images file
    mmap_path = path.replace(".npz", "_images.npy")

    if not os.path.exists(mmap_path):
        print(f"      Converting {path} to mmap-able format (one-time)...")
        data = np.load(path, allow_pickle=False)
        images = data["images"]  # loads into RAM once
        np.save(mmap_path, images)
        del images, data
        print(f"      Saved uncompressed images: {mmap_path}")

    # Memory-map the uncompressed .npy file (true zero-copy mmap)
    images = np.load(mmap_path, mmap_mode="r")  # (N, H, W, 3) uint8

    # Instructions and actions are small -- load from compressed npz
    data = np.load(path, allow_pickle=False)
    instructions = [str(s) for s in data["instructions"]]
    actions = data["actions"].astype(np.int64)

    n = len(images)
    raw_gb = n * images.shape[1] * images.shape[2] * 3 / 1e9
    print(f"      Memory-mapped {n} demos ({raw_gb:.1f}GB on disk, ~0 RAM)")
    return MmapPushTDataset(images, instructions, actions)


class PushTDataset(Dataset):
    """Dataset of (image_tensor, instruction, action) tuples from expert demos.

    Pre-converts all images to a single uint8 tensor at init time to avoid
    repeated numpy->tensor conversion in __getitem__ (which was bottlenecking CPU).
    """

    def __init__(self, demos: list[dict]) -> None:
        # Stack all images into one contiguous uint8 tensor (saves RAM vs list of arrays)
        # Shape: (N, 3, 224, 224) uint8
        print("      Converting demos to tensors...", end=" ")
        self.images = torch.from_numpy(np.stack([d["image"] for d in demos])).permute(
            0, 3, 1, 2
        )  # (N, H, W, C) -> (N, C, H, W), stays uint8
        self.instructions = [d["instruction"] for d in demos]
        self.actions = torch.tensor([d["action"] for d in demos], dtype=torch.long)
        print(f"done ({len(demos)} samples, {self.images.nbytes / 1e9:.1f}GB)")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        # Return uint8 tensor -- float conversion happens on GPU in the training loop
        return self.images[idx], self.instructions[idx], int(self.actions[idx].item())


class MmapPushTDataset(Dataset):
    """Memory-mapped dataset that streams images from disk on demand.

    Uses numpy mmap_mode='r' so the OS pages in only the image rows
    being accessed. RAM usage is proportional to batch_size, not dataset size.
    Compatible with DataLoader(num_workers>0) via copy-on-access semantics.
    """

    def __init__(
        self,
        images: np.ndarray,
        instructions: list[str],
        actions: np.ndarray,
    ) -> None:
        self.images = images  # memory-mapped (N, H, W, 3)
        self.instructions = instructions
        self.actions = torch.from_numpy(actions)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        # Read single image from mmap, convert HWC -> CHW uint8 tensor
        img = torch.from_numpy(self.images[idx].copy()).permute(2, 0, 1)
        return img, self.instructions[idx], int(self.actions[idx].item())


class EmbeddingDataset(Dataset):
    """Dataset of pre-computed (vision_embedding, instruction, action) tuples.

    Used for frozen encoders: run the ViT once over the full dataset, cache the
    embeddings, then train only the action head on cached features. This avoids
    re-running the frozen ViT every epoch.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        instructions: list[str],
        actions: torch.Tensor,
    ) -> None:
        self.embeddings = embeddings
        self.instructions = instructions
        self.actions = actions

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        return (
            self.embeddings[idx],
            self.instructions[idx],
            int(self.actions[idx].item()),
        )


def precompute_embeddings(
    vision_encoder: nn.Module,
    dataset: Dataset,
    device: str = "cpu",
    batch_size: int = 64,
) -> EmbeddingDataset:
    """Run frozen vision encoder once over dataset, return cached embeddings."""
    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()
    use_workers = device != "cpu"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if use_workers else 0,
        pin_memory=use_workers,
    )

    all_embeds = []
    all_instructions = []
    all_actions = []

    print("      Pre-computing vision embeddings (one-time ViT forward pass)...")
    with torch.no_grad():
        for batch_idx, (images, instructions, actions) in enumerate(loader):
            images = images.to(device).float() / 255.0
            embeds = vision_encoder(images)  # (B, embed_dim)
            all_embeds.append(embeds.cpu())
            all_instructions.extend(instructions)
            all_actions.append(actions)
            if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                line_end = "\r" if sys.stdout.isatty() else "\n"
                print(f"        batch {batch_idx + 1}/{len(loader)}", end=line_end)

    print(f"      Done. Cached {len(all_instructions)} embeddings.")
    return EmbeddingDataset(
        embeddings=torch.cat(all_embeds, dim=0),
        instructions=all_instructions,
        actions=torch.cat(all_actions, dim=0),
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def train(
    model: VLA,
    dataset: Dataset,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Behavior cloning training loop. Only updates parameters with requires_grad=True.

    Uses a streaming DataLoader that works with both in-memory and memory-mapped
    datasets. Batches are transferred to GPU on the fly (pin_memory + non_blocking).
    """
    model = model.to(device)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    use_workers = device != "cpu"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if use_workers else 0,
        pin_memory=use_workers,
        persistent_workers=use_workers,
    )
    n_batches = len(loader)
    loss_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, instructions, actions in loader:
            images = images.to(device, non_blocking=True).float() / 255.0
            actions = actions.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images, list(instructions))
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch + 1:02d}/{epochs}  loss={avg_loss:.4f}")

    return loss_history


def train_on_embeddings(
    language_encoder: LanguageEncoder,
    action_head: nn.Sequential,
    dataset: EmbeddingDataset,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Train only language encoder + action head on pre-computed vision embeddings.

    This is much faster than train() for frozen encoders because the ViT
    forward pass is skipped entirely -- embeddings were cached beforehand.
    """
    language_encoder = language_encoder.to(device)
    action_head = action_head.to(device)
    params = list(language_encoder.parameters()) + list(action_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history: list[float] = []
    for epoch in range(epochs):
        language_encoder.train()
        action_head.train()
        epoch_loss = 0.0
        for embeds, instructions, actions in loader:
            embeds = embeds.to(device)
            actions = actions.long().to(device)

            lang_feat = language_encoder(list(instructions))
            fused = torch.cat([embeds, lang_feat], dim=-1)
            logits = action_head(fused)

            optimizer.zero_grad()
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch + 1:02d}/{epochs}  loss={avg_loss:.4f}")

    return loss_history


def evaluate(
    model: VLA,
    env: MiniPushT,
    n_episodes: int = 50,
    device: str = "cpu",
) -> float:
    """Evaluate policy success rate over n_episodes."""
    model.eval()
    successes = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                image = (
                    torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                ).to(device)
                logits = model(image, info["instruction"])
                action = int(logits.argmax(dim=-1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if terminated:
                    successes += 1
    return successes / n_episodes


def _ood_reset(env: MiniPushT, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Reset env, then place agent + block in the OOD ring outside training bounds.

    Training reset samples agent_pos uniformly in [agent_lo, agent_hi] and
    block_pos in [block_lo, block_hi]. The OOD ring is the outer band
    [half, lo) ∪ (hi, size - half - 1] -- coordinates the policy never saw
    during demo collection. Goal stays in-distribution: it's the task target,
    not a "start position", and the per-handoff intent is to test
    generalization to novel starts.

    Both x and y coords are placed in the outer ring -> corner-region starts.
    Min-distance constraints (block-goal, agent-block) are preserved so the
    task remains solvable; up to 100 retries before falling back to any valid
    OOD coord.
    """
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    size = env.size

    def sample_outer(lo: int, hi: int, half: int) -> int:
        upper_max = size - half - 1
        choices = []
        if lo - 1 >= half:
            choices.append((half, lo - 1))
        if hi + 1 <= upper_max:
            choices.append((hi + 1, upper_max))
        if not choices:
            return int(rng.integers(half, upper_max + 1))
        side = choices[int(rng.integers(0, len(choices)))]
        return int(rng.integers(side[0], side[1] + 1))

    for _ in range(100):
        block = np.array(
            [
                sample_outer(env._block_lo, env._block_hi, env._obj_half),
                sample_outer(env._block_lo, env._block_hi, env._obj_half),
            ],
            dtype=np.int32,
        )
        if np.linalg.norm(block - env.goal_pos) > env._min_block_goal:
            break
    env.block_pos = block

    for _ in range(100):
        agent = np.array(
            [
                sample_outer(env._agent_lo, env._agent_hi, env._agent_half),
                sample_outer(env._agent_lo, env._agent_hi, env._agent_half),
            ],
            dtype=np.int32,
        )
        if np.linalg.norm(agent - env.block_pos) > env._min_agent_block:
            break
    env.agent_pos = agent

    return env._render_obs(), env._get_info()


def evaluate_ood(
    model: VLA,
    env: MiniPushT,
    n_episodes: int = 50,
    device: str = "cpu",
    seed: int = 12345,
) -> float:
    """Evaluate success rate with out-of-distribution agent/block start positions.

    Same loop as evaluate(), but uses _ood_reset() which forces starts into the
    outer ring beyond the training position distribution. A fixed `seed`
    guarantees identical OOD episodes across encoders (fair comparison).
    """
    model.eval()
    rng = np.random.default_rng(seed)
    successes = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = _ood_reset(env, rng)
            done = False
            while not done:
                image = (
                    torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                ).to(device)
                logits = model(image, info["instruction"])
                action = int(logits.argmax(dim=-1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if terminated:
                    successes += 1
    return successes / n_episodes


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def compute_embeddings_for_tsne(
    model: VLA,
    dataset: Dataset,
    n_samples: int = 500,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract vision embeddings and action labels for t-SNE.

    Returns:
        embeddings: (n_samples, embed_dim) numpy array
        labels: (n_samples,) numpy array of action ints
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_embeds = []
    all_labels = []
    count = 0

    with torch.no_grad():
        for images, instructions, actions in loader:
            images = images.to(device).float() / 255.0
            embeds = model.vision_encoder(images)  # (B, embed_dim)
            all_embeds.append(embeds.cpu().numpy())
            all_labels.append(actions.numpy())
            count += len(images)
            if count >= n_samples:
                break

    embeddings = np.concatenate(all_embeds, axis=0)[:n_samples]
    labels = np.concatenate(all_labels, axis=0)[:n_samples]
    return embeddings, labels


ACTION_NAMES = ["up", "down", "left", "right", "stay"]


def plot_tsne_comparison(
    embeddings_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    save_path: str = "../../assets/figures/ch02_tsne_comparison.png",
) -> None:
    """Plot t-SNE of vision embeddings for multiple encoders side by side.

    Args:
        embeddings_dict: {"EncoderName": (N, D) array, ...}
        labels: (N,) action labels shared across all encoders
        save_path: where to save the figure
    """
    n_encoders = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_encoders, figsize=(6 * n_encoders, 5))
    if n_encoders == 1:
        axes = [axes]

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    for ax, (name, embeds) in zip(axes, embeddings_dict.items()):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = tsne.fit_transform(embeds)

        for action_id in range(5):
            mask = labels == action_id
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors[action_id],
                label=ACTION_NAMES[action_id],
                s=10,
                alpha=0.6,
            )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Vision Embeddings (t-SNE) -- colored by expert action",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path_abs = save_path
    if not os.path.isabs(save_path):
        save_path_abs = os.path.join(os.path.dirname(__file__), save_path)
    os.makedirs(os.path.dirname(save_path_abs), exist_ok=True)
    plt.savefig(save_path_abs, dpi=150, bbox_inches="tight")
    print(f"  Saved t-SNE plot: {save_path_abs}")
    plt.close()


def save_training_curves(
    losses_dict: dict[str, list[float]],
    save_path: str = "../../assets/figures/ch02_training_curves.png",
) -> None:
    """Save overlaid training loss curves for all encoders.

    Args:
        losses_dict: {"EncoderName": [loss_epoch1, loss_epoch2, ...], ...}
        save_path: where to save the figure
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, losses in losses_dict.items():
        ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Chapter 2: Training loss by vision encoder")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path_abs = save_path
    if not os.path.isabs(save_path):
        save_path_abs = os.path.join(os.path.dirname(__file__), save_path)
    os.makedirs(os.path.dirname(save_path_abs), exist_ok=True)
    plt.savefig(save_path_abs, dpi=100, bbox_inches="tight")
    print(f"  Saved training curves: {save_path_abs}")
    plt.close()


def render_rollout_comparison(
    models: dict[str, VLA],
    env: MiniPushT,
    device: str = "cpu",
    n_frames: int = 6,
    seed: int = 4,
    save_path: str = "../../assets/figures/ch02_rollout_comparison.png",
) -> None:
    """Render a grid showing each encoder's rollout on the same episode seed.

    Produces an (N_encoders x n_frames) grid of frames with outcome labels,
    making it visually obvious which encoders solve the task and which fail.

    Args:
        models: {"EncoderName": trained_VLA_model, ...}
        env: MiniPushT instance
        device: inference device
        n_frames: evenly-spaced frames to show per encoder
        seed: environment reset seed (same for all encoders = fair comparison)
        save_path: where to save the figure
    """
    n_encoders = len(models)
    fig, axes = plt.subplots(
        n_encoders, n_frames, figsize=(n_frames * 2.2, n_encoders * 2.4)
    )
    if n_encoders == 1:
        axes = axes[np.newaxis, :]

    for row, (name, model) in enumerate(models.items()):
        model.eval()
        obs, info = env.reset(seed=seed)
        all_obs = [obs.copy()]
        success = False

        with torch.no_grad():
            done = False
            while not done:
                image = (
                    torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                ).to(device)
                logits = model(image, info["instruction"])
                action = int(logits.argmax(dim=-1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                all_obs.append(obs.copy())
                done = terminated or truncated
                if terminated:
                    success = True

        # Pick n_frames evenly spaced
        indices = [
            int(i * (len(all_obs) - 1) / (n_frames - 1)) for i in range(n_frames)
        ]
        frames = [all_obs[i] for i in indices]

        for col, (frame, idx) in enumerate(zip(frames, indices)):
            ax = axes[row, col]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                outcome = "SUCCESS" if success else "FAIL"
                color = "#2e7d32" if success else "#c62828"
                ax.set_ylabel(
                    f"{name}\n[{outcome}]",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                    rotation=0,
                    labelpad=70,
                    va="center",
                )
            if row == 0:
                ax.set_title(f"step {idx}", fontsize=8)

    fig.suptitle(
        "Chapter 2: Encoder Rollout Comparison (same start)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path_abs = save_path
    if not os.path.isabs(save_path):
        save_path_abs = os.path.join(os.path.dirname(__file__), save_path)
    os.makedirs(os.path.dirname(save_path_abs), exist_ok=True)
    plt.savefig(save_path_abs, dpi=120, bbox_inches="tight")
    print(f"  Saved rollout comparison: {save_path_abs}")
    plt.close()


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "tiny": {
        "size": 64,
        "n_demos": 200,
        "epochs": 10,
        "skip_pretrained": True,
    },
    "medium": {
        "size": 112,
        "n_demos": 500,
        "epochs": 15,
        "skip_pretrained": False,
    },
    "full": {
        "size": 224,
        "n_demos": 1000,
        "epochs": 20,
        "skip_pretrained": False,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare vision encoders on MiniPushT")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["tiny", "medium", "full"],
        default="full",
        help="configuration preset (default: full)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="override image resolution (e.g. 64, 112, 224)",
    )
    parser.add_argument(
        "--n-demos", type=int, default=None, help="override number of demo episodes"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="override training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device (auto-detects GPU if available)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="torch CPU threads (only used when device=cpu)",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="skip saving visualizations"
    )
    parser.add_argument(
        "--skip-pretrained",
        action="store_true",
        default=None,
        help="only run scratch CNN",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="directory for saving/loading demos and trained models",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="ignore cached checkpoints and retrain all",
    )
    parser.add_argument(
        "--force-recollect",
        action="store_true",
        help="ignore cached demos and recollect from expert",
    )
    parser.add_argument(
        "--attn-pool",
        action="store_true",
        help="include SigLIP with learned attention pooling (shows pooling impact)",
    )
    parser.add_argument(
        "--eval-ood",
        action="store_true",
        help="also evaluate on out-of-distribution start positions (corner ring)",
    )
    args = parser.parse_args()

    # Apply preset defaults, then override with explicit CLI args
    preset = PRESETS[args.preset]
    size = args.size if args.size is not None else preset["size"]
    n_demos = args.n_demos if args.n_demos is not None else preset["n_demos"]
    epochs = args.epochs if args.epochs is not None else preset["epochs"]
    skip_pretrained = (
        args.skip_pretrained
        if args.skip_pretrained is not None
        else preset["skip_pretrained"]
    )

    if args.device == "cpu":
        torch.set_num_threads(args.num_threads)

    print("=" * 60)
    print("Chapter 2: Vision Encoder Comparison")
    print("=" * 60)
    print(f"Preset: {args.preset} | Environment: MiniPushT {size}x{size}")
    print(f"Device: {args.device}")
    encoders_str = (
        "Scratch CNN only"
        if skip_pretrained
        else "Scratch CNN, CLIP ViT-B/16, SigLIP ViT-B/16"
        + (", SigLIP (attn pool)" if args.attn_pool else "")
    )
    print(f"Encoders: {encoders_str}")
    print(f"Demos: {n_demos} episodes, Epochs: {epochs}\n")

    # --- Step 1: collect or load expert demos ---
    ckpt_dir = os.path.join(os.path.dirname(__file__) or ".", args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    demos_path = os.path.join(ckpt_dir, f"demos_{size}x{size}_{n_demos}ep.npz")

    env = MiniPushT(size=size)
    expert = ScriptedExpert(size=size)

    if os.path.exists(demos_path) and not args.force_recollect:
        print("[1/4] Loading cached demos (memory-mapped)...")
        dataset = load_demos_mmap(demos_path)
    else:
        print("[1/4] Collecting expert demos...")
        collect_demos_to_disk(env, expert, demos_path, n_episodes=n_demos)
        dataset = load_demos_mmap(demos_path)

    print(f"      {len(dataset)} transitions from {n_demos} episodes.\n")

    # --- Step 2: define encoder configs ---
    encoder_configs = [
        ("Scratch CNN", ScratchCNN(), 128),
    ]
    if not skip_pretrained:
        print("      Loading pretrained models (first run downloads ~350MB each)...")
        encoder_configs.extend(
            [
                (
                    "CLIP ViT-B/16",
                    PretrainedVisionEncoder("openai/clip-vit-base-patch16"),
                    768,
                ),
                (
                    "SigLIP ViT-B/16",
                    PretrainedVisionEncoder("google/siglip-base-patch16-224"),
                    768,
                ),
            ]
        )
        if args.attn_pool:
            encoder_configs.append(
                (
                    "SigLIP (attn pool)",
                    PretrainedPatchEncoder("google/siglip-base-patch16-224"),
                    768,
                )
            )

    # --- Step 3: train and evaluate each ---
    results = {}
    losses_dict = {}
    embeddings_dict = {}
    trained_models = {}
    shared_labels = None

    for name, encoder, dim in encoder_configs:
        safe_name = name.lower().replace(" ", "_").replace("/", "-")
        ckpt_path = os.path.join(
            ckpt_dir, f"{safe_name}_{size}px_{n_demos}d_{epochs}ep.pt"
        )
        is_frozen = not any(p.requires_grad for p in encoder.parameters())

        model = VLA(vision_encoder=encoder, vision_dim=dim)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())

        # Check for cached checkpoint
        if os.path.exists(ckpt_path) and not args.force_retrain:
            print(f"\n[2/4] Loading cached: {name} ({ckpt_path})")
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            model.load_state_dict(ckpt["model_state"], strict=False)
            model = model.to(args.device)
            losses = ckpt["losses"]
            train_time = ckpt["train_time"]
            print(
                f"      Loaded {len(losses)} epochs, {train_time:.1f}s original train time"
            )
        elif is_frozen:
            print(
                f"\n[2/4] Training: {name} (frozen encoder -- pre-computing embeddings)"
            )
            print(f"      Vision dim: {dim}")
            print(f"      Parameters: {n_trainable:,} trainable / {n_total:,} total")

            embed_dataset = precompute_embeddings(
                encoder, dataset, device=args.device, batch_size=args.batch_size
            )

            start_time = time.time()
            losses = train_on_embeddings(
                model.language_encoder,
                model.action_head,
                embed_dataset,
                epochs=epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
            train_time = time.time() - start_time
            model = model.to(args.device)

            torch.save(
                {
                    "model_state": {k: v for k, v in model.state_dict().items()},
                    "losses": losses,
                    "train_time": train_time,
                },
                ckpt_path,
            )
            print(f"      Saved checkpoint: {ckpt_path}")
        else:
            print(f"\n[2/4] Training: {name}")
            print(f"      Vision dim: {dim}, Trainable action head + language encoder")
            print(f"      Parameters: {n_trainable:,} trainable / {n_total:,} total")

            start_time = time.time()
            losses = train(
                model,
                dataset,
                epochs=epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
            train_time = time.time() - start_time

            torch.save(
                {
                    "model_state": {k: v for k, v in model.state_dict().items()},
                    "losses": losses,
                    "train_time": train_time,
                },
                ckpt_path,
            )
            print(f"      Saved checkpoint: {ckpt_path}")

        print(f"\n[3/4] Evaluating: {name}")
        success_rate = evaluate(model, env, n_episodes=50, device=args.device)

        result = {
            "trainable_params": n_trainable,
            "train_time": train_time,
            "final_loss": losses[-1],
            "success_rate": success_rate,
        }
        if args.eval_ood:
            ood_rate = evaluate_ood(model, env, n_episodes=50, device=args.device)
            result["ood_success_rate"] = ood_rate
            print(f"      OOD (corner-start) success: {ood_rate * 100:.1f}%")
        results[name] = result
        losses_dict[name] = losses
        trained_models[name] = model

        embeds, labels = compute_embeddings_for_tsne(
            model, dataset, n_samples=500, device=args.device
        )
        embeddings_dict[name] = embeds
        if shared_labels is None:
            shared_labels = labels

    # --- Step 4: print comparison table ---
    print("\n" + "=" * 60)
    print("[4/4] RESULTS")
    print("=" * 60)
    if args.eval_ood:
        print(
            f"{'Encoder':<20} {'Trainable':>10} {'Time':>8} {'Loss':>8} "
            f"{'Success':>8} {'OOD':>7}"
        )
        print("-" * 67)
        for name, r in results.items():
            print(
                f"{name:<20} {r['trainable_params']:>10,} "
                f"{r['train_time']:>7.1f}s "
                f"{r['final_loss']:>8.4f} "
                f"{r['success_rate'] * 100:>7.1f}% "
                f"{r['ood_success_rate'] * 100:>6.1f}%"
            )
    else:
        print(
            f"{'Encoder':<20} {'Trainable':>10} {'Time':>8} {'Loss':>8} {'Success':>8}"
        )
        print("-" * 60)
        for name, r in results.items():
            print(
                f"{name:<20} {r['trainable_params']:>10,} "
                f"{r['train_time']:>7.1f}s "
                f"{r['final_loss']:>8.4f} "
                f"{r['success_rate'] * 100:>7.1f}%"
            )
    print()

    # --- Save visualizations ---
    if not args.no_viz:
        print("Saving visualizations...")
        save_training_curves(losses_dict)
        plot_tsne_comparison(embeddings_dict, shared_labels)
        render_rollout_comparison(trained_models, env, device=args.device)
        print("\nFigures saved to assets/figures/.")

    print(
        f"\nDone ({args.preset} preset, {size}x{size}). -> Chapter 3: swap LanguageEncoder for SmolLM2."
    )
