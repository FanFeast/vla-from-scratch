"""Chapter 3: Language Conditioning -- SmolLM2 + Cross-Attention Fusion.

Compares four configurations on MiniPushT with 11 instructions:
  1. Lookup + Concat    (ch01/02 baseline, no paraphrase generalization)
  2. SmolLM2-135M + Concat  (understands paraphrases, limited fusion)
  3. SmolLM2-135M + CrossAttn  (proper VLA fusion)
  4. SmolLM2-360M + CrossAttn  (scaling the language model)

Run standalone: python cross_attention.py
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
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(__file__))
from mini_pusht import (
    MiniPushT,
    ScriptedExpert,
    CANONICAL_INSTRUCTIONS,
    PARAPHRASES,
    PARAPHRASE_TO_CANONICAL,
    INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------


class VisionEncoder(nn.Module):
    """Frozen SigLIP or CLIP vision encoder.

    Can return either:
    - CLS/pooler embedding (B, embed_dim) when return_tokens=False
    - Full patch token sequence (B, N, embed_dim) when return_tokens=True

    All pretrained parameters are frozen (requires_grad=False).
    """

    def __init__(self, model_name: str, return_tokens: bool = False) -> None:
        super().__init__()
        processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_dim: int = self.model.config.vision_config.hidden_size
        self.return_tokens = return_tokens
        self.register_buffer(
            "mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        )
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) float in [0, 1]
        Returns:
            return_tokens=False: (B, embed_dim)
            return_tokens=True:  (B, N_patches, embed_dim)
        """
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            image = F.interpolate(
                image, size=(224, 224), mode="bilinear", align_corners=False
            )
        pixel_values = (image - self.mean) / self.std
        outputs = self.model.vision_model(pixel_values=pixel_values)
        if self.return_tokens:
            return outputs.last_hidden_state  # (B, N, D)
        return outputs.pooler_output  # (B, D)


# ---------------------------------------------------------------------------
# Language encoders
# ---------------------------------------------------------------------------


class LookupLanguageEncoder(nn.Module):
    """Lookup-table language encoder for canonical instructions only.

    Maps each of the 11 canonical instruction strings to a learned embedding.
    Unknown strings (paraphrases) get a zero vector -- the model cannot
    generalize, demonstrating why a real language model is needed.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.vocab = {instr: i for i, instr in enumerate(CANONICAL_INSTRUCTIONS)}
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.embed_dim = embed_dim

    def forward(self, instructions: list[str]) -> torch.Tensor:
        """Returns (B, embed_dim) embeddings. Unknown strings get zero vector."""
        device = self.embedding.weight.device
        embeds = []
        for instr in instructions:
            if instr in self.vocab:
                idx = torch.tensor([self.vocab[instr]], device=device)
                embeds.append(self.embedding(idx).squeeze(0))
            else:
                embeds.append(torch.zeros(self.embed_dim, device=device))
        return torch.stack(embeds)


class SmolLM2Encoder(nn.Module):
    """Frozen SmolLM2 language encoder with trainable projection.

    Tokenizes instruction strings and runs them through SmolLM2 (frozen).
    A learned linear projection maps from SmolLM2's hidden dim to d_model.

    Two modes:
    - pool=True: mean-pool hidden states -> project -> (B, d_model)
    - pool=False: project all token embeddings -> (B, T, d_model) + attention mask

    Caches frozen hidden states (pre-projection) to avoid redundant LM forward
    passes for repeated instruction strings.

    Args:
        model_name: HuggingFace model ID
        d_model: output projection dimension
        pool: if True, return pooled; if False, return token sequences
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        d_model: int = 256,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        hidden_dim = self.model.config.hidden_size
        self.projection = nn.Linear(hidden_dim, d_model)
        self.pool = pool
        self.embed_dim = d_model
        # Cache frozen hidden states (pre-projection) keyed by instruction string
        self._cache: dict[str, torch.Tensor] = {}

    def _get_hidden_states(
        self, instructions: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get frozen hidden states, using cache for seen instructions.

        Returns:
            hidden: (B, T, H) frozen hidden states
            attn_mask: (B, T) attention mask
        """
        # Check if all are cached (common during training with few unique instructions)
        all_cached = all(instr in self._cache for instr in instructions)

        if all_cached and self.pool:
            # Fast path: all cached, just stack pooled hidden states
            hiddens = torch.stack(
                [self._cache[instr].to(device) for instr in instructions]
            )
            return hiddens, torch.ones(
                hiddens.shape[0], 1, device=device, dtype=torch.long
            )

        # Forward pass through frozen LM
        tokens = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**tokens, use_cache=False)
        hidden = outputs.last_hidden_state  # (B, T, H)
        attn_mask = tokens["attention_mask"]  # (B, T)

        # Cache pooled hidden states for each instruction
        if self.pool:
            mask_expanded = attn_mask.unsqueeze(-1).float()  # (B, T, 1)
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(
                dim=1
            )  # (B, H)
            for i, instr in enumerate(instructions):
                if instr not in self._cache:
                    self._cache[instr] = pooled[i].detach().cpu()
            return pooled, attn_mask

        return hidden, attn_mask

    def forward(
        self, instructions: list[str]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            instructions: list of B instruction strings
        Returns:
            pool=True:  (B, d_model) projected mean-pooled embeddings
            pool=False: tuple of (B, T, d_model) projected tokens, (B, T) mask
        """
        device = self.projection.weight.device
        hidden, attn_mask = self._get_hidden_states(instructions, device)
        hidden = hidden.float()  # SmolLM2 outputs bfloat16; projection is float32

        if self.pool:
            return self.projection(hidden)  # (B, d_model)
        else:
            projected = self.projection(hidden)  # (B, T, d_model)
            return projected, attn_mask


# ---------------------------------------------------------------------------
# Fusion modules
# ---------------------------------------------------------------------------


class ConcatFusion(nn.Module):
    """Concatenation fusion: baseline from ch01/ch02.

    Projects vision and language embeddings to d_model, then concatenates.
    """

    def __init__(
        self, vision_dim: int = 768, lang_dim: int = 256, d_model: int = 256
    ) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.lang_proj = nn.Linear(lang_dim, d_model)

    def forward(
        self, vision_feat: torch.Tensor, lang_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vision_feat: (B, vision_dim) CLS token or pooled embedding
            lang_out: (B, lang_dim) pooled language embedding
        Returns:
            (B, 2 * d_model) concatenated projection
        """
        v = self.vision_proj(vision_feat)
        l = self.lang_proj(lang_out)
        return torch.cat([v, l], dim=-1)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: vision tokens attend to language tokens.

    This is the mechanism used by production VLAs (SmolVLA, RT-2, pi0).
    Vision tokens form the queries; language tokens form keys and values.
    Each vision patch "asks" the language tokens which instruction details
    are relevant to what it sees.

    Architecture:
        vision_proj: Linear(vision_dim, d_model)
        cross_attn: MultiheadAttention(d_model, n_heads, batch_first=True)
        layer_norm: LayerNorm(d_model)
        pool: mean over attended vision tokens -> (B, d_model)

    Args:
        vision_dim: dimension of input vision tokens (e.g. 768 for ViT-B)
        d_model: internal dimension for cross-attention
        n_heads: number of attention heads
    """

    def __init__(
        self,
        vision_dim: int = 768,
        d_model: int = 256,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vision_tokens: (B, N_v, vision_dim) raw patch tokens from ViT
            lang_tokens: (B, N_l, d_model) projected language token embeddings
            lang_mask: (B, N_l) attention mask (1=real, 0=padding). If provided,
                       padding positions are masked out so vision tokens don't
                       attend to them.
        Returns:
            (B, d_model) fused representation
        """
        v = self.vision_proj(vision_tokens)  # (B, N_v, d_model)

        # Build key_padding_mask: True where tokens should be ignored
        key_padding_mask = None
        if lang_mask is not None:
            key_padding_mask = lang_mask == 0  # (B, N_l), True = ignore

        attended, _ = self.cross_attn(
            query=v,
            key=lang_tokens,
            value=lang_tokens,
            key_padding_mask=key_padding_mask,
        )  # (B, N_v, d_model)
        attended = self.layer_norm(attended + v)  # residual + norm
        return attended.mean(dim=1)  # (B, d_model) mean pool





# ---------------------------------------------------------------------------
# VLA model
# ---------------------------------------------------------------------------


class VLA(nn.Module):
    """VLA with configurable language encoder and fusion method.

    Configs:
    - fusion="concat", lm="lookup": Lookup + Concat (ch01/02 baseline)
    - fusion="concat", lm="smollm2-*": SmolLM2 + Concat
    - fusion="cross-attn", lm="smollm2-*": SmolLM2 + Cross-Attention
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        lang_encoder: nn.Module,
        fusion: nn.Module,
        fusion_out_dim: int,
        fusion_type: str = "concat",
        n_actions: int = 5,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lang_encoder = lang_encoder
        self.fusion = fusion
        self.fusion_type = fusion_type
        self.action_head = nn.Sequential(
            nn.Linear(fusion_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(
        self, image: torch.Tensor, instructions: list[str]
    ) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) float in [0, 1]
            instructions: list of B instruction strings
        Returns:
            (B, n_actions) action logits
        """
        vision_out = self.vision_encoder(image)
        lang_out = self.lang_encoder(instructions)

        if self.fusion_type == "cross-attn":
            # lang_out is (tokens, mask) tuple for cross-attention
            lang_tokens, lang_mask = lang_out
            fused = self.fusion(vision_out, lang_tokens, lang_mask)
        else:
            fused = self.fusion(vision_out, lang_out)

        return self.action_head(fused)


def build_vla(
    fusion_type: str,
    lm_type: str,
    encoder_name: str = "google/siglip-base-patch16-224",
    d_model: int = 256,
) -> VLA:
    """Factory to build VLA with specified fusion and language encoder config.

    Args:
        fusion_type: "concat" or "cross-attn"
        lm_type: "lookup", "smollm2-135m", or "smollm2-360m"
        encoder_name: HuggingFace vision model name
        d_model: internal fusion dimension
    """
    use_cross_attn = fusion_type == "cross-attn"
    vision_encoder = VisionEncoder(encoder_name, return_tokens=use_cross_attn)
    vision_dim = vision_encoder.embed_dim

    if lm_type == "lookup":
        if use_cross_attn:
            raise ValueError("Cross-attention requires SmolLM2, not lookup encoder")
        lang_encoder = LookupLanguageEncoder(embed_dim=d_model)
        fusion = ConcatFusion(vision_dim=vision_dim, lang_dim=d_model, d_model=d_model)
        fusion_out_dim = 2 * d_model
    elif lm_type.startswith("smollm2"):
        model_map = {
            "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
            "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
        }
        if lm_type not in model_map:
            raise ValueError(f"Unknown lm_type: {lm_type}")
        hf_name = model_map[lm_type]

        if use_cross_attn:
            lang_encoder = SmolLM2Encoder(hf_name, d_model=d_model, pool=False)
            fusion = CrossAttentionFusion(
                vision_dim=vision_dim, d_model=d_model, n_heads=4
            )
            fusion_out_dim = d_model
        else:
            lang_encoder = SmolLM2Encoder(hf_name, d_model=d_model, pool=True)
            fusion = ConcatFusion(
                vision_dim=vision_dim, lang_dim=d_model, d_model=d_model
            )
            fusion_out_dim = 2 * d_model
    else:
        raise ValueError(f"Unknown lm_type: {lm_type}")

    return VLA(
        vision_encoder,
        lang_encoder,
        fusion,
        fusion_out_dim,
        fusion_type=fusion_type,
    )


# ---------------------------------------------------------------------------
# Data collection and dataset
# ---------------------------------------------------------------------------


def collect_demos(
    env: MiniPushT,
    expert: ScriptedExpert,
    n_episodes: int = 1100,
) -> list[dict]:
    """Collect expert demos balanced across all 11 canonical instructions.

    Each instruction gets n_episodes // 11 episodes.

    Returns:
        list of {"image": np.ndarray (H,W,3), "instruction": str, "action": int}
    """
    demos: list[dict] = []
    per_instr = max(1, n_episodes // len(CANONICAL_INSTRUCTIONS))
    for instr in CANONICAL_INSTRUCTIONS:
        for _ in range(per_instr):
            obs, info = env.reset(options={"instruction": instr})
            done = False
            while not done:
                action = expert.act(info)
                demos.append(
                    {
                        "image": obs.copy(),
                        "instruction": instr,
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
    n_episodes: int = 1100,
) -> int:
    """Collect expert demos balanced across instructions, stream to disk.

    Uses memory-mapped numpy arrays to avoid RAM explosion on large datasets.
    Returns the total number of transitions written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    size = env.observation_space.shape[0]
    per_instr = max(1, n_episodes // len(CANONICAL_INSTRUCTIONS))

    est_transitions = n_episodes * 200
    tmp_images = path + ".tmp_images.npy"

    images_mmap = np.lib.format.open_memmap(
        tmp_images,
        mode="w+",
        dtype=np.uint8,
        shape=(est_transitions, size, size, 3),
    )
    actions_arr = np.empty(est_transitions, dtype=np.int32)
    instructions_arr: list[str] = []

    idx = 0
    ep_count = 0
    for instr in CANONICAL_INSTRUCTIONS:
        for _ in range(per_instr):
            obs, info = env.reset(options={"instruction": instr})
            done = False
            while not done:
                action = expert.act(info)
                if idx >= len(images_mmap):
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
                instructions_arr.append(instr)
                idx += 1
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            ep_count += 1
            if ep_count % 500 == 0:
                print(
                    f"        {ep_count}/{n_episodes} episodes ({idx} transitions)..."
                )

    total = idx
    # Save metadata (instructions + actions) as compressed npz -- small
    print(f"      Saving {total} demos metadata to {path}...")
    np.savez_compressed(
        path,
        instructions=np.array(instructions_arr[:total]),
        actions=actions_arr[:total],
    )

    # Trim and rename mmap file as final images (avoids 100GB+ RAM copy)
    final_images = path.replace(".npz", "_images.npy")
    del images_mmap
    # Create properly-sized final mmap
    final_mmap = np.lib.format.open_memmap(
        final_images,
        mode="w+",
        dtype=np.uint8,
        shape=(total, size, size, 3),
    )
    # Copy in chunks from temp to avoid RAM spike
    tmp_mmap = np.lib.format.open_memmap(tmp_images, mode="r")
    chunk = 10000
    for i in range(0, total, chunk):
        end = min(i + chunk, total)
        final_mmap[i:end] = tmp_mmap[i:end]
    del tmp_mmap, final_mmap

    if os.path.exists(tmp_images):
        os.remove(tmp_images)

    disk_mb = os.path.getsize(path) / 1e6
    img_gb = os.path.getsize(final_images) / 1e9
    print(
        f"      Saved {total} demos (metadata: {disk_mb:.0f}MB, images: {img_gb:.1f}GB)"
    )

    return total


def save_demos(demos: list[dict], path: str) -> None:
    """Save collected demos to a .npz file on disk."""
    images = np.stack([d["image"] for d in demos])
    instructions = np.array([d["instruction"] for d in demos])
    actions = np.array([d["action"] for d in demos], dtype=np.int32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, images=images, instructions=instructions, actions=actions)
    print(f"      Saved {len(demos)} demos ({images.nbytes / 1e6:.0f}MB) to {path}")


def load_demos_mmap(path: str) -> "MmapPushTDataset":
    """Load demos using memory-mapped .npy files -- near-zero RAM usage.

    Compressed .npz files cannot be memory-mapped (numpy silently loads them
    into RAM). Extracts to uncompressed .npy on first call, then mmaps it.
    """
    mmap_path = path.replace(".npz", "_images.npy")

    if not os.path.exists(mmap_path):
        print(f"      Converting {path} to mmap-able format (one-time)...")
        data = np.load(path, allow_pickle=False)
        images = data["images"]
        np.save(mmap_path, images)
        del images, data
        print(f"      Saved uncompressed images: {mmap_path}")

    images = np.load(mmap_path, mmap_mode="r")
    data = np.load(path, allow_pickle=False)
    instructions = [str(s) for s in data["instructions"]]
    actions = data["actions"].astype(np.int64)

    n = len(images)
    raw_gb = n * images.shape[1] * images.shape[2] * 3 / 1e9
    print(f"      Memory-mapped {n} demos ({raw_gb:.1f}GB on disk, ~0 RAM)")
    return MmapPushTDataset(images, instructions, actions)


class PushTDataset(Dataset):
    """In-memory dataset of (image_tensor, instruction, action) tuples."""

    def __init__(self, demos: list[dict]) -> None:
        print("      Converting demos to tensors...", end=" ")
        self.images = torch.from_numpy(np.stack([d["image"] for d in demos])).permute(
            0, 3, 1, 2
        )
        self.instructions = [d["instruction"] for d in demos]
        self.actions = torch.tensor([d["action"] for d in demos], dtype=torch.long)
        print(f"done ({len(demos)} samples, {self.images.nbytes / 1e9:.1f}GB)")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        return self.images[idx], self.instructions[idx], int(self.actions[idx].item())


class MmapPushTDataset(Dataset):
    """Memory-mapped dataset that streams images from disk on demand."""

    def __init__(
        self,
        images: np.ndarray,
        instructions: list[str],
        actions: np.ndarray,
    ) -> None:
        self.images = images
        self.instructions = instructions
        self.actions = torch.from_numpy(actions)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        img = torch.from_numpy(self.images[idx].copy()).permute(2, 0, 1)
        return img, self.instructions[idx], int(self.actions[idx].item())


class EmbeddingDataset(Dataset):
    """Dataset of pre-computed (vision_embedding, instruction, action) tuples.

    Used for frozen encoders with concat fusion: run the ViT once, cache
    embeddings, then train only fusion + action head on cached features.
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
    vision_encoder: VisionEncoder,
    dataset: Dataset,
    device: str = "cpu",
    batch_size: int = 64,
) -> EmbeddingDataset:
    """Run frozen vision encoder once over dataset, return cached embeddings.

    Works for both pooled (B, D) and token (B, N, D) outputs.
    """
    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()
    # Use 0 workers -- precompute is GPU-bound, workers just waste RAM
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )

    all_embeds = []
    all_instructions: list[str] = []
    all_actions = []

    print("      Pre-computing vision embeddings (one-time ViT forward pass)...")
    with torch.no_grad():
        for batch_idx, (images, instructions, actions) in enumerate(loader):
            images = images.to(device).float() / 255.0
            embeds = vision_encoder(images)
            all_embeds.append(embeds.cpu())
            all_instructions.extend(instructions)
            all_actions.append(actions)
            if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                line_end = "\r" if sys.stdout.isatty() else "\n"
                print(
                    f"        batch {batch_idx + 1}/{len(loader)}", end=line_end
                )

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
    patience: int = 0,
) -> list[float]:
    """Behavior cloning training loop. Only updates parameters with requires_grad=True.

    Uses a streaming DataLoader with pin_memory for GPU training.
    Supports cosine LR schedule and early stopping with patience.
    """
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Mixed precision with bf16 (same exponent range as fp32, no overflow)
    use_amp = device != "cpu"

    use_workers = device != "cpu"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if use_workers else 0,
        pin_memory=use_workers,
        persistent_workers=use_workers,
    )
    n_batches = len(loader)
    loss_history: list[float] = []
    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, instructions, actions in loader:
            images = images.to(device, non_blocking=True).float() / 255.0
            actions = actions.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images, list(instructions))
                loss = criterion(logits, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch + 1:02d}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if patience > 0:
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                    break

    return loss_history


def train_on_embeddings(
    lang_encoder: nn.Module,
    fusion: nn.Module,
    action_head: nn.Module,
    dataset: EmbeddingDataset,
    fusion_type: str = "concat",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 0,
) -> list[float]:
    """Train fusion + action head on pre-computed vision embeddings.

    Much faster than train() for frozen encoders because the ViT forward
    pass is skipped entirely -- embeddings were cached beforehand.
    Supports cosine LR schedule and early stopping with patience.
    """
    lang_encoder = lang_encoder.to(device)
    fusion = fusion.to(device)
    action_head = action_head.to(device)

    params = (
        list(lang_encoder.parameters())
        + list(fusion.parameters())
        + list(action_head.parameters())
    )
    trainable_params = [p for p in params if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device != "cpu"))

    loss_history: list[float] = []
    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        lang_encoder.train()
        fusion.train()
        action_head.train()
        epoch_loss = 0.0

        for embeds, instructions, actions in loader:
            embeds = embeds.to(device)
            actions = actions.long().to(device)

            lang_out = lang_encoder(list(instructions))

            if fusion_type == "cross-attn":
                lang_tokens, lang_mask = lang_out
                fused = fusion(embeds, lang_tokens, lang_mask)
            else:
                fused = fusion(embeds, lang_out)

            logits = action_head(fused)

            optimizer.zero_grad()
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch + 1:02d}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if patience > 0:
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                    break

    return loss_history


def evaluate(
    model: VLA,
    env: MiniPushT,
    n_episodes: int = 55,
    device: str = "cpu",
    use_paraphrases: bool = False,
) -> float:
    """Evaluate policy success rate across all canonical instructions.

    Runs n_episodes total, cycling through canonical instructions.

    Args:
        model: trained VLA model
        env: MiniPushT instance
        n_episodes: total episodes (divided across 11 instructions)
        device: inference device
        use_paraphrases: if True, substitute random paraphrases for instructions
    """
    model.eval()
    rng = np.random.default_rng(42)
    per_instr = max(1, n_episodes // len(CANONICAL_INSTRUCTIONS))
    successes = 0
    total = 0

    with torch.no_grad():
        for canonical in CANONICAL_INSTRUCTIONS:
            for _ in range(per_instr):
                if use_paraphrases:
                    paras = PARAPHRASES[canonical]
                    instr = paras[int(rng.integers(0, len(paras)))]
                else:
                    instr = canonical

                obs, info = env.reset(options={"instruction": instr})
                done = False
                while not done:
                    image = (
                        torch.from_numpy(obs)
                        .permute(2, 0, 1)
                        .float()
                        .unsqueeze(0)
                        / 255.0
                    ).to(device)
                    logits = model(image, [instr])
                    action = int(logits.argmax(dim=-1).item())
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    if terminated:
                        successes += 1
                total += 1

    return successes / total


def evaluate_per_instruction(
    model: VLA,
    env: MiniPushT,
    n_per_instr: int = 10,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate per-instruction success rates. Returns dict of instruction -> rate."""
    model.eval()
    results: dict[str, float] = {}

    with torch.no_grad():
        for canonical in CANONICAL_INSTRUCTIONS:
            successes = 0
            for _ in range(n_per_instr):
                obs, info = env.reset(options={"instruction": canonical})
                done = False
                while not done:
                    image = (
                        torch.from_numpy(obs)
                        .permute(2, 0, 1)
                        .float()
                        .unsqueeze(0)
                        / 255.0
                    ).to(device)
                    logits = model(image, [canonical])
                    action = int(logits.argmax(dim=-1).item())
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    if terminated:
                        successes += 1
            results[canonical] = successes / n_per_instr

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

ACTION_NAMES = ["up", "down", "left", "right", "stay"]


def save_training_curves(
    losses_dict: dict[str, list[float]],
    save_path: str = "../../assets/figures/ch03_training_curves.png",
) -> None:
    """Save overlaid training loss curves for all configs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, losses in losses_dict.items():
        ax.plot(
            range(1, len(losses) + 1), losses, marker="o", markersize=3, label=name
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Chapter 3: Training loss by language conditioning config")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path_abs = save_path
    if not os.path.isabs(save_path):
        save_path_abs = os.path.join(os.path.dirname(__file__), save_path)
    os.makedirs(os.path.dirname(save_path_abs), exist_ok=True)
    plt.savefig(save_path_abs, dpi=100, bbox_inches="tight")
    print(f"  Saved training curves: {save_path_abs}")
    plt.close()


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "quick": {"size": 64, "n_demos": 220, "epochs": 10},
    "full": {"size": 224, "n_demos": 2200, "epochs": 40},
    "scale": {"size": 224, "n_demos": 10000, "epochs": 100, "batch_size": 512},
}

CONFIGS: list[dict] = [
    {"name": "Lookup + Concat", "fusion": "concat", "lm": "lookup"},
    {"name": "SmolLM2-135M + Concat", "fusion": "concat", "lm": "smollm2-135m"},
    {"name": "SmolLM2-135M + CrossAttn", "fusion": "cross-attn", "lm": "smollm2-135m"},
    {"name": "SmolLM2-360M + CrossAttn", "fusion": "cross-attn", "lm": "smollm2-360m"},
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare language conditioning strategies on MiniPushT"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "full", "scale"],
        default="quick",
        help="configuration preset (default: quick)",
    )
    parser.add_argument(
        "--size", type=int, default=None, help="override image resolution"
    )
    parser.add_argument(
        "--n-demos", type=int, default=None, help="override demo episode count"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="override training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=0,
                        help="early stopping patience (0=disabled)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["concat", "cross-attn"],
        default=None,
        help="run only this fusion type",
    )
    parser.add_argument(
        "--lm",
        type=str,
        choices=["lookup", "smollm2-135m", "smollm2-360m"],
        default=None,
        help="run only this language model",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["siglip", "clip"],
        default="siglip",
        help="vision encoder (default: siglip)",
    )
    parser.add_argument(
        "--eval-paraphrase",
        action="store_true",
        help="also evaluate on paraphrased instructions",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="skip saving visualizations"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="directory for saving/loading demos and models",
    )
    parser.add_argument(
        "--force-retrain", action="store_true", help="ignore cached checkpoints"
    )
    parser.add_argument(
        "--force-recollect", action="store_true", help="ignore cached demos"
    )
    args = parser.parse_args()

    # Apply preset
    preset = PRESETS[args.preset]
    size = args.size if args.size is not None else preset["size"]
    n_demos = args.n_demos if args.n_demos is not None else preset["n_demos"]
    epochs = args.epochs if args.epochs is not None else preset["epochs"]
    if args.batch_size == 64 and "batch_size" in preset:
        args.batch_size = preset["batch_size"]

    encoder_map = {
        "siglip": "google/siglip-base-patch16-224",
        "clip": "openai/clip-vit-base-patch16",
    }
    encoder_name = encoder_map[args.encoder]

    # Filter configs if specific fusion/lm requested
    configs = CONFIGS[:]
    if args.fusion is not None:
        configs = [c for c in configs if c["fusion"] == args.fusion]
    if args.lm is not None:
        configs = [c for c in configs if c["lm"] == args.lm]

    print("=" * 65)
    print("Chapter 3: Language Conditioning")
    print("=" * 65)
    print(f"Preset: {args.preset} | Env: MiniPushT {size}x{size}")
    print(f"Device: {args.device} | Vision: {args.encoder}")
    print(f"Demos: {n_demos} episodes | Epochs: {epochs} | Patience: {args.patience}")
    print(f"LR: {args.lr} (cosine decay) | Batch: {args.batch_size}")
    print(f"Configs: {', '.join(c['name'] for c in configs)}\n")

    # --- Step 1: collect or load expert demos ---
    ckpt_dir = os.path.join(os.path.dirname(__file__) or ".", args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    demos_path = os.path.join(ckpt_dir, f"demos_{size}x{size}_{n_demos}ep.npz")

    env = MiniPushT(size=size)
    expert = ScriptedExpert(size=size)

    if os.path.exists(demos_path) and not args.force_recollect:
        print("[1/4] Loading cached demos...")
        if size >= 112:
            dataset = load_demos_mmap(demos_path)
        else:
            data = np.load(demos_path, allow_pickle=False)
            demos_list = []
            for i in range(len(data["images"])):
                demos_list.append(
                    {
                        "image": data["images"][i],
                        "instruction": str(data["instructions"][i]),
                        "action": int(data["actions"][i]),
                    }
                )
            dataset = PushTDataset(demos_list)
            print(f"      Loaded {len(dataset)} cached demos")
    else:
        print("[1/4] Collecting expert demos...")
        if size >= 112:
            collect_demos_to_disk(env, expert, demos_path, n_episodes=n_demos)
            dataset = load_demos_mmap(demos_path)
        else:
            demos_list = collect_demos(env, expert, n_episodes=n_demos)
            save_demos(demos_list, demos_path)
            dataset = PushTDataset(demos_list)

    print(f"      {len(dataset)} transitions from {n_demos} episodes.\n")

    # --- Step 2+3: train and evaluate each config ---
    results: dict[str, dict] = {}
    losses_dict: dict[str, list[float]] = {}

    for cfg in configs:
        name = cfg["name"]
        safe_name = name.lower().replace(" ", "_").replace("+", "").replace("-", "_")
        ckpt_path = os.path.join(
            ckpt_dir, f"{safe_name}_{size}px_{n_demos}d_{epochs}ep.pt"
        )

        print(f"\n[2/4] Building: {name}")
        model = build_vla(
            fusion_type=cfg["fusion"],
            lm_type=cfg["lm"],
            encoder_name=encoder_name,
        )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"      Parameters: {n_trainable:,} trainable / {n_total:,} total")

        # Check for cached checkpoint
        if os.path.exists(ckpt_path) and not args.force_retrain:
            print(f"      Loading cached checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            model.load_state_dict(ckpt["model_state"], strict=False)
            model = model.to(args.device)
            losses = ckpt["losses"]
            train_time = ckpt["train_time"]
            print(f"      {len(losses)} epochs, {train_time:.1f}s original time")
        else:
            # Decide training strategy
            is_frozen_vision = not any(
                p.requires_grad for p in model.vision_encoder.parameters()
            )
            use_embed_cache = is_frozen_vision and cfg["fusion"] == "concat"

            if use_embed_cache:
                print("      Pre-computing vision embeddings...")
                embed_dataset = precompute_embeddings(
                    model.vision_encoder,
                    dataset,
                    device=args.device,
                    batch_size=512,  # inference only, no gradient impact
                )
                print(f"      Training on cached embeddings ({len(embed_dataset)} samples)")
                start_time = time.time()
                losses = train_on_embeddings(
                    model.lang_encoder,
                    model.fusion,
                    model.action_head,
                    embed_dataset,
                    fusion_type=cfg["fusion"],
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    patience=args.patience,
                )
                train_time = time.time() - start_time
                model = model.to(args.device)
            else:
                # Cross-attn runs full ViT+LLM forward — use moderate batch for good gradient signal
                cross_batch = min(args.batch_size, 512)
                print(f"      Training with full forward pass + AMP (bs={cross_batch})...")
                start_time = time.time()
                losses = train(
                    model,
                    dataset,
                    epochs=epochs,
                    batch_size=cross_batch,
                    lr=args.lr,
                    device=args.device,
                    patience=args.patience,
                )
                train_time = time.time() - start_time

            # Save checkpoint
            torch.save(
                {
                    "model_state": {k: v for k, v in model.state_dict().items()},
                    "losses": losses,
                    "train_time": train_time,
                },
                ckpt_path,
            )
            print(f"      Saved: {ckpt_path}")

        # Evaluate
        print(f"\n[3/4] Evaluating: {name}")
        success_rate = evaluate(
            model, env, n_episodes=55, device=args.device, use_paraphrases=False
        )
        print(f"      Success rate: {success_rate * 100:.1f}%")

        result = {
            "trainable_params": n_trainable,
            "train_time": train_time,
            "final_loss": losses[-1],
            "success_rate": success_rate,
        }

        if args.eval_paraphrase:
            para_rate = evaluate(
                model, env, n_episodes=55, device=args.device, use_paraphrases=True
            )
            result["paraphrase_rate"] = para_rate
            print(f"      Paraphrase success: {para_rate * 100:.1f}%")

        results[name] = result
        losses_dict[name] = losses

    # --- Step 4: print comparison table ---
    print("\n" + "=" * 65)
    print("[4/4] RESULTS")
    print("=" * 65)

    has_para = args.eval_paraphrase
    if has_para:
        print(
            f"{'Config':<28} {'Trainable':>10} {'Time':>8} {'Loss':>8} "
            f"{'Success':>8} {'Paraph.':>8}"
        )
        print("-" * 76)
        for name, r in results.items():
            print(
                f"{name:<28} {r['trainable_params']:>10,} "
                f"{r['train_time']:>7.1f}s "
                f"{r['final_loss']:>8.4f} "
                f"{r['success_rate'] * 100:>7.1f}% "
                f"{r.get('paraphrase_rate', 0) * 100:>7.1f}%"
            )
    else:
        print(
            f"{'Config':<28} {'Trainable':>10} {'Time':>8} {'Loss':>8} {'Success':>8}"
        )
        print("-" * 60)
        for name, r in results.items():
            print(
                f"{name:<28} {r['trainable_params']:>10,} "
                f"{r['train_time']:>7.1f}s "
                f"{r['final_loss']:>8.4f} "
                f"{r['success_rate'] * 100:>7.1f}%"
            )

    print()

    # --- Save visualizations ---
    if not args.no_viz and losses_dict:
        print("Saving visualizations...")
        save_training_curves(losses_dict)
        print("Figures saved to assets/figures/.\n")

    print(
        f"Done ({args.preset} preset, {size}x{size}). "
        "-> Chapter 4: swap discrete actions for continuous regression / diffusion."
    )
