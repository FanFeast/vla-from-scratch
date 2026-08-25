"""Configuration for Chapter 7: Full VLA (SmolVLA-like architecture).

Defines hyperparameters for the SmolVLA model, training recipe, and dataset
presets. All defaults match the SmolVLA paper (Shukor et al., 2025) except
where scaled down to fit our data budget (~200 episodes vs 481 datasets).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SmolVLAConfig:
    """All hyperparameters for SmolVLA training."""

    # --- Pretrained backbone ---
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    vlm_hidden_dim: int = 960
    vlm_num_layers: int = 16  # first L/2 of 32 total
    vision_dim: int = 768
    vision_image_size: int = 512
    vision_patch_size: int = 16
    pixel_shuffle_scale: int = 4
    num_vision_tokens: int = 64  # 1024 / (4*4)

    # --- Action expert ---
    # Scaled down from SmolVLA's 720-dim/16-layer expert to ~20M params,
    # appropriate for our ~90K frame data budget.
    expert_dim: int = 512
    expert_num_layers: int = 6  # 3 SA + 3 CA interleaved
    expert_nhead: int = 8
    expert_ffn_dim: int = 2048  # 4 * expert_dim
    expert_dropout: float = 0.1

    # --- Action / state space ---
    action_dim: int = 6
    state_dim: int = 6
    chunk_size: int = 10

    # --- Training ---
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    batch_size: int = 32
    epochs: int = 200
    ema_decay: float = 0.999

    # --- Flow matching ---
    beta_a: float = 1.5  # Beta distribution shape for timestep sampling
    beta_b: float = 1.0
    inference_steps: int = 10

    # --- Data ---
    max_lang_tokens: int = 48
    val_ratio: float = 0.13
    num_workers: int = 4

    # VLM layers whose hidden states feed cross-attention in the expert.
    # Indices into hidden_states tuple (0=input_embeds, 1=layer0_out, ...).
    # With vlm_num_layers=16 and 3 CA layers, evenly spaced at layers 4,9,15.
    ca_layer_indices: tuple[int, ...] = (5, 10, 16)


# ---- Dataset presets -------------------------------------------------------

DATASETS_SO100: list[dict[str, Any]] = [
    {
        "repo_id": "lerobot/svla_so100_pickplace",
        "image_key": "observation.images.top",
        "state_key": "observation.state",
        "action_key": "action",
        "task": "Pick and place the object.",
    },
    {
        "repo_id": "lerobot/svla_so100_stacking",
        "image_key": "observation.images.top",
        "state_key": "observation.state",
        "action_key": "action",
        "task": "Stack the blocks.",
    },
    {
        "repo_id": "lerobot/svla_so100_sorting",
        "image_key": "observation.images.top",
        "state_key": "observation.state",
        "action_key": "action",
        "task": "Sort the objects by color.",
    },
]

DATASETS_PUSHT: list[dict[str, Any]] = [
    {
        "repo_id": "lerobot/pusht",
        "image_key": "observation.image",
        "state_key": "observation.state",
        "action_key": "action",
        "task": "Push the T-shaped block to the goal.",
    },
]

PRESETS: dict[str, dict[str, Any]] = {
    "so100": {
        "datasets": DATASETS_SO100,
        "action_dim": 6,
        "state_dim": 6,
        "chunk_size": 10,
        "epochs": 200,
        "batch_size": 32,
    },
    "pusht": {
        "datasets": DATASETS_PUSHT,
        "action_dim": 2,
        "state_dim": 2,
        "chunk_size": 10,
        "epochs": 200,
        "batch_size": 64,
    },
}
