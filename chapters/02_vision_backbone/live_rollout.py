"""Live side-by-side rollout viewer for trained Ch02 models.

Opens a Pygame window showing all 4 encoders attempting MiniPushT simultaneously.
Each encoder gets its own panel with a label and outcome indicator.

Usage:
    python live_rollout.py                    # default: 224px, all encoders
    python live_rollout.py --seed 10          # specific starting seed
    python live_rollout.py --fps 10           # slower playback
    python live_rollout.py --skip-scratch     # skip slow scratch CNN

Controls:
    SPACE  — pause/resume
    R      — reset with new random seed
    N      — next seed (increment)
    Q/ESC  — quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pygame
import torch

from compare_encoders import (
    MiniPushT,
    PretrainedPatchEncoder,
    PretrainedVisionEncoder,
    ScratchCNN,
    VLA,
)


def load_models(
    ckpt_dir: str, device: str, skip_scratch: bool = False
) -> dict[str, VLA]:
    """Load all trained checkpoints from disk."""
    configs: list[tuple[str, str, torch.nn.Module, int]] = []

    if not skip_scratch:
        configs.append(
            ("Scratch CNN", "scratch_cnn_224px_2000d_40ep.pt", ScratchCNN(), 128)
        )

    configs.extend(
        [
            (
                "CLIP ViT-B/16",
                "clip_vit-b-16_224px_2000d_40ep.pt",
                PretrainedVisionEncoder("openai/clip-vit-base-patch16"),
                768,
            ),
            (
                "SigLIP ViT-B/16",
                "siglip_vit-b-16_224px_2000d_40ep.pt",
                PretrainedVisionEncoder("google/siglip-base-patch16-224"),
                768,
            ),
            (
                "SigLIP (attn pool)",
                "siglip_(attn_pool)_224px_2000d_40ep.pt",
                PretrainedPatchEncoder("google/siglip-base-patch16-224"),
                768,
            ),
        ]
    )

    models: dict[str, VLA] = {}
    for name, ckpt_file, encoder, dim in configs:
        path = os.path.join(ckpt_dir, ckpt_file)
        if not os.path.exists(path):
            print(f"  Skipping {name} (no checkpoint at {path})")
            continue
        model = VLA(vision_encoder=encoder, vision_dim=dim)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model = model.to(device)
        model.eval()
        models[name] = model
        print(f"  Loaded: {name}")

    return models


def run_viewer(
    models: dict[str, VLA],
    device: str,
    initial_seed: int = 4,
    fps: int = 15,
    panel_size: int = 256,
) -> None:
    """Run the live Pygame viewer."""
    n_models = len(models)
    if n_models == 0:
        print("No models loaded!")
        return

    # Layout: horizontal panels with labels
    label_height = 36
    window_w = panel_size * n_models
    window_h = panel_size + label_height

    pygame.init()
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("Ch02 Live Rollout — SPACE=pause R=reset N=next Q=quit")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)

    # State per model
    envs = {name: MiniPushT(size=224) for name in models}
    seed = initial_seed

    def reset_all() -> None:
        for name, env in envs.items():
            env.reset(seed=seed)

    reset_all()
    paused = False
    running = True
    # Track outcomes
    outcomes: dict[str, str | None] = {name: None for name in models}
    steps: dict[str, int] = {name: 0 for name in models}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    seed = int(time.time()) % 10000
                    reset_all()
                    outcomes = {name: None for name in models}
                    steps = {name: 0 for name in models}
                elif event.key == pygame.K_n:
                    seed += 1
                    reset_all()
                    outcomes = {name: None for name in models}
                    steps = {name: 0 for name in models}

        if not paused:
            # Step each model
            with torch.no_grad():
                for name, model in models.items():
                    if outcomes[name] is not None:
                        continue  # episode done

                    env = envs[name]
                    obs = env._render_obs()
                    info = env._get_info()
                    image = (
                        torch.from_numpy(obs)
                        .permute(2, 0, 1)
                        .float()
                        .unsqueeze(0)
                        / 255.0
                    ).to(device)
                    logits = model(image, info["instruction"])
                    action = int(logits.argmax(dim=-1).item())
                    _, reward, terminated, truncated, _ = env.step(action)
                    steps[name] += 1

                    if terminated:
                        outcomes[name] = "SUCCESS"
                    elif truncated:
                        outcomes[name] = "FAIL"

        # Draw
        screen.fill((30, 30, 30))
        for i, name in enumerate(models):
            x_offset = i * panel_size
            env = envs[name]
            obs = env._render_obs()

            # Scale 224x224 to panel_size x panel_size
            surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (panel_size, panel_size))
            screen.blit(surf, (x_offset, label_height))

            # Label
            outcome = outcomes[name]
            if outcome == "SUCCESS":
                color = (80, 200, 80)
                label = f"{name} [OK] s{steps[name]}"
            elif outcome == "FAIL":
                color = (200, 80, 80)
                label = f"{name} [FAIL] s{steps[name]}"
            else:
                color = (220, 220, 220)
                label = f"{name} s{steps[name]}"

            text_surf = font.render(label, True, color)
            screen.blit(text_surf, (x_offset + 4, 8))

        if paused:
            pause_surf = font.render("PAUSED", True, (255, 255, 0))
            screen.blit(pause_surf, (window_w // 2 - 40, window_h - 24))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live rollout viewer for Ch02 models")
    parser.add_argument("--seed", type=int, default=4, help="initial episode seed")
    parser.add_argument("--fps", type=int, default=15, help="playback speed (frames/sec)")
    parser.add_argument("--panel-size", type=int, default=256, help="pixel size per panel")
    parser.add_argument("--skip-scratch", action="store_true", help="skip scratch CNN")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    print("Loading models...")
    models = load_models(args.checkpoint_dir, args.device, args.skip_scratch)

    if not models:
        print("No checkpoints found! Run the benchmark first:")
        print("  python compare_encoders.py --preset full --attn-pool")
        sys.exit(1)

    print(f"\nStarting viewer (seed={args.seed}, fps={args.fps})...")
    print("Controls: SPACE=pause, R=random reset, N=next seed, Q=quit")
    run_viewer(models, args.device, args.seed, args.fps, args.panel_size)


if __name__ == "__main__":
    main()
