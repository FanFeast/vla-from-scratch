"""Pygame live rollout viewer for Chapter 4 action heads.

Shows 3 models side-by-side (discrete, regression, diffusion) running
the same episode in real-time.

Controls:
  SPACE  Pause/resume
  R      Reset all to same seed
  N      Next seed
  K      Toggle chunk size (K=1 vs K=4)
  Q/ESC  Quit
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from mini_pusht_continuous import ContinuousMiniPushT
from action_heads import VLA, VisionEncoder, build_vla, HEAD_TYPES, PRESETS

try:
    import pygame
except ImportError:
    print("pip install pygame")
    sys.exit(1)


PANEL_SIZE = 300
GAP = 10
FPS = 15


def load_model(
    head: str, chunk_size: int, ckpt_dir: str, n_demos: int, epochs: int
) -> VLA | None:
    """Load a trained model from checkpoint."""
    path = os.path.join(ckpt_dir, f"{head}_K{chunk_size}_{n_demos}d_{epochs}ep.pt")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return None
    model = build_vla(head, chunk_size=chunk_size)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    print(f"  Loaded: {head} K={chunk_size}")
    return model


def scale_surface(obs: np.ndarray, target_size: int) -> pygame.Surface:
    """Scale observation image to target panel size."""
    # obs is (H, W, 3) uint8
    h, w = obs.shape[:2]
    surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (target_size, target_size))


def run_viewer(
    models: dict[str, VLA],
    vision_encoder: VisionEncoder,
    chunk_size: int,
    device: str = "cpu",
) -> None:
    """Run the Pygame viewer."""
    n = len(models)
    width = n * PANEL_SIZE + (n - 1) * GAP
    height = PANEL_SIZE + 60

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Ch04: Action Heads Comparison")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    envs = {name: ContinuousMiniPushT(size=224) for name in models}
    seed = 0
    paused = False

    names = list(models.keys())
    dones: dict[str, bool] = {}
    steps: dict[str, int] = {}
    successes: dict[str, bool] = {}

    def reset_all():
        for name in names:
            envs[name].reset(seed=seed)
        dones.clear()
        steps.clear()
        successes.clear()
        for name in names:
            dones[name] = False
            steps[name] = 0
            successes[name] = False

    reset_all()

    running = True
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
                    reset_all()
                elif event.key == pygame.K_n:
                    seed += 1
                    reset_all()

        if not paused:
            for name in names:
                if dones[name]:
                    continue
                env = envs[name]
                model = models[name]

                obs = env.render()
                img_t = (
                    torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                )
                img_t = img_t.to(device)

                with torch.no_grad():
                    features = vision_encoder(img_t)
                    action = model(features)

                if action.dim() == 3:
                    # Chunked: execute first action for smooth viz
                    a = action[0, 0].cpu().numpy()
                else:
                    a = action[0].cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(a)
                steps[name] += 1
                if terminated:
                    successes[name] = True
                    dones[name] = True
                elif truncated:
                    dones[name] = True

        # Draw
        screen.fill((30, 30, 30))
        for i, name in enumerate(names):
            x_off = i * (PANEL_SIZE + GAP)
            obs = envs[name].render()
            surf = scale_surface(obs, PANEL_SIZE)
            screen.blit(surf, (x_off, 0))

            # Label
            if successes[name]:
                status = "OK"
                color = (0, 255, 0)
            elif dones[name]:
                status = "FAIL"
                color = (255, 0, 0)
            else:
                status = f"s{steps[name]}"
                color = (255, 255, 255)

            label = font.render(f"{name} [{status}]", True, color)
            screen.blit(label, (x_off + 5, PANEL_SIZE + 5))

            seed_label = font.render(f"seed={seed}", True, (180, 180, 180))
            screen.blit(seed_label, (x_off + 5, PANEL_SIZE + 25))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live rollout viewer for ch04 action heads"
    )
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--chunk-size", type=int, choices=[1, 4], default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for checkpoint filename")
    parser.add_argument("--n-demos", type=int, default=None,
                        help="Override n_demos for checkpoint filename")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    n_demos = args.n_demos or preset["n_demos"]
    epochs = args.epochs or preset["epochs"]

    ckpt_dir = args.checkpoint_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(os.path.dirname(__file__) or ".", ckpt_dir)

    print("Loading models...")
    vision_encoder = VisionEncoder()
    vision_encoder = vision_encoder.to(args.device)
    vision_encoder.eval()

    models: dict[str, VLA] = {}
    for head in HEAD_TYPES:
        model = load_model(head, args.chunk_size, ckpt_dir, n_demos, epochs)
        if model is not None:
            model = model.to(args.device)
            models[f"{head.capitalize()} K={args.chunk_size}"] = model

    if not models:
        print("No models found! Run training first.")
        sys.exit(1)

    print(f"\nStarting viewer (chunk_size={args.chunk_size}, seed=0, fps={FPS})...")
    print("Controls: SPACE=pause R=reset N=next Q=quit")
    run_viewer(models, vision_encoder, args.chunk_size, device=args.device)
