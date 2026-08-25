"""Pygame live rollout viewer for Chapter 6 flow matching models.

Shows FM vs DDPM side-by-side running the PushT task in real-time,
so you can visually compare trajectory quality.

Controls:
  SPACE  Pause/resume
  R      Reset all to same seed
  N      Next seed
  P      Previous seed
  1/2    Switch chunk size (K=4 / K=16)
  Q/ESC  Quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from action_expert import build_action_expert
from flow_matching import (
    PRESETS,
    EMAModel,
    FlowMatchingTrainer,
    DDPMTransformerTrainer,
    NormStats,
    VisionEncoder,
    CHECKPOINT_DIR,
)

try:
    import pygame
except ImportError:
    print("pip install pygame")
    sys.exit(1)

try:
    import gymnasium as gym
    import gym_pusht  # noqa: F401
except ImportError:
    print("pip install gymnasium gym-pusht")
    sys.exit(1)

from PIL import Image


PANEL_SIZE = 300
GAP = 10
INFO_HEIGHT = 80
FPS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelRunner:
    """Wraps a trained model for live inference."""

    def __init__(
        self,
        name: str,
        trainer: FlowMatchingTrainer | DDPMTransformerTrainer,
        ema: EMAModel,
        state_stats: NormStats,
        action_stats: NormStats,
        chunk_size: int,
        inference_steps: int,
        device: torch.device = DEVICE,
    ):
        self.name = name
        self.trainer = trainer
        self.ema = ema
        self.state_stats = state_stats
        self.action_stats = action_stats
        self.chunk_size = chunk_size
        self.inference_steps = inference_steps
        self.device = device
        self.ema.shadow.to(device)
        self.ema.shadow.eval()

    def predict(
        self, vision_emb: torch.Tensor, state: np.ndarray
    ) -> list[np.ndarray]:
        """Predict a chunk of actions from vision embedding and state."""
        state_t = torch.tensor(state, dtype=torch.float32)
        state_norm = self.state_stats.normalize(state_t).unsqueeze(0).to(self.device)
        vision = vision_emb.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.trainer.sample(
                vision,
                state_norm,
                ema_model=self.ema.shadow,
                num_steps=self.inference_steps,
            )

        actions = []
        for k in range(self.chunk_size):
            a_norm = pred[0, k].cpu()
            a = self.action_stats.denormalize(a_norm).numpy().astype(np.float32)
            actions.append(a)
        return actions


def load_model(
    method: str,
    chunk_size: int,
    cfg: dict,
    device: torch.device = DEVICE,
) -> ModelRunner | None:
    """Load a trained model from checkpoint."""
    ckpt_path = CHECKPOINT_DIR / f"ch06_pusht_{method}_K{chunk_size}.pt"
    stats_path = CHECKPOINT_DIR / "ch06_pusht_stats.pt"

    if not ckpt_path.exists():
        print(f"  Warning: {ckpt_path} not found")
        return None
    if not stats_path.exists():
        print(f"  Warning: {stats_path} not found")
        return None

    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    action_stats = NormStats(stats["action_min"], stats["action_max"])
    state_stats = NormStats(stats["state_min"], stats["state_max"])

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

    if method == "flow_matching":
        trainer = FlowMatchingTrainer(model, inference_steps=10)
        infer_steps = cfg.get("fm_inference_steps", 10)
    else:
        trainer = DDPMTransformerTrainer(model)
        infer_steps = cfg.get("ddpm_inference_steps", 20)

    name = f"{'FM' if method == 'flow_matching' else 'DDPM'} K={chunk_size}"
    print(f"  Loaded: {name}")
    return ModelRunner(
        name=name,
        trainer=trainer,
        ema=ema,
        state_stats=state_stats,
        action_stats=action_stats,
        chunk_size=chunk_size,
        inference_steps=infer_steps,
        device=device,
    )


def scale_surface(obs: np.ndarray, target_size: int) -> pygame.Surface:
    """Scale observation image to target panel size."""
    surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (target_size, target_size))


def run_viewer(runners: list[ModelRunner], device: torch.device = DEVICE) -> None:
    """Run the Pygame viewer with side-by-side model comparison."""
    n = len(runners)
    width = n * PANEL_SIZE + (n - 1) * GAP
    height = PANEL_SIZE + INFO_HEIGHT

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Ch06: Flow Matching vs DDPM — PushT Live Rollout")
    font = pygame.font.SysFont("monospace", 14)
    font_big = pygame.font.SysFont("monospace", 16, bold=True)
    clock = pygame.time.Clock()

    encoder = VisionEncoder(device=device)
    envs = [
        gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        for _ in runners
    ]

    seed = 0
    paused = False

    # Per-runner state
    action_buffers: list[list[np.ndarray]] = [[] for _ in runners]
    steps = [0] * n
    coverages = [0.0] * n
    dones = [False] * n
    successes = [False] * n
    observations: list[dict] = [{}] * n
    last_infer_ms = [0.0] * n

    def reset_all():
        nonlocal observations
        for i in range(n):
            obs, info = envs[i].reset(seed=seed)
            observations[i] = obs
            action_buffers[i].clear()
            steps[i] = 0
            coverages[i] = 0.0
            dones[i] = False
            successes[i] = False
            last_infer_ms[i] = 0.0

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
                elif event.key == pygame.K_p:
                    seed = max(0, seed - 1)
                    reset_all()

        if not paused:
            for i, runner in enumerate(runners):
                if dones[i]:
                    continue

                obs = observations[i]

                # Get new actions if buffer empty
                if not action_buffers[i]:
                    pil_img = Image.fromarray(obs["pixels"])
                    vision_emb = encoder.encode_pil(pil_img).to(device)

                    t0 = time.time()
                    actions = runner.predict(vision_emb, obs["agent_pos"])
                    last_infer_ms[i] = (time.time() - t0) * 1000
                    action_buffers[i] = actions

                # Execute next action
                action = action_buffers[i].pop(0)
                obs, reward, terminated, truncated, info = envs[i].step(action)
                observations[i] = obs
                steps[i] += 1
                coverages[i] = max(coverages[i], reward)

                if info.get("is_success", False):
                    successes[i] = True
                    dones[i] = True
                elif terminated or truncated:
                    dones[i] = True

        # Draw
        screen.fill((30, 30, 30))
        for i, runner in enumerate(runners):
            x_off = i * (PANEL_SIZE + GAP)

            # Render environment
            frame = envs[i].render()
            surf = scale_surface(frame, PANEL_SIZE)
            screen.blit(surf, (x_off, 0))

            # Status color
            if successes[i]:
                status_text = "SUCCESS"
                color = (0, 255, 0)
            elif dones[i]:
                status_text = "DONE"
                color = (255, 100, 100)
            else:
                status_text = f"step {steps[i]}"
                color = (255, 255, 255)

            # Model name
            label = font_big.render(runner.name, True, (100, 200, 255))
            screen.blit(label, (x_off + 5, PANEL_SIZE + 4))

            # Status + coverage
            info_text = f"{status_text} | cov={coverages[i]:.0%}"
            info_label = font.render(info_text, True, color)
            screen.blit(info_label, (x_off + 5, PANEL_SIZE + 24))

            # Inference time + seed
            timing = f"{last_infer_ms[i]:.0f}ms | seed={seed}"
            timing_label = font.render(timing, True, (150, 150, 150))
            screen.blit(timing_label, (x_off + 5, PANEL_SIZE + 44))

        # Controls hint
        hint = font.render("SPACE=pause R=reset N/P=seed Q=quit", True, (100, 100, 100))
        screen.blit(hint, (5, height - 18))

        pygame.display.flip()
        clock.tick(FPS)

    for env in envs:
        env.close()
    del encoder
    torch.cuda.empty_cache()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Live rollout viewer for Ch06 flow matching models"
    )
    parser.add_argument(
        "--chunk-size", type=int, choices=[4, 16], default=4,
        help="Action chunk size (default: 4)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["flow_matching", "ddpm"],
        choices=["flow_matching", "ddpm"],
        help="Which methods to show (default: both)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = PRESETS["pusht"]

    print("Loading models...")
    runners: list[ModelRunner] = []
    for method in args.methods:
        runner = load_model(method, args.chunk_size, cfg, device)
        if runner is not None:
            runners.append(runner)

    if not runners:
        print("No models found! Run training first:")
        print("  python flow_matching.py --preset pusht")
        sys.exit(1)

    print(f"\nStarting viewer ({len(runners)} models, K={args.chunk_size}, seed=0)")
    print("Controls: SPACE=pause  R=reset  N/P=seed  Q=quit")
    run_viewer(runners, device)


if __name__ == "__main__":
    main()
