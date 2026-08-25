"""Live side-by-side rollout viewer for trained Ch03 language-conditioned models.

Opens a Pygame window showing all 4 model configs attempting MiniPushT with
the same instruction, so you can compare how each handles language conditioning.

Usage:
    python live_rollout.py                    # default settings
    python live_rollout.py --seed 10          # specific starting seed
    python live_rollout.py --fps 10           # slower playback

Controls:
    SPACE  -- pause/resume
    R      -- reset with new random seed
    N      -- next seed (increment)
    I      -- cycle to next instruction
    P      -- toggle paraphrase mode
    Q/ESC  -- quit
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import pygame
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cross_attention import build_vla, CANONICAL_INSTRUCTIONS
from mini_pusht import MiniPushT, PARAPHRASES

# -- Model configurations ---------------------------------------------------

PRESET_CONFIGS = {
    "scale": [
        {"name": "Lookup+Concat", "fusion": "concat", "lm": "lookup",
         "ckpt": "lookup__concat_224px_10000d_200ep.pt"},
        {"name": "135M+Concat", "fusion": "concat", "lm": "smollm2-135m",
         "ckpt": "smollm2_135m__concat_224px_10000d_100ep.pt"},
        {"name": "135M+CrossAttn", "fusion": "cross-attn", "lm": "smollm2-135m",
         "ckpt": "smollm2_135m__crossattn_224px_10000d_100ep.pt"},
        {"name": "360M+CrossAttn", "fusion": "cross-attn", "lm": "smollm2-360m",
         "ckpt": "smollm2_360m__crossattn_224px_10000d_100ep.pt"},
    ],
    "full": [
        {"name": "Lookup+Concat", "fusion": "concat", "lm": "lookup",
         "ckpt": "lookup__concat_224px_2200d_40ep.pt"},
        {"name": "135M+Concat", "fusion": "concat", "lm": "smollm2-135m",
         "ckpt": "smollm2_135m__concat_224px_2200d_40ep.pt"},
        {"name": "135M+CrossAttn", "fusion": "cross-attn", "lm": "smollm2-135m",
         "ckpt": "smollm2_135m__crossattn_224px_2200d_40ep.pt"},
        {"name": "360M+CrossAttn", "fusion": "cross-attn", "lm": "smollm2-360m",
         "ckpt": "smollm2_360m__crossattn_224px_2200d_40ep.pt"},
    ],
    "quick": [
        {"name": "Lookup+Concat", "fusion": "concat", "lm": "lookup",
         "ckpt": "lookup__concat_64px_220d_10ep.pt"},
        {"name": "135M+Concat", "fusion": "concat", "lm": "smollm2-135m",
         "ckpt": "smollm2_135m__concat_64px_220d_10ep.pt"},
        {"name": "135M+CrossAttn", "fusion": "cross-attn", "lm": "smollm2-135m",
         "ckpt": "smollm2_135m__crossattn_64px_220d_10ep.pt"},
        {"name": "360M+CrossAttn", "fusion": "cross-attn", "lm": "smollm2-360m",
         "ckpt": "smollm2_360m__crossattn_64px_220d_10ep.pt"},
    ],
}


def load_models(
    configs: list[dict], ckpt_dir: str, device: str
) -> list[dict]:
    """Build each model via build_vla() and load its checkpoint.

    Returns a list of dicts with keys: name, model, loaded.
    """
    entries: list[dict] = []
    for cfg in configs:
        path = os.path.join(ckpt_dir, cfg["ckpt"])
        if not os.path.exists(path):
            print(f"  Skipping {cfg['name']} (no checkpoint at {path})")
            continue
        model = build_vla(fusion_type=cfg["fusion"], lm_type=cfg["lm"])
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model = model.to(device)
        model.eval()
        entries.append({"name": cfg["name"], "model": model})
        print(f"  Loaded: {cfg['name']}")
    return entries


def _pick_paraphrase(canonical: str) -> str:
    """Return a random paraphrase for a canonical instruction."""
    paras = PARAPHRASES.get(canonical, [])
    if not paras:
        return canonical
    return random.choice(paras)


def run_viewer(
    entries: list[dict],
    device: str,
    initial_seed: int = 4,
    fps: int = 15,
    panel_size: int = 256,
    env_size: int = 224,
) -> None:
    """Run the live Pygame viewer."""
    n_models = len(entries)
    if n_models == 0:
        print("No models loaded!")
        return

    # Layout
    label_height = 56
    window_w = panel_size * n_models
    window_h = panel_size + label_height

    pygame.init()
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption(
        "Ch03 Live Rollout -- SPACE=pause R=reset N=next I=instr P=para Q=quit"
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14, bold=True)
    font_small = pygame.font.SysFont("monospace", 12)

    names = [e["name"] for e in entries]
    models = [e["model"] for e in entries]

    envs = [MiniPushT(size=env_size) for _ in entries]

    seed = initial_seed
    instr_idx = 0
    paraphrase_mode = False

    def current_instruction() -> str:
        canonical = CANONICAL_INSTRUCTIONS[instr_idx]
        if paraphrase_mode:
            return _pick_paraphrase(canonical)
        return canonical

    instruction = current_instruction()

    outcomes: list[str | None] = [None] * n_models
    steps: list[int] = [0] * n_models

    def reset_all() -> None:
        nonlocal instruction, outcomes, steps
        instruction = current_instruction()
        outcomes = [None] * n_models
        steps = [0] * n_models
        for env in envs:
            env.reset(seed=seed, options={"instruction": instruction})

    reset_all()
    paused = False
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
                    seed = int(time.time()) % 10000
                    reset_all()
                elif event.key == pygame.K_n:
                    seed += 1
                    reset_all()
                elif event.key == pygame.K_i:
                    instr_idx = (instr_idx + 1) % len(CANONICAL_INSTRUCTIONS)
                    reset_all()
                elif event.key == pygame.K_p:
                    paraphrase_mode = not paraphrase_mode
                    reset_all()

        if not paused:
            with torch.no_grad():
                for i, model in enumerate(models):
                    if outcomes[i] is not None:
                        continue
                    env = envs[i]
                    obs = env._render_obs()
                    image = (
                        torch.from_numpy(obs)
                        .permute(2, 0, 1)
                        .float()
                        .unsqueeze(0)
                        / 255.0
                    ).to(device)
                    logits = model(image, [instruction])
                    action = int(logits.argmax(dim=-1).item())
                    _, reward, terminated, truncated, _ = env.step(action)
                    steps[i] += 1
                    if terminated:
                        outcomes[i] = "SUCCESS"
                    elif truncated:
                        outcomes[i] = "FAIL"

        # -- Draw -------------------------------------------------------
        screen.fill((30, 30, 30))
        for i, name in enumerate(names):
            x_off = i * panel_size

            obs = envs[i]._render_obs()
            surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (panel_size, panel_size))
            screen.blit(surf, (x_off, label_height))

            # Top line: model name + outcome
            outcome = outcomes[i]
            if outcome == "SUCCESS":
                color = (80, 200, 80)
                label = f"{name} [OK] s{steps[i]}"
            elif outcome == "FAIL":
                color = (200, 80, 80)
                label = f"{name} [FAIL] s{steps[i]}"
            else:
                color = (220, 220, 220)
                label = f"{name} s{steps[i]}"
            screen.blit(font.render(label, True, color), (x_off + 4, 4))

            # Second line: instruction (truncated to fit panel)
            mode_tag = "[P] " if paraphrase_mode else ""
            instr_text = f"{mode_tag}{instruction}"
            max_chars = panel_size // 7
            if len(instr_text) > max_chars:
                instr_text = instr_text[: max_chars - 1] + "..."
            screen.blit(
                font_small.render(instr_text, True, (180, 180, 255)),
                (x_off + 4, 22),
            )

        if paused:
            pause_surf = font.render("PAUSED", True, (255, 255, 0))
            screen.blit(pause_surf, (window_w // 2 - 40, window_h - 24))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live rollout viewer for Ch03 language-conditioned models"
    )
    parser.add_argument("--seed", type=int, default=4, help="initial episode seed")
    parser.add_argument("--fps", type=int, default=15, help="playback speed")
    parser.add_argument("--panel-size", type=int, default=256, help="pixels per panel")
    parser.add_argument(
        "--preset", choices=["scale", "full", "quick"], default="scale",
        help="which checkpoint set to load (default: scale)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    configs = PRESET_CONFIGS[args.preset]
    print(f"Loading models (preset={args.preset})...")
    entries = load_models(configs, args.checkpoint_dir, args.device)

    if not entries:
        print("No checkpoints found! Train models first:")
        print("  python cross_attention.py")
        sys.exit(1)

    print(f"\nStarting viewer (seed={args.seed}, fps={args.fps})...")
    print("Controls: SPACE=pause R=reset N=next I=instruction P=paraphrase Q=quit")
    env_size = 64 if args.preset == "quick" else 224
    run_viewer(entries, args.device, args.seed, args.fps, args.panel_size, env_size)


if __name__ == "__main__":
    main()
