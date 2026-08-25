"""Tiny VLA: CNN vision encoder + lookup language encoder + MLP action head.

Also contains ScriptedExpert, NoisyExpert, data collection, and training loop.
Run standalone: python tiny_vla.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Union

sys.path.insert(0, os.path.dirname(__file__))
from mini_pusht import MiniPushT


# ---------------------------------------------------------------------------
# Expert policies
# ---------------------------------------------------------------------------


class ScriptedExpert:
    """Deterministic two-phase oracle for MiniPushT.

    Phase 1: navigate agent toward the block.
    Phase 2: push block toward goal.
    Uses ground-truth state from env info dict (not the image).
    """

    def act(self, info: dict) -> int:
        """Return optimal discrete action given environment state.

        Args:
            info: dict with keys agent_pos, block_pos, goal_pos (np.ndarray)
        Returns:
            action in {0=up, 1=down, 2=left, 3=right, 4=stay}
        """
        agent = info["agent_pos"].astype(float)
        block = info["block_pos"].astype(float)
        goal = info["goal_pos"].astype(float)

        if np.linalg.norm(agent - block) > 4:
            # Phase 1: move toward block
            return self._move_toward(agent, block)
        # Phase 2: push block toward goal
        return self._move_toward(agent, goal)

    def _move_toward(self, src: np.ndarray, dst: np.ndarray) -> int:
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 2  # right or left
        else:
            return 1 if dy > 0 else 0  # down or up


class NoisyExpert:
    """Expert with epsilon-greedy noise (appendix: data augmentation demo).

    See README appendix: try training with epsilon=0.1 and compare results.
    """

    def __init__(self, expert: ScriptedExpert, epsilon: float = 0.1) -> None:
        self.expert = expert
        self.epsilon = epsilon
        self._rng = np.random.default_rng()

    def act(self, info: dict) -> int:
        """Return action: expert action with probability 1-epsilon, else random."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, 5))
        return self.expert.act(info)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class VisionEncoder(nn.Module):
    """3-layer CNN: maps 64x64 RGB images to 128-dim embeddings.

    Architecture:
        Conv(3->32, 3x3, s=2) -> ReLU  # (B,32,31,31)
        Conv(32->64, 3x3, s=2) -> ReLU  # (B,64,15,15)
        Conv(64->64, 3x3, s=2) -> ReLU  # (B,64,7,7)
        Flatten -> Linear(3136, 128)
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Args: image (B, 3, 64, 64) float in [0,1]. Returns: (B, 128)."""
        return self.net(image)


class LanguageEncoder(nn.Module):
    """Lookup-table language encoder for 3 fixed instructions.

    Interface is intentionally identical to Chapter 3's SmolLM2 encoder --
    swap this class out and nothing else changes.
    """

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


class TinyVLA(nn.Module):
    """Minimal VLA: CNN vision + lookup language + MLP action head.

    forward(image, instruction) -> action logits
    """

    def __init__(self, n_actions: int = 5) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.language_encoder = LanguageEncoder()
        self.action_head = nn.Sequential(
            nn.Linear(256, 256),
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
            image: (B, 3, 64, 64) float in [0, 1]
            instruction: str or list[str], length must match B
        Returns:
            (B, n_actions) logits
        """
        vision_feat = self.vision_encoder(image)  # (B, 128)
        lang_feat = self.language_encoder(instruction)  # (B, 128)
        fused = torch.cat([vision_feat, lang_feat], dim=-1)  # (B, 256)
        return self.action_head(fused)  # (B, n_actions)


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

    Args:
        env: MiniPushT instance
        expert: ScriptedExpert (or NoisyExpert) instance
        n_episodes: number of episodes to collect
        instruction: language command passed to env at each reset
    Returns:
        list of {"image": np.ndarray (64,64,3), "instruction": str, "action": int}
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


class PushTDataset(Dataset):
    """Dataset of (image_tensor, instruction, action) tuples from expert demos."""

    def __init__(self, demos: list[dict]) -> None:
        self.demos = demos

    def __len__(self) -> int:
        return len(self.demos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        demo = self.demos[idx]
        # HWC uint8 -> CHW float32 in [0, 1]
        image = torch.from_numpy(demo["image"]).permute(2, 0, 1).float() / 255.0
        return image, demo["instruction"], int(demo["action"])


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def train(
    model: TinyVLA,
    dataset: PushTDataset,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Behavior cloning training loop.

    Args:
        model: TinyVLA instance
        dataset: PushTDataset of expert demos
        epochs: number of full passes over the dataset
        batch_size: samples per gradient step
        lr: Adam learning rate
        device: "cpu" or "cuda"
    Returns:
        list of per-epoch average loss values
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history: list[float] = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, instructions, actions in loader:
            images = images.to(device)
            actions = actions.long().to(device)

            optimizer.zero_grad()
            logits = model(images, list(instructions))
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1:02d}/{epochs}  loss={avg_loss:.4f}")

    return loss_history


def evaluate(
    model: TinyVLA,
    env: MiniPushT,
    n_episodes: int = 20,
    device: str = "cpu",
) -> float:
    """Evaluate policy success rate over n_episodes.

    Args:
        model: trained TinyVLA
        env: MiniPushT instance
        n_episodes: number of evaluation episodes
        device: "cpu" or "cuda"
    Returns:
        fraction of episodes where block reached goal
    """
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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def render_episode(
    env: MiniPushT,
    policy_fn,
    title: str = "Episode",
    n_frames: int = 6,
    save_path: str | None = None,
) -> None:
    """Render a grid of frames from one episode and optionally save to disk.

    Args:
        env: MiniPushT instance
        policy_fn: callable(obs, info) -> int  (action)
        title: figure title
        n_frames: how many evenly-spaced frames to show in the grid
        save_path: if given, save figure to this path instead of showing it
    """
    frames = []
    obs, info = env.reset(seed=0)
    done = False
    step = 0
    all_obs = [obs.copy()]
    all_rewards = [0.0]

    while not done:
        action = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        all_obs.append(obs.copy())
        all_rewards.append(reward)
        done = terminated or truncated
        step += 1

    success = any(r > 0 for r in all_rewards)
    total_steps = step

    # Pick n_frames evenly spaced indices
    indices = [int(i * (len(all_obs) - 1) / (n_frames - 1)) for i in range(n_frames)]
    frames = [all_obs[i] for i in indices]

    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2, 2.5))
    for ax, frame, idx in zip(axes, frames, indices):
        ax.imshow(frame)
        ax.set_title(f"step {idx}", fontsize=8)
        ax.axis("off")

    outcome = "SUCCESS" if success else f"fail ({total_steps} steps)"
    fig.suptitle(f"{title}  [{outcome}]", fontsize=10, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def save_training_curve(
    losses: list[float],
    save_path: str = "../../assets/figures/ch01_training_loss.png",
) -> None:
    """Save the training loss curve as a PNG.

    Args:
        losses: per-epoch loss values from train()
        save_path: where to save the figure
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Chapter 1: Behavior cloning training loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Live interactive evaluation
# ---------------------------------------------------------------------------

ACTION_NAMES = ["up", "down", "left", "right", "stay"]


def run_live_eval(model: TinyVLA, env: MiniPushT, device: str = "cpu") -> None:
    """Interactive terminal + live window eval loop.

    Shows a real-time matplotlib window with:
      - Left panel: the 64x64 env rendered live (blue=agent, red=block, green=goal)
      - Right panel: bar chart of model's action probabilities for each step

    The user types an instruction in the terminal and watches the model execute it.
    Type 'q' to quit.

    What is happening at each step:
      1. The env renders the current state as a 64x64 RGB image.
      2. The image is passed through VisionEncoder (CNN) -> 128-dim vector.
      3. The instruction string is passed through LanguageEncoder (lookup) -> 128-dim vector.
      4. Both vectors are concatenated -> 256-dim and fed into the MLP action head.
      5. The MLP outputs 5 logits (one per action: up/down/left/right/stay).
      6. We take argmax -> the action with the highest probability is executed.
      7. The env steps forward, returns the next image, and we repeat.
    """
    vocab = list(LanguageEncoder.VOCAB.keys())

    print("\n" + "=" * 50)
    print("INTERACTIVE EVAL")
    print("=" * 50)
    print("The 5 actions the model can choose:")
    for i, name in enumerate(ACTION_NAMES):
        delta = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)][i]
        print(f"  {i} = {name:5s}  (dx={delta[0]:+d}, dy={delta[1]:+d})")
    print("\nValid instructions:")
    for i, instr in enumerate(vocab):
        print(f"  [{i + 1}] {instr}")
    print("\nClose the plot window or type 'q' to quit.\n")

    plt.ion()
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    ax_env = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    fig.tight_layout(pad=2)

    while True:
        raw = input("Instruction (1/2/3 or full text, q to quit): ").strip()
        if raw.lower() == "q":
            break
        if raw in ("1", "2", "3"):
            instruction = vocab[int(raw) - 1]
        elif raw in LanguageEncoder.VOCAB:
            instruction = raw
        else:
            print(f"  Unknown. Choose from: {vocab} or type 1/2/3")
            continue

        print(f"\n  Running: '{instruction}'")
        print("  Steps: agent(blue) -> block(red) -> goal(green)")
        obs, info = env.reset(seed=None, options={"instruction": instruction})
        done = False
        step = 0

        model.eval()
        with torch.no_grad():
            while not done:
                # --- forward pass ---
                img_t = (
                    torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                ).to(device)
                logits = model(img_t, instruction)  # (1, 5)
                probs = torch.softmax(logits, dim=-1)[0]  # (5,)
                action = int(logits.argmax(dim=-1).item())

                # --- left panel: env ---
                ax_env.clear()
                ax_env.imshow(obs, interpolation="nearest")
                ax_env.set_title(
                    f"Step {step:03d}  action -> {ACTION_NAMES[action]}\n"
                    f"agent={info['agent_pos']}  block={info['block_pos']}",
                    fontsize=8,
                )
                ax_env.axis("off")

                # --- right panel: action probabilities ---
                colors = ["tomato" if i == action else "steelblue" for i in range(5)]
                ax_bar.clear()
                bars = ax_bar.barh(ACTION_NAMES, probs.cpu().numpy(), color=colors)
                ax_bar.set_xlim(0, 1)
                ax_bar.set_xlabel("probability")
                ax_bar.set_title(
                    f"Model output  (chosen = {ACTION_NAMES[action]})", fontsize=9
                )
                # annotate each bar with its probability
                for bar, p in zip(bars, probs.cpu().numpy()):
                    ax_bar.text(
                        min(p + 0.02, 0.95),
                        bar.get_y() + bar.get_height() / 2,
                        f"{p:.2f}",
                        va="center",
                        fontsize=8,
                    )

                fig.canvas.draw()
                plt.pause(0.08)  # ~12 fps

                # --- env step ---
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

            # episode over
            outcome = "SUCCESS" if terminated else "timeout (200 steps)"
            ax_env.set_title(
                f"{outcome} after {step} steps",
                fontsize=9,
                color="green" if terminated else "red",
            )
            fig.canvas.draw()
            plt.pause(1.5)
            print(f"  -> {outcome} ({step} steps)\n")

    plt.ioff()
    plt.close(fig)
    print("Exited interactive eval.")


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tiny VLA on MiniPushT")
    parser.add_argument("--n-demos", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="torch CPU threads (4 is faster than default on most machines)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="skip saving visualizations (useful for CI/headless runs)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="after training, open live window and accept instruction input",
    )
    args = parser.parse_args()

    viz = not args.no_viz

    # Cap CPU threads: PyTorch's default (num_cores) often hurts small models
    # due to thread-launch overhead. 4 threads is a sweet spot for CPU training.
    if args.device == "cpu":
        torch.set_num_threads(args.num_threads)

    print("=== Tiny VLA: Chapter 1 ===")
    print(
        "Environment: MiniPushT -- 64x64 grid, blue agent pushes red block onto green goal."
    )
    print("Task: learn from expert demos to reproduce the push behavior.\n")

    # --- Step 1: collect expert demonstrations ---
    print(f"[1/4] Collecting {args.n_demos} expert demos...")
    env = MiniPushT()
    expert = ScriptedExpert()
    demos = collect_demos(env, expert, n_episodes=args.n_demos)
    n_success = sum(1 for d in demos if d["action"] != 4)  # rough proxy
    print(f"      Collected {len(demos)} transitions from {args.n_demos} episodes.")
    print("      Each transition = (image 64x64x3, instruction str, action int 0-4)\n")

    if viz:
        print("      Saving expert episode visualization...")
        render_episode(
            env,
            policy_fn=lambda obs, info: expert.act(info),
            title="Expert policy",
            save_path=os.path.join(
                os.path.dirname(__file__),
                "../../assets/figures/ch01_expert_episode.png",
            ),
        )

    # --- Step 2: build dataset + model ---
    dataset = PushTDataset(demos)
    model = TinyVLA()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2/4] Model: TinyVLA ({n_params:,} params)")
    print("      Architecture: VisionEncoder(CNN)->128 + LanguageEncoder(lookup)->128")
    print("      -> concat 256 -> MLP -> 5 action logits\n")

    # --- Step 3: behavior cloning ---
    print(
        f"[3/4] Training for {args.epochs} epochs (behavior cloning = cross-entropy loss)..."
    )
    print("      Goal: predict the expert's action given (image, instruction).")
    losses = train(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    print(
        f"\n      Loss: {losses[0]:.4f} -> {losses[-1]:.4f}  "
        f"({'improving' if losses[-1] < losses[0] else 'not converging'})"
    )

    if viz:
        print("      Saving training loss curve...")
        save_training_curve(losses)

    # --- Step 4: evaluate ---
    print("\n[4/4] Evaluating over 50 episodes...")
    print("      Success = block reaches green goal within 200 steps.")
    print("      The model gets the image only (not ground-truth positions).")
    rate = evaluate(model, env, n_episodes=50, device=args.device)
    print(f"\n      Success rate: {rate * 100:.1f}%  ({int(rate * 50)}/50 episodes)")
    if rate >= 0.7:
        print("      Good! The policy learned to push the block to the goal.")
    elif rate >= 0.3:
        print("      Partial learning. Try --n-demos 2000 or --epochs 40.")
    else:
        print("      Low success rate -- this is expected with few demos/epochs.")
        print("      With --n-demos 1000 --epochs 20 you should see ~70-80%.")

    print("\n      Compounding errors: on unseen starting positions success drops.")
    print("      This motivates Chapter 2 (better vision = more robust features).\n")

    if viz:
        print("      Saving trained policy episode visualization...")
        model.eval()

        def _policy(obs: np.ndarray, info: dict) -> int:
            with torch.no_grad():
                img = (
                    torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                )
                return int(model(img, info["instruction"]).argmax(dim=-1).item())

        render_episode(
            env,
            policy_fn=_policy,
            title="Trained policy (after BC)",
            save_path=os.path.join(
                os.path.dirname(__file__),
                "../../assets/figures/ch01_policy_episode.png",
            ),
        )
        print(
            "\nFigures saved to assets/figures/. Open them to see what the model learned."
        )

    print("Done. -> Chapter 2: swap VisionEncoder for pretrained CLIP/SigLIP.")

    if args.interactive:
        run_live_eval(model, env, device=args.device)
