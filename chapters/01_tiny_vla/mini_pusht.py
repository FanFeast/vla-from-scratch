"""MiniPushT: simplified 2D block-pushing gym environment.

A 64x64 grid world where an agent must push a block onto a goal.
Observations are rendered as RGB images using pure numpy (no display needed).
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional


class MiniPushT(gymnasium.Env):
    """Simplified 2D block-pushing task with discrete actions.

    Observation: 64x64x3 RGB image (uint8).
    Actions: Discrete(5) -- 0=up, 1=down, 2=left, 3=right, 4=stay.
    Reward: +1.0 when block overlaps goal (within 4px), 0 otherwise.
    Episode ends when block reaches goal or after 200 steps.
    """

    metadata = {"render_modes": ["rgb_array"]}

    INSTRUCTIONS: dict[str, int] = {
        "push block to goal": 0,
        "move left": 1,
        "move right": 2,
    }

    # Action deltas: up, down, left, right, stay
    _DELTAS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]], dtype=np.int32)

    def __init__(self, render_mode: str = "rgb_array") -> None:
        super().__init__()
        self.render_mode = render_mode
        self.size = 64
        self.max_steps = 200
        self.goal_tolerance = 4

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(5)

        self.agent_pos: np.ndarray = np.zeros(2, dtype=np.int32)
        self.block_pos: np.ndarray = np.zeros(2, dtype=np.int32)
        self.goal_pos: np.ndarray = np.zeros(2, dtype=np.int32)
        self.instruction: str = "push block to goal"
        self._step_count: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment. Pass instruction via options={"instruction": ...}."""
        super().reset(seed=seed)

        if options and "instruction" in options:
            self.instruction = options["instruction"]
        else:
            self.instruction = "push block to goal"

        rng = self.np_random

        # Place goal in inner region
        self.goal_pos = rng.integers(12, 52, size=2).astype(np.int32)

        # Place block away from goal
        while True:
            self.block_pos = rng.integers(8, 56, size=2).astype(np.int32)
            if np.linalg.norm(self.block_pos - self.goal_pos) > 12:
                break

        # Place agent away from block
        while True:
            self.agent_pos = rng.integers(4, 60, size=2).astype(np.int32)
            if np.linalg.norm(self.agent_pos - self.block_pos) > 10:
                break

        self._step_count = 0
        return self._render_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one step. Agent pushes block if they collide."""
        self._step_count += 1
        delta = self._DELTAS[action]

        new_agent = np.clip(self.agent_pos + delta, 4, 59).astype(np.int32)

        # Push block if agent moves adjacent to it
        if np.linalg.norm(new_agent.astype(float) - self.block_pos.astype(float)) < 6:
            self.block_pos = np.clip(self.block_pos + delta, 6, 57).astype(np.int32)

        self.agent_pos = new_agent

        terminated = bool(
            np.linalg.norm(self.block_pos.astype(float) - self.goal_pos.astype(float))
            < self.goal_tolerance
        )
        truncated = self._step_count >= self.max_steps
        reward = 1.0 if terminated else 0.0

        return self._render_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> np.ndarray:
        """Return a 64x64x3 RGB image of the current state."""
        return self._render_obs()

    def _render_obs(self) -> np.ndarray:
        img = np.full((self.size, self.size, 3), 255, dtype=np.uint8)

        # Goal: green square (drawn first so others appear on top)
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        img[max(0, gy - 3) : min(64, gy + 3), max(0, gx - 3) : min(64, gx + 3)] = [
            0,
            200,
            0,
        ]

        # Block: red square
        bx, by = int(self.block_pos[0]), int(self.block_pos[1])
        img[max(0, by - 3) : min(64, by + 3), max(0, bx - 3) : min(64, bx + 3)] = [
            220,
            50,
            50,
        ]

        # Agent: blue square (smallest, drawn last)
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        img[max(0, ay - 2) : min(64, ay + 2), max(0, ax - 2) : min(64, ax + 2)] = [
            50,
            100,
            220,
        ]

        return img

    def _get_info(self) -> dict:
        return {
            "agent_pos": self.agent_pos.copy(),
            "block_pos": self.block_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "instruction": self.instruction,
            "step": self._step_count,
        }
