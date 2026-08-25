"""MiniPushT: configurable-resolution environment for Chapter 2+.

Accepts any resolution via size parameter. All physics constants scale
proportionally from 64x64 base values so gameplay feels identical at any size.
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional


def _scale(base: int, size: int) -> int:
    """Scale a base-64 constant to target size, minimum 1."""
    return max(1, round(base * size / 64))


class MiniPushT(gymnasium.Env):
    """2D block-pushing task at configurable resolution.

    Args:
        size: Image resolution (size x size). Default 224. Common values: 64, 112, 224.
        render_mode: Only "rgb_array" is supported.

    Observation: (size, size, 3) RGB image (uint8).
    Actions: Discrete(5) -- 0=up, 1=down, 2=left, 3=right, 4=stay.
    Reward: +1.0 when block overlaps goal (within tolerance), 0 otherwise.
    Episode ends when block reaches goal or after 200 steps.
    """

    metadata = {"render_modes": ["rgb_array"]}

    INSTRUCTIONS: dict[str, int] = {
        "push block to goal": 0,
        "move left": 1,
        "move right": 2,
    }

    def __init__(self, size: int = 224, render_mode: str = "rgb_array") -> None:
        super().__init__()
        self.render_mode = render_mode
        self.size = size
        self.max_steps = 200

        # All constants scale from 64x64 base values
        self._speed = _scale(1, size)
        self.goal_tolerance = _scale(4, size)
        self._push_dist = _scale(6, size)
        self._goal_lo = _scale(12, size)
        self._goal_hi = size - self._goal_lo
        self._block_lo = _scale(8, size)
        self._block_hi = size - self._block_lo
        self._agent_lo = _scale(4, size)
        self._agent_hi = size - self._agent_lo
        self._min_block_goal = _scale(12, size)
        self._min_agent_block = _scale(10, size)
        self._obj_half = _scale(3, size)
        self._agent_half = _scale(2, size)

        self._DELTAS = np.array(
            [
                [0, -self._speed],
                [0, self._speed],
                [-self._speed, 0],
                [self._speed, 0],
                [0, 0],
            ],
            dtype=np.int32,
        )

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

        self.goal_pos = rng.integers(self._goal_lo, self._goal_hi, size=2).astype(
            np.int32
        )

        while True:
            self.block_pos = rng.integers(
                self._block_lo, self._block_hi, size=2
            ).astype(np.int32)
            if np.linalg.norm(self.block_pos - self.goal_pos) > self._min_block_goal:
                break

        while True:
            self.agent_pos = rng.integers(
                self._agent_lo, self._agent_hi, size=2
            ).astype(np.int32)
            if np.linalg.norm(self.agent_pos - self.block_pos) > self._min_agent_block:
                break

        self._step_count = 0
        return self._render_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one step. Agent pushes block if they collide."""
        self._step_count += 1
        delta = self._DELTAS[action]

        new_agent = np.clip(
            self.agent_pos + delta, self._agent_lo, self.size - self._agent_lo - 1
        ).astype(np.int32)

        if (
            np.linalg.norm(new_agent.astype(float) - self.block_pos.astype(float))
            < self._push_dist
        ):
            self.block_pos = np.clip(
                self.block_pos + delta, self._obj_half, self.size - self._obj_half - 1
            ).astype(np.int32)

        self.agent_pos = new_agent

        terminated = bool(
            np.linalg.norm(self.block_pos.astype(float) - self.goal_pos.astype(float))
            < self.goal_tolerance
        )
        truncated = self._step_count >= self.max_steps
        reward = 1.0 if terminated else 0.0

        return self._render_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> np.ndarray:
        """Return a (size, size, 3) RGB image of the current state."""
        return self._render_obs()

    def _render_obs(self) -> np.ndarray:
        img = np.full((self.size, self.size, 3), 255, dtype=np.uint8)
        s = self.size
        oh = self._obj_half
        ah = self._agent_half

        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        img[max(0, gy - oh) : min(s, gy + oh), max(0, gx - oh) : min(s, gx + oh)] = [
            0,
            200,
            0,
        ]

        bx, by = int(self.block_pos[0]), int(self.block_pos[1])
        img[max(0, by - oh) : min(s, by + oh), max(0, bx - oh) : min(s, bx + oh)] = [
            220,
            50,
            50,
        ]

        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        img[max(0, ay - ah) : min(s, ay + ah), max(0, ax - ah) : min(s, ax + ah)] = [
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


if __name__ == "__main__":
    for sz in [64, 112, 224]:
        env = MiniPushT(size=sz)
        obs, info = env.reset(seed=42)
        print(
            f"size={sz}: obs shape={obs.shape}, speed={env._speed}, tol={env.goal_tolerance}"
        )

        total_reward = 0.0
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f"  total_reward after 10 steps: {total_reward}")
    print("MiniPushT multi-resolution OK")
