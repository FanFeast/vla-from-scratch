"""ContinuousMiniPushT: continuous-action wrapper for Chapter 4.

Wraps ch03's MiniPushT, replacing discrete actions with continuous (dx, dy)
in [-1, 1]. The environment is restricted to a single instruction
("push block to goal") to keep focus on action representations.
"""
from __future__ import annotations

import sys
import os
import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional

# Import ch03 MiniPushT
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03_language_conditioning"))
from mini_pusht import MiniPushT as DiscreteMiniPushT


class ContinuousMiniPushT(gymnasium.Env):
    """MiniPushT with continuous (dx, dy) actions.

    Actions are (dx, dy) in [-1, 1], scaled by the environment's speed constant.
    Only supports the "push block to goal" instruction.

    Args:
        size: Image resolution (size x size). Default 224.
        render_mode: Only "rgb_array" is supported.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size: int = 224, render_mode: str = "rgb_array") -> None:
        super().__init__()
        self._env = DiscreteMiniPushT(size=size, render_mode=render_mode)
        self.size = size
        self.max_steps = self._env.max_steps

        self.observation_space = self._env.observation_space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._speed = self._env._speed

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment. Always uses 'push block to goal' instruction."""
        opts = {"instruction": "push block to goal"}
        return self._env.reset(seed=seed, options=opts)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one step with continuous (dx, dy) action.

        Args:
            action: (2,) array in [-1, 1], representing normalized velocity.
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Scale to pixel movement
        dx = action[0] * self._speed
        dy = action[1] * self._speed
        delta = np.round(np.array([dx, dy])).astype(np.int32)

        # Apply movement (same physics as discrete env)
        self._env._step_count += 1
        new_agent = np.clip(
            self._env.agent_pos + delta,
            self._env._agent_lo,
            self._env.size - self._env._agent_lo - 1,
        ).astype(np.int32)

        if (
            np.linalg.norm(new_agent.astype(float) - self._env.block_pos.astype(float))
            < self._env._push_dist
        ):
            self._env.block_pos = np.clip(
                self._env.block_pos + delta,
                self._env._obj_half,
                self._env.size - self._env._obj_half - 1,
            ).astype(np.int32)

        self._env.agent_pos = new_agent

        terminated = self._env._check_success()
        truncated = self._env._step_count >= self._env.max_steps
        reward = 1.0 if terminated else 0.0

        return self._env._render_obs(), reward, terminated, truncated, self._env._get_info()

    def render(self) -> np.ndarray:
        return self._env.render()

    @property
    def agent_pos(self) -> np.ndarray:
        return self._env.agent_pos

    @property
    def block_pos(self) -> np.ndarray:
        return self._env.block_pos

    @property
    def goal_pos(self) -> np.ndarray:
        return self._env.goal_pos


class ContinuousExpert:
    """Scripted expert for ContinuousMiniPushT.

    Computes direction vector to target, normalizes to [-1, 1].
    Adds small Gaussian noise for trajectory diversity.
    """

    def __init__(self, size: int = 224, noise_std: float = 0.1) -> None:
        self._threshold = max(1, round(4 * size / 64))
        self._noise_std = noise_std

    def act(self, info: dict) -> np.ndarray:
        """Return (dx, dy) continuous action in [-1, 1]."""
        agent = info["agent_pos"].astype(float)
        block = info["block_pos"].astype(float)
        goal = info["goal_pos"].astype(float)

        # Phase 1: approach block
        if np.linalg.norm(agent - block) > self._threshold:
            target = block
        else:
            # Phase 2: push block toward goal
            target = goal

        direction = target - agent
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            action = np.zeros(2, dtype=np.float32)
        else:
            action = (direction / norm).astype(np.float32)

        # Add noise for diversity
        noise = np.random.randn(2).astype(np.float32) * self._noise_std
        action = np.clip(action + noise, -1.0, 1.0)
        return action


if __name__ == "__main__":
    print("ContinuousMiniPushT: testing expert demos\n")
    env = ContinuousMiniPushT(size=64)
    expert = ContinuousExpert(size=64, noise_std=0.05)

    successes = 0
    n_trials = 50
    for _ in range(n_trials):
        obs, info = env.reset()
        done = False
        while not done:
            action = expert.act(info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if terminated:
            successes += 1
    print(f"  Expert success: {successes}/{n_trials} = {100*successes/n_trials:.1f}%")
