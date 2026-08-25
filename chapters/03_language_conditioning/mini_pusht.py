"""MiniPushT: enhanced environment for Chapter 3 (Language Conditioning).

Extends Chapter 2's MiniPushT with 11 canonical instructions, paraphrases,
multi-behavior expert, and instruction-specific success conditions.

All physics constants scale proportionally from 64x64 base values so gameplay
feels identical at any resolution.
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional


def _scale(base: int, size: int) -> int:
    """Scale a base-64 constant to target size, minimum 1."""
    return max(1, round(base * size / 64))


# ---------------------------------------------------------------------------
# Instruction taxonomy
# ---------------------------------------------------------------------------

BEHAVIOR_GOAL = "goal"
BEHAVIOR_PUSH_DIR = "push_dir"
BEHAVIOR_AGENT_MOVE = "agent_move"
BEHAVIOR_PUSH_LOC = "push_loc"

INSTRUCTIONS: dict[str, dict] = {
    "push block to goal": {"behavior": BEHAVIOR_GOAL, "direction": None},
    "push block up": {"behavior": BEHAVIOR_PUSH_DIR, "direction": "up"},
    "push block down": {"behavior": BEHAVIOR_PUSH_DIR, "direction": "down"},
    "push block left": {"behavior": BEHAVIOR_PUSH_DIR, "direction": "left"},
    "push block right": {"behavior": BEHAVIOR_PUSH_DIR, "direction": "right"},
    "move agent up": {"behavior": BEHAVIOR_AGENT_MOVE, "direction": "up"},
    "move agent down": {"behavior": BEHAVIOR_AGENT_MOVE, "direction": "down"},
    "move agent left": {"behavior": BEHAVIOR_AGENT_MOVE, "direction": "left"},
    "move agent right": {"behavior": BEHAVIOR_AGENT_MOVE, "direction": "right"},
    "push block to top-left corner": {"behavior": BEHAVIOR_PUSH_LOC, "direction": "top-left"},
    "push block to center": {"behavior": BEHAVIOR_PUSH_LOC, "direction": "center"},
}

CANONICAL_INSTRUCTIONS: list[str] = list(INSTRUCTIONS.keys())

PARAPHRASES: dict[str, list[str]] = {
    "push block to goal": [
        "move the block to the target",
        "shove the block toward the goal",
        "get the block to the green zone",
        "push it to the destination",
    ],
    "push block up": [
        "nudge the block upward",
        "send the block to the top",
        "push the block toward the top edge",
    ],
    "push block down": [
        "nudge the block downward",
        "send the block to the bottom",
        "push the block toward the bottom edge",
    ],
    "push block left": [
        "nudge the block to the left",
        "send the block leftward",
        "push the block toward the left edge",
    ],
    "push block right": [
        "nudge the block to the right",
        "send the block rightward",
        "push the block toward the right edge",
    ],
    "move agent up": [
        "go up",
        "move yourself upward",
        "walk to the top",
    ],
    "move agent down": [
        "go down",
        "move yourself downward",
        "walk to the bottom",
    ],
    "move agent left": [
        "go left",
        "move yourself to the left",
        "walk left",
    ],
    "move agent right": [
        "go right",
        "move yourself to the right",
        "walk right",
    ],
    "push block to top-left corner": [
        "move the block to the upper left",
        "push it to the top-left",
        "get the block to the corner",
    ],
    "push block to center": [
        "push the block to the middle",
        "move the block to the center",
        "get the block to the middle of the arena",
    ],
}

# Reverse lookup: paraphrase string -> canonical instruction
PARAPHRASE_TO_CANONICAL: dict[str, str] = {}
for _canonical, _paras in PARAPHRASES.items():
    for _p in _paras:
        PARAPHRASE_TO_CANONICAL[_p] = _canonical


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MiniPushT(gymnasium.Env):
    """2D block-pushing task with multi-instruction support.

    Supports 11 canonical instructions across 4 behavior categories:
    - goal: push block to a randomly placed goal
    - push_dir: push block toward an edge (up/down/left/right)
    - agent_move: move agent toward an edge (ignore block)
    - push_loc: push block to a specific location (corner, center)

    Also accepts paraphrases of each canonical instruction.

    Args:
        size: Image resolution (size x size). Default 224.
        render_mode: Only "rgb_array" is supported.

    Observation: (size, size, 3) RGB image (uint8).
    Actions: Discrete(5) -- 0=up, 1=down, 2=left, 3=right, 4=stay.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size: int = 224, render_mode: str = "rgb_array") -> None:
        super().__init__()
        self.render_mode = render_mode
        self.size = size
        self.max_steps = 200

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
        self._edge_tol = _scale(6, size)

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
        self.canonical_instruction: str = "push block to goal"
        self._behavior: str = BEHAVIOR_GOAL
        self._target_entity: str = "block"  # "block" or "agent"
        self._step_count: int = 0

    def _resolve_instruction(self, instruction: str) -> str:
        """Resolve a paraphrase to its canonical instruction."""
        if instruction in INSTRUCTIONS:
            return instruction
        if instruction in PARAPHRASE_TO_CANONICAL:
            return PARAPHRASE_TO_CANONICAL[instruction]
        raise ValueError(
            f"Unknown instruction: '{instruction}'. "
            f"Must be a canonical instruction or known paraphrase."
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment. Pass instruction via options={'instruction': ...}."""
        super().reset(seed=seed)

        if options and "instruction" in options:
            self.instruction = options["instruction"]
        else:
            self.instruction = "push block to goal"

        self.canonical_instruction = self._resolve_instruction(self.instruction)
        meta = INSTRUCTIONS[self.canonical_instruction]
        self._behavior = meta["behavior"]

        if self._behavior == BEHAVIOR_AGENT_MOVE:
            self._target_entity = "agent"
        else:
            self._target_entity = "block"

        rng = self.np_random

        # Place goal (used for BEHAVIOR_GOAL and as visual target)
        self.goal_pos = rng.integers(self._goal_lo, self._goal_hi, size=2).astype(
            np.int32
        )

        # For non-goal behaviors, set goal_pos to the instruction target
        # so the green marker shows where the block/agent should go
        if self._behavior == BEHAVIOR_PUSH_DIR:
            self.goal_pos = self._direction_target(meta["direction"]).astype(np.int32)
        elif self._behavior == BEHAVIOR_PUSH_LOC:
            self.goal_pos = self._location_target(meta["direction"]).astype(np.int32)
        elif self._behavior == BEHAVIOR_AGENT_MOVE:
            self.goal_pos = self._direction_target(meta["direction"]).astype(np.int32)

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

        terminated = self._check_success()
        truncated = self._step_count >= self.max_steps
        reward = 1.0 if terminated else 0.0

        return self._render_obs(), reward, terminated, truncated, self._get_info()

    def _check_success(self) -> bool:
        """Check instruction-specific success condition."""
        meta = INSTRUCTIONS[self.canonical_instruction]

        if self._behavior == BEHAVIOR_GOAL:
            return bool(
                np.linalg.norm(
                    self.block_pos.astype(float) - self.goal_pos.astype(float)
                )
                < self.goal_tolerance
            )

        direction = meta["direction"]

        if self._behavior == BEHAVIOR_PUSH_DIR:
            return self._at_edge(self.block_pos, direction)

        if self._behavior == BEHAVIOR_AGENT_MOVE:
            return self._at_edge(self.agent_pos, direction)

        if self._behavior == BEHAVIOR_PUSH_LOC:
            target = self._location_target(direction)
            return bool(
                np.linalg.norm(self.block_pos.astype(float) - target)
                < self.goal_tolerance
            )

        return False

    def _at_edge(self, pos: np.ndarray, direction: str) -> bool:
        """Check if position has reached the specified edge."""
        tol = self._edge_tol
        if direction == "up":
            return bool(pos[1] <= tol)
        elif direction == "down":
            return bool(pos[1] >= self.size - tol - 1)
        elif direction == "left":
            return bool(pos[0] <= tol)
        elif direction == "right":
            return bool(pos[0] >= self.size - tol - 1)
        return False

    def _direction_target(self, direction: str) -> np.ndarray:
        """Get target position for directional instructions."""
        s = self.size
        targets = {
            "up": np.array([s // 2, self._edge_tol // 2], dtype=float),
            "down": np.array([s // 2, s - self._edge_tol // 2 - 1], dtype=float),
            "left": np.array([self._edge_tol // 2, s // 2], dtype=float),
            "right": np.array([s - self._edge_tol // 2 - 1, s // 2], dtype=float),
        }
        return targets[direction]

    def _location_target(self, location: str) -> np.ndarray:
        """Get target position for location-based instructions."""
        s = self.size
        targets = {
            "top-left": np.array([s // 6, s // 6], dtype=float),
            "center": np.array([s // 2, s // 2], dtype=float),
        }
        return targets[location]

    def render(self) -> np.ndarray:
        """Return a (size, size, 3) RGB image of the current state."""
        return self._render_obs()

    def _render_obs(self) -> np.ndarray:
        img = np.full((self.size, self.size, 3), 255, dtype=np.uint8)
        s = self.size
        oh = self._obj_half
        ah = self._agent_half

        # Goal/target marker (green)
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        img[max(0, gy - oh) : min(s, gy + oh), max(0, gx - oh) : min(s, gx + oh)] = [
            0,
            200,
            0,
        ]

        # Block (red)
        bx, by = int(self.block_pos[0]), int(self.block_pos[1])
        img[max(0, by - oh) : min(s, by + oh), max(0, bx - oh) : min(s, bx + oh)] = [
            220,
            50,
            50,
        ]

        # Agent (blue)
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
            "canonical_instruction": self.canonical_instruction,
            "step": self._step_count,
        }


# ---------------------------------------------------------------------------
# Scripted Expert
# ---------------------------------------------------------------------------


class ScriptedExpert:
    """Multi-behavior oracle that interprets instruction strings.

    Dispatches to the appropriate behavior based on the instruction's category:
    - goal: go to block, push toward goal
    - push_dir: go to block, push toward edge
    - agent_move: move agent toward edge (ignore block)
    - push_loc: go to block, push toward target coordinate

    Args:
        size: Environment resolution. Scales the phase-switch threshold.
    """

    def __init__(self, size: int = 224) -> None:
        self._threshold = max(1, round(4 * size / 64))
        self._size = size

    def act(self, info: dict) -> int:
        """Return optimal discrete action given environment state and instruction."""
        instruction = info.get("canonical_instruction", info["instruction"])
        canonical = PARAPHRASE_TO_CANONICAL.get(instruction, instruction)
        meta = INSTRUCTIONS[canonical]

        agent = info["agent_pos"].astype(float)
        block = info["block_pos"].astype(float)

        if meta["behavior"] == BEHAVIOR_AGENT_MOVE:
            target = self._edge_target(meta["direction"])
            return self._move_toward(agent, target)

        # All push behaviors: first go to block, then push toward target
        if np.linalg.norm(agent - block) > self._threshold:
            return self._move_toward(agent, block)

        if meta["behavior"] == BEHAVIOR_GOAL:
            return self._move_toward(agent, info["goal_pos"].astype(float))
        elif meta["behavior"] == BEHAVIOR_PUSH_DIR:
            target = self._edge_target(meta["direction"])
            return self._move_toward(agent, target)
        else:  # BEHAVIOR_PUSH_LOC
            target = self._location_target(meta["direction"])
            return self._move_toward(agent, target)

    def _edge_target(self, direction: str) -> np.ndarray:
        s = self._size
        targets = {
            "up": np.array([s // 2, 0], dtype=float),
            "down": np.array([s // 2, s - 1], dtype=float),
            "left": np.array([0, s // 2], dtype=float),
            "right": np.array([s - 1, s // 2], dtype=float),
        }
        return targets[direction]

    def _location_target(self, location: str) -> np.ndarray:
        s = self._size
        targets = {
            "top-left": np.array([s // 6, s // 6], dtype=float),
            "center": np.array([s // 2, s // 2], dtype=float),
        }
        return targets[location]

    def _move_toward(self, src: np.ndarray, dst: np.ndarray) -> int:
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 2  # right or left
        else:
            return 1 if dy > 0 else 0  # down or up


if __name__ == "__main__":
    print("MiniPushT Enhanced: testing all 11 instructions\n")
    for sz in [64, 224]:
        env = MiniPushT(size=sz)
        expert = ScriptedExpert(size=sz)

        print(f"--- size={sz} ---")
        for instr in CANONICAL_INSTRUCTIONS:
            successes = 0
            n_trials = 20
            for trial in range(n_trials):
                obs, info = env.reset(
                    seed=trial, options={"instruction": instr}
                )
                done = False
                while not done:
                    action = expert.act(info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                if terminated:
                    successes += 1
            print(f"  {instr:35s}  {successes}/{n_trials} ({100*successes/n_trials:.0f}%)")
        print()

    # Test paraphrase acceptance
    env = MiniPushT(size=64)
    print("Paraphrase acceptance test:")
    for canonical, paras in PARAPHRASES.items():
        for para in paras:
            obs, info = env.reset(seed=0, options={"instruction": para})
            assert info["canonical_instruction"] == canonical
    print("  All paraphrases accepted and resolved correctly.")
    print("\nMiniPushT enhanced OK")
