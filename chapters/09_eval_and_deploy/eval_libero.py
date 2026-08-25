"""Chapter 9: Policy evaluation (LIBERO-style closed-loop rollouts).

Training loss tells you the action expert fit the data. It does *not* tell you
whether the policy completes the task on a robot -- small per-step errors
compound over a 300-step episode. The only honest metric for a VLA is
**closed-loop success rate**: roll the policy out in the environment and count
how often it finishes the task.

This module provides:

- Metric helpers: success rate, episode length, action smoothness.
- `MockManipulationEnv`: a tiny, dependency-free goal-reaching environment so
  the eval harness (and its tests) run anywhere, no simulator required.
- `evaluate_policy`: the closed-loop loop, driven through the async action-
  chunking controller from `async_inference.py`.
- A gated `make_libero_env` adapter for the real
  [LIBERO](https://libero-project.github.io/) benchmark, with graceful
  guidance if the `libero` package is not installed.

LIBERO is the standard VLA manipulation benchmark (OpenVLA, SmolVLA, pi0 all
report on it). It has four task suites of 10 tasks each plus two larger ones:

    libero_spatial  -- same objects, different spatial arrangements
    libero_object   -- same layout, different objects
    libero_goal     -- same objects/layout, different goals
    libero_10       -- 10 long-horizon tasks (a.k.a. LIBERO-LONG)
    libero_90       -- 90 short-horizon tasks (the big training/eval pool)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from async_inference import (
    AsyncChunkController,
    ChunkPolicy,
    SyncChunkController,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class EpisodeMetrics:
    """Per-episode evaluation outcome."""

    success: bool
    length: int
    smoothness: float
    n_inferences: int


def compute_smoothness(actions: list[np.ndarray]) -> float:
    """Mean L2 norm of consecutive action differences (lower = smoother).

    A proxy for jerk: jerky open-loop chunks show up as large jumps at chunk
    boundaries. Returns 0.0 for trajectories shorter than two steps.
    """
    if len(actions) < 2:
        return 0.0
    arr = np.stack([np.asarray(a, dtype=np.float64) for a in actions])
    diffs = np.linalg.norm(arr[1:] - arr[:-1], axis=-1)
    return float(diffs.mean())


def aggregate_metrics(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    """Aggregate per-episode metrics into benchmark-level numbers."""
    if not episodes:
        return {
            "success_rate": 0.0,
            "mean_length": 0.0,
            "mean_smoothness": 0.0,
            "n_episodes": 0.0,
        }
    n = len(episodes)
    return {
        "success_rate": sum(e.success for e in episodes) / n,
        "mean_length": sum(e.length for e in episodes) / n,
        "mean_smoothness": sum(e.smoothness for e in episodes) / n,
        "n_episodes": float(n),
    }


# ---------------------------------------------------------------------------
# Environment protocol + mock
# ---------------------------------------------------------------------------


class EvalEnv(Protocol):
    """Minimal gym-like environment interface used by the eval loop."""

    def reset(self) -> np.ndarray:
        """Reset and return the initial observation."""
        ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply an action; return (obs, reward, done, info with 'success')."""
        ...


class MockManipulationEnv:
    """A 1-D goal-reaching task with a VLA-shaped observation/action interface.

    The agent starts at position 0 and must reach `goal`. The first action
    dimension is the commanded velocity (clipped to [-1, 1]); the rest are
    ignored, mimicking a 6-DOF action where only some dims matter. `success`
    is True when the agent gets within `tolerance` of the goal.

    This makes the eval harness fully runnable and deterministic for tests:
    a policy that drives toward the goal succeeds; one that drives away fails.
    """

    def __init__(
        self,
        obs_dim: int = 6,
        action_dim: int = 6,
        goal: float = 5.0,
        dt: float = 0.25,
        tolerance: float = 0.25,
        max_steps: int = 60,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal = goal
        self.dt = dt
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.pos = 0.0
        self.t = 0

    def reset(self) -> np.ndarray:
        """Reset position and step counter, return the first observation."""
        self.pos = 0.0
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = self.pos
        if self.obs_dim > 1:
            obs[1] = self.goal - self.pos  # error signal
        return obs

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one control step using action[0] as velocity."""
        vel = float(np.clip(np.asarray(action)[0], -1.0, 1.0))
        self.pos += vel * self.dt
        self.t += 1
        success = abs(self.pos - self.goal) <= self.tolerance
        done = success or self.t >= self.max_steps
        reward = 1.0 if success else 0.0
        return self._obs(), reward, done, {"success": success}


# ---------------------------------------------------------------------------
# A scripted policy for the runnable demo (drives toward the goal)
# ---------------------------------------------------------------------------


class ScriptedReachPolicy:
    """Chunk policy that commands velocity toward the goal (for the demo).

    Reads the error signal from observation[1] and outputs a constant push in
    that direction for the whole chunk -- an open-loop chunk that the async
    controller will refresh as the error shrinks.
    """

    def __init__(self, action_dim: int = 6, chunk_size: int = 10) -> None:
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def predict_chunk(self, observation: np.ndarray) -> np.ndarray:
        error = float(np.asarray(observation)[1])
        vel = float(np.clip(error, -1.0, 1.0))
        chunk = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        chunk[:, 0] = vel
        return chunk


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_episode(
    policy: ChunkPolicy,
    env: EvalEnv,
    use_async: bool = True,
    replan_threshold: int = 3,
    max_steps: int = 300,
) -> EpisodeMetrics:
    """Run a single closed-loop episode and return its metrics.

    Drives the policy through a chunk controller, stepping the environment
    until it reports done or max_steps is reached.
    """
    controller: SyncChunkController | AsyncChunkController = (
        AsyncChunkController(policy, replan_threshold=replan_threshold)
        if use_async
        else SyncChunkController(policy)
    )

    obs = env.reset()
    actions: list[np.ndarray] = []
    success = False

    for _ in range(max_steps):
        action = controller.get_action(obs)
        actions.append(action)
        obs, _reward, done, info = env.step(action)
        if done:
            success = bool(info.get("success", False))
            break

    controller.close()
    return EpisodeMetrics(
        success=success,
        length=len(actions),
        smoothness=compute_smoothness(actions),
        n_inferences=controller.n_inferences,
    )


def evaluate_policy(
    policy: ChunkPolicy,
    env_factory,  # Callable[[], EvalEnv]
    n_episodes: int = 10,
    use_async: bool = True,
    max_steps: int = 300,
) -> dict[str, float]:
    """Evaluate a policy over several episodes, returning aggregate metrics.

    Args:
        policy: A chunk policy (predict_chunk -> (K, action_dim)).
        env_factory: Zero-arg callable returning a fresh EvalEnv per episode.
        n_episodes: Number of episodes to roll out.
        use_async: Use the async controller (True) or the sync baseline.
        max_steps: Hard cap on control steps per episode.

    Returns:
        Aggregate metrics dict from aggregate_metrics.
    """
    episodes: list[EpisodeMetrics] = []
    for _ in range(n_episodes):
        env = env_factory()
        episodes.append(
            evaluate_episode(
                policy, env, use_async=use_async, max_steps=max_steps
            )
        )
    return aggregate_metrics(episodes)


# ---------------------------------------------------------------------------
# Real LIBERO adapter (gated)
# ---------------------------------------------------------------------------


LIBERO_TASK_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


def describe_libero_suites() -> str:
    """Return a readable description of the LIBERO task suites."""
    return (
        "LIBERO task suites:\n"
        "  libero_spatial  -- 10 tasks, varied spatial arrangements\n"
        "  libero_object   -- 10 tasks, varied objects\n"
        "  libero_goal     -- 10 tasks, varied goals\n"
        "  libero_10       -- 10 long-horizon tasks (LIBERO-LONG)\n"
        "  libero_90       -- 90 short-horizon tasks"
    )


def make_libero_env(task_suite: str, task_id: int = 0):  # type: ignore[no-untyped-def]
    """Construct a real LIBERO environment, if the `libero` package is present.

    Raises RuntimeError with installation guidance if LIBERO is unavailable.
    The returned environment must be wrapped to match the EvalEnv interface
    (LIBERO uses the robosuite API); see the README for the wrapper sketch.
    """
    import importlib.util

    if task_suite not in LIBERO_TASK_SUITES:
        raise ValueError(
            f"Unknown task suite {task_suite!r}. "
            f"Choose from {LIBERO_TASK_SUITES}."
        )

    if importlib.util.find_spec("libero") is None:
        raise RuntimeError(
            "LIBERO is not installed. Install from source:\n"
            "  pip install robosuite mujoco\n"
            "  pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
            "See https://github.com/Lifelong-Robot-Learning/LIBERO . "
            "Until then, use MockManipulationEnv to exercise the harness."
        )

    from libero.libero import benchmark  # type: ignore

    suite = benchmark.get_benchmark_dict()[task_suite]()
    task = suite.get_task(task_id)
    return suite.get_task_env(task)


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Evaluate the scripted reach policy on the mock env (sync vs async)."""
    print("=" * 60)
    print("  Chapter 9: Policy Evaluation Demo (MockManipulationEnv)")
    print("=" * 60)
    print(describe_libero_suites())
    print()

    policy = ScriptedReachPolicy(action_dim=6, chunk_size=10)

    for label, use_async in [("sync ", False), ("async", True)]:
        metrics = evaluate_policy(
            policy,
            env_factory=lambda: MockManipulationEnv(),
            n_episodes=10,
            use_async=use_async,
            max_steps=120,
        )
        print(
            f"  {label} | success={metrics['success_rate'] * 100:5.1f}% | "
            f"len={metrics['mean_length']:5.1f} | "
            f"smooth={metrics['mean_smoothness']:.4f}"
        )

    print(
        "\n  The scripted policy reaches the goal, so success rate is high and\n"
        "  sync/async produce the same behavior -- async only changes timing.\n"
        "  Swap in make_libero_env(...) for the real benchmark."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 9: Policy evaluation")
    parser.add_argument(
        "--task-suite",
        default="mock",
        help="'mock' for the built-in env, or a LIBERO suite name",
    )
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--sync", action="store_true", help="Use sync controller")
    args = parser.parse_args()

    if args.task_suite == "mock":
        _demo()
        return

    print(f"[eval] Requested LIBERO suite: {args.task_suite}")
    try:
        make_libero_env(args.task_suite)
    except RuntimeError as exc:
        print(f"[eval] {exc}")
        return


if __name__ == "__main__":
    main()
