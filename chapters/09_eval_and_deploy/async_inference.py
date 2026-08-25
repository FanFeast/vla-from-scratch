"""Chapter 9: Asynchronous action-chunking inference.

A VLA forward pass is slow: a flow-matching policy runs the VLM plus ~10 Euler
integration steps, which is tens to hundreds of milliseconds. A robot control
loop wants a fresh action every ~20-50 ms. If you call the policy synchronously
every step, the robot stalls while the GPU thinks.

Two ideas fix this, and modern VLAs (ACT, SmolVLA, pi0) use both:

1. **Action chunking.** The policy predicts a chunk of K future actions at
   once. You execute all K open-loop before replanning, so you call the
   (expensive) policy only once per K control steps.

2. **Asynchronous inference.** Instead of stopping to replan when the chunk
   runs low, you kick off the next prediction in a background thread while you
   keep executing the actions you already have. The robot never waits (except
   on the very first prediction).

A third, optional refinement smooths the seams between chunks:

3. **Temporal ensembling** (from ACT). Overlapping chunks each predict the
   same future timestep; average those predictions with an exponential recency
   weight to remove the discontinuity at chunk boundaries.

This module implements all three with plain NumPy so the control logic is
independent of the policy framework. `MockPolicy` lets you study latency
hiding without a GPU.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Policy protocol + mock
# ---------------------------------------------------------------------------


class ChunkPolicy(Protocol):
    """Anything that maps an observation to a (chunk_size, action_dim) array."""

    chunk_size: int
    action_dim: int

    def predict_chunk(self, observation: np.ndarray) -> np.ndarray:
        """Return a chunk of future actions, shape (chunk_size, action_dim)."""
        ...


class MockPolicy:
    """A deterministic, configurable-latency stand-in for a real VLA.

    The chunk it returns depends only on the observation, so two controllers
    fed the same observation stream produce identical actions -- which lets us
    test that async inference does not change behavior, only timing.
    """

    def __init__(
        self,
        action_dim: int = 6,
        chunk_size: int = 10,
        latency_s: float = 0.0,
    ) -> None:
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latency_s = latency_s
        self.n_calls = 0

    def predict_chunk(self, observation: np.ndarray) -> np.ndarray:
        """Sleep for the configured latency, then return a deterministic chunk."""
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        self.n_calls += 1
        seed = float(np.asarray(observation, dtype=np.float64).sum())
        steps = np.arange(self.chunk_size)[:, None]
        dims = np.arange(self.action_dim)[None, :]
        # Smooth, bounded, fully determined by the observation.
        return np.tanh(0.1 * (seed + steps + dims)).astype(np.float32)


# ---------------------------------------------------------------------------
# Action chunk buffer
# ---------------------------------------------------------------------------


class ActionChunkBuffer:
    """A FIFO queue of single-step actions popped from predicted chunks."""

    def __init__(self) -> None:
        self._actions: deque[np.ndarray] = deque()

    def __len__(self) -> int:
        return len(self._actions)

    def is_empty(self) -> bool:
        """Return True if no buffered actions remain."""
        return len(self._actions) == 0

    def push_chunk(self, chunk: np.ndarray) -> None:
        """Append every action in a (K, action_dim) chunk to the queue."""
        for action in chunk:
            self._actions.append(np.asarray(action))

    def replace(self, chunk: np.ndarray) -> None:
        """Discard buffered actions and load a fresh chunk."""
        self._actions.clear()
        self.push_chunk(chunk)

    def pop(self) -> np.ndarray:
        """Remove and return the next action (raises if empty)."""
        if not self._actions:
            raise IndexError("pop from empty ActionChunkBuffer")
        return self._actions.popleft()


# ---------------------------------------------------------------------------
# Temporal ensembling (ACT-style)
# ---------------------------------------------------------------------------


class TemporalEnsembler:
    """Average overlapping chunk predictions for each timestep.

    Each call to add_chunk registers a chunk that predicts actions for
    timesteps [start_t, start_t + K). When several chunks predict the same
    timestep, step() returns their exponentially recency-weighted average:

        weight(age) = exp(-m * age)

    where age 0 is the most recent prediction. Larger m trusts recent
    predictions more; m -> 0 approaches a uniform average.
    """

    def __init__(self, m: float = 0.1) -> None:
        self.m = m
        self.t = 0
        # timestep -> list of predictions, oldest first.
        self._table: dict[int, list[np.ndarray]] = defaultdict(list)

    def add_chunk(self, start_t: int, chunk: np.ndarray) -> None:
        """Register a chunk's predictions, one per covered future timestep."""
        for k, action in enumerate(chunk):
            self._table[start_t + k].append(np.asarray(action, dtype=np.float64))

    def step(self) -> np.ndarray:
        """Return the ensembled action for the current timestep and advance."""
        preds = self._table.pop(self.t, None)
        self.t += 1
        if not preds:
            raise IndexError(f"No predictions registered for timestep {self.t - 1}")
        # age 0 = most recent (last appended); weight decays for older preds.
        n = len(preds)
        ages = np.arange(n - 1, -1, -1)  # oldest has the largest age
        weights = np.exp(-self.m * ages)
        weights /= weights.sum()
        stacked = np.stack(preds, axis=0)  # (n, action_dim)
        return (weights[:, None] * stacked).sum(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class SyncChunkController:
    """Synchronous baseline: replan (blocking) whenever the buffer empties."""

    def __init__(self, policy: ChunkPolicy) -> None:
        self.policy = policy
        self.buffer = ActionChunkBuffer()
        self.n_inferences = 0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Return the next action, blocking to replan if the buffer is empty."""
        if self.buffer.is_empty():
            self.buffer.push_chunk(self.policy.predict_chunk(observation))
            self.n_inferences += 1
        return self.buffer.pop()

    def close(self) -> None:
        """No-op; present for API parity with the async controller."""


class AsyncChunkController:
    """Replan in a background thread before the buffer runs dry.

    When the number of buffered actions drops to `replan_threshold` and no
    inference is already in flight, this submits the next prediction to a
    single-worker thread pool. The control loop keeps popping buffered actions
    meanwhile, so it only ever blocks on the very first (cold-start) chunk.
    """

    def __init__(self, policy: ChunkPolicy, replan_threshold: int = 3) -> None:
        self.policy = policy
        self.replan_threshold = replan_threshold
        self.buffer = ActionChunkBuffer()
        self.n_inferences = 0
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None

    def _collect_ready(self) -> None:
        """Fold a finished background prediction into the buffer."""
        if self._future is not None and self._future.done():
            self.buffer.push_chunk(self._future.result())
            self._future = None

    def _maybe_replan(self, observation: np.ndarray) -> None:
        """Kick off a background prediction if running low and none in flight."""
        if self._future is None and len(self.buffer) <= self.replan_threshold:
            self._future = self._executor.submit(
                self.policy.predict_chunk, observation
            )
            self.n_inferences += 1

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Return the next action, hiding inference latency when possible."""
        self._collect_ready()
        self._maybe_replan(observation)

        if self.buffer.is_empty():
            # Cold start (or we fell behind): we must wait for the prediction.
            assert self._future is not None
            self.buffer.push_chunk(self._future.result())
            self._future = None

        return self.buffer.pop()

    def close(self) -> None:
        """Shut down the background worker."""
        self._executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Episode runner (used by the demo and by eval_libero.py)
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """Outcome of running a controller against an environment."""

    actions: list[np.ndarray] = field(default_factory=list)
    n_inferences: int = 0
    wall_time_s: float = 0.0

    @property
    def n_steps(self) -> int:
        """Number of control steps executed."""
        return len(self.actions)


def run_rollout(
    controller: SyncChunkController | AsyncChunkController,
    observe: Callable[[int], np.ndarray],
    step_env: Callable[[np.ndarray], None],
    n_steps: int,
) -> RolloutResult:
    """Drive a controller for n_steps, returning actions and timing.

    Args:
        controller: A sync or async chunk controller.
        observe: Callable mapping a step index to an observation array.
        step_env: Callable that applies an action (may sleep to simulate the
            robot's control period).
        n_steps: Number of control steps to execute.

    Returns:
        RolloutResult with the action sequence, inference count, and wall time.
    """
    result = RolloutResult()
    t0 = time.perf_counter()
    for i in range(n_steps):
        obs = observe(i)
        action = controller.get_action(obs)
        step_env(action)
        result.actions.append(action)
    result.wall_time_s = time.perf_counter() - t0
    result.n_inferences = controller.n_inferences
    return result


# ---------------------------------------------------------------------------
# Demo: show async hiding latency vs the synchronous baseline
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Compare sync vs async wall-clock time on a mock policy with latency."""
    n_steps = 40
    control_period_s = 0.005   # 5 ms "robot" step
    inference_latency_s = 0.03  # 30 ms policy forward pass

    def observe(i: int) -> np.ndarray:
        return np.full(6, float(i), dtype=np.float32)

    def step_env(_action: np.ndarray) -> None:
        time.sleep(control_period_s)

    print("=" * 60)
    print("  Chapter 9: Async Inference Demo")
    print("=" * 60)
    print(
        f"  steps={n_steps}, control_period={control_period_s * 1e3:.0f}ms, "
        f"inference_latency={inference_latency_s * 1e3:.0f}ms\n"
    )

    for name, ctor in [
        ("sync ", lambda p: SyncChunkController(p)),
        ("async", lambda p: AsyncChunkController(p, replan_threshold=3)),
    ]:
        policy = MockPolicy(chunk_size=10, latency_s=inference_latency_s)
        controller = ctor(policy)
        res = run_rollout(controller, observe, step_env, n_steps)
        controller.close()
        ideal = n_steps * control_period_s
        print(
            f"  {name} | wall={res.wall_time_s * 1e3:6.0f}ms | "
            f"inferences={res.n_inferences} | "
            f"overhead={100 * (res.wall_time_s - ideal) / ideal:5.1f}%"
        )

    print(
        "\n  Async hides inference behind execution: its wall time stays close\n"
        "  to the ideal (steps x control_period), while sync pays the full\n"
        "  inference latency every time the chunk runs out."
    )


if __name__ == "__main__":
    _demo()
