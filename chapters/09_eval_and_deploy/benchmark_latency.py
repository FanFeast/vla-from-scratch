"""Chapter 9: real VLA inference-latency benchmark (sync vs async).

`async_inference.py` demonstrates latency hiding with a `MockPolicy` that just
sleeps. This script swaps in the **actual Chapter 7 SmolVLA** -- SigLIP tokens
(cached) -> frozen SmolLM2 prefix -> flow-matching action expert (10 Euler
steps) -- so the numbers are a real policy forward pass, not a stand-in.

It measures two things:

1. **Raw per-chunk latency** of the real policy (mean +/- std over N calls).
2. **Sync vs async rollout** over a fixed number of control steps at a realistic
   robot control period, reporting wall time and overhead vs the ideal
   (steps x control_period). Async should stay close to ideal; sync pays the
   full policy latency every time a chunk runs out.

Requires the Ch07 model stack and cache:
    uv sync --extra vla
    uv run python benchmark_latency.py --epoch 100 --steps 60 --control-ms 33
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the Chapter 7 model/trainer and the Chapter 9 controllers.
CH07_DIR = Path(__file__).resolve().parent.parent / "07_full_vla"
sys.path.insert(0, str(CH07_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from async_inference import (  # noqa: E402
    AsyncChunkController,
    SyncChunkController,
    run_rollout,
)

CH07_CHECKPOINTS = CH07_DIR / "checkpoints"


class SmolVLAChunkPolicy:
    """ChunkPolicy adapter around the Chapter 7 SmolVLA flow-matching policy.

    Holds a small pool of real cached SO100 observations (vision tokens +
    language + state) on the target device. `predict_chunk` runs the true
    VLM-prefix + flow-matching sampling and returns a denormalized
    (chunk_size, action_dim) action chunk. Inference latency does not depend on
    the observation values, so cycling through a few real frames is enough for
    an honest measurement.
    """

    def __init__(self, epoch: int, n_obs: int = 8) -> None:
        import torch

        from config import SmolVLAConfig
        from model import build_smolvla
        from train import (
            DEVICE,
            DTYPE,
            EMAModel,
            FlowMatchingVLA,
            compute_norm_stats,
        )

        self.torch = torch
        self.DEVICE = DEVICE
        self.DTYPE = DTYPE
        config = SmolVLAConfig()
        self.config = config
        self.chunk_size = config.chunk_size
        self.action_dim = config.action_dim

        model = build_smolvla(config)

        ckpt_path = CH07_CHECKPOINTS / f"vla_epoch_{epoch}.pt"
        if not ckpt_path.exists():
            ckpt_path = CH07_CHECKPOINTS / "vla_so100_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.state_proj.load_state_dict(ckpt["trainable"]["state_proj"])
        model.action_expert.load_state_dict(ckpt["trainable"]["action_expert"])

        # Apply EMA weights once (deployment uses the smoothed weights); then we
        # can sample with ema=None and avoid a per-call deepcopy in the hot path.
        ema = EMAModel(model, decay=config.ema_decay)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_to(model)

        # Vision encoder stays on CPU -- tokens are pre-cached (extraction is a
        # one-time cost in Ch07); the online policy cost is VLM + expert.
        model.vlm.to(DEVICE)
        model.connector.to(DEVICE)
        model.state_proj.to(DEVICE)
        model.action_expert.to(DEVICE)
        model.eval()
        self.model = model
        self.trainer = FlowMatchingVLA(model, config)
        self.action_key = ckpt_path.name

        # Load a pool of real observations from the shared Ch07 cache.
        cache = torch.load(
            CH07_CHECKPOINTS / "vla_cache.pt", map_location="cpu", weights_only=True
        )
        action_stats = compute_norm_stats(cache["actions"])
        state_stats = compute_norm_stats(cache["states"])
        self._action_mean = action_stats.mean
        self._action_std = action_stats.std

        n = cache["vision_tokens"].shape[0]
        idx = np.linspace(0, n - 1, n_obs).astype(int)
        norm_states = state_stats.normalize(cache["states"])
        self._obs = [
            {
                "vision_tokens": cache["vision_tokens"][i].unsqueeze(0).to(DEVICE),
                "lang_token_ids": cache["lang_token_ids"][i].unsqueeze(0).to(DEVICE),
                "lang_attention_mask": cache["lang_attention_mask"][i]
                .unsqueeze(0)
                .to(DEVICE),
                "state": norm_states[i].unsqueeze(0).to(DEVICE),
            }
            for i in idx
        ]
        self.n_calls = 0

    def predict_chunk(self, observation: np.ndarray) -> np.ndarray:
        """Run the real flow-matching policy; return (chunk_size, action_dim)."""
        torch = self.torch
        # The controller passes a step index; cycle through the real obs pool.
        i = int(np.asarray(observation).ravel()[0]) % len(self._obs)
        obs = self._obs[i]
        with torch.no_grad():
            with torch.amp.autocast(
                "cuda", dtype=self.DTYPE, enabled=(self.DEVICE.type == "cuda")
            ):
                chunk = self.trainer.sample(
                    obs["vision_tokens"],
                    obs["lang_token_ids"],
                    obs["lang_attention_mask"],
                    obs["state"],
                    ema=None,
                )
        if self.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        self.n_calls += 1
        # Denormalize to raw action units (not needed for latency, but realistic).
        chunk = chunk[0].float().cpu() * self._action_std + self._action_mean
        return chunk.numpy().astype(np.float32)


def measure_raw_latency(policy: SmolVLAChunkPolicy, n_calls: int, warmup: int) -> dict:
    """Time single predict_chunk calls after a warmup, return ms stats."""
    for w in range(warmup):
        policy.predict_chunk(np.array([w]))
    times_ms: list[float] = []
    for c in range(n_calls):
        t0 = time.perf_counter()
        policy.predict_chunk(np.array([c]))
        times_ms.append((time.perf_counter() - t0) * 1e3)
    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.pstdev(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def run_benchmark(epoch: int, steps: int, control_ms: float, threshold: int) -> None:
    """Print raw latency and a sync-vs-async rollout comparison."""
    control_s = control_ms / 1e3
    print("=" * 62)
    print("  Chapter 9: Real VLA Inference Latency (SmolVLA, Ch07)")
    print("=" * 62)

    policy = SmolVLAChunkPolicy(epoch=epoch)
    print(f"  checkpoint      : {policy.action_key}")
    print(f"  device          : {policy.DEVICE}")
    print(f"  chunk_size (K)  : {policy.chunk_size}")
    print(f"  control period  : {control_ms:.0f} ms  ({1e3 / control_ms:.0f} Hz)")
    print(f"  rollout steps   : {steps}\n")

    raw = measure_raw_latency(policy, n_calls=20, warmup=3)
    print(
        f"  raw policy latency : {raw['mean_ms']:.1f} +/- {raw['std_ms']:.1f} ms "
        f"(min {raw['min_ms']:.1f}, max {raw['max_ms']:.1f})"
    )
    print(
        f"  -> a blocking control loop stalls ~{raw['mean_ms']:.0f} ms every "
        f"{policy.chunk_size} steps\n"
    )

    def observe(i: int) -> np.ndarray:
        return np.array([i], dtype=np.float32)

    def step_env(_action: np.ndarray) -> None:
        time.sleep(control_s)

    ideal_ms = steps * control_s * 1e3
    print(f"  ideal wall (steps x period) = {ideal_ms:.0f} ms\n")
    for name, ctor in [
        ("sync ", lambda p: SyncChunkController(p)),
        ("async", lambda p: AsyncChunkController(p, replan_threshold=threshold)),
    ]:
        policy.n_calls = 0
        controller = ctor(policy)
        res = run_rollout(controller, observe, step_env, steps)
        controller.close()
        overhead = 100 * (res.wall_time_s * 1e3 - ideal_ms) / ideal_ms
        print(
            f"  {name} | wall={res.wall_time_s * 1e3:7.0f} ms | "
            f"inferences={res.n_inferences:2d} | overhead={overhead:6.1f}%"
        )

    print(
        "\n  Async runs the policy in a background thread while the robot keeps\n"
        "  executing buffered actions, so it only blocks on the cold-start chunk.\n"
        "  Sync pays the full policy latency every time the chunk empties."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Real VLA latency benchmark")
    parser.add_argument("--epoch", type=int, default=100, help="Ch07 checkpoint epoch")
    parser.add_argument("--steps", type=int, default=60, help="rollout control steps")
    parser.add_argument(
        "--control-ms", type=float, default=33.0, help="robot control period (ms)"
    )
    parser.add_argument("--replan-threshold", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(args.epoch, args.steps, args.control_ms, args.replan_threshold)


if __name__ == "__main__":
    main()
