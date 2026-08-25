# Chapter 9: Evaluation & Deployment — Does the Policy Actually Work?

**Goal:** Measure a VLA the only way that matters — **closed-loop success rate**
on a benchmark — and run it in **real time** on a robot using asynchronous
action chunking. You will build the evaluation harness, the LIBERO adapter,
and the async inference engine from scratch.

![Sync vs async inference](../../assets/diagrams/ch09_async.svg)

Low training loss is not success. A regression head can hit MAE 0.05 and still
fail every episode (we saw exactly this in Chapter 5): tiny per-step errors
compound over a 300-step rollout until the arm drifts off task. The honest
question is *"how often does it finish the task when you actually run it?"*

## Part 1: The Evaluation Problem

```
   Training metric              Deployment metric
   ───────────────              ─────────────────
   MSE / flow loss      vs.     success rate (%)
   per-step error               whole-episode outcome
   teacher-forced               closed-loop (errors compound)
```

A VLA evaluation is a loop: observe -> policy predicts -> execute -> repeat,
until the task is solved or time runs out. We track three numbers:

| Metric | Meaning |
|--------|---------|
| **Success rate** | Fraction of episodes that reach the goal. The headline number. |
| **Episode length** | Steps to completion (lower = more efficient). |
| **Smoothness** | Mean step-to-step action change (jerk proxy; spikes at chunk seams). |

`eval_libero.py` implements these with a dependency-free `MockManipulationEnv`
so the whole harness runs and is tested without a simulator:

```bash
cd chapters/09_eval_and_deploy
uv sync
uv run python eval_libero.py            # mock-env demo (sync vs async)
uv run pytest test_chapter_09.py -v
```

## Part 2: LIBERO — The Standard VLA Benchmark

[LIBERO](https://libero-project.github.io/) is what OpenVLA, SmolVLA, and pi0
report on. It has four 10-task suites plus two larger pools:

| Suite | Tasks | Varies |
|-------|-------|--------|
| `libero_spatial` | 10 | Spatial arrangement of the same objects |
| `libero_object` | 10 | Which objects are present |
| `libero_goal` | 10 | The goal, with fixed objects/layout |
| `libero_10` (LONG) | 10 | Long-horizon, multi-step tasks |
| `libero_90` | 90 | Short-horizon pool (training + eval) |

For reference, SmolVLA reports ~74% on LIBERO-90; OpenVLA-7B is in a similar
range after fine-tuning. Our Chapter 7 model is far smaller and trained on 3
datasets, so it is not LIBERO-competitive — but the *harness* is identical.

`make_libero_env(task_suite, task_id)` returns a real LIBERO env when the
`libero` package is installed, and otherwise raises a clear install message.
LIBERO is not on PyPI — install it from source:

```bash
pip install robosuite mujoco
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
uv run python eval_libero.py --task-suite libero_spatial
```

LIBERO uses the robosuite API, so wrap it to the `EvalEnv` interface
(`reset() -> obs`, `step(action) -> (obs, reward, done, info)`); the success
flag lives in `info["success"]`.

## Part 3: Real-Time Inference (`async_inference.py`)

A flow-matching VLA forward pass (VLM + ~10 Euler steps) takes tens to hundreds
of milliseconds. A robot wants an action every 20-50 ms. Three techniques make
this work, and we implement all three:

### 1. Action chunking

The policy predicts **K future actions at once**; you execute the whole chunk
before replanning. The expensive policy runs once per K steps instead of every
step.

```
predict 10 actions ──► [a0 a1 a2 a3 a4 a5 a6 a7 a8 a9] ──► execute open-loop
                         then predict the next chunk
```

### 2. Asynchronous inference

Synchronous chunking still **stalls** every K steps while the GPU replans.
Async inference launches the next prediction in a background thread *before*
the buffer empties, so execution and inference overlap. The robot only waits
on the very first (cold-start) chunk.

```
sync :  [exec chunk]████STALL████[exec chunk]████STALL████   <- pays latency K times
async:  [exec chunk][exec chunk][exec chunk] ...             <- inference hidden underneath
              └─ predict next ─┘ (in background thread)
```

The `AsyncChunkController` kicks off a replan when the buffer drops to
`replan_threshold` and no inference is in flight:

```python
def get_action(self, obs):
    self._collect_ready()        # fold in any finished background prediction
    self._maybe_replan(obs)      # start next prediction if running low
    if self.buffer.is_empty():   # cold start: we have no choice but to wait
        self.buffer.push_chunk(self._future.result())
    return self.buffer.pop()
```

The demo shows async beating the synchronous baseline in wall-clock time while
producing **identical actions** (verified in the tests):

```
  sync  | wall=325ms | inferences=4 | overhead=62.3%
  async | wall=286ms | inferences=5 | overhead=43.1%
```

(That is the mock demo with a 30 ms synthetic latency; see
[Deploying Our Chapter 7 Model](#deploying-our-chapter-7-model--real-latency-numbers)
below for the same comparison with the **real** SmolVLA forward pass.)

### 3. Temporal ensembling (ACT)

Overlapping chunks each predict the same future timestep. Averaging them with
an exponential recency weight removes the discontinuity at chunk boundaries:

```
weight(age) = exp(-m * age)      # age 0 = most recent prediction
action(t)   = sum_i weight_i * prediction_i(t) / sum_i weight_i
```

`TemporalEnsembler` maintains a per-timestep table and returns the weighted
average. Larger `m` trusts the newest chunk more; `m -> 0` is a plain average.

## What's Here

| File | What it does |
|------|--------------|
| `async_inference.py` | `ActionChunkBuffer`, `TemporalEnsembler`, sync/async controllers, mock policy, rollout runner, latency demo. |
| `eval_libero.py` | Metrics, `MockManipulationEnv`, `evaluate_policy`, gated `make_libero_env` LIBERO adapter. |
| `benchmark_latency.py` | `SmolVLAChunkPolicy` wrapping the real Ch07 model; measures true sync-vs-async inference latency (needs `--extra vla`). |
| `test_chapter_09.py` | Buffer FIFO, ensembling weights, async==sync behavior, async latency hiding, success-rate aggregation, mock-env pass/fail. |

## Deploying Our Chapter 7 Model — Real Latency Numbers

The demo above uses a mock policy that just sleeps. `benchmark_latency.py`
swaps in the **actual Chapter 7 SmolVLA** (`SmolVLAChunkPolicy`) — real SigLIP
tokens -> frozen SmolLM2 prefix -> flow-matching action expert (10 Euler steps)
— and measures the same sync-vs-async comparison with a genuine forward pass:

```python
class SmolVLAChunkPolicy:
    chunk_size = 10
    action_dim = 6
    def predict_chunk(self, observation):
        chunk = self.trainer.sample(vision, lang, lang_mask, state, ema=None)
        return (chunk[0].cpu() * self.action_std + self.action_mean).numpy()
```

```bash
uv sync --extra vla     # torch + transformers (the core harness stays numpy-only)
uv run python benchmark_latency.py --epoch 100 --steps 60 --control-ms 33
```

**Measured on an RTX 4090 (Ch07 model, epoch 100, 30 Hz control loop):**

| | Raw policy latency | Rollout wall (60 steps) | Overhead vs ideal | Policy calls |
|---|---|---|---|---|
| — | **72.6 +/- 1.8 ms** | ideal = 1980 ms | — | — |
| **Sync** | (stalls every K steps) | 2432 ms | **+22.8%** | 6 |
| **Async** | (hidden in background) | 2081 ms | **+5.1%** | 7 |

A single real VLA forward pass is **~73 ms** — far longer than a 33 ms control
period, so a synchronous loop stalls ~73 ms every 10 steps and burns **+22.8%**
wall time. The async controller runs each replan in a background thread while the
robot keeps executing buffered actions, so it pays that latency **only on the
cold-start chunk** — overhead drops to **+5.1%** (the residual is essentially
that one unavoidable first prediction). The action sequences are identical; only
the timing changes.

Drop the same `SmolVLAChunkPolicy` into `evaluate_policy(policy, env_factory,
...)` for closed-loop success-rate eval — the controller and metrics are
policy-agnostic. (Full log: `benchmark_log.txt`.)

## What This Chapter Demonstrates

1. **Success rate is the metric.** Training loss and even per-step MAE can lie;
   only closed-loop rollouts tell the truth.
2. **The eval harness is simple and framework-agnostic.** A clean `EvalEnv`
   interface plus three metrics covers mock envs and real LIBERO alike.
3. **Real-time VLAs need async chunking.** Chunking amortizes the policy cost;
   async hides it; temporal ensembling smooths the seams.
4. **Async changes timing, not behavior.** Given the same observations it emits
   the same actions — verified in the tests.

## The Arc Is Complete

From a 3-layer CNN on a toy pushing task (Chapter 1) to a SmolVLA-like model
with flow matching (Chapter 7), LoRA fine-tuning (Chapter 8), and real-time
evaluated deployment (Chapter 9), you have built a Vision-Language-Action model
from first principles.

**Where to go next** (out of scope for this guide, mentioned for direction):
- **RL fine-tuning** on top of the BC policy (e.g. residual RL, RLPD).
- **Sim-to-real** transfer and domain randomization.
- **Multi-embodiment** training across robot types (see X-VLA).

## References

- [LIBERO](https://arxiv.org/abs/2306.03310) — Liu et al., 2023 (lifelong robot learning benchmark)
- [ACT: Action Chunking with Transformers](https://arxiv.org/abs/2304.13705) — Zhao et al., 2023 (chunking + temporal ensembling)
- [SmolVLA](https://arxiv.org/abs/2506.01844) — Shukor et al., 2025 (async inference for real-time VLAs)
- [OpenVLA](https://arxiv.org/abs/2406.09246) — Kim et al., 2024 (LIBERO evaluation protocol)
