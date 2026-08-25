# Flow matching

*Chapter 6 of [vla-from-scratch](https://github.com/FanFeast/vla-from-scratch).*

Chapter 5 diagnosed two problems: the denoiser had no capacity, and MSE regression
averages multi-modal human demonstrations into paths nobody drove. This chapter
fixes both — a 12.4M-parameter transformer action expert, trained generatively —
and gets the first architecture in the series that does something real on robot
data.

It also runs the comparison the field mostly asserts: **flow matching vs DDPM**,
same transformer, same data, same compute.

## Straight paths beat curved ones

![Flow matching vs DDPM](../assets/diagrams/ch06_flow_vs_ddpm.svg)

Both methods learn to turn noise into an action. They differ in the path.

**Flow matching** (rectified flow) defines a straight line between noise `x0` and
data `x1`:

```
x_t = (1 - t) * x0 + t * x1
target = x1 - x0
```

The target velocity is `x1 - x0` — **constant along the path**, independent of `t`.
The model learns a velocity field whose integral curves are straight lines, so an
Euler solver with a handful of steps integrates it almost exactly.

**DDPM** follows the curve implied by its noise schedule. Same endpoints, but the
velocity changes as you go, so coarse integration accumulates error.

That is the entire mechanism behind the result below.

## Results (PushT, 206 episodes, 200 epochs, EMA)

| Config | Coverage | Inference |
|---|---|---|
| **FM K=4** | **29.0%** | 13.0 ms |
| FM K=16 | 26.2% | 13.9 ms |
| DDPM K=4 | 5.3% | 28.4 ms |
| DDPM K=16 | 5.5% | 30.3 ms |

**5× the coverage at half the latency.** `avg_coverage` is the fraction of the
target region the T-block was pushed into — a graded score, more informative than
binary success when nothing is near 100%.

## DDPM wins on validation loss and loses on the task

On ALOHA, DDPM gets **0.028** val loss to flow matching's 0.048 — and samples
worse.

Do not rank generative policies by training loss. The two objectives predict
different quantities (noise vs velocity) on different scales; the numbers are not
comparable across methods. Only rollout tells you which sampler works. This is
Chapter 5's lesson one level up: there, offline metrics missed compounding error;
here, they cannot even be compared across two models.

## The fix was making the model smaller

The first version of this action expert was **96.3M parameters**. It overfit
22,000 frames catastrophically.

Cutting it to **12.4M** with dropout=0.1 fixed it — comparable to Diffusion
Policy's ~16M UNet. Chapter 4's diffusion head failed for too *little* capacity;
the obvious fix overshot in the other direction. The lesson is not "bigger" or
"smaller" but *matched to your data*, and 206 episodes is not much data.

## How many Euler steps?

| Steps | Coverage | Inference |
|---|---|---|
| 3 | 22.3% | 4.9 ms |
| 5 | 27.1% | 7.0 ms |
| **10** | **29.6%** | **12.8 ms** |
| 20 | 20.4% | 25.6 ms |
| 50 | 36.0% | 63.4 ms |

Coverage climbs to 10 steps, dips at 20, and only beats 10 again at 50 steps and
63 ms — far too slow for a 30 Hz control loop. **10 steps is the sweet spot**, and
the fact that a good answer exists at 10 steps at all is what the straight path
buys you.

## Architecture

Six blocks, each **self-attention over the action chunk → cross-attention into
visual context → FFN**, pre-LayerNorm with residuals.

The cross-attention is interleaved on purpose: every block re-reads the
observation, rather than conditioning once at the input and hoping the signal
survives six layers.

Two details that made training stable:

- **Zero-initialized output projection**, so the model starts predicting identity.
- **EMA (decay 0.999)** for inference. Generative models sample noticeably better
  from averaged weights.

## Being honest about 29%

Coverage of 29% is not success. 206 demonstrations is not enough data, and we are
not going to dress that up.

But the policy is genuinely pushing the block toward the target, which nothing in
Chapter 5 did. That is the first real signal in this series on real robot data.

**Next:** [Chapter 7](07-full-vla.md) puts the vision encoder, the language model,
and this action expert together into one model.
