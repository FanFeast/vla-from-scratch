# How do you represent an action?

*Chapter 4 of [vla-from-scratch](https://github.com/FanFeast/vla-from-scratch).*

Everything so far picked from five discrete actions. Real robots take continuous
ones, and the moment actions are continuous, "what shape is the output?" becomes a
real question with three competing answers.

![Three action heads](../assets/diagrams/ch04_action_heads.svg)

## The three heads

**Discrete.** Chop each dimension into 256 bins and predict a bin like a language
token. RT-2 and OpenVLA do this, because it lets you reuse an LM head unchanged.
Over a `[-1, 1]` range each bin is 0.0078 wide, so the round-trip error is bounded
by roughly one bin:

```python
>>> from action_heads import continuous_to_bins, bins_to_continuous
>>> a = torch.tensor([[0.137, -0.42]])
>>> continuous_to_bins(a)
tensor([[144,  73]])
>>> bins_to_continuous(continuous_to_bins(a))
tensor([[ 0.1294, -0.4275]])      # max abs error 0.0076, bin width 0.0078
```

**Regression.** Predict the number, MSE loss. One forward pass, exact, cheapest.

**Diffusion.** Learn to denoise: start from noise, run 10 steps, get an action.
This is what pi0 and SmolVLA use.

And crossed with all three, **action chunking**: predict the next K actions
instead of just the next one.

## Results (224×224, 3000 demos, 200 epochs)

| Config | Success | Smoothness ↓ | Avg length |
|---|---|---|---|
| Discrete K=1 | 69% | 0.630 | 113 |
| Discrete K=4 | 82% | 0.206 | 97 |
| Regression K=1 | 86% | 0.215 | 94 |
| **Regression K=4** | **95%** | **0.101** | **83** |
| Diffusion K=1 | 1% | 1.177 | 200 |
| Diffusion K=4 | 0% | 1.086 | 200 |

## Chunking is the biggest lever in the series

Read the table in pairs. Every head improves with K=4, and the trajectories get
markedly smoother — smoothness drops 3× for discrete, 2× for regression.

The intuition: a policy predicting one step at a time can dither, reversing
direction every frame because each decision is independent. Predicting four
actions as one coherent unit forces commitment. You are not just reducing
inference calls; you are changing what the policy is allowed to be.

This is bigger than the choice of head, and it is the cheapest change in the
chapter.

## Regression wins here, for a reason that will not last

95% with K=4, and it is the simplest and fastest option. Why?

**MiniPushT is unimodal.** From any state there is essentially one right way to
push the block. When the answer is a single number, predicting a single number is
exactly correct, and anything generative is overhead.

Hold onto that sentence. Chapter 5 puts human demonstrations in front of the same
head and it stops being true.

## Diffusion gets 0%, and that is the most useful row

A 2-layer MLP denoiser with 10 inference steps scores **1% and 0%**.

It would be easy to conclude diffusion is wrong for robot policies. That is not
what happened. Two things went wrong, both fixable:

1. **No capacity.** A 2-layer MLP cannot represent a useful score function. Real
   diffusion policies use 16M+ parameter UNets or transformers.
2. **Nothing to model.** Diffusion's advantage is capturing *multi-modal*
   distributions. On a unimodal toy task there is no second mode to capture, so
   the extra machinery buys nothing and costs 10 forward passes per action.

Chapter 6 rebuilds this head as a 12.4M-parameter transformer trained with flow
matching, on data that *is* multi-modal, and it works. The failure here is the
control that makes that result mean something.

## Discrete is more competitive than its reputation

82% at K=4. The quantization tax people worry about is mostly imaginary at 256
bins. Its real problem shows up at high dimension — Chapter 5's 14-DOF ALOHA arm
turns into 14 × 256 independent classification problems per timestep, with no
coupling between joints.

## What is still missing

Everything so far ran on a toy environment with a scripted oracle: noiseless,
deterministic, always optimal. Real demonstrations are none of those.

**Next:** [Chapter 5](05-scaling-up.md) loads real robot data, and the 95% policy
scores zero.
