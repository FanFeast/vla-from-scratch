# Teaching it to read

*Chapter 3 of [vla-from-scratch](https://github.com/FanFeast/vla-from-scratch).*

Chapters 1 and 2 had one instruction, so the language input was decoration. This
chapter gives the policy **eleven instructions** across four behavior categories —
push the block to a goal, push it in a direction, move the agent, push the block
to a named location — and then asks the question that actually matters.

Not "can it follow instructions it trained on." Can it follow **rewordings it has
never seen**?

## Paraphrase generalization is the real metric

Each canonical instruction has held-out paraphrases:

```
"push block to goal"  ->  "move the block to the target"
                          "shove the block toward the goal"
                          "get the block to the green zone"
```

A lookup table maps a *string* to an embedding. Show it a string it has never
seen and it has nothing — it falls back to guessing from vision alone. A real
language model maps both phrasings to nearby points in embedding space, because
that is what pretraining on text buys you.

This is the cleanest demonstration in the series of why a **pretrained language
model** and not an embedding table.

## Two ways to fuse

![Concat vs cross-attention](../assets/diagrams/ch03_fusion.svg)

**Concat** pools the language to a single vector and concatenates it with the
vision vector. Cheap. The policy cannot ask which word applies to which pixel.

**Cross-attention** keeps every language token. Vision tokens are queries,
language tokens are keys and values, so each patch can attend to individual words.
This is the sophisticated option, and it is what you would pick from first
principles.

## Results (224×224, 10K demos, 100 epochs, RTX 4090)

| Config | Trainable | Time | Success | Paraphrase |
|---|---|---|---|---|
| Lookup + Concat | 332K | ~900s | 83.6% | 23.6% |
| **SmolLM2-135M + Concat** | 477K | 910s | **89.1%** | **40.0%** |
| SmolLM2-135M + CrossAttn | 642K | 37,897s | 81.8% | 38.2% |
| SmolLM2-360M + CrossAttn | 740K | 40,464s | 87.3% | 29.1% |

## The pretrained LM is the big lever

Lookup gets **23.6%** on paraphrases. SmolLM2 gets **40.0%** — a 70% relative
improvement, purely from having read the internet. The lookup table's 23.6% is
roughly what you would get by ignoring the instruction and guessing from the scene.

## Cross-attention lost, and that is the interesting part

It trained **40× slower** (37,897s vs 910s) and scored *worse* on both metrics.

The reason is that MiniPushT has nothing to ground. There is one block, one goal,
five actions, and instructions of five words. Cross-attention's advantage is
resolving *which object among many* a phrase refers to — "pick up the **red** mug,
not the blue one." With a single object, per-token attention is machinery with
nothing to do, and you pay for it in compute and in optimization difficulty.

**Architectural sophistication is not free, and it is not always an improvement.**
Cross-attention earns its place from Chapter 5 onward, once scenes have multiple
objects and instructions get specific. Measuring it on a toy task and concluding
it is bad would be exactly the wrong lesson — the right one is that the benchmark
has to be able to *express* the advantage you are testing for.

## Three implementation details that cost real time

**Cache the pre-projection hidden states.** SmolLM2's outputs get cached *before*
the trainable projection, or gradients cannot flow to it.

**Use `key_padding_mask`.** Without it, vision tokens attend to padding, and short
instructions get systematically worse.

**bf16, not fp16.** fp16 produces NaN in the cross-attention softmax — the
exponent range is too small. bf16 has fp32's exponent range at half the width and
just works. This one cost an afternoon.

## What is still missing

Five discrete actions. Real robots take continuous ones.

**Next:** [Chapter 4](04-action-representations.md) moves to continuous control and
finds a lever bigger than anything so far.
