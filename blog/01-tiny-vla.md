# A VLA in 200 lines

*Chapter 1 of [vla-from-scratch](https://github.com/FanFeast/vla-from-scratch).*

The fastest way to stop being intimidated by VLAs is to build one where you wrote
every line. No pretrained encoder, no HuggingFace, no config framework. Three
convolutions, an embedding table, and an MLP.

It reaches **70–80% success** on a toy pushing task, on CPU, in about three
minutes.

## The environment

`MiniPushT` is a 64×64 grid with three squares: a blue **agent** you control, a
red **block**, and a green **goal**. Push the block onto the goal. Five discrete
actions: up, down, left, right, stay.

Building our own environment instead of importing one is deliberate. Every pixel
is accounted for — you can see exactly where the image comes from and how the
expert generates demonstrations. When Chapter 2 swaps in a real vision encoder,
`VisionEncoder` is the only thing that changes.

A scripted oracle solves it in 20–40 steps, in two phases: walk to the block,
then push it toward the goal. Run it 1000 times and you have ~30k
`(image, instruction, action)` tuples.

## The model

```
Image 64x64x3
  -> Conv(3->32, s=2) -> ReLU
  -> Conv(32->64, s=2) -> ReLU
  -> Conv(64->128, s=2) -> ReLU
  -> AdaptiveAvgPool -> 128-dim vector
                                        \
Instruction -> embedding lookup -> 32-dim -> concat -> MLP -> 5 logits
```

The language side is an **embedding table**, not a language model. With one
instruction there is nothing to understand — it is a constant. That is the point:
it makes the seam obvious, so when Chapter 3 replaces it with SmolLM2 you can see
exactly what a language model buys you.

Training is a cross-entropy classification loop. If you have trained MNIST, you
have trained this.

## Results

| Parameter | Value |
|---|---|
| Demos | 1000 episodes (~30k transitions) |
| Epochs | 20 |
| Hardware | CPU / Colab T4 |
| Time | ~2–3 min |
| Success rate | ~70–80% |

## What is actually being learned

Not much, and that is worth being honest about. The policy learns a mapping from
"where are the coloured squares" to "which direction to move". It has no concept
of pushing, and it will not transfer to a differently-coloured block.

Two things do transfer, and they carry through all ten chapters:

**The data tuple.** `(image, instruction, action)` is the unit for every model in
this series, including the 324M-parameter one in Chapter 7. Only the encoders and
the head change.

**Behavior cloning is just supervised learning.** No reward, no rollouts during
training. The hard parts of VLAs are not the training loop — they are
representation choices and the gap between offline loss and closed-loop success.
Chapter 5 makes that gap painfully concrete.

## The obvious next question

The CNN is 308k parameters trained on 30k images of coloured squares. It has never
seen a real object. Meanwhile CLIP has seen 400 million image-text pairs.

What happens if we just... use that instead?

**Next:** [Chapter 2](02-vision-backbone.md) freezes a pretrained encoder, and the
scratch CNN does not survive the comparison.
