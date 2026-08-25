# Borrowing a vision encoder

*Chapter 2 of [vla-from-scratch](https://github.com/FanFeast/vla-from-scratch).*

Chapter 1's CNN saw 30,000 images of coloured squares. CLIP saw 400 million
image-text pairs. This chapter swaps one for the other and measures what happens.

The setup is a controlled experiment: identical task, identical action head,
identical training loop. **Only the vision encoder changes**, and it is frozen —
we train just the head on top, so any difference is attributable to the features.

## Three encoders

- **Scratch CNN** — Chapter 1's, 4 layers, 308k trainable params.
- **CLIP ViT-B/16** — [Radford et al., 2021](https://arxiv.org/abs/2103.00020),
  softmax contrastive loss, CLS-token pooling.
- **SigLIP ViT-B/16** — [Zhai et al., 2023](https://arxiv.org/abs/2303.15343),
  same architecture, **sigmoid** loss instead of softmax. This is what SmolVLA
  uses, so we carry it forward.

## Results at native resolution (224×224, 2000 demos, 40 epochs)

| Encoder | Trainable | Time | Loss | Success | OOD success |
|---|---|---|---|---|---|
| Scratch CNN | 308,293 | 838s | 1.386 | **0%** | 0% |
| CLIP ViT-B/16 | 231,301 | 95s | 0.350 | 68% | 18% |
| SigLIP ViT-B/16 | 231,301 | 94s | 0.418 | 34% | 6% |
| SigLIP + attention pooling | 232,070 | 9,949s | 0.329 | 50% | **30%** |

**OOD success** starts the agent and block in corner positions never seen during
training — it tests whether spatial understanding generalizes past the training
distribution.

Three things in that table are worth stopping on.

## 1. The scratch CNN gets 0%, and it is not a bug

Its loss plateaus at 1.386. Random guessing over 5 classes is ln(5) ≈ 1.609, so
it is barely better than chance.

The cause is pooling. The CNN reduces 224×224 to a 13×13 feature map, then
global-average-pools to 128 dims. At 64px the pre-pool map was 3×3 — coarse, but
the average still carried *where* things were. At 13×13, averaging destroys it.
The network can tell you a red block exists; it cannot tell you where.

It also trains **9× slower** than the frozen ViTs, because the frozen encoders
need no backward pass through the vision tower.

## 2. Frozen beats trained, with fewer parameters

CLIP hits 68% with **fewer trainable parameters** than the CNN (231k vs 308k) in
**one ninth** the time. The features were already there. We only had to learn a
linear map from them to five actions.

This is the whole argument for the VLA approach, visible at toy scale: do not
teach the policy what objects look like. Borrow that, and spend your parameters
on the part that is actually robot-specific.

## 3. CLIP beat SigLIP — and the reason is instructive

At 112×112 the gap was enormous (74% vs 36%), which was surprising given SigLIP
is the more modern model and the one SmolVLA chose.

Both were pretrained at 224×224. At 112 we bicubic-upsample to 224 first, and the
two models degrade differently:

- **CLIP** pools with a **CLS token** — one global summary that attends to all
  patches. Relatively robust to a blurry input.
- **SigLIP** pools by **averaging spatial patch tokens**. Blur every patch, then
  average the degraded patches, and errors compound.

At native 224 the gap narrows (68% vs 34%), and swapping SigLIP's mean-pool for a
**learned attention pool** — 769 extra parameters — takes it to 50% success and
the best OOD score in the table at 30%.

So the honest conclusion is not "CLIP is better than SigLIP." It is that **how you
pool matters as much as which encoder you pick**, and a benchmark run off-resolution
will lie to you.

## Cost

The attention-pooling run took 9,949 seconds — over 100× the plain SigLIP run —
because attention pooling puts a trainable module back in the gradient path. It
bought the best generalization in the table. Whether that trade is worth it
depends on whether you are optimizing for iteration speed or final quality.

## What is still missing

There is exactly one instruction. The "language" input is a constant, so the model
never has to read anything.

**Next:** [Chapter 3](03-language-conditioning.md) gives it eleven instructions and
a real language model, and finds that the sophisticated fusion mechanism loses.
