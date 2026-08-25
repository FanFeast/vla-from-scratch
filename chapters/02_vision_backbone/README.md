# Chapter 2: Vision Backbone

**Goal:** Swap the scratch CNN for frozen pretrained vision encoders (CLIP, SigLIP)
and measure exactly how much pretrained representations improve over training from scratch.

## Training Recipe

| Parameter | Default (`--preset full`) | Benchmark run |
|-----------|--------------------------|---------------|
| Demos | 1000 episodes | 2000 episodes |
| Transitions | ~107k | ~215k |
| Batch size | 64 | 64 |
| Learning rate | 1e-3 | 1e-3 |
| Epochs | 20 | 40 |
| Hardware | GPU recommended (CLIP, SigLIP) | RTX 4090 (24GB) |

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Quick test (~30s, no pretrained models needed)
python compare_encoders.py --preset tiny

# Full comparison (~8min on GPU, downloads CLIP + SigLIP ~700MB)
python compare_encoders.py --preset full

# Full comparison with attention pooling + OOD eval (~3hrs on GPU)
python compare_encoders.py --preset full --attn-pool --eval-ood --n-demos 2000 --epochs 40
```

First run of `medium` or `full` preset downloads CLIP (~350MB) and SigLIP (~350MB)
from HuggingFace. The `tiny` preset skips pretrained models entirely.

## What Changed From Chapter 1

Two things changed. Everything else is identical.

1. **Resolution: 64x64 -> 224x224.** CLIP and SigLIP expect 224x224 input.
   Rather than resizing at inference (a hack we would remove next chapter),
   we bump the environment to 224x224. This is the standard for all future chapters.

2. **Vision encoder is now swappable.** The `VLA` class takes any encoder that
   implements `forward(image) -> (B, embed_dim)`. We compare three encoders
   that all satisfy this interface.

## The Three Encoders

### Scratch CNN (baseline)

A 4-layer CNN that trains from scratch, identical in spirit to Chapter 1 but
deeper to handle 224x224 input:

```
Conv(3->32, 3x3, s=2)  -> ReLU    # (B, 32, 111, 111)
Conv(32->64, 3x3, s=2) -> ReLU    # (B, 64, 55, 55)
Conv(64->128, 3x3, s=2) -> ReLU   # (B, 128, 27, 27)
Conv(128->128, 3x3, s=2) -> ReLU  # (B, 128, 13, 13)
AdaptiveAvgPool2d(1)                # (B, 128, 1, 1)
Flatten -> 128-dim
```

This encoder sees only MiniPushT images during training. It has to learn
what "block", "goal", and "agent" look like from scratch using ~30k training
frames.

### CLIP ViT-B/16

**Contrastive Language-Image Pretraining** ([Radford et al., 2021](https://arxiv.org/abs/2103.00020))
trains a vision encoder and a text encoder jointly on 400M image-text pairs
from the internet. The training objective: given a batch of (image, caption)
pairs, the model learns to match each image to its correct caption and vice versa.

Architecture: **Vision Transformer** (ViT) with patch size 16. The image is split
into 14x14 = 196 patches of 16x16 pixels each. Each patch is linearly projected
to a token, position embeddings are added, and the sequence is processed by 12
transformer layers. The **CLS token** output (768-dim) summarizes the full image.

We freeze all 86M parameters and use only the CLS token output as our vision
embedding. No CLIP weights change during training -- only the action head and
language encoder learn.

### SigLIP ViT-B/16

**Sigmoid Loss for Language-Image Pretraining** ([Zhai et al., 2023](https://arxiv.org/abs/2303.15343))
is similar to CLIP but replaces the softmax-based contrastive loss with a
**sigmoid loss** applied to each image-text pair independently.

Why does this matter?

- **CLIP's loss** computes softmax over the entire batch. Each image competes
  against every other image for its caption. This requires large batch sizes
  (32k+) to work well, because with small batches there aren't enough negatives.

- **SigLIP's loss** treats each (image, text) pair as an independent binary
  classification: "does this image match this text?" This means it works well
  with smaller batches and scales more efficiently.

Same architecture as CLIP (ViT-B/16, 768-dim output), but trained with the
sigmoid objective. SigLIP is what **SmolVLA** uses as its vision encoder --
we will carry it forward through the rest of this guide.

## Benchmark Results

> **Author's note:** These are real numbers from a single machine (RTX GPU, CUDA).
> Your results will vary slightly due to random seeds, but the relative ordering
> should be consistent.

### Medium preset — 112×112, 2000 demos, 40 epochs

| Encoder | Trainable Params | Train Time | Final Loss | Success Rate |
|---------|-----------------|------------|------------|-------------|
| Scratch CNN | 308,293 | 301s | 0.105 | 16% |
| CLIP ViT-B/16 | 231,301 | 123s | 0.318 | 74% |
| SigLIP ViT-B/16 | 231,301 | 122s | 0.364 | 36% |

### Medium preset — 112×112, 4000 demos, 40 epochs

| Encoder | Trainable Params | Train Time | Final Loss | Success Rate |
|---------|-----------------|------------|------------|-------------|
| Scratch CNN | 308,293 | ~600s | 0.105 | 26% |
| CLIP ViT-B/16 | 231,301 | ~245s | 0.318 | 76% |
| SigLIP ViT-B/16 | 231,301 | ~244s | 0.364 | 28% |

**Key takeaway:** CLIP dominates at 112×112. But why such a big gap over SigLIP?

> **⚠️ Why CLIP >> SigLIP at non-native resolutions**
>
> Both CLIP and SigLIP were trained on **224×224** images. At 112×112 we
> `F.interpolate` (bicubic upscale) to 224 before feeding the ViT — this is
> a lossy transform that affects the two models differently:
>
> - **CLIP** uses a **CLS token** for pooling — a single global summary vector.
>   It is relatively robust to spatial distortion from resizing because the CLS
>   token attends to all patches equally regardless of fine-grained positions.
>
> - **SigLIP** uses a **pooler output** that averages over spatial patch tokens.
>   When the input is blurry (upscaled from 112→224), the per-patch features
>   degrade, and averaging over degraded patches compounds the error.
>
> - SigLIP's **sigmoid loss** training produces more tightly calibrated embeddings
>   than CLIP's softmax contrastive loss. This makes SigLIP features more
>   sensitive to distribution shift (i.e., blurry upscaled inputs ≠ the crisp
>   224px images it was trained on).
>
> **At native 224×224 (`--preset full`) the gap should narrow significantly**
> because both models receive the input resolution they expect. Try it yourself
> to verify!

### Full preset — 224×224, 2000 demos, 40 epochs

| Encoder | Trainable Params | Train Time | Final Loss | Success Rate | OOD Success |
|---------|-----------------|------------|------------|-------------|-------------|
| Scratch CNN | 308,293 | 838s | 1.386 | 0% | 0% |
| CLIP ViT-B/16 | 231,301 | 95s | 0.350 | 68% | 18% |
| SigLIP ViT-B/16 | 231,301 | 94s | 0.418 | 34% | 6% |
| SigLIP (attn pool) | 232,070 | 9,949s | 0.329 | 50% | 30% |

> **OOD Success** = out-of-distribution evaluation using `--eval-ood`. Agent and block
> start in corner positions the policy never saw during training. This tests whether
> the encoder's spatial understanding generalizes beyond the training position distribution.

> **🔍 Why Scratch CNN gets 0% at 224×224 despite high loss**
>
> The loss plateaus at 1.386 (random chance for 5 classes = ln(5) ≈ 1.609, so barely
> better than random). The 4-layer CNN (308K params) is too small for 224×224 input.
> Here's what goes wrong:
>
> 1. **AdaptiveAvgPool crushes spatial info.** The CNN reduces 224×224 down to
>    a 13×13 feature map, then global average-pools it to 128 dims. At 64px
>    the pre-pool map is 3×3 (still informative); at 224px it's 13×13
>    (averaging kills "where" information).
>
> 2. **Capacity mismatch.** 308K parameters cannot represent the function needed to
>    map 224×224 images → 5 actions. The model barely learns anything — its loss
>    plateau (1.386) is only marginally better than random guessing (1.609).
>
> 3. **Compounding errors.** Even if it memorized some training patterns, at
>    evaluation time its own mistakes lead to states it never saw — and the
>    fragile 128-dim features can't generalise.
>
> **Lesson:** A small scratch CNN can work at low resolutions (16% at 112px)
> but completely falls apart at higher resolutions. This is *exactly* why
> pretrained encoders matter.

> **🤔 "But I expected SigLIP to beat CLIP!"**
>
> A reasonable expectation — SigLIP is newer (2023), generally considered an
> improvement over CLIP, and is what SmolVLA uses. So why does CLIP win here?
>
> **It's about pooling strategy, not encoder quality.**
>
> - **CLIP `pooler_output`** = CLS token → LayerNorm → Linear. The CLS token
>   learns selective attention during pre-training — it focuses on the most
>   salient parts of the image. For MiniPushT (tiny colored squares on a
>   black canvas), this selective focus is exactly what the action head needs.
>
> - **SigLIP `pooler_output`** = mean of all 196 patch tokens → Dense → **tanh**.
>   In MiniPushT, ~190 of those patches are pure black background. Mean pooling
>   *dilutes* the signal from the ~6 patches that actually contain the block,
>   agent, and goal. Then tanh squashes the dynamic range to [-1, 1], making it
>   even harder for the action head to discriminate spatial configurations.
>
> - **SigLIP's advantages don't apply here.** Its sigmoid loss helps with
>   efficient pre-training at scale and better zero-shot classification — none
>   of which matters for a frozen encoder on a 2D toy task.
>
> **The fix:** Use `--attn-pool` to replace SigLIP's built-in mean+tanh pooler
> with a tiny learned attention pooler (769 extra params) that focuses on the
> patches that matter. See the section below.

### Unlocking SigLIP: learned attention pooling

The `--attn-pool` flag adds a 4th encoder — **SigLIP (attn pool)** — that
replaces the default mean+tanh pooler with a learned single-head attention
pooler over SigLIP's raw patch tokens:

```
Frozen SigLIP ViT → last_hidden_state (B, 197, 768)
    → AttentionPool (769 trainable params)
    → weighted sum → (B, 768)
```

The attention pooler is just `nn.Linear(768, 1)` → softmax over patches →
weighted sum. It learns to attend to the patches containing the block, agent,
and goal while ignoring the ~190 black background patches.

```bash
# Compare all 4 encoders (adds ~3-5min for the attn pool variant)
python compare_encoders.py --preset medium --attn-pool

# Full resolution comparison
python compare_encoders.py --preset full --attn-pool
```

This demonstrates that the **architecture around the encoder matters as much
as the encoder itself**. SigLIP's raw patch features are excellent — its
default pooler is just a poor fit for sparse-object environments.

### OOD generalization: the real test

In-distribution success rate can be misleading — a model that memorises
the training start positions might look good but fail in deployment. The
`--eval-ood` flag tests exactly this: agent and block start in the **outer
ring** of the environment (corner positions never seen during training).

```bash
python compare_encoders.py --preset full --attn-pool --eval-ood
```

Results at 224×224 tell a clear story:

- **Scratch CNN (0% OOD):** Can't even solve in-distribution, let alone novel starts.
- **SigLIP plain (6% OOD):** Mean pooling dilutes spatial info → fragile features.
- **CLIP (18% OOD):** CLS token's selective attention helps, but limited.
- **SigLIP attn pool (30% OOD):** Task-specific pooling learns the *right* patches
  to attend to, giving the most robust spatial understanding.

The attn-pool variant's OOD advantage is even more striking than its in-distribution
lead. It suggests that learned attention over patches captures **relative spatial
relationships** that transfer to novel configurations, while fixed pooling strategies
overfit to the training position distribution.

## How Pretrained Features Help

The scratch CNN starts with random weights. It must learn *everything* from
MiniPushT data alone: edges, colors, shapes, spatial relationships. With only
a few hundred thousand training frames of simple colored squares, it can learn
enough to work at low resolutions (~16% at 112×112) but its features are fragile
— and at 224×224 it fails entirely (0% success).

CLIP and SigLIP were trained on hundreds of millions of real-world images. They
already know what objects are, where they are in the frame, and how they relate
spatially. Even though they have never seen MiniPushT, their features encode
useful structure:

- "There is a red thing near the center" -> meaningful spatial feature
- "The blue thing is far from the red thing" -> relative position encoded

This connects directly to Chapter 1's compounding error problem: when the
scratch CNN sees a slightly unfamiliar state (due to a small mistake), its
features degrade and cause bigger mistakes. Pretrained features are more
**robust** because they were trained on far more visual diversity than our
toy environment can provide.

## Embedding Visualization

Run the script and look at `assets/figures/ch02_tsne_comparison.png`. This shows
a t-SNE plot of vision embeddings colored by the expert's action at each step.

What you should see:
- **Scratch CNN:** noisy clusters with significant overlap between actions.
  The encoder learned *some* structure but the boundaries are fuzzy.
- **CLIP/SigLIP:** tighter, more separated clusters. The pretrained features
  already group similar scenes together, even before any task-specific training.

This visualization is the core insight of this chapter: pretrained features
give the action head a much easier classification problem to solve.

## Live Rollout Viewer

Want to *watch* the models in action? The live viewer opens a Pygame window
showing all 4 encoders attempting MiniPushT **simultaneously**, side by side:

```bash
# All 4 encoders (requires trained checkpoints from --preset full --attn-pool)
python live_rollout.py

# Skip scratch CNN (it just sits there), slower playback
python live_rollout.py --skip-scratch --fps 10

# Try a specific starting seed
python live_rollout.py --seed 10
```

**Controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `R` | Reset with random seed |
| `N` | Next seed (increment by 1) |
| `Q` / `ESC` | Quit |

Each panel shows the environment from one encoder's perspective. Labels turn
**green** on success and **red** on failure. You'll immediately see the contrast:
the scratch CNN barely moves, CLIP confidently pushes the block to the goal,
and SigLIP variants show varying degrees of spatial understanding. Try pressing
`N` a few times to see how consistent each encoder is across different starting
positions.

## Generalization: OOD Evaluation

How robust is each encoder to unseen starting positions? The `--eval-ood` flag
evaluates each trained policy on **out-of-distribution** start positions — agent
and block placed in the outer ring of the environment, positions never seen
during training.

```bash
python compare_encoders.py --preset full --attn-pool --eval-ood
```

Results confirm the intuition from Chapter 1's compounding error analysis:
- **Scratch CNN** (0% OOD): completely fails — fragile features collapse on novel states
- **SigLIP plain** (6% OOD): mean pooling dilutes spatial signal → poor generalization
- **CLIP** (18% OOD): CLS token's selective attention helps, but limited
- **SigLIP attn pool** (30% OOD): learned patch attention captures spatial relationships
  that transfer to novel configurations

## What's Next

- **Chapter 3** swaps the lookup-table language encoder for SmolLM2.
  The model will understand arbitrary instructions, not just three hardcoded strings.
- **Chapter 8** explores fine-tuning these frozen encoders with LoRA.
  Frozen features are good -- but task-adapted features are better.

## Usage & Configuration

### Quick Start

```bash
# Fast experiment (~30s, scratch CNN only, 64x64)
python compare_encoders.py --preset tiny

# Balanced run (~3min, all encoders, 112x112)
python compare_encoders.py --preset medium

# Full comparison (~8min on GPU, all encoders, 224x224)
python compare_encoders.py --preset full
```

### Presets

| Preset | Resolution | Demo Episodes | Epochs | Encoders | Est. Time (GPU) |
|--------|-----------|---------------|--------|----------|-----------------|
| `tiny` | 64x64 | 200 | 10 | Scratch CNN only | ~15s |
| `medium` | 112x112 | 500 | 15 | All 3 (+ attn pool with `--attn-pool`) | ~2min |
| `full` | 224x224 | 1000 | 20 | All 3 (+ attn pool with `--attn-pool`) | ~8min (3hrs with attn-pool) |

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--preset {tiny,medium,full}` | `full` | Load preset defaults |
| `--size N` | from preset | Override image resolution (any integer >= 32) |
| `--n-demos N` | from preset | Override number of expert demo episodes |
| `--epochs N` | from preset | Override training epochs |
| `--batch-size N` | 64 | Training batch size |
| `--lr F` | 1e-3 | Learning rate |
| `--device {cuda,cpu}` | auto-detect | Training device |
| `--skip-pretrained` | from preset | Only run scratch CNN (skip CLIP/SigLIP) |
| `--attn-pool` | off | Include SigLIP with learned attention pooling |
| `--eval-ood` | off | Run OOD generalization eval (corner start positions) |
| `--checkpoint-dir DIR` | `checkpoints/` | Where to save demos and model checkpoints |
| `--force-retrain` | off | Ignore cached model checkpoints, retrain all |
| `--force-recollect` | off | Ignore cached demos, recollect from expert |
| `--no-viz` | off | Skip saving figures |
| `--num-threads N` | 4 | CPU threads (only used when device=cpu) |

Individual flags override preset values. For example, `--preset tiny --epochs 20`
uses the tiny preset but trains for 20 epochs instead of 10.

### Caching

The script caches two things to avoid redundant work:

1. **Expert demos** (`checkpoints/demos_NxN_Mep.npz`): Collected once per
   resolution + episode count combination. Use `--force-recollect` to regenerate.
   For large datasets (>10GB), the script automatically extracts images to an
   uncompressed `.npy` file for memory-mapped loading (avoids RAM explosion).
2. **Trained models** (`checkpoints/encoder_name_Npx_Md_Eep.pt`): Saved after
   training, keyed by resolution + demos + epochs. Use `--force-retrain` to retrain.

On a second run with the same settings, both demo collection and training are
skipped entirely -- the script loads from cache and only runs evaluation.

### What Each Preset Produces

**tiny:** Trains only the scratch CNN at 64x64. Produces a training loss curve
(`assets/figures/ch02_training_curves.png`) and evaluates success rate. No
encoder comparison or t-SNE since only one encoder runs. Best for verifying
the pipeline works.

**medium:** Trains all 3 encoders at 112x112 (4 with `--attn-pool`). Produces
training curves, t-SNE comparison (`assets/figures/ch02_tsne_comparison.png`),
and a results table. Good balance of speed and insight.

**full:** Trains all 3 encoders at 224x224 (4 with `--attn-pool`), the native
CLIP/SigLIP resolution. Produces all visualizations. Note: attn-pool training
is slow (~3hrs on RTX 4090) because it cannot precompute embeddings — the ViT
forward pass runs every batch. Without `--attn-pool`, finishes in ~8min.

### Custom Experiments

```bash
# Run full comparison on CPU
python compare_encoders.py --preset full --device cpu

# Quick scratch CNN at a custom resolution
python compare_encoders.py --size 96 --n-demos 300 --epochs 12 --skip-pretrained

# Force fresh data collection but keep trained models
python compare_encoders.py --force-recollect

# Retrain everything from scratch
python compare_encoders.py --force-retrain --force-recollect
```
