# Chapter 5: Scaling Up — From Toy Environments to Real Robot Data

**Goal:** Bridge the gap from toy MiniPushT experiments (Chapters 1-4) to real robot
data from the [LeRobot Hub](https://huggingface.co/lerobot). Train the same Ch04-style
action heads on two real datasets — **PushT** (warm-up, familiar task) and
**ALOHA insertion** (reality check, 14-DOF bimanual) — and discover that simple MLP
heads fail on real robot complexity.

## Training Recipe

| Parameter | Value |
|-----------|-------|
| Vision encoder | SigLIP-B/16 (frozen, 768D) |
| Proprio encoder | Linear(state_dim, 64) + ReLU |
| Fused features | 768 + 64 = 832D |
| Epochs | 200 max |
| Early stopping | patience=20 |
| Learning rate | 1e-3 (cosine decay to 0) |
| Batch size | 128 |
| Mixed precision | bf16 AMP |
| Gradient clipping | max_norm=1.0 |
| Configs per dataset | 6 (3 heads x 2 chunk sizes) |
| Hardware | RTX 4090 (24GB) |

## Setup

```bash
cd chapters/05_scaling_up

# Install dependencies (LeRobot + gym environments)
# - the [dataset] extra provides LeRobotDataset, which extract_and_cache imports
# - gym-pusht still calls pymunk's add_collision_handler, removed in pymunk 7
pip install 'lerobot[dataset]' gym-pusht gym-aloha 'pymunk<7'

# Run PushT experiments (~25 min: 1 min extract + 6 min train + 18 min live eval)
python data_pipeline.py --preset pusht

# Run ALOHA experiments (~30 min: 3 min extract + 8 min train + 19 min live eval)
python data_pipeline.py --preset aloha

# Re-run with cached embeddings (skip SigLIP extraction)
python data_pipeline.py --preset pusht --skip-precompute

# Evaluation only (load trained checkpoints)
python data_pipeline.py --preset pusht --eval-only
```

First run downloads datasets from HuggingFace Hub and extracts SigLIP embeddings
(cached as `.pt` files in `checkpoints/` for reuse).

## What Changed From Chapter 4

1. **Real data.** No more scripted demonstrations — we load human teleoperation data
   from the LeRobot Hub. Videos are decoded frame-by-frame, SigLIP embeddings are
   extracted in a single pass, then cached for fast reuse.

2. **Proprioception.** The model now receives robot state (2D agent position for PushT,
   14D joint positions for ALOHA) through a learned `ProprioEncoder`, concatenated with
   vision features: `fused = [SigLIP_768D | proprio_64D] = 832D`.

3. **Episode-based splits.** Train/val splits use whole episodes (not random frames)
   to prevent data leakage — adjacent frames are nearly identical, so random splitting
   would inflate validation metrics.

4. **Two datasets.** PushT is a familiar warm-up (2D actions, 96x96 images, 206
   episodes). ALOHA insertion is the reality check (14D bimanual actions, 480x640
   images, 50 episodes).

5. **Live rollout evaluation.** Models are tested by actually running in the simulator
   (gym-pusht, gym-aloha) with real-time SigLIP encoding per frame — not just offline
   loss metrics.

## The Datasets

### PushT (lerobot/pusht)

The same T-shaped block pushing task from earlier chapters, but now with
**206 episodes of human teleoperation data** (25,650 frames) from the LeRobot Hub.
Actions are 2D pixel coordinates controlling the end-effector position.

- **Observation:** 96x96 RGB image + 2D agent position
- **Action:** 2D absolute pixel coordinates [0, 512]
- **Task:** Push the T-block onto the T-shaped target

### ALOHA Insertion (lerobot/aloha_sim_insertion_human)

A bimanual peg insertion task with **50 episodes** (25,000 frames) of human
teleoperation. Two 7-DOF robot arms must coordinate to insert a peg into a socket.

- **Observation:** 480x640 RGB image (top camera) + 14D joint positions
- **Action:** 14D joint velocities [-1, 1]
- **Task:** Insert the peg into the socket using both arms

This is a massive jump in complexity: 7x more action dimensions, visual complexity
from two robot arms, and precise bimanual coordination where small errors cascade.

## The Data Pipeline

The key engineering challenge is **efficiency**. Each LeRobot frame access decodes
video on-the-fly, so we must minimize dataset iterations:

```
LeRobot Hub
    │
    ▼
extract_and_cache()     ← SINGLE PASS: decode video + SigLIP + states + actions
    │
    ├── pusht_embeddings.pt    (25650, 768) float32 = ~75MB
    ├── pusht_data.pt          (states, actions, episode_indices)
    │
    ▼
RobotDataset            ← Pure tensor indexing, no I/O during training
    │
    ▼
DataLoader → train()    ← Standard PyTorch training loop
```

**Why single-pass matters:** A naive approach that iterates the dataset separately
for embedding extraction, normalization stats, and training would decode each video
frame 3-5 times. With 25k frames of video decoding, that's hours vs. minutes.

## Architecture

Same three action heads from Chapter 4, now with proprioception:

```
Image ──→ SigLIP (frozen) ──→ 768D ──┐
                                      ├──→ 832D fused ──→ Action Head ──→ actions
State ──→ ProprioEncoder ──→ 64D ──┘

Action Heads:
  Discrete:   832D → MLP → K*D*256 logits → argmax → de-quantize
  Regression: 832D → MLP (256→128) → Tanh → K*D continuous
  Diffusion:  832D → DDPM denoiser (100 train / 10 inference steps) → K*D continuous
```

## Results

### PushT (206 episodes, 25,650 frames)

| Config | Live Success | MAE ↓ | Val Loss ↓ | Epochs |
|--------|-------------|-------|-----------|--------|
| Discrete K=1 | 0% | 0.062 | 3.727 | 33 |
| Discrete K=4 | 0% | 0.085 | 4.008 | 30 |
| Regression K=1 | 0% | **0.052** | **0.005** | 26 |
| Regression K=4 | 0% | 0.071 | 0.012 | 51 |
| Diffusion K=1 | 0% | 0.636 | 0.093 | 62 |
| Diffusion K=4 | 0% | 0.651 | 0.097 | 61 |

### ALOHA Insertion (50 episodes, 25,000 frames)

| Config | Live Success | MAE ↓ | Val Loss ↓ | Epochs |
|--------|-------------|-------|-----------|--------|
| Discrete K=1 | 0% | 0.194 | 5.159 | 21 |
| Discrete K=4 | 0% | 0.200 | 5.195 | 21 |
| Regression K=1 | 0% | 0.051 | 0.006 | 31 |
| Regression K=4 | 0% | **0.049** | **0.006** | 46 |
| Diffusion K=1 | 0% | 0.674 | 0.085 | 63 |
| Diffusion K=4 | 0% | 0.687 | 0.070 | 80 |

*MAE = Mean Absolute Error on normalized [-1, 1] actions (lower = better).
Live Success = success rate over 50 rollout episodes in the simulator.*

## Key Takeaways

### 1. The Wall: Low Loss != Task Success

The most striking result: **regression achieves excellent offline metrics (MAE ~0.05)
but 0% live success** on both tasks. The model learns to predict "close to the right
action" but can't actually solve the task.

**Why:** Real robot tasks require **temporal coherence** over hundreds of steps.
Even 5% average error compounds — after 300 steps, the robot has drifted far from
any demonstrated trajectory. An MLP that maps each frame independently to an action
has no concept of trajectory-level planning.

### 2. Real Data is Harder Than Toy Data

In Chapter 4, regression K=4 hit **95% success** on MiniPushT with scripted demos.
Here, the same architecture gets **0%** on the PushT task with human demos.

The difference:
- **Human demonstrations are noisy.** Scripted oracles produce clean, optimal
  trajectories. Humans hesitate, overshoot, and take suboptimal paths.
- **Multi-modal behavior.** Different human demonstrators solve the task differently.
  Regression averages these modes and produces actions that follow none of them.
- **No error recovery.** The MLP never sees recovery behavior in training data,
  so any deviation from the demonstrated state distribution leads to cascading failure.

### 3. Discrete Tokenization Struggles with High-DOF

Discrete heads plateau at much higher loss on ALOHA (14D) vs PushT (2D). With 14
action dimensions x 256 bins = 3,584 independent classification problems per timestep,
the model spreads its capacity too thin. The 256-bin quantization also becomes coarser
as dimensionality increases.

### 4. Diffusion Has Potential But Needs Scale

The diffusion head converges to the lowest validation loss on ALOHA (0.070 for K=4),
but its high MAE reflects the mismatch between diffusion's strengths and our setup:
- Our 2-layer MLP denoiser is far too small for 14D action spaces
- 10 DDIM-style inference steps produce noisy samples from this undertrained model
- The distributional modeling power of diffusion only pays off with a proper
  transformer-based denoiser and flow matching (which we'll build in **Chapter 6**)

### 5. The Case for Specialized Action Experts

This chapter demonstrates why production VLAs (pi0, SmolVLA) use dedicated
**action expert** networks — typically 100M+ parameter transformers with
cross-attention to vision/language features — rather than simple MLP action heads.

The leap from "predicts roughly the right action" (low MAE) to "actually solves
the task" (non-zero success rate) requires:
- **Temporal reasoning** via attention over action sequences
- **Multi-modal distribution modeling** via flow matching
- **Scale** — orders of magnitude more denoiser parameters

We'll build exactly this in **Chapter 6: Flow Matching**.

## What's Next

**Chapter 6** replaces the toy MLP denoiser with a proper **flow-matching action
expert** — a ~100M parameter transformer with interleaved cross-attention and
self-attention blocks, trained with rectified flow instead of DDPM. This is the
architecture that makes diffusion-based VLAs (pi0, SmolVLA) actually work.

## File Structure

```
chapters/05_scaling_up/
  data_pipeline.py       # Everything: extraction, models, training, eval, CLI
  test_chapter_05.py     # 18 unit tests (models, normalization, splitting, dataset)
  checkpoints/           # Cached embeddings + trained models (gitignored)
  .gitignore             # Excludes checkpoints/
```
