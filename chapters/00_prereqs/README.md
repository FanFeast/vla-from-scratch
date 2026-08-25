# Chapter 0: Prerequisites

> **What you need to know before starting:**
> PyTorch basics, how transformers work (attention, encoder/decoder),
> and what a vision encoder does (e.g. you've heard of CLIP or ViT).
> You do not need to know anything about robotics or VLAs.

---

In 2023, a team at Google asked a deceptively simple question: a robot's next
move is just a sequence of numbers. A language model predicts the next token in
a sequence of numbers. What if you trained the language model to predict robot
actions the same way?

That question produced RT-2 -- and with it, the **Vision-Language-Action model**
(VLA). The insight was not complicated. The consequences were. This guide builds
the answer from scratch, one component at a time. This chapter is the map before
the territory.

## What is a VLA?

A Vision-Language-Action model takes three inputs and produces one output.
Inputs: a camera image, a natural-language instruction ("pick up the red cup"),
and optionally the robot's current joint state. Output: an action -- the next
move the robot should make.

The pipeline has three stages: **see** (encode the image into a vector),
**read** (encode the instruction into a vector), **act** (fuse both vectors and
decode an action). Everything in this guide is about making each of those three
stages better.

If you have used a **VLM** (vision-language model) like GPT-4V or LLaVA, you
already know 80% of a VLA. A VLM takes an image and text as input and outputs
text. A VLA does the same, but outputs robot actions instead of words. The
remaining 20% -- the part that is not obvious -- is how those actions are
represented. That is the subject of Chapter 4, and it is where most of the
interesting architectural decisions live.

## Key Design Decisions

Every VLA paper makes three foundational choices. Understanding them lets you
read any VLA paper and immediately know which "family" it belongs to.

**1. Discrete vs continuous actions.**
Do you represent robot movements as discrete tokens from a fixed vocabulary
(e.g., bin each joint angle into one of 256 values) or as raw floating-point
numbers? Discrete tokens make the action prediction problem identical to
next-token prediction in a language model -- easy to train, fast to sample,
but limited in precision. Continuous actions are more expressive but require
a different training objective (regression or diffusion rather than
cross-entropy). RT-2 and OpenVLA use discrete. pi0 and SmolVLA use continuous.

**2. Single-system vs dual-system.**
Does one transformer handle everything -- language understanding, visual
reasoning, and action generation -- or do you use two separate models: a
language backbone for thinking and a dedicated action expert for moving?
Single-system is simpler and easier to train end-to-end. Dual-system lets you
independently scale the language brain (pretrained, frozen) and the motor
expert (trained from scratch on robot data). RT-2 and OpenVLA are
single-system; pi0 and SmolVLA are dual-system. The dual-system approach is
where the field is currently moving.

**3. Single-step vs action chunking.**
Do you predict only the very next robot command, or do you predict a chunk of
the next K commands at once? Single-step is simpler. Chunking produces smoother
trajectories (each action in the chunk is consistent with its neighbors),
handles the latency of running inference on a real robot, and helps the model
commit to a strategy rather than being indecisive step by step. Most modern
VLAs use chunking with K between 10 and 100 -- ACT uses K=100, SmolVLA uses
K=50.

## The VLA Landscape

Six models define the current VLA space. Three shaped the architecture this
guide builds toward; three others are worth knowing by name.

### The three you need to understand deeply

**RT-2** ([Zitkovich et al., 2023](https://arxiv.org/abs/2307.15818)) is where
VLAs began. Google took PaLM-E, a 55-billion-parameter vision-language model,
and fine-tuned it to predict discretized robot actions alongside language tokens.
The model was impractical to run -- 55B parameters on real-robot inference is
a serious engineering problem -- but the idea was the unlock: language
pretraining at internet scale transfers to robotic manipulation. Everything
since has been about making that idea practical.

**pi0** ([Black et al., 2024](https://arxiv.org/abs/2410.24164)) from Physical
Intelligence introduced the dual-system architecture: a large VLM backbone
handles perception and language understanding, while a separate flow-matching
diffusion model generates the actual actions. The action expert is trained to
denoise a trajectory from Gaussian noise, conditioned on the VLM's output.
This produces smooth, precise movements and handles the multimodality of robot
tasks well (multiple valid ways to complete the same instruction). pi0 is the
direct architectural ancestor of what we build in Chapter 7.

**SmolVLA** ([Shukor et al., 2025](https://arxiv.org/abs/2506.01844)) is
HuggingFace's compact implementation of the pi0 idea. At 450M parameters, it
runs on a single GPU, trains on the LeRobot dataset format, and achieves
competitive results on standard benchmarks. SmolVLA is the target architecture
for this guide -- Chapter 7 builds a version of it from scratch.

### Three others worth knowing

**OpenVLA** ([Kim et al., 2024](https://arxiv.org/abs/2406.09246)) -- Stanford's
open-source 7B VLA using discrete action tokens. The main open-source baseline
for discrete-action VLAs and the reference point for comparisons in Chapter 4.

**Helix** (NVIDIA, 2024) -- A dual-system VLA targeting humanoid robots with
high degree-of-freedom manipulation. Notable for its work on whole-body
coordination and real-time inference at humanoid-robot speeds.

**GR00T N1** (NVIDIA, 2024) -- NVIDIA's foundation model for humanoid
generalist behavior, trained on a broad mixture of real and synthetic robot
data. Represents the current push toward cross-embodiment generalization.

For a structured reading list across the full VLA literature, see
[awesome-vla-study](https://github.com/MilkClouds/awesome-vla-study).

## What We Build

Each chapter adds one component to the architecture. Nothing is thrown away --
every addition builds directly on the previous one.

**Chapter 1** starts from scratch: a 3-layer CNN encodes the image, a small
learned embedding table encodes the instruction, and an MLP predicts the next
discrete action. The full model fits in your head. It trains in minutes on a
CPU.

**Chapter 2** swaps the CNN for a frozen pretrained vision encoder (CLIP or
SigLIP). The rest stays the same. The goal is to see exactly how much
pretrained representations improve performance -- and why we never go back to
training vision from scratch.

**Chapter 3** swaps the embedding table for a frozen SmolLM2 language model
and adds cross-attention between the vision and language features. Now the
model actually understands language, and can generalize to instruction phrasings
it has not seen before.

**Chapter 4** is the pivotal chapter: we build three action heads on the same
backbone and compare them directly. Discrete tokens (RT-2 style), continuous
MSE regression, and flow-matching diffusion. This is where the "20%" that
separates VLMs from VLAs becomes concrete.

**Chapter 5** scales to real robot data. We use the same architecture from
Chapter 4 but train on a subset of Open X-Embodiment via the LeRobot dataset
format. The data engineering problem becomes real.

**Chapter 6** replaces the MLP action head with a proper flow-matching action
expert: a ~100M-parameter transformer with interleaved cross-attention and
causal self-attention, generating 10-step action chunks conditioned on VLM
features.

**Chapter 7** is the full assembly: SigLIP vision encoder, SmolLM2 language
backbone with layer skipping, and the flow-matching action expert from
Chapter 6. This is SmolVLA built from scratch.

**Chapter 8** covers fine-tuning: LoRA adapting a pretrained VLA (OpenVLA or
SmolVLA) to a new task without retraining from scratch, and when fine-tuning
beats training from scratch.

**Chapter 9** closes the loop: evaluation on the LIBERO benchmark, what an
async inference stack looks like, and the open problems that define where the
field is heading next.

Every chapter adds one piece. Nothing is thrown away.
