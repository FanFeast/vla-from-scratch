# Chapter 1: Tiny VLA

**Goal:** Build a complete VLA pipeline from scratch. No pretrained anything.

## Training Recipe

| Parameter | Value |
|-----------|-------|
| Demos | 1000 episodes |
| Transitions | ~30k |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Epochs | 20 |
| Hardware | CPU / Colab T4 |
| Expected time | ~2-3 min |
| Expected success rate | ~70-80% |

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python tiny_vla.py
```

## What We're Building

A VLA takes three inputs and produces one output:
- **Input 1:** camera image (64x64 RGB)
- **Input 2:** language instruction ("push block to goal")
- **Input 3:** (optional) robot state -- we skip this in Chapter 1
- **Output:** action (one of: up, down, left, right, stay)

The data tuple `(image, instruction, action)` is the fundamental unit of every
VLA in this guide. Chapter 1 builds the simplest possible model that learns
from these tuples.

## MiniPushT

`mini_pusht.py` implements a custom `gymnasium.Env`. The world is a 64x64 grid:

- **Agent** (blue square): the robot we control
- **Block** (red square): the object to push
- **Goal** (green square): where the block needs to go

The agent gets +1 reward only when the block overlaps the goal. Each episode
resets with random positions. The scripted expert always solves it in ~20-40 steps.

Why build our own environment? Because every line is transparent. You see exactly
where the image comes from, how the expert generates demonstrations, and how the
training loop connects them. When Chapter 2 swaps in a real vision encoder, the
only thing that changes is `VisionEncoder`.

## The VisionEncoder

Three convolutional layers map the raw 64x64 image to a 128-dimensional vector:

```
Conv(3->32, 3x3, s=2)  # (B, 32, 31, 31)
Conv(32->64, 3x3, s=2) # (B, 64, 15, 15)
Conv(64->64, 3x3, s=2) # (B, 64, 7, 7)
Flatten -> Linear(3136, 128)
```

Each stride-2 convolution halves the spatial dimensions. The final 3136 values
encode the full scene into a 128-dim vector. This encoder trains from scratch --
Chapter 2 shows how much pretrained representations improve on this.

## The LanguageEncoder

```python
class LanguageEncoder(nn.Module):
    VOCAB = {"push block to goal": 0, "move left": 1, "move right": 2}

    def forward(self, instruction: str | list[str]) -> Tensor:
        ...  # returns (B, 128)
```

For now this is a lookup table: three instructions, each mapped to a learned
128-dim vector. The interface (`forward` takes a string, returns a tensor) is
designed so Chapter 3 can swap in a SmolLM2 language model with zero changes
to `TinyVLA.forward`.

## Behavior Cloning

The scripted expert generates demonstrations by watching the ground-truth state
(not the image). It uses a two-phase strategy:
1. Move toward the block
2. Push the block toward the goal

We collect 1000 episodes of `(image, instruction, action)` tuples and train
with cross-entropy loss:

```python
logits = model(image, instruction)   # (B, 5)
loss = cross_entropy(logits, action) # scalar
```

This is identical to supervised text classification -- except the labels are
robot actions.

## Results and Compounding Errors

After training, the policy reaches ~70-80% success rate on episodes with the
same position distribution as training. On slightly different starting positions,
performance drops noticeably.

This is **compounding errors**: at test time, the policy sees states it was never
trained on (because it makes small mistakes), which leads to more unfamiliar states,
which leads to more mistakes. A single 1% error per step compounds to 26% failure
over 200 steps.

This is the fundamental problem with naive behavior cloning, and it motivates
Chapter 2: better vision representations give the model more robust features,
making it less sensitive to distribution shift.

## What Are We Actually Learning From This Experiment?

Run the interactive eval (`python tiny_vla.py --interactive`) and watch the
probability bar chart on the right. You will notice the model looks like it
navigates to the block, then pushes it toward the goal -- as if it learned a
strategy. It did not. Here is what actually happened.

**The expert cheats.** `ScriptedExpert` never sees the image. It reads exact
pixel coordinates from the environment's `info` dict and runs a hardcoded
two-phase rule: move toward block, then push toward goal. It always knows the
shortest path because it always knows exactly where everything is.

**The model copies, not plans.** During training the model sees `(image, action)`
pairs. It learns one thing: *"when the image looks like this, the expert chose
action X."* No coordinates. No distance. No concept of phases. No memory across
steps. Every step is completely independent -- the model looks at one frame and
asks what the expert did when they saw something similar.

```python
# The entire model "thought process" at each step:
logits = model(current_image, instruction)  # one frame, nothing else
action = logits.argmax()                    # highest probability wins
```

The two-phase behavior emerges from imitation, not from understanding. Images
of "agent far from block" are almost always paired with "move toward block"
actions in the training data, so the model absorbs that correlation. It looks
intelligent because it is mimicking someone who was intelligent.

**What the model does not know:**

| | Expert (data source) | Trained model |
|---|---|---|
| Input | Ground-truth coordinates | Raw pixels only |
| Shortest path? | Computes it explicitly | No concept of it |
| Memory across steps? | Yes | No -- each step independent |
| Understands "push"? | Hardcoded rule | Statistical correlation |
| Generalizes to new layouts? | Always | Only if pixels look familiar |

**So what is the experiment actually teaching?**

1. Behavior cloning is just supervised learning. The loss function, training
   loop, and data pipeline are identical to text classification -- only the
   labels are robot actions instead of class names.

2. The gap between the expert and the learned policy reveals exactly what is
   hard about robot learning. The expert used ground-truth state; the model
   must infer everything from pixels. That gap -- state vs. image -- is where
   most of the difficulty lives.

3. The language encoder is a fake. "Push block to goal" is just the integer 0.
   The model has no idea what "push", "block", or "goal" mean. Chapter 3 fixes
   this by swapping in a real language model.

4. Compounding errors are the core failure mode of naive imitation learning.
   A small mistake at step 10 produces a state the model has never seen, which
   causes a bigger mistake at step 11, and so on. This motivates every
   architectural improvement in Chapters 2 through 6.

The point of Chapter 1 is not to build a good robot. It is to build the
simplest possible thing that has all the moving parts -- image, language,
action, training loop -- so that every later chapter is a targeted upgrade
to one specific component, and you can see exactly what each upgrade buys.

## Appendix: Noisy Expert

What happens if training data comes from a noisy expert?

```python
noisy = NoisyExpert(expert, epsilon=0.1)  # 10% random actions
demos = collect_demos(env, noisy, n_episodes=1000)
```

Try it: does noisy data help or hurt success rate? Does it change the compounding
error failure mode? (Answer: slightly worse initial performance but sometimes more
robust to distribution shift.)
