# Real robots break everything

*Chapter 5 of [vla-from-scratch](https://github.com/FanFeast/vla-from-scratch).*

Chapter 4 ended at 95% success. This chapter changes exactly one thing — real data
instead of a scripted oracle — and the same architecture scores **0%**.

That is the most useful result in the series, so this post is mostly about why.

## The data

Two datasets from the LeRobot Hub:

| Dataset | Episodes | Frames | DOF |
|---|---|---|---|
| `lerobot/pusht` | 206 | 25,650 | 2 |
| `lerobot/aloha_sim_insertion_human` | 50 | 25,000 | 14 |

The critical word is **human**. MiniPushT demos came from a scripted oracle:
deterministic, noiseless, always optimal. PushT demos come from people with a
mouse. They hesitate, overshoot, correct, and take genuinely different routes to
the same goal.

We also add **proprioception** — the robot's own joint state, projected to 64 dims
and concatenated with SigLIP's 768. A real robot knows where its arm is; not using
that would be strange.

## Results (200 epochs, both datasets)

**PushT:**

| Config | Live success | MAE ↓ | Val loss ↓ |
|---|---|---|---|
| Discrete K=1 | 0% | 0.062 | 3.727 |
| Regression K=1 | 0% | **0.052** | **0.005** |
| Regression K=4 | 0% | 0.071 | 0.012 |
| Diffusion K=1 | 0% | 0.636 | 0.093 |

**ALOHA Insertion:**

| Config | Live success | MAE ↓ | Val loss ↓ |
|---|---|---|---|
| Discrete K=1 | 0% | 0.194 | 5.159 |
| Regression K=4 | 0% | **0.049** | 0.006 |
| Diffusion K=4 | 0% | 0.687 | 0.070 |

Every row is 0%.

## Low loss is not task success

Regression reaches **0.052 MAE** on held-out frames and **0% success** closed-loop.
Both numbers are correct. They measure different things:

- **MAE** asks: given this exact frame from a human demo, is your action close to
  what the human did?
- **Rollout** asks: does a chain of 300 of *your own* actions reach the goal?

Small errors compound. Step 1 is slightly off, so step 2 sees a state a bit
outside the demonstrations, so its error is larger, and by step 40 the policy is
in a state no human ever visited — where its predictions were never trained to be
right. This is the classic covariate shift of behavior cloning, and offline
metrics are structurally blind to it.

**If a VLA result does not come from rollouts, it does not mean much.**

## MSE averages modes, and executes the average

Here is the concrete failure. Two demonstrators push the T-block toward the same
goal. One goes around the left side; the other goes right. Both succeed.

MSE regression minimizes squared error against both. The minimizer is the
**mean** of the two trajectories — which goes straight into the block. The policy
executes a path no human ever took, and which does not work.

This is not a bug in the loss. It is what "minimize squared error" *means* when
the target distribution is multi-modal. Chapter 4's regression head won precisely
because MiniPushT had one mode; here there are several.

That is the argument for generative action heads, and it is why Chapter 6 exists.

## Two details that matter more than they look

**Split by episode, not by frame.** Frame 501 and 502 are nearly the same image.
Random frame splitting leaks — half of any validation frame's neighbours are in
training, validation loss looks wonderful, and it means nothing.

**Extract once, cache forever.** Video decoding costs more than SigLIP itself. We
iterate the dataset a single time, extract embeddings + states + actions +
episode indices together, and cache them. Every later experiment reads the cache.
This is the difference between a 20-minute chapter and a four-hour one.

## What is still missing

Diffusion has the lowest val loss on ALOHA (0.070) and still 0% success. The idea
is right; a 2-layer MLP is not enough model.

**Next:** [Chapter 6](06-flow-matching.md) gives it a real transformer, and picks a
better path through noise.
