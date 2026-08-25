# Companion blog posts

One post per chapter. They are drafts: the code, numbers, and figures are final,
the prose is meant to be edited before publishing anywhere.

Each post is built to stand alone — you can read #6 without reading #5 — but the
series has a spine, and it is not the usual one. Most tutorials show you a thing
that works. This one is a chain of honest failures, where each chapter's result
is the argument for the next chapter's architecture:

| # | Post | The result that motivates the next chapter |
|---|------|--------------------------------------------|
| 0 | [What a VLA actually is](00-what-is-a-vla.md) | Three design axes, and where the real forks are |
| 1 | [A VLA in 200 lines](01-tiny-vla.md) | 70–80% on a toy task with no pretrained anything |
| 2 | [Borrowing a vision encoder](02-vision-backbone.md) | Frozen CLIP beats a scratch CNN 68% to 0% |
| 3 | [Teaching it to read](03-language-conditioning.md) | Cross-attention loses to concat, 40× slower |
| 4 | [How do you represent an action?](04-action-representations.md) | Chunking doubles success; diffusion scores 0% |
| 5 | [Real robots break everything](05-scaling-up.md) | MAE 0.05 and 0% success, simultaneously |
| 6 | [Flow matching](06-flow-matching.md) | 5× better than DDPM at half the latency |
| 7 | [Assembling the whole thing](07-full-vla.md) | 20.8M trainable of 323.7M; best checkpoint is not the last |
| 8 | [LoRA, from the adapter up](08-lora-finetuning.md) | +5.3% from 0.13% of the parameters |
| 9 | [Latency is the last boss](09-eval-and-deploy.md) | Async: +5.1% overhead vs sync's +22.8% |

## House rules for these posts

- **Report what happened.** Chapter 5 gets 0% and Chapter 6 gets 29% coverage.
  Both numbers are in the posts, unsoftened. A tutorial that only shows wins
  teaches you nothing about what going wrong looks like.
- **Every number is measured**, on hardware named in the post. Nothing is
  estimated or copied from a paper unless it is cited as such.
- **Diagrams before equations.** Figures live in [`../assets/diagrams`](../assets/diagrams)
  and [`../assets/figures`](../assets/figures).
