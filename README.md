# VLA From Scratch

A build-it-yourself tutorial for **Vision-Language-Action
(VLA)** models. Start with a 3-layer CNN on a toy pushing task and end with a
SmolVLA-like architecture — SigLIP vision + SmolLM2 language + a flow-matching
action expert — trained on real robot data, LoRA fine-tuned, and evaluated
closed-loop.

Every chapter adds exactly one concept, with runnable code at every step.
nanoGPT did this for LLMs; this does it for VLAs.

## Who it's for

ML and robotics engineers who know PyTorch and basic transformers but have
never built a VLA. Chapters 0-4 run on a Colab T4; later chapters use a single
A100/4090.

![Anatomy of a VLA](assets/diagrams/vla_anatomy.svg)

## The arc

Each chapter is self-contained with its own README, runnable scripts, tests,
and a locked `uv` environment.

| # | Chapter | Adds | Status |
|---|---------|------|--------|
| 0 | [Prerequisites](chapters/00_prereqs) | VLA taxonomy, design decisions | done |
| 1 | [Tiny VLA](chapters/01_tiny_vla) | 3-layer CNN + MLP action head on MiniPushT | done |
| 2 | [Vision Backbone](chapters/02_vision_backbone) | Frozen CLIP / SigLIP vs scratch CNN | done |
| 3 | [Language Conditioning](chapters/03_language_conditioning) | SmolLM2 + cross-attention fusion | done |
| 4 | [Action Representations](chapters/04_action_representations) | Discrete vs regression vs diffusion heads, action chunking | done |
| 5 | [Scaling Up](chapters/05_scaling_up) | Real robot data (PushT, ALOHA) via LeRobot | done |
| 6 | [Flow Matching](chapters/06_flow_matching) | Transformer action expert, flow matching vs DDPM | done |
| 7 | [Full VLA](chapters/07_full_vla) | SigLIP + SmolLM2 + flow-matching expert, assembled | done |
| 8 | [Fine-tuning](chapters/08_finetuning) | LoRA from scratch + OpenVLA-7B recipe | done |
| 9 | [Eval & Deploy](chapters/09_eval_and_deploy) | LIBERO eval + async action-chunking inference | done |

## Final architecture (Chapter 7)

```
Image    -> SigLIP (frozen) -> pixel shuffle 4x4 -> connector -> 64 tokens x 960
Language -> SmolLM2 tokenizer ----------------------------------\
State    -> Linear(6 -> 960) -> 1 token -------------------------> prefix
Prefix   -> SmolLM2 (frozen, first 16/32 layers) -> hidden states @ L5, L10, L16
Actions  -> Flow-matching action expert (trainable, interleaved SA/CA) -> chunk (10 x 6)
```

Only the action expert + state projector train (~20M of ~324M params). Chapter
8 then shows how to adapt the frozen backbone with LoRA, and Chapter 9 runs the
result in real time.

## Quickstart

Each chapter has its own `uv`-managed environment with a committed lockfile:

```bash
cd chapters/01_tiny_vla
uv sync                       # install exact locked deps into .venv/
uv run python tiny_vla.py     # run a script
uv run pytest -q              # sanity tests
```

No global install — every chapter lists only what it needs.

Or open that chapter's `notebook.ipynb` in Colab: every chapter except Ch 0
(prose only) ships one, and each opens with its own `pip install` cell.

## Repo layout

```
chapters/XX_name/   README + notebook.ipynb + scripts + test_chapter_XX.py
                    + pyproject.toml + uv.lock
environments/       MiniPushT toy environment
assets/             diagrams and figures
blog/               companion post drafts
```

## Companion blog posts

One draft post per chapter in [`blog/`](blog/), written around the measured
results rather than the happy path — Chapter 5 scores 0% and says so.

## Key references

- [SmolVLA](https://arxiv.org/abs/2506.01844) — primary architecture reference (Ch 7)
- [OpenVLA](https://arxiv.org/abs/2406.09246) — open 7B VLA, LoRA fine-tuning (Ch 8)
- [pi0](https://arxiv.org/abs/2410.24164) — flow-matching action expert (Ch 6-7)
- [ACT](https://arxiv.org/abs/2304.13705) — action chunking + temporal ensembling (Ch 9)
- [RT-2](https://arxiv.org/abs/2307.15818) — coined the "VLA" term (Ch 0, 4)

Each chapter's README carries its own training recipe, results table, and
walkthrough — start with [Chapter 0](chapters/00_prereqs) or jump to whichever
component you came for.
