# LoRA, from the adapter up

*Chapter 8 of [vla-from-scratch](https://github.com/FanFeast/vla-from-scratch).*

Chapter 7 froze SmolLM2 completely. That backbone was pretrained on web text — it
has never seen a robot arm. **LoRA** lets us adapt it with 0.13% of the parameters.

We implement it from scratch, because it is two matrices and a scalar, and doing
so makes the `peft` API read like a config file instead of magic.

## The adapter

![LoRA adapter](../assets/diagrams/ch08_lora.svg)

A frozen linear layer `W` gets a parallel low-rank branch:

```
h = Wx + (alpha / r) * B A x        A: (r, d_in)    B: (d_out, r)
```

Two details do the work.

**`B` is zero-initialized**, so `BA = 0` at step 0 and the adapted model starts
*exactly* equal to the original. No warm-up shock, no risk of destroying a good
backbone on the first batch.

```python
>>> layer = LoRALinear(base, LoRAConfig(r=8, alpha=16))
>>> torch.allclose(layer(x), base(x), atol=1e-6)
True
```

**The `alpha/r` scaling** means changing `r` does not force you to retune the
learning rate — the branch's contribution is normalized by rank.

## Injection targets q and v

`inject_lora` walks the module tree and wraps every `nn.Linear` whose name
contains one of `target_modules`. For a Llama-style backbone — which SmolLM2 is —
those are `q_proj`, `k_proj`, `v_proj`, `o_proj`.

Targeting **q and v only** is the [LoRA paper's](https://arxiv.org/abs/2106.09685)
finding: most of the benefit at half the parameters of all four.

## Verify the freeze, do not trust it

"The base model is frozen" deserves a check. Take one optimizer step and compare
every parameter:

```
parameters that changed:   12
parameters that did not:   60
all changed are LoRA:      True
```

There is a subtlety worth noticing here. On the *first* step, only `lora_B` moves —
never `lora_A`. The gradient with respect to `A` is proportional to `B`, and `B`
starts at zero. `A` only begins moving once `B` is non-zero.

## Merging away the overhead

At inference you do not want an extra matmul per layer. The branch is linear, so
fold it:

```
W' = W + (alpha / r) B A
```

```
merged adapters: 12
outputs match:   True
max difference:  2.24e-08
```

Identical outputs, zero added latency, and the model is a plain `nn.Linear` again.

## Does it actually help?

One fine-tune against one baseline proves nothing — the gap could be sampling
noise. `verify_baseline.py` evaluates both on **8 seeds with identical noise draws
per seed**, so the comparison is paired.

| Model | Val loss (8 seeds) | Trainable |
|---|---|---|
| Ch07 frozen backbone | 0.13474 ± 0.00123 | 0 |
| + r=8 LoRA on backbone | **0.12763 ± 0.00129** | 0.41M (0.13%) |
| **Paired improvement** | **0.00711 ± 0.00014 (+5.3%)** | |

The paired delta is **~50× its own standard deviation**. That is the difference
between a result and a hopeful anecdote, and it costs one afternoon of extra
compute to establish.

## It overfits fast, and one run diverged

Best validation at **epoch ~6**, then it drifts back above baseline. 158 episodes
is not much data for even 0.41M new parameters. Early stopping is not optional.

We also tried `--train-expert` (r=16, ~21M trainable). It **diverged around epoch
25** under Chapter 7's constant learning rate with no warmup. It is left in the
repo as an exercise rather than quietly dropped — larger adapters need a schedule
the original recipe does not provide, and pretending every experiment worked would
be dishonest about what this work looks like.

## The 7B recipe

`lora_openvla.py` holds the canonical [OpenVLA](https://arxiv.org/abs/2406.09246)
recipe: `peft` + 4-bit NF4, `r=32`, adapters on all linear layers.

| Approach | Trainable | % of model |
|---|---|---|
| Ch07 action expert from scratch | 20.8M | 6.4% |
| Ch08 LoRA on backbone (r=8, q/v) | 0.41M | 0.13% |
| OpenVLA-7B full fine-tune | 7B | 100% |
| OpenVLA-7B LoRA (r=32) | ~110M | ~1.5% |

Full 7B fine-tuning needs ~140 GB of optimizer state. LoRA at 4-bit fits on one
24 GB GPU. That is the difference between needing a cluster and needing a desk.

**Next:** [Chapter 9](09-eval-and-deploy.md) asks the question that finally decides
everything — how fast is it?
