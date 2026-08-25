"""Chapter 8: LoRA fine-tuning of our Chapter 7 SmolVLA.

In Chapter 7 we froze the SigLIP + SmolLM2 backbone and trained a 20.8M
action expert from scratch. That works, but it leaves the language backbone
completely unadapted to the robot domain.

This script shows the alternative: keep the action expert (optionally warm-
started from Ch07) and inject **LoRA adapters into the frozen SmolLM2 layers**
so the backbone itself adapts to manipulation data -- with ~100x fewer
trainable parameters than full fine-tuning.

Two important implementation details that the from-scratch view makes obvious:

1. Chapter 7's `SmolVLA.forward` wraps the VLM call in `torch.no_grad()`,
   because the backbone was frozen. For LoRA we MUST let gradients flow into
   the adapters, so we subclass and drop that `no_grad` guard. The base
   weights stay frozen (requires_grad=False); only the rank-r factors train.

2. The optimizer is given only LoRA parameters (plus, optionally, the action
   expert). Everything else is frozen.

Usage:
    # Reuses the Ch07 cache at ../07_full_vla/checkpoints/vla_cache.pt
    python finetune_smolvla.py --preset so100 --rank 8
    python finetune_smolvla.py --preset so100 --rank 16 --train-expert
    python finetune_smolvla.py --smoke   # tiny wiring check, no downloads
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

# Reuse the Chapter 7 implementation (model, dataset, trainer, helpers).
CH07_DIR = Path(__file__).resolve().parent.parent / "07_full_vla"
sys.path.insert(0, str(CH07_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lora import (  # noqa: E402  (after sys.path setup)
    LoRAConfig,
    count_parameters,
    inject_lora,
    lora_state_dict,
    summarize,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
CH07_CHECKPOINTS = CH07_DIR / "checkpoints"


# ---------------------------------------------------------------------------
# Parameter collection (pure helper, unit-tested)
# ---------------------------------------------------------------------------


def collect_finetune_parameters(
    model: nn.Module, train_expert: bool
) -> list[nn.Parameter]:
    """Return the parameters the optimizer should update.

    Always includes LoRA factors. If train_expert is True, also includes the
    action expert and state projector (the parts trained in Chapter 7), so you
    can compare "LoRA only" against "LoRA + expert".

    Args:
        model: A LoRA-injected SmolVLA (or any module with matching names).
        train_expert: Whether to also train action_expert / state_proj.

    Returns:
        List of nn.Parameter objects with requires_grad enabled.
    """
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        is_expert = "action_expert" in name or "state_proj" in name
        if is_lora or (train_expert and is_expert):
            param.requires_grad = True
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
        elif not is_lora:
            # Keep the rest frozen unless it is an expert weight we opted in to.
            if not (train_expert and is_expert):
                param.requires_grad = False
    return params


# ---------------------------------------------------------------------------
# LoRA-aware SmolVLA (gradients flow through the backbone adapters)
# ---------------------------------------------------------------------------


def build_lora_vla(
    config: "SmolVLAConfig",  # noqa: F821 - imported lazily below
    lora_config: LoRAConfig,
    warm_start: bool = True,
) -> nn.Module:
    """Build a SmolVLA, inject LoRA into the VLM, and optionally warm-start.

    Imports Chapter 7 lazily so that `import finetune_smolvla` is cheap and
    does not trigger a model download (important for the test suite).
    """
    from model import build_smolvla
    from train import CHECKPOINT_DIR as _CH07_CKPT

    model = build_smolvla(config)

    # Optionally warm-start the action expert + state projector from Ch07.
    if warm_start:
        best = _CH07_CKPT / "vla_so100_best.pt"
        if best.exists():
            ckpt = torch.load(best, map_location="cpu", weights_only=False)
            model.state_proj.load_state_dict(ckpt["trainable"]["state_proj"])
            model.action_expert.load_state_dict(
                ckpt["trainable"]["action_expert"]
            )
            print(f"[finetune] Warm-started expert from {best.name}")
        else:
            print("[finetune] No Ch07 checkpoint found; expert starts random")

    # Inject LoRA adapters into the (previously frozen) SmolLM2 layers.
    n_wrapped = inject_lora(model.vlm, lora_config)
    print(f"[finetune] Injected LoRA into {n_wrapped} VLM linear layers")
    print(summarize(model, lora_config))

    # Patch the forward so the VLM runs WITH gradients enabled (so the LoRA
    # adapters receive a learning signal). Chapter 7 used no_grad here.
    model.forward = _lora_forward.__get__(model, type(model))  # type: ignore
    return model


def _lora_forward(
    self: nn.Module,
    vision_tokens: Tensor,
    lang_token_ids: Tensor,
    lang_attention_mask: Tensor,
    states: Tensor,
    noisy_actions: Tensor,
    timesteps: Tensor,
) -> Tensor:
    """SmolVLA.forward variant that lets gradients reach the VLM LoRA adapters.

    Identical to Chapter 7's forward except the VLM call is NOT wrapped in
    torch.no_grad(). Base weights remain frozen via requires_grad=False; only
    the LoRA factors accumulate gradients.
    """
    B = vision_tokens.shape[0]

    # Language + state embeddings. embed_tokens stays frozen (no LoRA there).
    with torch.no_grad():
        lang_embeds = self.vlm.embed_tokens(lang_token_ids)
    state_token = self.state_proj(states).unsqueeze(1)

    prefix = torch.cat([vision_tokens, lang_embeds, state_token], dim=1)

    vision_mask = torch.ones(
        B, vision_tokens.shape[1],
        device=vision_tokens.device, dtype=lang_attention_mask.dtype,
    )
    state_mask = torch.ones(
        B, 1, device=vision_tokens.device, dtype=lang_attention_mask.dtype
    )
    prefix_mask = torch.cat([vision_mask, lang_attention_mask, state_mask], dim=1)

    # NOTE: no torch.no_grad() here -- this is the whole point of LoRA tuning.
    vlm_out = self.vlm(
        inputs_embeds=prefix,
        attention_mask=prefix_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    ca_features = [
        vlm_out.hidden_states[i] for i in self.config.ca_layer_indices
    ]
    return self.action_expert(noisy_actions, timesteps, ca_features)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def evaluate_val_loss(trainer: "FlowMatchingVLA", val_loader: DataLoader) -> float:  # noqa: F821
    """Mean flow-matching loss over the validation set (model set to eval)."""
    trainer.model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            with torch.amp.autocast(
                "cuda", dtype=DTYPE, enabled=(DEVICE.type == "cuda")
            ):
                total += trainer.compute_loss(batch, DEVICE).item()
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def finetune(
    preset_name: str,
    rank: int,
    alpha: int,
    train_expert: bool,
    epochs: int,
    warm_start: bool,
) -> None:
    """Run LoRA fine-tuning on the Chapter 7 SO100 cache."""
    from config import PRESETS, SmolVLAConfig
    from train import (
        FlowMatchingVLA,
        VLADataset,
        compute_norm_stats,
        extract_and_cache,
        split_by_episode,
    )

    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}")
    preset = PRESETS[preset_name]
    config = SmolVLAConfig(
        action_dim=preset["action_dim"],
        state_dim=preset["state_dim"],
        chunk_size=preset["chunk_size"],
        epochs=epochs,
        batch_size=preset["batch_size"],
    )
    lora_config = LoRAConfig(r=rank, alpha=alpha, target_modules=("q_proj", "v_proj"))

    print(f"\n{'=' * 60}")
    print(f"  Chapter 8: LoRA fine-tuning -- {preset_name}")
    print(f"{'=' * 60}")
    print(f"  LoRA rank={rank}, alpha={alpha}, train_expert={train_expert}")
    print(f"  Device: {DEVICE}\n")

    model = build_lora_vla(config, lora_config, warm_start=warm_start)

    # Load (or build) the cached vision tokens shared with Chapter 7.
    model.vision_encoder.to(DEVICE)
    model.connector.to(DEVICE)
    data = extract_and_cache(
        model, preset["datasets"], config, CH07_CHECKPOINTS, skip=True
    )
    model.vision_encoder.cpu()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    state_stats = compute_norm_stats(data["states"])
    action_stats = compute_norm_stats(data["actions"])
    norm_states = state_stats.normalize(data["states"])
    norm_actions = action_stats.normalize(data["actions"])
    train_mask, val_mask = split_by_episode(
        data["episode_indices"], config.val_ratio
    )

    def make_ds(mask: Tensor) -> VLADataset:
        return VLADataset(
            vision_tokens=data["vision_tokens"][mask],
            lang_token_ids=data["lang_token_ids"][mask],
            lang_attention_mask=data["lang_attention_mask"][mask],
            states=norm_states[mask],
            actions=norm_actions[mask],
            episode_indices=data["episode_indices"][mask],
            chunk_size=config.chunk_size,
        )

    train_ds, val_ds = make_ds(train_mask), make_ds(val_mask)

    model.vlm.to(DEVICE)
    model.connector.to(DEVICE)
    model.state_proj.to(DEVICE)
    model.action_expert.to(DEVICE)

    params = collect_finetune_parameters(model, train_expert)
    counts = count_parameters(model)
    print(
        f"[finetune] Optimizing {sum(p.numel() for p in params) / 1e6:.2f}M "
        f"params (LoRA={counts['lora'] / 1e6:.2f}M)"
    )

    trainer = FlowMatchingVLA(model, config)
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    # Baseline: with zero-init LoRA (B=0) and the frozen warm-started expert,
    # the model at step 0 is byte-for-byte the Chapter 7 frozen-backbone policy.
    # This epoch-0 val loss is the reference the LoRA-adapted result is measured
    # against -- on the identical split, norm stats, and code path.
    baseline_val = evaluate_val_loss(trainer, val_loader)
    print(f"[finetune] Baseline (Ch07 frozen backbone) val loss: {baseline_val:.5f}")

    best_val = float("inf")
    history: list[dict[str, float]] = [
        {"epoch": 0, "train_loss": float("nan"), "val_loss": baseline_val, "time_s": 0.0}
    ]
    tag = f"{preset_name}_r{rank}" + ("_expert" if train_expert else "")
    for epoch in range(epochs):
        model.train()
        model.vision_encoder.eval()  # frozen, no LoRA
        t0 = time.time()
        total, n = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(DEVICE.type == "cuda")):
                loss = trainer.compute_loss(batch, DEVICE)
            loss.backward()
            nn.utils.clip_grad_norm_(params, config.grad_clip_norm)
            optimizer.step()
            total += loss.item()
            n += 1
        train_loss = total / max(n, 1)

        val_loss = evaluate_val_loss(trainer, val_loader)
        epoch_time = time.time() - t0
        print(
            f"  Epoch {epoch + 1:3d}/{epochs} | train={train_loss:.5f} | "
            f"val={val_loss:.5f} | {epoch_time:.1f}s"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_s": epoch_time,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            adapter_path = CHECKPOINT_DIR / f"lora_{tag}.pt"
            torch.save(
                {
                    "lora": lora_state_dict(model),
                    "expert": (
                        {
                            "state_proj": model.state_proj.state_dict(),
                            "action_expert": model.action_expert.state_dict(),
                        }
                        if train_expert
                        else None
                    ),
                    "lora_config": lora_config,
                    "norm": {"state": state_stats, "action": action_stats},
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "history": history,
                },
                adapter_path,
            )

    # Persist the full loss history for plotting/reporting (matches the
    # per-chapter convention of a reproducible training curve artifact).
    history_path = CHECKPOINT_DIR / f"lora_{tag}_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "preset": preset_name,
                "rank": rank,
                "alpha": alpha,
                "train_expert": train_expert,
                "warm_start": warm_start,
                "epochs": epochs,
                "best_val_loss": best_val,
                "trainable_params": sum(p.numel() for p in params),
                "lora_params": counts["lora"],
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"\n[finetune] Best val loss: {best_val:.5f}")
    print(f"[finetune] Adapter saved to {CHECKPOINT_DIR}")
    print(f"[finetune] History saved to {history_path}")


# ---------------------------------------------------------------------------
# Smoke test (no downloads): validate the LoRA training wiring end to end
# ---------------------------------------------------------------------------


def smoke() -> None:
    """Validate parameter collection + a LoRA gradient step on a tiny stand-in.

    Builds a small module that mimics the attention-projection naming, injects
    LoRA, runs one optimizer step, and checks only LoRA params changed.
    """
    torch.manual_seed(0)

    class TinyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(16, 16)
            self.v_proj = nn.Linear(16, 16)
            self.action_expert = nn.Linear(16, 4)

        def forward(self, x: Tensor) -> Tensor:
            return self.action_expert(self.q_proj(x) + self.v_proj(x))

    model = TinyBackbone()
    inject_lora(model, LoRAConfig(r=4, alpha=8, target_modules=("q_proj", "v_proj")))
    params = collect_finetune_parameters(model, train_expert=False)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.SGD(params, lr=0.1)
    x = torch.randn(8, 16)
    target = torch.randn(8, 4)
    loss = F.mse_loss(model(x), target)
    loss.backward()
    opt.step()

    changed = [
        n for n, p in model.named_parameters()
        if not torch.equal(p.detach(), before[n])
    ]
    assert all("lora_" in n for n in changed), f"Non-LoRA weights moved: {changed}"
    assert changed, "No LoRA weights updated"
    print(f"[smoke] OK -- updated only LoRA params: {changed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 8: LoRA fine-tuning")
    parser.add_argument("--preset", default="so100")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-expert", action="store_true")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny wiring check")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return

    finetune(
        preset_name=args.preset,
        rank=args.rank,
        alpha=args.alpha,
        train_expert=args.train_expert,
        epochs=args.epochs,
        warm_start=not args.no_warm_start,
    )


if __name__ == "__main__":
    main()
