"""Action Expert Transformer for flow-matching and DDPM denoising.

A ~100M parameter transformer denoiser with interleaved self-attention and
cross-attention blocks.  The model takes noisy action chunks, a scalar timestep,
vision conditioning embeddings, and proprioceptive state, then predicts the
velocity field (flow matching) or noise (DDPM).

Architecture follows SmolVLA / pi0 style:
  - Sinusoidal timestep embedding + MLP
  - Learnable positional embedding over the action chunk
  - N blocks of [Self-Attn, Cross-Attn, FFN] with pre-LayerNorm + residual
  - Zero-initialized output projection for stable training start
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Map scalar timestep (B,) to (B, d_model) via sinusoidal encoding + MLP.

    Works for both continuous t in [0, 1] (flow matching) and discrete t (DDPM).
    """

    def __init__(self, d_model: int, max_period: int = 10000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Compute timestep embedding.

        Args:
            t: Scalar timesteps of shape (B,).

        Returns:
            Embedding of shape (B, d_model).
        """
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embedding)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class ActionExpertBlock(nn.Module):
    """One transformer block: Self-Attn + Cross-Attn + FFN, all pre-LayerNorm."""

    def __init__(
        self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        # Self-attention
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Cross-attention
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Feed-forward network
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Run one transformer block.

        Args:
            x: Action tokens of shape (B, K, d_model).
            cond: Conditioning tokens of shape (B, C, d_model).

        Returns:
            Updated action tokens of shape (B, K, d_model).
        """
        # Self-attention with residual
        x_norm = self.norm_self(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # Cross-attention with residual
        x_norm = self.norm_cross(x)
        x = x + self.cross_attn(x_norm, cond, cond, need_weights=False)[0]

        # FFN with residual
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Full action expert transformer
# ---------------------------------------------------------------------------

class ActionExpertTransformer(nn.Module):
    """Transformer denoiser for action chunk prediction.

    Combines sinusoidal timestep embedding, learnable positional embeddings,
    and N interleaved self/cross-attention blocks.  Supports both CLS-token
    (2-D) and patch-level (3-D) vision conditioning.
    """

    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 10,
        ffn_dim: int = 3072,
        cond_dim: int = 768,
        state_dim: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Input / output projections
        self.input_proj = nn.Linear(action_dim, d_model)
        self.output_proj = nn.Linear(d_model, action_dim)

        # Timestep embedding
        self.timestep_emb = SinusoidalTimestepEmbedding(d_model)

        # Conditioning projections
        self.vision_proj = nn.Linear(cond_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, chunk_size, d_model) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(d_model, nhead, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Weight initialisation
        self._init_weights()

    # -- initialisation ------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform for all Linear layers; zero-init output projection."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Zero-initialise output projection for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        noisy_actions: Tensor,
        t: Tensor,
        vision_emb: Tensor,
        state: Tensor,
    ) -> Tensor:
        """Predict velocity (flow matching) or noise (DDPM).

        Args:
            noisy_actions: (B, K, action_dim) noisy action chunk.
            t: (B,) float timestep.
            vision_emb: (B, cond_dim) CLS mode or (B, N, cond_dim) patch mode.
            state: (B, state_dim) proprioceptive state.

        Returns:
            Predicted velocity/noise of shape (B, K, action_dim).
        """
        B = noisy_actions.shape[0]

        # Project actions and add positional + timestep embeddings
        x = self.input_proj(noisy_actions) + self.pos_embed
        t_emb = self.timestep_emb(t)  # (B, d_model)
        x = x + t_emb.unsqueeze(1)  # broadcast over chunk dim

        # Build conditioning tokens
        if vision_emb.dim() == 2:
            # CLS mode: (B, cond_dim) -> (B, 1, d_model)
            vis_tokens = self.vision_proj(vision_emb).unsqueeze(1)
        else:
            # Patch mode: (B, N, cond_dim) -> (B, N, d_model)
            vis_tokens = self.vision_proj(vision_emb)

        state_tokens = self.state_proj(state).unsqueeze(1)  # (B, 1, d_model)
        cond = torch.cat([vis_tokens, state_tokens], dim=1)  # (B, C, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        # Output
        x = self.final_norm(x)
        return self.output_proj(x)

    # -- utilities -----------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_action_expert(
    action_dim: int,
    chunk_size: int,
    state_dim: int,
    cond_dim: int = 768,
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 10,
    ffn_dim: int = 3072,
    dropout: float = 0.0,
) -> ActionExpertTransformer:
    """Build an ActionExpertTransformer with the given hyperparameters.

    Args:
        action_dim: Dimensionality of the action space.
        chunk_size: Number of action steps predicted at once.
        state_dim: Dimensionality of proprioceptive state.
        cond_dim: Dimensionality of vision conditioning input.
        d_model: Hidden dimension of the transformer.
        nhead: Number of attention heads.
        num_layers: Number of transformer blocks.
        ffn_dim: Hidden dimension of the feed-forward network.
        dropout: Dropout rate for attention and FFN layers.

    Returns:
        Configured ActionExpertTransformer instance.
    """
    return ActionExpertTransformer(
        action_dim=action_dim,
        chunk_size=chunk_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        cond_dim=cond_dim,
        state_dim=state_dim,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_action_expert(action_dim=7, chunk_size=16, state_dim=14)
    print(f"Action expert: {model.count_parameters() / 1e6:.1f}M parameters")

    B, K = 4, 16
    out = model(
        noisy_actions=torch.randn(B, K, 7),
        t=torch.rand(B),
        vision_emb=torch.randn(B, 196, 768),
        state=torch.randn(B, 14),
    )
    print(f"Output shape: {out.shape}")  # (4, 16, 7)
