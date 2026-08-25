"""Chapter 7: Full VLA -- SmolVLA-like Architecture.

Assembles SigLIP vision encoder, SmolLM2 language backbone, and a
flow-matching action expert into a complete Vision-Language-Action model
following SmolVLA (Shukor et al., 2025).

Architecture:
    1. SigLIP encodes 512x512 images into 1024 patches (dim=768).
    2. Pixel-shuffle reduces to 64 tokens, connector projects to dim=960.
    3. SmolLM2 (first 16/32 layers, frozen) processes the prefix:
       [vision_tokens, language_embeds, state_token].
    4. Action expert (8 trainable layers, interleaved SA/CA) generates
       flow-matching velocity predictions conditioned on VLM hidden states.

Key difference from Chapter 6's action expert:
    - Ch06: every block has SA + CA + FFN (combined in each block)
    - Ch07: blocks alternate SA-only and CA-only (SmolVLA interleaving)
    - Ch06: standalone expert; Ch07: expert is conditioned layer-by-layer
      on per-layer VLM hidden states via cross-attention
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from config import SmolVLAConfig


# ---------------------------------------------------------------------------
# Pixel Shuffle (token reduction)
# ---------------------------------------------------------------------------

def pixel_shuffle(x: Tensor, scale_factor: int = 4) -> Tensor:
    """Merge neighboring vision patches to reduce token count.

    Rearranges a grid of patches so that each scale_factor x scale_factor
    neighborhood is merged into a single token with concatenated features.

    Args:
        x: Patch embeddings of shape (B, H*W, C) where H=W=sqrt(H*W).
        scale_factor: Spatial reduction factor. 4 means 32x32 -> 8x8.

    Returns:
        Reduced tokens of shape (B, H*W/s^2, C*s^2).
    """
    bsz, seq, embed_dim = x.shape
    height = width = int(seq ** 0.5)
    x = x.reshape(bsz, height, width, embed_dim)
    x = x.reshape(
        bsz, height, width // scale_factor, embed_dim * scale_factor
    )
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(
        bsz,
        width // scale_factor,
        height // scale_factor,
        embed_dim * (scale_factor ** 2),
    )
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, -1, embed_dim * (scale_factor ** 2))
    return x


# ---------------------------------------------------------------------------
# Timestep embedding (same as Ch06, included for self-containment)
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Map scalar timestep (B,) -> (B, d_model) via sinusoidal + MLP."""

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
        """Compute embedding for timesteps t of shape (B,)."""
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
# Expert blocks
# ---------------------------------------------------------------------------

class ExpertSelfAttnBlock(nn.Module):
    """Causal self-attention + FFN for action tokens (pre-LayerNorm)."""

    def __init__(
        self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Self-attend over action tokens with optional causal mask."""
        h = self.norm_sa(x)
        x = x + self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ExpertCrossAttnBlock(nn.Module):
    """Cross-attention (action -> VLM features) + FFN (pre-LayerNorm).

    Handles the dimension mismatch between action expert (720) and
    VLM hidden states (960) via kdim/vdim in MultiheadAttention.
    """

    def __init__(
        self,
        d_model: int,
        kv_dim: int,
        nhead: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_ca = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
            kdim=kv_dim, vdim=kv_dim,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: Tensor, kv: Tensor) -> Tensor:
        """Cross-attend: action tokens query VLM features as key/value."""
        h = self.norm_ca(x)
        kv_norm = self.norm_kv(kv)
        x = x + self.cross_attn(h, kv_norm, kv_norm, need_weights=False)[0]
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Action Expert (interleaved SA/CA)
# ---------------------------------------------------------------------------

class ActionExpert(nn.Module):
    """Interleaved self-attention / cross-attention action expert.

    Even layers (0, 2, 4, 6): causal self-attention among action tokens
    Odd layers (1, 3, 5, 7): cross-attention to VLM per-layer hidden states

    This interleaving pattern follows SmolVLA's ablation results showing
    interleaved > pure CA > pure SA for action prediction quality.
    """

    def __init__(self, config: SmolVLAConfig) -> None:
        super().__init__()
        d = config.expert_dim

        self.input_proj = nn.Linear(config.action_dim, d)
        self.output_proj = nn.Linear(d, config.action_dim)
        self.timestep_emb = SinusoidalTimestepEmbedding(d)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.chunk_size, d) * 0.02
        )

        # Build interleaved blocks
        self.blocks = nn.ModuleList()
        for i in range(config.expert_num_layers):
            if i % 2 == 0:
                self.blocks.append(ExpertSelfAttnBlock(
                    d, config.expert_nhead, config.expert_ffn_dim,
                    config.expert_dropout,
                ))
            else:
                self.blocks.append(ExpertCrossAttnBlock(
                    d, config.vlm_hidden_dim, config.expert_nhead,
                    config.expert_ffn_dim, config.expert_dropout,
                ))

        self.final_norm = nn.LayerNorm(d)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform everywhere, zero-init output projection."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_actions: Tensor,
        timesteps: Tensor,
        vlm_features: list[Tensor],
    ) -> Tensor:
        """Predict flow velocity from noisy actions + VLM conditioning.

        Args:
            noisy_actions: (B, K, action_dim) noisy action chunk.
            timesteps: (B,) float in [0, 1].
            vlm_features: List of (B, N_prefix, vlm_hidden_dim) tensors,
                one per cross-attention layer.

        Returns:
            Predicted velocity: (B, K, action_dim).
        """
        B, K, _ = noisy_actions.shape

        # Project actions + add timestep + positional embedding
        x = self.input_proj(noisy_actions)
        t_emb = self.timestep_emb(timesteps)
        x = x + t_emb.unsqueeze(1) + self.pos_embed[:, :K, :]

        # Causal mask for self-attention
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            K, device=x.device, dtype=x.dtype
        )

        # Interleaved SA/CA
        ca_idx = 0
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                x = block(x, attn_mask=causal_mask)
            else:
                x = block(x, vlm_features[ca_idx])
                ca_idx += 1

        x = self.final_norm(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Full SmolVLA Model
# ---------------------------------------------------------------------------

class SmolVLA(nn.Module):
    """Full SmolVLA: SigLIP + SmolLM2 + flow-matching action expert.

    Frozen components (~300M params):
        - SigLIP vision encoder (512x512 -> 1024 patches, dim=768)
        - Pixel-shuffle connector (1024 -> 64 tokens, projects to dim=960)
        - SmolLM2-360M (first 16 of 32 layers, dim=960)

    Trainable components (~21M params):
        - State projector: Linear(state_dim -> 960)
        - Action expert: 8-layer interleaved SA/CA transformer (dim=720)
    """

    def __init__(self, config: SmolVLAConfig) -> None:
        super().__init__()
        self.config = config
        self._load_pretrained(config)

        # Trainable projectors
        self.state_proj = nn.Linear(config.state_dim, config.vlm_hidden_dim)

        # Action expert
        self.action_expert = ActionExpert(config)

    def _load_pretrained(self, config: SmolVLAConfig) -> None:
        """Load and freeze SigLIP + connector + SmolLM2 from SmolVLM2."""
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

        print(f"[SmolVLA] Loading {config.vlm_model_name}...")
        full_model = AutoModel.from_pretrained(
            config.vlm_model_name, torch_dtype=torch.float32
        )

        # Extract components
        self.vision_encoder = full_model.vision_model
        self.connector = full_model.connector
        self.vlm = full_model.text_model

        # Truncate VLM to first L/2 layers
        total_layers = len(self.vlm.layers)
        self.vlm.layers = self.vlm.layers[:config.vlm_num_layers]
        print(
            f"[SmolVLA] VLM truncated: {total_layers} -> "
            f"{len(self.vlm.layers)} layers"
        )

        # Freeze all pretrained components
        for component in [self.vision_encoder, self.connector, self.vlm]:
            component.eval()
            for p in component.parameters():
                p.requires_grad = False

        # Load tokenizer for language processing
        self.tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load image processor for SigLIP input preparation
        self.image_processor = AutoImageProcessor.from_pretrained(
            config.vlm_model_name
        )

        del full_model

        vision_params = sum(
            p.numel() for p in self.vision_encoder.parameters()
        )
        vlm_params = sum(p.numel() for p in self.vlm.parameters())
        conn_params = sum(p.numel() for p in self.connector.parameters())
        print(
            f"[SmolVLA] Frozen: vision={vision_params / 1e6:.1f}M, "
            f"VLM={vlm_params / 1e6:.1f}M, connector={conn_params / 1e6:.1f}M"
        )

    @torch.no_grad()
    def extract_vision_tokens(
        self, pixel_values: Tensor
    ) -> Tensor:
        """Run SigLIP + pixel_shuffle + connector on preprocessed images.

        Args:
            pixel_values: (B, 3, 512, 512) normalized images.

        Returns:
            Vision tokens: (B, 64, 960).
        """
        self.vision_encoder.eval()

        # SmolVLM2's vision encoder expects (B, C, H, W) — standard 4D.
        # The SmolVLM2 processor produces 5D (B, num_tiles, C, H, W) with
        # tiling, which we bypass by using simple torchvision transforms.
        # We also need a patch_attention_mask: (B, num_patches) of ones.
        B = pixel_values.shape[0]
        H = W = self.config.vision_image_size // self.config.vision_patch_size
        patch_mask = torch.ones(
            B, H, W,
            device=pixel_values.device, dtype=torch.bool,
        )

        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            patch_attention_mask=patch_mask,
        )
        patches = outputs.last_hidden_state  # (B, 1024, 768)

        # Pixel shuffle + connector projection (using pretrained weights)
        shuffled = pixel_shuffle(patches, self.config.pixel_shuffle_scale)
        vision_tokens = self.connector.modality_projection(shuffled)
        return vision_tokens

    def forward(
        self,
        vision_tokens: Tensor,
        lang_token_ids: Tensor,
        lang_attention_mask: Tensor,
        states: Tensor,
        noisy_actions: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Full forward pass: VLM prefix encoding + action expert.

        Args:
            vision_tokens: (B, 64, 960) pre-cached from extract_vision_tokens.
            lang_token_ids: (B, T) tokenized language instructions.
            lang_attention_mask: (B, T) attention mask for language.
            states: (B, state_dim) proprioceptive state.
            noisy_actions: (B, K, action_dim) noisy action chunk.
            timesteps: (B,) float in [0, 1].

        Returns:
            Predicted velocity: (B, K, action_dim).
        """
        B = vision_tokens.shape[0]

        # 1. Language embeddings from frozen VLM embedding layer
        with torch.no_grad():
            lang_embeds = self.vlm.embed_tokens(lang_token_ids)

        # 2. State token from trainable projector
        state_token = self.state_proj(states).unsqueeze(1)  # (B, 1, 960)

        # 3. Concatenate prefix: [vision | language | state]
        prefix = torch.cat(
            [vision_tokens, lang_embeds, state_token], dim=1
        )

        # 4. Build attention mask for prefix
        vision_mask = torch.ones(
            B, vision_tokens.shape[1],
            device=vision_tokens.device, dtype=lang_attention_mask.dtype,
        )
        state_mask = torch.ones(
            B, 1,
            device=vision_tokens.device, dtype=lang_attention_mask.dtype,
        )
        prefix_mask = torch.cat(
            [vision_mask, lang_attention_mask, state_mask], dim=1
        )

        # 5. Run VLM (frozen) -- collect per-layer hidden states
        with torch.no_grad():
            vlm_out = self.vlm(
                inputs_embeds=prefix,
                attention_mask=prefix_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # 6. Select hidden states for cross-attention layers
        ca_features = [
            vlm_out.hidden_states[i] for i in self.config.ca_layer_indices
        ]

        # 7. Action expert (trainable) predicts flow velocity
        velocity = self.action_expert(noisy_actions, timesteps, ca_features)
        return velocity

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only trainable parameters for the optimizer."""
        params = list(self.state_proj.parameters())
        params += list(self.action_expert.parameters())
        return params

    def count_parameters(self) -> dict[str, int]:
        """Count total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_smolvla(config: SmolVLAConfig) -> SmolVLA:
    """Build SmolVLA model and print parameter summary."""
    model = SmolVLA(config)
    counts = model.count_parameters()
    print(
        f"[SmolVLA] Parameters: total={counts['total'] / 1e6:.1f}M, "
        f"trainable={counts['trainable'] / 1e6:.1f}M, "
        f"frozen={counts['frozen'] / 1e6:.1f}M"
    )
    return model
