"""Mammo-CLIP encoder wrapper template.

Mammo-CLIP (Ghosh et al., MICCAI 2024) is a vision-language foundation model for
mammography. Reference: github.com/batmanlab/Mammo-CLIP.

This template wraps the Mammo-CLIP image encoder. Once the user provides the
actual checkpoint path, instantiate the appropriate backbone in __init__ — the
attention pool + LayerNorm output head is already wired in.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from mammo_sde.encoders.base_encoder import BaseEncoder


class MammoCLIPEncoder(BaseEncoder):
    """Mammo-CLIP image encoder + multi-view attention pool + LayerNorm.

    The Mammo-CLIP backbone is typically an EfficientNetV1-B2 or ConvNeXt-tiny
    image encoder pretrained with image-text contrastive loss on mammography
    pairs. Per-image output is a feature vector; we attention-pool over the
    4 views to get one exam-level embedding.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        image_feature_dim: int,
        out_dim: int = 256,
        pool_heads: int = 4,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_feature_dim = int(image_feature_dim)
        self._out_dim = int(out_dim)

        self._learned_query = nn.Parameter(torch.zeros(1, 1, image_feature_dim))
        nn.init.trunc_normal_(self._learned_query, std=0.02)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=image_feature_dim,
            num_heads=pool_heads,
            batch_first=True,
        )

        self.proj = nn.Linear(image_feature_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def load_checkpoint(self, checkpoint_path: str | Path, map_location: str = "cpu") -> None:
        """Load Mammo-CLIP image encoder weights.

        The Mammo-CLIP checkpoint typically contains a state dict with keys
        prefixed by ``image_encoder.``. We extract those into our backbone.
        """
        state = torch.load(Path(checkpoint_path), map_location=map_location)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            if "model" in state:
                state = state["model"]
        # Extract image encoder weights
        image_state: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            new_key = k
            for prefix in ("module.", "image_encoder.", "visual.", "encoder."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            image_state[new_key] = v
        missing, unexpected = self.image_encoder.load_state_dict(image_state, strict=False)
        if missing:
            print(f"[MammoCLIPEncoder] Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"[MammoCLIPEncoder] Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
        self.freeze()

    @property
    def embedding_dim(self) -> int:
        return self._out_dim

    @torch.no_grad()
    def encode_exam(self, views: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(views)  # (n_views, d_image)
        if feats.dim() > 2:
            feats = feats.flatten(start_dim=1).mean(dim=1, keepdim=False)
        feats = feats.unsqueeze(0)  # (1, n_views, d)
        q = self._learned_query
        pooled, _ = self.attention_pool(q, feats, feats)
        out = self.layer_norm(self.proj(pooled.squeeze(0).squeeze(0)))
        return out

    @torch.no_grad()
    def encode_batch(self, batch_views: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = batch_views.shape
        flat = batch_views.reshape(B * V, C, H, W)
        feats = self.image_encoder(flat)
        if feats.dim() > 2:
            feats = feats.flatten(start_dim=1).mean(dim=1, keepdim=False)
        feats = feats.view(B, V, self.image_feature_dim)
        q = self._learned_query.expand(B, -1, -1)
        pooled, _ = self.attention_pool(q, feats, feats)
        out = self.layer_norm(self.proj(pooled.squeeze(1)))
        return out
