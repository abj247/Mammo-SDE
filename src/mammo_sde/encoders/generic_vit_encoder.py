"""Generic Vision Transformer encoder wrapper.

Works for I-JEPA ViT-Small, HuggingFace ViTs, or any ViT-style backbone where
the per-image embedding is the CLS token (or mean-pooled tokens) from the final
transformer block.

This is a template that the user can adapt once they provide the actual
checkpoint path. The architecture instantiation is left as a callback so the
user can wire in any timm/HuggingFace model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from mammo_sde.encoders.base_encoder import BaseEncoder


class GenericViTEncoder(BaseEncoder):
    """Generic ViT encoder with attention pooling over views + LayerNorm output.

    Pipeline per exam:
        (n_views, C, H, W) → ViT → (n_views, d_hidden) per-image features
                          → attention pool over views → (d_hidden,) exam feature
                          → Linear(d_hidden → d_out) → LayerNorm(d_out) → (d_out,)
    """

    def __init__(
        self,
        backbone_factory: Callable[[], nn.Module],
        backbone_feature_dim: int,
        out_dim: int = 256,
        pool_heads: int = 4,
        feature_extractor: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
    ):
        """
        Parameters
        ----------
        backbone_factory : Callable
            No-arg factory that constructs the ViT backbone, e.g. ``lambda: timm.create_model(...)``.
        backbone_feature_dim : int
            Output feature dim of the backbone (e.g., 384 for ViT-Small, 768 for ViT-Base).
        out_dim : int
            Final exam-embedding dimension (the d in the SDE pipeline).
        pool_heads : int
            Number of attention heads for the multi-view attention pool.
        feature_extractor : Optional[Callable]
            Optional function ``(backbone, x_views) -> features`` that returns
            ``(n_views, backbone_feature_dim)``. If None, uses a default that
            calls ``backbone(x_views).mean(dim=1)`` or treats the output as already
            being per-image features.
        """
        super().__init__()
        self.backbone = backbone_factory()
        self.backbone_feature_dim = int(backbone_feature_dim)
        self._out_dim = int(out_dim)

        self._learned_query = nn.Parameter(torch.zeros(1, 1, backbone_feature_dim))
        nn.init.trunc_normal_(self._learned_query, std=0.02)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=backbone_feature_dim,
            num_heads=pool_heads,
            batch_first=True,
        )

        self.proj = nn.Linear(backbone_feature_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

        self._feature_extractor = feature_extractor or self._default_feature_extractor

    @staticmethod
    def _default_feature_extractor(backbone: nn.Module, x_views: torch.Tensor) -> torch.Tensor:
        """Default: assume backbone returns either (n, d) directly or (n, T, d) tokens."""
        out = backbone(x_views)
        if isinstance(out, dict) and "last_hidden_state" in out:
            out = out["last_hidden_state"]
        if out.dim() == 3:
            out = out.mean(dim=1)
        return out

    def load_checkpoint(self, checkpoint_path: str | Path, map_location: str = "cpu") -> None:
        state = torch.load(Path(checkpoint_path), map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Strip common prefixes
        cleaned = {}
        for k, v in state.items():
            new_key = k
            for prefix in ("module.", "encoder.", "backbone.", "target_encoder."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            cleaned[new_key] = v
        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[GenericViTEncoder] Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"[GenericViTEncoder] Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
        self.freeze()

    @property
    def embedding_dim(self) -> int:
        return self._out_dim

    @torch.no_grad()
    def encode_exam(self, views: torch.Tensor) -> torch.Tensor:
        per_view = self._feature_extractor(self.backbone, views)  # (n_views, d_hidden)
        per_view = per_view.unsqueeze(0)  # (1, n_views, d_hidden)
        q = self._learned_query.expand(per_view.size(0), -1, -1)
        pooled, _ = self.attention_pool(q, per_view, per_view)
        pooled = pooled.squeeze(0).squeeze(0)  # (d_hidden,)
        out = self.layer_norm(self.proj(pooled))
        return out

    @torch.no_grad()
    def encode_batch(self, batch_views: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = batch_views.shape
        flat = batch_views.reshape(B * V, C, H, W)
        per_view = self._feature_extractor(self.backbone, flat)  # (B*V, d_hidden)
        per_view = per_view.view(B, V, self.backbone_feature_dim)
        q = self._learned_query.expand(B, -1, -1)
        pooled, _ = self.attention_pool(q, per_view, per_view)
        pooled = pooled.squeeze(1)  # (B, d_hidden)
        out = self.layer_norm(self.proj(pooled))
        return out
