"""Abstract interface for frozen encoder wrappers.

The encoder takes an exam (4 views) and produces one embedding vector h_1 ∈ R^d.
This contract is the same regardless of which underlying foundation model is used.

Subclasses must:
    1. Load the checkpoint from a user-provided path
    2. Set the model to eval mode (frozen)
    3. Implement encode_exam(views) → (d,) tensor
    4. Expose embedding_dim property
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for frozen foundation-model encoders.

    All implementations must:
        - Be inference-only (no gradient flow); call .eval() in __init__ or load_checkpoint
        - Produce ONE d-dimensional vector per exam (per the SDE pipeline contract)
        - Apply multi-view aggregation INSIDE the encoder (attention pool over the 4 views)
        - End with LayerNorm to satisfy the bounded-norm requirement of the SDE
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | Path, map_location: str = "cpu") -> None:
        """Load model weights from disk. Sets model to eval mode."""

    @abstractmethod
    def encode_exam(self, views: torch.Tensor) -> torch.Tensor:
        """Encode one exam to one d-dimensional vector.

        Parameters
        ----------
        views : torch.Tensor
            Shape (n_views, C, H, W). Typically n_views=4 (L-CC, R-CC, L-MLO, R-MLO).

        Returns
        -------
        torch.Tensor
            Shape (d,). Single embedding vector for the exam.
        """

    @abstractmethod
    def encode_batch(self, batch_views: torch.Tensor) -> torch.Tensor:
        """Encode a batch of exams.

        Parameters
        ----------
        batch_views : torch.Tensor
            Shape (B, n_views, C, H, W).

        Returns
        -------
        torch.Tensor
            Shape (B, d).
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output embedding dimension d."""

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward dispatches to encode_batch or encode_exam by rank."""
        if x.dim() == 4:
            return self.encode_exam(x)
        if x.dim() == 5:
            return self.encode_batch(x)
        raise ValueError(f"Expected 4D (single exam) or 5D (batch) input, got shape {tuple(x.shape)}")

    def freeze(self) -> None:
        """Set all parameters to requires_grad=False and eval mode."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
