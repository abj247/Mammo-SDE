"""σ-VAE (Rybkin, Daniilidis, Levine — ICML 2021).

"Simple and Effective VAE Training with Calibrated Decoders"
https://arxiv.org/abs/2006.13202

Core insight: Vanilla VAE has a hidden flaw — it implicitly assumes the decoder
noise variance σ²_x = 1 (because MSE loss = -log p with unit variance). This
creates a wrong balance between reconstruction and KL terms.

Two fixes proposed:
    (a) Learnable σ_x   — add a single scalar parameter ``log_sigma_x`` and learn it
    (b) Optimal σ_x    — compute the analytically optimal σ²_x per batch:
                          σ²_x* = mean((x - x_recon)²) over all elements

Both improve over vanilla VAE. ``optimal`` is hyperparameter-free.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from mammo_sde.models.base_vae import BaseVAE


class SigmaVAE(BaseVAE):
    """σ-VAE with either learnable or analytically optimal decoder variance.

    Parameters
    ----------
    sigma_mode : {"learned", "optimal"}
        - "learned": add ``log_sigma_x`` as a learnable parameter
        - "optimal": compute σ²_x* analytically each forward pass (paper's preferred form)
    """

    name = "sigma_vae"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        activation: str = "gelu",
        sigma_mode: Literal["learned", "optimal"] = "optimal",
        init_log_sigma_x: float = 0.0,
    ):
        super().__init__(input_dim, latent_dim, encoder_hidden, decoder_hidden, activation)
        if sigma_mode not in ("learned", "optimal"):
            raise ValueError(f"sigma_mode must be 'learned' or 'optimal', got {sigma_mode!r}")
        self.sigma_mode = sigma_mode
        # Learnable scalar — always created so checkpoint shapes are stable.
        # In 'optimal' mode it's overwritten analytically per batch and not used in backward.
        self.log_sigma_x = nn.Parameter(torch.tensor(float(init_log_sigma_x)), requires_grad=(sigma_mode == "learned"))

    def _current_log_sigma_x(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        if self.sigma_mode == "learned":
            return self.log_sigma_x
        # Optimal σ²_x* = mean((x - x_recon)²) over all dimensions of the batch
        mse = (x - x_recon).pow(2).mean().detach()
        return 0.5 * mse.clamp(min=1e-8).log()

    def log_likelihood(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Per-sample Gaussian log-likelihood with calibrated σ_x.

        log p(x | z) = -0.5 * D * log(2π σ²_x) - 0.5 * Σ_d (x_d - x_recon_d)² / σ²_x
        """
        D = self.input_dim
        log_sigma = self._current_log_sigma_x(x, x_recon)
        sq = (x - x_recon).pow(2).sum(dim=-1)
        inv_var = torch.exp(-2.0 * log_sigma)
        return -0.5 * (D * math.log(2.0 * math.pi) + 2.0 * D * log_sigma + sq * inv_var)
