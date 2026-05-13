"""Shared base class for VAE variants.

All three variants (Vanilla, σ-VAE, VampPrior) share:
- MLP encoder: input → hidden_dims → (μ, log σ²)
- Reparameterization trick
- MLP decoder: latent → hidden_dims → output

What differs:
- Reconstruction likelihood (vanilla uses MSE with implicit σ=1; σ-VAE learns σ)
- KL prior (vanilla and σ-VAE use N(0,I); VampPrior uses learned mixture)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VAELossOutput:
    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor
    extras: dict[str, torch.Tensor]


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, activation: str = "gelu") -> nn.Sequential:
    """Build an MLP: in → hidden_dims (with activation) → out (no activation)."""
    act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class BaseVAE(nn.Module):
    """Shared VAE architecture: encoder + decoder MLPs + reparameterize.

    Subclasses must override:
        - ``log_likelihood(x, x_recon)`` — returns per-sample log p(x | z)
        - ``kl_divergence(mu, logvar, z)`` — returns per-sample KL(q(z|x) || prior)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        encoder_hidden = encoder_hidden if encoder_hidden is not None else [512, 256]
        decoder_hidden = decoder_hidden if decoder_hidden is not None else list(reversed(encoder_hidden))

        # Encoder outputs concat(mu, logvar)
        self.encoder = _build_mlp(self.input_dim, encoder_hidden, 2 * self.latent_dim, activation)
        # Decoder outputs reconstructed input
        self.decoder = _build_mlp(self.latent_dim, decoder_hidden, self.input_dim, activation)

    # ----------------------------------------------------------------- core ops

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x → (μ, log σ²)."""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        logvar = logvar.clamp(min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z}

    # ----------------------------------------------------------------- to override

    def log_likelihood(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Per-sample log p(x | z), shape (B,). Default = Gaussian with σ²=1 (= -0.5 * MSE - const)."""
        D = self.input_dim
        sq = (x - x_recon).pow(2).sum(dim=-1)
        return -0.5 * (sq + D * math.log(2.0 * math.pi))

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Per-sample KL(q(z|x) || prior), shape (B,). Default = closed-form vs N(0, I)."""
        return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1)

    def loss(self, x: torch.Tensor, beta: float = 1.0) -> VAELossOutput:
        out = self.forward(x)
        log_lik = self.log_likelihood(x, out["x_recon"])  # (B,)
        kl = self.kl_divergence(out["mu"], out["logvar"], out["z"])  # (B,)
        # ELBO per sample = log p(x|z) - KL; negate for minimization
        recon_term = -log_lik.mean()
        kl_term = kl.mean()
        total = recon_term + beta * kl_term
        return VAELossOutput(total=total, recon=recon_term, kl=kl_term, extras={})

    # ----------------------------------------------------------------- inference helpers

    @torch.no_grad()
    def sample_prior(self, n: int, device: str | torch.device | None = None) -> torch.Tensor:
        """Sample n new ĥ from the prior. Default = N(0, I)."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def reconstruction(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """Reconstruct x. If use_mean, use posterior μ (not a sample)."""
        mu, logvar = self.encode(x)
        z = mu if use_mean else self.reparameterize(mu, logvar)
        return self.decode(z)

    @torch.no_grad()
    def posterior_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return mu, logvar
