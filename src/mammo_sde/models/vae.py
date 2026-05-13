"""Vanilla VAE (Kingma & Welling 2014) from scratch.

Architecture:
    Encoder MLP: x → (μ, log σ²)
    Decoder MLP: z → x_recon
    Loss = recon (Gaussian with implicit σ²_x=1) + KL(q(z|x) || N(0, I))

This is the baseline against which σ-VAE and VampPrior VAE are compared.
"""

from __future__ import annotations

from mammo_sde.models.base_vae import BaseVAE


class VanillaVAE(BaseVAE):
    """Vanilla VAE — uses BaseVAE defaults (Gaussian σ²=1 likelihood, N(0,I) prior)."""

    name = "vanilla_vae"


__all__ = ["VanillaVAE"]
