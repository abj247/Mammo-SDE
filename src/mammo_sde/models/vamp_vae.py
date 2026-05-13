"""VampPrior VAE (Tomczak & Welling — AISTATS 2018).

"VAE with a VampPrior"
https://arxiv.org/abs/1705.07120

The prior p(z) is replaced by a mixture of variational posteriors evaluated at K
learned "pseudoinputs" u_1, ..., u_K:

    p(z) = (1/K) Σ_k q(z | u_k)

This makes the prior data-driven and naturally multimodal — a perfect fit for
embeddings that cluster (e.g., by BI-RADS density). Synergy with our GMM Phase 1:
we initialize the pseudoinputs from GMM cluster centers.

Because p(z) is no longer a simple N(0, I), the KL has no closed form. We
compute it via Monte Carlo estimate using the sample z from the encoder:

    KL = E_q[log q(z|x) - log p(z)]
       ≈ log q(z|x) - log [(1/K) Σ_k q(z | u_k)]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mammo_sde.models.base_vae import BaseVAE


def _diag_gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-sample log N(z | mu, diag(exp(logvar))).

    Shapes:
        z      : (B, D)  or (B, 1, D) broadcastable
        mu     : (..., D)
        logvar : (..., D)

    Returns
    -------
    Tensor of shape z.shape[:-1]
    """
    var = logvar.exp()
    sq = (z - mu).pow(2) / var
    return -0.5 * (sq + logvar + math.log(2.0 * math.pi)).sum(dim=-1)


class VampPriorVAE(BaseVAE):
    """VAE with a VampPrior: prior is a mixture over q(z | u_k) for K pseudoinputs.

    Parameters
    ----------
    n_pseudoinputs : int
        Number of pseudoinputs K. Common choice: 50-500. Default 50.
    pseudoinput_init : Optional[torch.Tensor]
        Optional initialization tensor of shape (K, input_dim). If provided, used
        as the initial pseudoinputs. Otherwise initialized from a small Gaussian.
        **Strong recommendation:** pass GMM cluster centers here.
    pseudoinput_trainable : bool
        Whether pseudoinputs are learnable parameters. Default True.
    """

    name = "vamp_vae"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        activation: str = "gelu",
        n_pseudoinputs: int = 50,
        pseudoinput_init: torch.Tensor | None = None,
        pseudoinput_trainable: bool = True,
    ):
        super().__init__(input_dim, latent_dim, encoder_hidden, decoder_hidden, activation)
        self.K = int(n_pseudoinputs)

        if pseudoinput_init is not None:
            assert pseudoinput_init.shape == (
                self.K,
                self.input_dim,
            ), f"pseudoinput_init must be ({self.K}, {self.input_dim}), got {tuple(pseudoinput_init.shape)}"
            pseudoinputs = pseudoinput_init.detach().clone().float()
        else:
            pseudoinputs = torch.randn(self.K, self.input_dim) * 0.05

        self.pseudoinputs = nn.Parameter(pseudoinputs, requires_grad=pseudoinput_trainable)

    def _prior_components(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get (mu_k, logvar_k) for the K mixture components by encoding pseudoinputs."""
        return self.encode(self.pseudoinputs)  # (K, D), (K, D)

    def log_prior(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z) = log [(1/K) Σ_k q(z | u_k)].

        Shapes:
            z       : (B, D)
        Returns
        -------
        Tensor of shape (B,)
        """
        mu_k, logvar_k = self._prior_components()  # (K, D), (K, D)
        # Expand: z (B, 1, D), mu_k (1, K, D)
        z_e = z.unsqueeze(1)
        mu_e = mu_k.unsqueeze(0)
        logvar_e = logvar_k.unsqueeze(0)
        log_qk = _diag_gaussian_log_prob(z_e, mu_e, logvar_e)  # (B, K)
        log_p = torch.logsumexp(log_qk, dim=1) - math.log(self.K)  # (B,)
        return log_p

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || p_vamp(z)) via Monte Carlo with single sample z.

        log q(z|x) is the diagonal Gaussian density of z under N(mu, σ²)
        log p(z) is the VampPrior mixture density.
        """
        log_q = _diag_gaussian_log_prob(z, mu, logvar)  # (B,)
        log_p = self.log_prior(z)  # (B,)
        return log_q - log_p

    @torch.no_grad()
    def sample_prior(self, n: int, device: str | torch.device | None = None) -> torch.Tensor:
        """Sample from the VampPrior mixture, decode to ĥ."""
        device = device or next(self.parameters()).device
        mu_k, logvar_k = self._prior_components()  # (K, D), (K, D)
        # Choose a component uniformly, then sample from it
        comp = torch.randint(0, self.K, (n,), device=device)
        mu = mu_k[comp]
        logvar = logvar_k[comp]
        eps = torch.randn_like(mu)
        z = mu + (0.5 * logvar).exp() * eps
        return self.decode(z)


def init_pseudoinputs_from_gmm(gmm_state_dict: dict, n_pseudoinputs: int | None = None) -> torch.Tensor:
    """Helper: extract GMM cluster centers to initialize VampPrior pseudoinputs.

    Parameters
    ----------
    gmm_state_dict : dict
        Output of GMM.state_dict(). Must contain ``means``: (K, D).
    n_pseudoinputs : Optional[int]
        If specified, use only the first n_pseudoinputs centers. Otherwise use all K.

    Returns
    -------
    torch.Tensor of shape (K, D)
    """
    means = gmm_state_dict["means"]
    if n_pseudoinputs is not None and n_pseudoinputs < means.shape[0]:
        means = means[:n_pseudoinputs]
    return means.detach().clone().float()
