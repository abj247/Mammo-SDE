"""VAE-specific analysis: reconstruction quality, latent structure, active dims, etc."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from mammo_sde.models.base_vae import BaseVAE


@torch.no_grad()
def reconstruction_metrics(model: BaseVAE, loader: DataLoader, device: str = "cpu") -> dict:
    """Compute reconstruction MSE statistics over a dataloader."""
    model.eval()
    device_t = torch.device(device)
    mse_per_sample: list[np.ndarray] = []
    for batch in loader:
        x, _ = batch
        x = x.to(device_t)
        x_recon = model.reconstruction(x, use_mean=True)
        mse = (x - x_recon).pow(2).mean(dim=1).cpu().numpy()
        mse_per_sample.append(mse)
    mse_arr = np.concatenate(mse_per_sample)
    return {
        "mse_mean": float(mse_arr.mean()),
        "mse_std": float(mse_arr.std()),
        "mse_median": float(np.median(mse_arr)),
        "mse_p95": float(np.percentile(mse_arr, 95)),
        "mse_p99": float(np.percentile(mse_arr, 99)),
        "n_samples": int(mse_arr.size),
    }


@torch.no_grad()
def posterior_statistics(model: BaseVAE, loader: DataLoader, device: str = "cpu") -> dict:
    """Compute posterior μ and log σ² statistics across the dataset."""
    model.eval()
    device_t = torch.device(device)
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    for batch in loader:
        x, _ = batch
        x = x.to(device_t)
        mu, logvar = model.posterior_params(x)
        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
    mu_arr = np.concatenate(mus)
    logvar_arr = np.concatenate(logvars)
    var_arr = np.exp(logvar_arr)
    return {
        "mu_mean_per_dim": mu_arr.mean(axis=0).tolist(),
        "mu_std_per_dim": mu_arr.std(axis=0).tolist(),
        "logvar_mean_per_dim": logvar_arr.mean(axis=0).tolist(),
        "var_mean_per_dim": var_arr.mean(axis=0).tolist(),
    }


@torch.no_grad()
def active_dimensions(model: BaseVAE, loader: DataLoader, threshold: float = 0.01, device: str = "cpu") -> dict:
    """Identify which latent dimensions are 'active' vs collapsed (var of μ near 0).

    Definition (per Burda et al. 2016): a dimension d is active if
        Var_q[μ_d(x)] > threshold (over the dataset).
    Collapsed dimensions have μ_d effectively constant, σ_d ≈ 1 — same as prior.
    """
    model.eval()
    device_t = torch.device(device)
    mus: list[np.ndarray] = []
    for batch in loader:
        x, _ = batch
        x = x.to(device_t)
        mu, _ = model.posterior_params(x)
        mus.append(mu.cpu().numpy())
    mu_arr = np.concatenate(mus)
    var_of_mu = mu_arr.var(axis=0)
    active = var_of_mu > threshold
    return {
        "var_of_mu_per_dim": var_of_mu.tolist(),
        "active_mask": active.tolist(),
        "n_active": int(active.sum()),
        "n_total": int(active.size),
        "fraction_active": float(active.mean()),
        "threshold": threshold,
    }


@torch.no_grad()
def kl_per_dim(model: BaseVAE, loader: DataLoader, device: str = "cpu") -> dict:
    """Average KL divergence per latent dimension across the dataset.

    For each dim, KL_d = 0.5 (μ²_d + σ²_d - log σ²_d - 1). High → dim carries info.
    """
    model.eval()
    device_t = torch.device(device)
    kl_sums = None
    n_total = 0
    for batch in loader:
        x, _ = batch
        x = x.to(device_t)
        mu, logvar = model.posterior_params(x)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        if kl_sums is None:
            kl_sums = kl.sum(dim=0).cpu().numpy()
        else:
            kl_sums = kl_sums + kl.sum(dim=0).cpu().numpy()
        n_total += x.size(0)
    kl_mean = kl_sums / max(1, n_total)
    return {
        "kl_per_dim_mean": kl_mean.tolist(),
        "n_samples": int(n_total),
        "total_kl": float(kl_mean.sum()),
    }


@torch.no_grad()
def prior_sample_quality(
    model: BaseVAE, real_embeddings: np.ndarray, n_samples: int = 5000, device: str = "cpu"
) -> dict:
    """Compare moments of real embeddings to samples decoded from the prior."""
    model.eval()
    samples = model.sample_prior(n_samples, device=device).cpu().numpy()
    return {
        "real_per_dim_mean": real_embeddings.mean(axis=0).tolist(),
        "sample_per_dim_mean": samples.mean(axis=0).tolist(),
        "real_per_dim_std": real_embeddings.std(axis=0).tolist(),
        "sample_per_dim_std": samples.std(axis=0).tolist(),
        "mean_diff_l2": float(np.linalg.norm(real_embeddings.mean(axis=0) - samples.mean(axis=0))),
        "std_diff_l2": float(np.linalg.norm(real_embeddings.std(axis=0) - samples.std(axis=0))),
    }
