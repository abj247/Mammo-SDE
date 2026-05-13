"""Unit tests for vanilla VAE + σ-VAE + VampPrior VAE."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mammo_sde.models import SigmaVAE, VampPriorVAE, VanillaVAE, init_pseudoinputs_from_gmm
from mammo_sde.models.gmm import GMM
from mammo_sde.training.train_vae import train_vae


def _make_data(n=400, d=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 3.0, size=(4, d))
    cluster_ids = rng.integers(0, 4, size=n)
    X = centers[cluster_ids] + rng.normal(0.0, 1.0, size=(n, d))
    return torch.from_numpy(X.astype(np.float32))


def _make_loader(X, bs=64):
    ds = TensorDataset(X, torch.zeros(len(X)))
    return DataLoader(ds, batch_size=bs, shuffle=True)


def test_vanilla_vae_loss_decreases():
    X = _make_data(n=400, d=8, seed=0)
    loader = _make_loader(X)
    model = VanillaVAE(input_dim=8, latent_dim=4, encoder_hidden=[16, 8])
    h = train_vae(model, loader, val_loader=None, epochs=10, lr=1e-2, device="cpu", log_every=1000)
    assert h.train_total[-1] < h.train_total[0], "Vanilla VAE loss did not decrease"


def test_sigma_vae_loss_decreases_optimal():
    X = _make_data(n=400, d=8, seed=1)
    loader = _make_loader(X)
    model = SigmaVAE(input_dim=8, latent_dim=4, encoder_hidden=[16, 8], sigma_mode="optimal")
    h = train_vae(model, loader, val_loader=None, epochs=10, lr=1e-2, device="cpu", log_every=1000)
    assert h.train_total[-1] < h.train_total[0], "σ-VAE (optimal) loss did not decrease"


def test_sigma_vae_loss_decreases_learned():
    X = _make_data(n=400, d=8, seed=2)
    loader = _make_loader(X)
    model = SigmaVAE(input_dim=8, latent_dim=4, encoder_hidden=[16, 8], sigma_mode="learned")
    h = train_vae(model, loader, val_loader=None, epochs=10, lr=1e-2, device="cpu", log_every=1000)
    assert h.train_total[-1] < h.train_total[0], "σ-VAE (learned) loss did not decrease"


def test_vamp_vae_loss_decreases():
    X = _make_data(n=400, d=8, seed=3)
    loader = _make_loader(X)
    model = VampPriorVAE(
        input_dim=8, latent_dim=4, encoder_hidden=[16, 8], n_pseudoinputs=10,
    )
    h = train_vae(model, loader, val_loader=None, epochs=10, lr=1e-2, device="cpu", log_every=1000)
    assert h.train_total[-1] < h.train_total[0], "VampPrior VAE loss did not decrease"


def test_vamp_init_from_gmm():
    X = _make_data(n=300, d=6, seed=4)
    gmm = GMM(n_components=5, cov_type="diag", max_iter=50, seed=4)
    gmm.fit(X)
    state = gmm.state_dict()
    pseudoinit = init_pseudoinputs_from_gmm(state, n_pseudoinputs=5)
    assert pseudoinit.shape == (5, 6)

    model = VampPriorVAE(
        input_dim=6, latent_dim=4, encoder_hidden=[16, 8], n_pseudoinputs=5,
        pseudoinput_init=pseudoinit,
    )
    # Pseudoinputs were initialized from GMM centers
    assert torch.allclose(model.pseudoinputs.detach(), pseudoinit, atol=1e-6)


def test_vae_reparameterize_shapes():
    model = VanillaVAE(input_dim=8, latent_dim=4)
    x = torch.randn(16, 8)
    out = model.forward(x)
    assert out["x_recon"].shape == x.shape
    assert out["mu"].shape == (16, 4)
    assert out["logvar"].shape == (16, 4)
    assert out["z"].shape == (16, 4)


def test_vae_sample_prior_shape():
    model = VanillaVAE(input_dim=8, latent_dim=4)
    sample = model.sample_prior(5)
    assert sample.shape == (5, 8)


def test_vamp_sample_prior_shape():
    model = VampPriorVAE(input_dim=8, latent_dim=4, n_pseudoinputs=10)
    sample = model.sample_prior(5)
    assert sample.shape == (5, 8)


def test_kl_divergence_nonneg_with_standard_prior():
    model = VanillaVAE(input_dim=8, latent_dim=4)
    x = torch.randn(16, 8)
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    kl = model.kl_divergence(mu, logvar, z)
    assert (kl >= -1e-3).all(), f"KL went negative: {kl}"
