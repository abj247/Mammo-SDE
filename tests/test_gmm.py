"""Unit tests for the from-scratch GMM."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mammo_sde.models.gmm import GMM


def _make_synthetic_gmm_data(n_per_cluster=200, K=3, dim=4, sep=5.0, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, sep, size=(K, dim))
    samples = []
    labels = []
    for k in range(K):
        s = rng.normal(0.0, 1.0, size=(n_per_cluster, dim)) + centers[k]
        samples.append(s)
        labels.extend([k] * n_per_cluster)
    X = np.concatenate(samples)
    y = np.array(labels)
    perm = rng.permutation(len(y))
    return torch.from_numpy(X[perm]).float(), torch.from_numpy(y[perm]).long(), centers


@pytest.mark.parametrize("cov_type", ["diag", "full", "spherical", "tied"])
def test_gmm_fits_well_separated_clusters(cov_type):
    X, y_true, _ = _make_synthetic_gmm_data(n_per_cluster=200, K=3, dim=4, sep=8.0, seed=0)
    gmm = GMM(n_components=3, cov_type=cov_type, max_iter=100, seed=0)
    gmm.fit(X)
    y_pred = gmm.predict(X).cpu().numpy()

    # Check purity: most-common true label per cluster should account for >85% of cluster
    purities = []
    for k in range(3):
        mask = y_pred == k
        if mask.sum() == 0:
            continue
        purest = np.bincount(y_true.numpy()[mask]).max()
        purities.append(purest / mask.sum())
    assert min(purities) > 0.85, f"GMM purity too low: {purities}"


def test_gmm_log_likelihood_increases():
    X, _, _ = _make_synthetic_gmm_data(n_per_cluster=100, K=2, dim=2, sep=4.0, seed=1)
    gmm = GMM(n_components=2, cov_type="diag", max_iter=50, seed=1)
    gmm.fit(X)
    ll_history = gmm.history_.log_likelihoods
    # LL should be monotonically non-decreasing
    for prev, nxt in zip(ll_history, ll_history[1:]):
        assert nxt >= prev - 1e-3, f"LL decreased: {prev} → {nxt}"


def test_gmm_bic_picks_correct_k():
    X, _, _ = _make_synthetic_gmm_data(n_per_cluster=300, K=3, dim=3, sep=6.0, seed=2)
    bics = []
    for k in range(1, 7):
        gmm = GMM(n_components=k, cov_type="diag", max_iter=100, seed=2)
        gmm.fit(X)
        bics.append(gmm.bic(X))
    best_k = int(np.argmin(bics)) + 1
    assert best_k == 3, f"Expected best K=3, got {best_k}. BICs: {bics}"


def test_gmm_predict_proba_sums_to_one():
    X, _, _ = _make_synthetic_gmm_data(n_per_cluster=50, K=3, dim=2, sep=4.0, seed=3)
    gmm = GMM(n_components=3, cov_type="diag", max_iter=30, seed=3)
    gmm.fit(X)
    probs = gmm.predict_proba(X)
    sums = probs.sum(dim=1).cpu().numpy()
    assert np.allclose(sums, 1.0, atol=1e-5), f"Responsibilities don't sum to 1: {sums[:5]}"


def test_gmm_sample_recovers_distribution():
    X, _, _ = _make_synthetic_gmm_data(n_per_cluster=500, K=3, dim=2, sep=6.0, seed=4)
    gmm = GMM(n_components=3, cov_type="diag", max_iter=100, seed=4)
    gmm.fit(X)
    samples, _ = gmm.sample(5000, seed=99)
    real_mean = X.mean(dim=0)
    samp_mean = samples.mean(dim=0)
    # Means should be close
    assert torch.allclose(real_mean, samp_mean.cpu(), atol=0.5), \
        f"Sample mean too different: real={real_mean.tolist()} vs sample={samp_mean.tolist()}"


def test_gmm_state_dict_roundtrip():
    X, _, _ = _make_synthetic_gmm_data(n_per_cluster=100, K=2, dim=3, sep=4.0, seed=5)
    gmm = GMM(n_components=2, cov_type="full", max_iter=50, seed=5)
    gmm.fit(X)
    state = gmm.state_dict()
    restored = GMM.from_state_dict(state)
    pred1 = gmm.predict(X)
    pred2 = restored.predict(X)
    assert torch.equal(pred1.cpu(), pred2.cpu()), "Predictions differ after state_dict roundtrip"
