"""GMM-specific analysis: cluster-vs-label alignment, cluster sizes, etc."""

from __future__ import annotations

import numpy as np
import torch

from mammo_sde.models.gmm import GMM


def cluster_label_alignment(cluster_ids: np.ndarray, labels: np.ndarray) -> dict:
    """How well do GMM clusters align with a discrete label (e.g., BI-RADS density)?

    Returns confusion matrix (clusters x labels, normalized per cluster) and
    cluster-purity (per-cluster, the fraction of the dominant label).
    """
    cluster_ids = cluster_ids.astype(int)
    labels = labels.astype(int)
    K = int(cluster_ids.max()) + 1
    L = int(labels.max()) + 1
    confusion = np.zeros((K, L), dtype=np.int64)
    for c, lab in zip(cluster_ids, labels, strict=False):
        confusion[c, lab] += 1
    row_sums = confusion.sum(axis=1, keepdims=True) + 1e-9
    confusion_norm = confusion / row_sums
    purity = confusion.max(axis=1) / (confusion.sum(axis=1) + 1e-9)
    overall_purity = float(confusion.max(axis=1).sum() / confusion.sum())
    return {
        "confusion": confusion.tolist(),
        "confusion_normalized": confusion_norm.tolist(),
        "cluster_purity": purity.tolist(),
        "overall_purity": overall_purity,
        "n_clusters": K,
        "n_labels": L,
    }


def cluster_sizes(cluster_ids: np.ndarray, K: int) -> dict:
    counts = np.bincount(cluster_ids.astype(int), minlength=K)
    fractions = counts / counts.sum()
    return {
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "min_fraction": float(fractions.min()),
        "max_fraction": float(fractions.max()),
        "balanced_score": float(1.0 - fractions.std() * np.sqrt(K)),
    }


def cluster_centroids_distances(gmm: GMM) -> dict:
    """Pairwise distances between GMM cluster centers (in embedding space)."""
    if gmm.means_ is None:
        raise RuntimeError("GMM not fitted")
    means = gmm.means_.detach().cpu().numpy()
    K = means.shape[0]
    dists = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            dists[i, j] = np.linalg.norm(means[i] - means[j])
    iu = np.triu_indices(K, k=1)
    pairwise = dists[iu]
    return {
        "pairwise_distances": dists.tolist(),
        "mean_pairwise_distance": float(pairwise.mean()),
        "min_pairwise_distance": float(pairwise.min()) if pairwise.size > 0 else 0.0,
        "max_pairwise_distance": float(pairwise.max()) if pairwise.size > 0 else 0.0,
    }


def sample_quality(gmm: GMM, real_embeddings: torch.Tensor, n_samples: int = 5000) -> dict:
    """Compare real embedding stats to samples drawn from the fitted GMM."""
    samples, _ = gmm.sample(n_samples)
    samples_np = samples.detach().cpu().numpy()
    real_np = real_embeddings.detach().cpu().numpy() if isinstance(real_embeddings, torch.Tensor) else real_embeddings
    return {
        "real_per_dim_mean": real_np.mean(axis=0).tolist(),
        "sample_per_dim_mean": samples_np.mean(axis=0).tolist(),
        "real_per_dim_std": real_np.std(axis=0).tolist(),
        "sample_per_dim_std": samples_np.std(axis=0).tolist(),
        "mean_diff_l2": float(np.linalg.norm(real_np.mean(axis=0) - samples_np.mean(axis=0))),
        "std_diff_l2": float(np.linalg.norm(real_np.std(axis=0) - samples_np.std(axis=0))),
    }
