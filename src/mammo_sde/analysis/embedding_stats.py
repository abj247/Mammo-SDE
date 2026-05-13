"""Basic descriptive stats on encoder embeddings (no model fitting)."""

from __future__ import annotations

import numpy as np


def basic_stats(embeddings: np.ndarray) -> dict:
    """Compute per-dim and global stats on a set of embeddings."""
    N, D = embeddings.shape
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "n_samples": int(N),
        "embedding_dim": int(D),
        "per_dim_mean": embeddings.mean(axis=0).tolist(),
        "per_dim_std": embeddings.std(axis=0).tolist(),
        "global_mean_norm": float(norms.mean()),
        "global_std_norm": float(norms.std()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "expected_norm_sqrt_d": float(np.sqrt(D)),
        "norm_ratio_to_sqrt_d": float(norms.mean() / np.sqrt(D)),
        "n_dead_dims": int((embeddings.std(axis=0) < 1e-4).sum()),
    }


def pca_scree(embeddings: np.ndarray, top_k: int | None = None) -> dict:
    """Return cumulative explained variance ratios from PCA."""
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    # Use SVD on X (avoid forming Σ explicitly for high-dim)
    # Subsample to 20k for speed if data is huge
    if X.shape[0] > 20_000:
        idx = np.random.default_rng(42).choice(X.shape[0], size=20_000, replace=False)
        X = X[idx]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    explained_var = (S**2) / (X.shape[0] - 1)
    total = explained_var.sum()
    ratios = explained_var / total
    cum = np.cumsum(ratios)
    if top_k is not None:
        ratios = ratios[:top_k]
        cum = cum[:top_k]
    return {
        "explained_variance_ratio": ratios.tolist(),
        "cumulative_explained_variance": cum.tolist(),
        "n_components_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "n_components_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
    }


def pairwise_cosine_sample(embeddings: np.ndarray, n_pairs: int = 10_000, seed: int = 42) -> np.ndarray:
    """Sample random pairs and return cosine similarities."""
    rng = np.random.default_rng(seed)
    N = embeddings.shape[0]
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    a = embeddings[i]
    b = embeddings[j]
    num = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / denom
