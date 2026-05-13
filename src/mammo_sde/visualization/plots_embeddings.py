"""Plots that describe the raw encoder embedding space (no model fit)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_per_dim_histograms(
    embeddings: np.ndarray,
    output_path: str | Path,
    n_dims_to_show: int = 16,
    bins: int = 50,
) -> None:
    """Small-multiples histogram: distribution of first n dims of the embeddings."""
    n_show = min(n_dims_to_show, embeddings.shape[1])
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for d in range(n_show):
        axes[d].hist(embeddings[:, d], bins=bins, color="steelblue", alpha=0.85)
        axes[d].set_title(f"dim {d}", fontsize=9)
        axes[d].tick_params(labelsize=7)
    for d in range(n_show, len(axes)):
        axes[d].axis("off")
    fig.suptitle("Per-dim embedding histograms (first 16 dims)", fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_norm_distribution(embeddings: np.ndarray, output_path: str | Path) -> None:
    norms = np.linalg.norm(embeddings, axis=1)
    sqrt_d = np.sqrt(embeddings.shape[1])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(norms, bins=60, color="seagreen", alpha=0.85)
    ax.axvline(sqrt_d, color="crimson", linestyle="--", linewidth=2, label=f"√d = {sqrt_d:.2f}")
    ax.set_xlabel("||z||₂")
    ax.set_ylabel("count")
    ax.set_title("Embedding norm distribution\n(should peak near √d if LayerNorm applied)")
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pca_scree(explained_variance_ratio: list[float], cumulative: list[float], output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    n = min(len(explained_variance_ratio), 50)
    x = np.arange(1, n + 1)
    ax.bar(x, explained_variance_ratio[:n], color="steelblue", alpha=0.7, label="per-component")
    ax2 = ax.twinx()
    ax2.plot(x, cumulative[:n], color="crimson", marker="o", linewidth=2, label="cumulative")
    ax2.axhline(0.90, color="gray", linestyle=":", linewidth=1)
    ax2.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylim([0, 1.05])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio", color="steelblue")
    ax2.set_ylabel("Cumulative", color="crimson")
    ax.set_title("PCA scree plot")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pairwise_cosine(cos_sims: np.ndarray, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(cos_sims, bins=60, color="purple", alpha=0.75)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.axvline(float(cos_sims.mean()), color="crimson", linestyle="--", label=f"mean={cos_sims.mean():.3f}")
    ax.set_xlabel("cos(z_i, z_j)")
    ax.set_ylabel("count")
    ax.set_title("Pairwise cosine similarity (random samples)")
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
