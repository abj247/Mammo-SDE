"""GMM-specific plots: BIC/AIC curves, cluster heatmaps, t-SNE colored by cluster."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_bic_aic_curve(
    k_values: list[int],
    bic: list[float],
    aic: list[float],
    best_k_bic: int,
    best_k_aic: int,
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(k_values, bic, marker="o", linewidth=2, color="crimson", label="BIC")
    ax.plot(k_values, aic, marker="s", linewidth=2, color="steelblue", label="AIC")
    ax.axvline(best_k_bic, color="crimson", linestyle="--", alpha=0.6, label=f"best K (BIC) = {best_k_bic}")
    ax.axvline(best_k_aic, color="steelblue", linestyle=":", alpha=0.6, label=f"best K (AIC) = {best_k_aic}")
    ax.set_xlabel("Number of components K")
    ax.set_ylabel("Information criterion (lower = better)")
    ax.set_title("GMM model selection: BIC / AIC vs K")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ll_convergence(
    ll_per_k: list[float],
    k_values: list[int],
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(k_values, ll_per_k, marker="o", linewidth=2, color="seagreen")
    ax.set_xlabel("Number of components K")
    ax.set_ylabel("Final log-likelihood (higher = better fit)")
    ax.set_title("Final log-likelihood vs K")
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cluster_scatter(
    embedding_2d: np.ndarray,
    cluster_ids: np.ndarray,
    title: str,
    output_path: str | Path,
    point_size: float = 4.0,
) -> None:
    """Generic 2D scatter colored by cluster id."""
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    K = int(cluster_ids.max()) + 1
    palette = sns.color_palette("tab20", n_colors=max(K, 20))
    for k in range(K):
        mask = cluster_ids == k
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=point_size,
            color=palette[k % len(palette)],
            alpha=0.65,
            label=f"cluster {k}",
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if K <= 20:
        ax.legend(markerscale=2, fontsize=8, loc="best", framealpha=0.9)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_categorical_overlay(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    label_name: str,
    title: str,
    output_path: str | Path,
    point_size: float = 4.0,
) -> None:
    """2D scatter colored by a categorical label (BI-RADS, cancer y/n, etc.)."""
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    unique = np.unique(labels)
    palette = sns.color_palette("husl", n_colors=len(unique))
    for i, val in enumerate(unique):
        mask = labels == val
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=point_size,
            color=palette[i],
            alpha=0.7,
            label=f"{label_name}={val}",
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(markerscale=2, fontsize=9, loc="best", framealpha=0.9)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_continuous_overlay(
    embedding_2d: np.ndarray,
    values: np.ndarray,
    label_name: str,
    title: str,
    output_path: str | Path,
    point_size: float = 4.0,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    sc = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=values, s=point_size, cmap=cmap, alpha=0.7)
    plt.colorbar(sc, ax=ax, label=label_name)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cluster_label_confusion(
    confusion: np.ndarray,
    cluster_purity: list[float],
    label_name: str,
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, gridspec_kw={"width_ratios": [3, 1]})
    sns.heatmap(
        confusion,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "fraction"},
        ax=axes[0],
    )
    axes[0].set_xlabel(label_name)
    axes[0].set_ylabel("GMM cluster")
    axes[0].set_title(f"Cluster vs {label_name} (row-normalized)")

    axes[1].barh(np.arange(len(cluster_purity)), cluster_purity, color="seagreen")
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel("purity")
    axes[1].set_ylabel("cluster")
    axes[1].invert_yaxis()
    axes[1].set_title("Per-cluster purity")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cluster_sizes(counts: list[int], output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(np.arange(len(counts)), counts, color="steelblue", alpha=0.85)
    ax.set_xlabel("cluster id")
    ax.set_ylabel("# exams")
    ax.set_title("Cluster sizes")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cluster_mean_heatmap(means: np.ndarray, output_path: str | Path) -> None:
    """Heatmap of K cluster means (rows) × first 64 dims (cols)."""
    show_dims = min(64, means.shape[1])
    fig, ax = plt.subplots(figsize=(min(20, show_dims * 0.3), max(4, means.shape[0] * 0.3)), constrained_layout=True)
    sns.heatmap(means[:, :show_dims], cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "value"})
    ax.set_xlabel(f"embedding dim (first {show_dims})")
    ax.set_ylabel("cluster id")
    ax.set_title("Cluster mean embeddings (centroid heatmap)")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_real_vs_sampled_histograms(
    real: np.ndarray,
    sampled: np.ndarray,
    output_path: str | Path,
    n_dims_to_show: int = 8,
    bins: int = 50,
) -> None:
    n_show = min(n_dims_to_show, real.shape[1])
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.5), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for d in range(n_show):
        axes[d].hist(real[:, d], bins=bins, alpha=0.55, label="real", color="steelblue", density=True)
        axes[d].hist(sampled[:, d], bins=bins, alpha=0.55, label="GMM samples", color="crimson", density=True)
        axes[d].set_title(f"dim {d}", fontsize=9)
        axes[d].tick_params(labelsize=7)
        if d == 0:
            axes[d].legend(fontsize=7)
    for d in range(n_show, len(axes)):
        axes[d].axis("off")
    fig.suptitle("Real embeddings vs GMM samples", fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
