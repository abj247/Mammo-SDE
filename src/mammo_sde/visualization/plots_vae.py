"""VAE-specific plots: loss curves, reconstructions, latent space, active dims."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict, output_path: str | Path) -> None:
    epochs = np.arange(1, len(history["train_total"]) + 1)
    has_val = len(history.get("val_total", [])) > 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    axes[0].plot(epochs, history["train_total"], label="train", linewidth=2, color="steelblue")
    if has_val:
        axes[0].plot(epochs, history["val_total"], label="val", linewidth=2, color="crimson", linestyle="--")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_recon"], label="train", linewidth=2, color="steelblue")
    if has_val:
        axes[1].plot(epochs, history["val_recon"], label="val", linewidth=2, color="crimson", linestyle="--")
    axes[1].set_title("Reconstruction loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["train_kl"], label="train", linewidth=2, color="steelblue")
    if has_val:
        axes[2].plot(epochs, history["val_kl"], label="val", linewidth=2, color="crimson", linestyle="--")
    axes[2].set_title("KL divergence")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_reconstruction_scatter(
    real: np.ndarray,
    recon: np.ndarray,
    output_path: str | Path,
    n_dims_to_show: int = 6,
    n_points: int = 500,
) -> None:
    """Scatter of real vs reconstructed value on randomly sampled dims."""
    rng = np.random.default_rng(42)
    n_show = min(n_dims_to_show, real.shape[1])
    dim_idx = rng.choice(real.shape[1], size=n_show, replace=False)
    sample_idx = rng.choice(real.shape[0], size=min(n_points, real.shape[0]), replace=False)
    cols = 3
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i, d in enumerate(dim_idx):
        ax = axes[i]
        x = real[sample_idx, d]
        y = recon[sample_idx, d]
        ax.scatter(x, y, alpha=0.5, s=15, color="steelblue")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="crimson", linestyle="--", linewidth=1.5)
        ax.set_xlabel("real")
        ax.set_ylabel("recon")
        ax.set_title(f"dim {d}")
        ax.grid(alpha=0.3)
    for i in range(len(dim_idx), len(axes)):
        axes[i].axis("off")
    fig.suptitle("Reconstruction quality (real vs recon on random dims)", fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_recon_mse_hist(mse_per_sample: np.ndarray, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(mse_per_sample, bins=60, color="steelblue", alpha=0.85)
    ax.axvline(
        float(np.median(mse_per_sample)),
        color="crimson",
        linestyle="--",
        label=f"median={np.median(mse_per_sample):.4f}",
    )
    ax.set_xlabel("per-sample MSE")
    ax.set_ylabel("count")
    ax.set_title("Reconstruction MSE distribution")
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_posterior_stats(mu: np.ndarray, logvar: np.ndarray, output_path: str | Path) -> None:
    n_dims = mu.shape[1]
    var = np.exp(logvar)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    axes[0].bar(np.arange(n_dims), mu.mean(axis=0), color="steelblue", alpha=0.85)
    axes[0].set_title("Mean of μ per dim")
    axes[0].set_xlabel("latent dim")
    axes[1].bar(np.arange(n_dims), mu.var(axis=0), color="seagreen", alpha=0.85)
    axes[1].axhline(0.01, color="crimson", linestyle="--", label="active threshold")
    axes[1].set_title("Var of μ per dim (active-dim signature)")
    axes[1].set_xlabel("latent dim")
    axes[1].legend()
    axes[2].bar(np.arange(n_dims), var.mean(axis=0), color="orange", alpha=0.85)
    axes[2].axhline(1.0, color="crimson", linestyle="--", label="prior var")
    axes[2].set_title("Mean of σ² per dim")
    axes[2].set_xlabel("latent dim")
    axes[2].legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_kl_per_dim(kl_per_dim: np.ndarray, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    n = len(kl_per_dim)
    ax.bar(np.arange(n), kl_per_dim, color="purple", alpha=0.85)
    ax.set_xlabel("latent dim")
    ax.set_ylabel("KL divergence")
    ax.set_title("Average KL per latent dim (higher = more informative)")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_latent_scatter_2d(
    latent_2d: np.ndarray,
    color_values: np.ndarray,
    color_name: str,
    output_path: str | Path,
    point_size: float = 4.0,
    cmap: str = "viridis",
    categorical: bool = False,
) -> None:
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if categorical:
        unique = np.unique(color_values)
        palette = sns.color_palette("husl", n_colors=len(unique))
        for i, v in enumerate(unique):
            mask = color_values == v
            ax.scatter(
                latent_2d[mask, 0],
                latent_2d[mask, 1],
                s=point_size,
                color=palette[i],
                alpha=0.7,
                label=f"{color_name}={v}",
            )
        ax.legend(markerscale=2, fontsize=8)
    else:
        sc = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=color_values, s=point_size, cmap=cmap, alpha=0.7)
        plt.colorbar(sc, ax=ax, label=color_name)
    ax.set_title(f"Latent space (t-SNE/UMAP) — colored by {color_name}")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_real_vs_prior_samples(
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
        axes[d].hist(sampled[:, d], bins=bins, alpha=0.55, label="VAE prior sample", color="crimson", density=True)
        axes[d].set_title(f"dim {d}", fontsize=9)
        axes[d].tick_params(labelsize=7)
        if d == 0:
            axes[d].legend(fontsize=7)
    for d in range(n_show, len(axes)):
        axes[d].axis("off")
    fig.suptitle("Real embeddings vs samples decoded from VAE prior", fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_latent_traversal(
    decoded_trajectories: np.ndarray,
    dim_traversed: int,
    output_path: str | Path,
    n_output_dims_to_show: int = 6,
) -> None:
    """decoded_trajectories: (T, output_dim) — decoded vectors at T traversal steps."""
    T, D = decoded_trajectories.shape
    n_show = min(n_output_dims_to_show, D)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    steps = np.linspace(-3, 3, T)
    for d in range(n_show):
        ax.plot(steps, decoded_trajectories[:, d], linewidth=2, label=f"out dim {d}")
    ax.set_xlabel(f"latent dim {dim_traversed} value")
    ax.set_ylabel("decoded value")
    ax.set_title(f"Latent traversal along dim {dim_traversed}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
