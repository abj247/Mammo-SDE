"""Side-by-side comparison plots: vanilla VAE vs σ-VAE vs VampPrior."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve_comparison(
    histories: dict[str, dict],
    output_path: str | Path,
    metric: str = "train_recon",
) -> None:
    """One curve per VAE variant on a chosen metric."""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, hist in histories.items():
        values = hist.get(metric, [])
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, linewidth=2, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"VAE variant comparison — {metric}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_bar_comparison(
    metric_dict: dict[str, dict[str, float]],
    metrics: Sequence[str],
    output_path: str | Path,
) -> None:
    """Grouped bar chart: rows of metrics, bars per variant."""
    variants = list(metric_dict.keys())
    n_metrics = len(metrics)
    n_variants = len(variants)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4.0, 4.5), constrained_layout=True)
    if n_metrics == 1:
        axes = [axes]
    palette = plt.cm.tab10.colors
    for i, m in enumerate(metrics):
        values = [metric_dict[v].get(m, float("nan")) for v in variants]
        axes[i].bar(variants, values, color=[palette[j % len(palette)] for j in range(n_variants)])
        axes[i].set_title(m)
        axes[i].tick_params(axis="x", rotation=20, labelsize=9)
        axes[i].grid(axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_kl_per_dim_comparison(
    kl_per_variant: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    n_variants = len(kl_per_variant)
    width = 0.8 / n_variants
    palette = plt.cm.tab10.colors
    base_x = None
    for i, (name, kl) in enumerate(kl_per_variant.items()):
        x = np.arange(len(kl))
        if base_x is None:
            base_x = x
        ax.bar(x + (i - n_variants / 2 + 0.5) * width, kl, width=width, label=name, color=palette[i % len(palette)])
    ax.set_xlabel("latent dim")
    ax.set_ylabel("KL")
    ax.set_title("KL per dim — active-dim comparison across variants")
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
