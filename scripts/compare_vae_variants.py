#!/usr/bin/env python
"""Compare results across VAE variants (vanilla / σ-VAE / VampPrior).

Reads results from multiple --runs and produces side-by-side comparison plots
and a summary table.

Usage:
    python scripts/compare_vae_variants.py \\
        --runs outputs/vae/run outputs/sigma_vae/run outputs/vamp_vae/run \\
        --names vanilla sigma vamp \\
        --output-dir outputs/comparison/run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mammo_sde.utils.io import load_json, save_json
from mammo_sde.utils.logger import StepLogger
from mammo_sde.visualization.plots_compare import (
    plot_kl_per_dim_comparison,
    plot_metric_bar_comparison,
    plot_training_curve_comparison,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="Output dirs of finished VAE runs.")
    p.add_argument("--names", nargs="+", default=None, help="Display names for each run (same length as --runs).")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = args.names or [Path(r).name for r in args.runs]
    if len(names) != len(args.runs):
        raise SystemExit("--names must have the same length as --runs")

    output_dir = Path(args.output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger = StepLogger(
        total_steps=4,
        log_file=output_dir / "logs" / "compare.log",
        title="VAE Variant Comparison",
    )

    with logger.step("Load histories and metrics"):
        histories: dict[str, dict] = {}
        recon_metrics: dict[str, dict] = {}
        active: dict[str, dict] = {}
        klpd: dict[str, np.ndarray] = {}
        prior_q: dict[str, dict] = {}
        for run_dir, name in zip(args.runs, names, strict=False):
            run_dir = Path(run_dir)
            histories[name] = load_json(run_dir / "results" / "history.json")
            recon_metrics[name] = load_json(run_dir / "results" / "reconstruction_metrics.json")
            active[name] = load_json(run_dir / "results" / "active_dimensions.json")
            kl_data = load_json(run_dir / "results" / "kl_per_dim.json")
            klpd[name] = np.array(kl_data["kl_per_dim_mean"])
            prior_q[name] = load_json(run_dir / "results" / "prior_sample_quality.json")
            logger.metric(f"Loaded {name}", run_dir)

    with logger.step("Training curve comparisons"):
        for metric in ("train_total", "train_recon", "train_kl", "val_total", "val_recon", "val_kl"):
            try:
                plot_training_curve_comparison(
                    histories,
                    output_dir / "plots" / f"curve_{metric}.png",
                    metric=metric,
                )
            except Exception as exc:
                logger.warn(f"Skipping {metric}: {exc}")

    with logger.step("Bar-chart metric comparison"):
        summary_metrics = {}
        for name in names:
            summary_metrics[name] = {
                "recon_mse_mean": recon_metrics[name]["mse_mean"],
                "recon_mse_p95": recon_metrics[name]["mse_p95"],
                "active_dims_fraction": active[name]["fraction_active"],
                "total_kl": float(klpd[name].sum()),
                "prior_mean_diff_l2": prior_q[name]["mean_diff_l2"],
                "prior_std_diff_l2": prior_q[name]["std_diff_l2"],
            }
        save_json(output_dir / "results" / "comparison_summary.json", summary_metrics)
        plot_metric_bar_comparison(
            summary_metrics,
            metrics=("recon_mse_mean", "active_dims_fraction", "total_kl", "prior_mean_diff_l2"),
            output_path=output_dir / "plots" / "metric_comparison.png",
        )
        plot_kl_per_dim_comparison(klpd, output_dir / "plots" / "kl_per_dim_comparison.png")

    with logger.step("Print summary table"):
        headers = ["variant", "recon MSE", "p95 MSE", "active %", "total KL", "prior μ-diff"]
        rows = []
        for name in names:
            sm = summary_metrics[name]
            rows.append(
                [
                    name,
                    f"{sm['recon_mse_mean']:.4f}",
                    f"{sm['recon_mse_p95']:.4f}",
                    f"{sm['active_dims_fraction'] * 100:.1f}%",
                    f"{sm['total_kl']:.3f}",
                    f"{sm['prior_mean_diff_l2']:.3f}",
                ]
            )
        logger.table(headers, rows, title="VAE Variants Comparison")

    logger.summary()
    logger.close()


if __name__ == "__main__":
    main()
