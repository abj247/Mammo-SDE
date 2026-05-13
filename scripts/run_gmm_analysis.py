#!/usr/bin/env python
"""End-to-end GMM analysis pipeline on encoder embeddings.

Steps:
    1. Load embeddings + metadata
    2. Basic embedding stats + plots
    3. Sweep K, pick best by BIC, fit, save
    4. Compute cluster-vs-label alignment (BI-RADS, cancer outcome, age)
    5. t-SNE / UMAP visualizations colored by cluster + clinical variables
    6. Sample from fitted GMM, compare to real
    7. Save all plots + JSON metrics

Usage:
    python scripts/run_gmm_analysis.py \\
        --embeddings-path outputs/embeddings/synthetic/dev10k/embeddings.h5 \\
        --output-dir outputs/gmm/synthetic_dev10k \\
        --k-min 2 --k-max 20 --cov-type diag
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mammo_sde.analysis.embedding_stats import basic_stats, pairwise_cosine_sample, pca_scree
from mammo_sde.analysis.gmm_analysis import (
    cluster_centroids_distances,
    cluster_label_alignment,
    cluster_sizes,
    sample_quality,
)
from mammo_sde.data.embedding_dataset import EmbeddingDataset
from mammo_sde.models.gmm import GMM
from mammo_sde.training.fit_gmm import save_sweep_result, sweep_k
from mammo_sde.utils.io import save_json
from mammo_sde.utils.logger import StepLogger
from mammo_sde.utils.seed import set_seed
from mammo_sde.visualization.plots_embeddings import (
    plot_norm_distribution,
    plot_pairwise_cosine,
    plot_pca_scree,
    plot_per_dim_histograms,
)
from mammo_sde.visualization.plots_gmm import (
    plot_bic_aic_curve,
    plot_categorical_overlay,
    plot_cluster_label_confusion,
    plot_cluster_mean_heatmap,
    plot_cluster_scatter,
    plot_cluster_sizes,
    plot_continuous_overlay,
    plot_ll_convergence,
    plot_real_vs_sampled_histograms,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run end-to-end GMM analysis.")
    p.add_argument("--embeddings-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=20)
    p.add_argument("--cov-type", choices=["full", "diag", "spherical", "tied"], default="diag")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--reg-covar", type=float, default=1e-6)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tsne-subsample", type=int, default=10_000, help="Subsample size for t-SNE/UMAP.")
    p.add_argument("--n-samples-for-quality", type=int, default=5000)
    p.add_argument("--skip-tsne", action="store_true")
    p.add_argument("--skip-umap", action="store_true")
    return p.parse_args()


def _tsne(X: np.ndarray, perplexity: int = 30) -> np.ndarray:
    import inspect

    from sklearn.manifold import TSNE

    # sklearn renamed n_iter → max_iter in v1.5+; accept either
    kwargs = dict(n_components=2, perplexity=perplexity, init="pca", random_state=42)
    sig = inspect.signature(TSNE.__init__).parameters
    if "max_iter" in sig:
        kwargs["max_iter"] = 750
    elif "n_iter" in sig:
        kwargs["n_iter"] = 750
    return TSNE(**kwargs).fit_transform(X)


def _umap(X: np.ndarray) -> np.ndarray:
    import umap

    return umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1).fit_transform(X)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    for d in (plots_dir, results_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = StepLogger(total_steps=7, log_file=logs_dir / "gmm.log", title="GMM Analysis Pipeline")

    with logger.step("Load embeddings"):
        ds = EmbeddingDataset(args.embeddings_path, return_metadata=True)
        X_np = ds.embeddings
        X_t = ds.all_embeddings_tensor(device=args.device)
        logger.metric("N samples", X_np.shape[0])
        logger.metric("Embedding dim", X_np.shape[1])
        logger.metric("Metadata cols", list(ds.metadata.keys()))

    with logger.step("Basic embedding statistics + plots"):
        stats = basic_stats(X_np)
        save_json(results_dir / "embedding_stats.json", stats)
        logger.metric("Mean norm", f"{stats['global_mean_norm']:.3f}")
        logger.metric("Expected √d", f"{stats['expected_norm_sqrt_d']:.3f}")
        logger.metric("Dead dims", stats["n_dead_dims"])

        pca = pca_scree(X_np, top_k=50)
        save_json(results_dir / "pca_scree.json", pca)
        cosines = pairwise_cosine_sample(X_np, n_pairs=10_000)
        save_json(
            results_dir / "pairwise_cosine.json",
            {
                "mean": float(cosines.mean()),
                "std": float(cosines.std()),
            },
        )

        plot_per_dim_histograms(X_np, plots_dir / "01_per_dim_histograms.png")
        plot_norm_distribution(X_np, plots_dir / "02_norm_distribution.png")
        plot_pca_scree(
            pca["explained_variance_ratio"], pca["cumulative_explained_variance"], plots_dir / "03_pca_scree.png"
        )
        plot_pairwise_cosine(cosines, plots_dir / "04_pairwise_cosine.png")
        logger.success("4 embedding plots saved")

    with logger.step(f"GMM sweep K={args.k_min}..{args.k_max} (cov={args.cov_type})"):
        sweep = sweep_k(
            X_t,
            k_min=args.k_min,
            k_max=args.k_max,
            cov_type=args.cov_type,
            max_iter=args.max_iter,
            tol=args.tol,
            reg_covar=args.reg_covar,
            seed=args.seed,
            device=args.device,
            selection_metric="bic",
            logger=logger,
        )
        save_sweep_result(sweep, output_dir)
        logger.metric("Best K (BIC)", sweep.best_k_bic)
        logger.metric("Best K (AIC)", sweep.best_k_aic)
        plot_bic_aic_curve(
            sweep.k_values,
            sweep.bic_per_k,
            sweep.aic_per_k,
            sweep.best_k_bic,
            sweep.best_k_aic,
            plots_dir / "05_bic_aic_curve.png",
        )
        plot_ll_convergence(sweep.log_likelihood_per_k, sweep.k_values, plots_dir / "06_log_likelihood_vs_k.png")

    with logger.step(f"Predict + analyze best GMM (K={sweep.best_k})"):
        gmm = GMM.from_state_dict(sweep.best_gmm_state, device=args.device)
        cluster_ids = gmm.predict(X_t).cpu().numpy()

        sizes = cluster_sizes(cluster_ids, gmm.K)
        centroids = cluster_centroids_distances(gmm)
        save_json(results_dir / "cluster_sizes.json", sizes)
        save_json(results_dir / "cluster_centroid_distances.json", centroids)
        plot_cluster_sizes(sizes["counts"], plots_dir / "07_cluster_sizes.png")

        means_np = gmm.means_.detach().cpu().numpy()
        plot_cluster_mean_heatmap(means_np, plots_dir / "08_cluster_mean_heatmap.png")

        alignments = {}
        for label_col in ("birads_density", "cancer_label", "synthetic_cluster_id"):
            if label_col in ds.metadata:
                labels = ds.get_metadata_column(label_col).astype(int)
                ali = cluster_label_alignment(cluster_ids, labels)
                alignments[label_col] = ali
                plot_cluster_label_confusion(
                    np.array(ali["confusion_normalized"]),
                    ali["cluster_purity"],
                    label_col,
                    plots_dir / f"09_cluster_vs_{label_col}_confusion.png",
                )
                logger.metric(f"Purity vs {label_col}", f"{ali['overall_purity']:.3f}")
        save_json(results_dir / "cluster_label_alignment.json", alignments)

    with logger.step("Compare real vs samples from GMM"):
        sample_q = sample_quality(gmm, X_t, n_samples=args.n_samples_for_quality)
        save_json(results_dir / "sample_quality.json", sample_q)
        samples, _ = gmm.sample(args.n_samples_for_quality)
        plot_real_vs_sampled_histograms(
            X_np[: args.n_samples_for_quality],
            samples.detach().cpu().numpy(),
            plots_dir / "10_real_vs_gmm_samples.png",
        )

    with logger.step("2D projections (t-SNE / UMAP)"):
        n_proj = min(args.tsne_subsample, X_np.shape[0])
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X_np.shape[0], size=n_proj, replace=False)
        X_sub = X_np[idx]
        cluster_sub = cluster_ids[idx]

        if not args.skip_tsne:
            logger.info(f"Computing t-SNE on {n_proj} points...")
            tsne_2d = _tsne(X_sub)
            plot_cluster_scatter(
                tsne_2d, cluster_sub, f"t-SNE colored by GMM cluster (K={gmm.K})", plots_dir / "11_tsne_clusters.png"
            )
            for col, fname in [
                ("birads_density", "12_tsne_birads.png"),
                ("cancer_label", "13_tsne_cancer.png"),
                ("age", "14_tsne_age.png"),
            ]:
                if col in ds.metadata:
                    vals = ds.get_metadata_column(col)[idx]
                    if col == "age":
                        plot_continuous_overlay(tsne_2d, vals, "age", "t-SNE colored by age", plots_dir / fname)
                    else:
                        plot_categorical_overlay(tsne_2d, vals, col, f"t-SNE colored by {col}", plots_dir / fname)

        if not args.skip_umap:
            logger.info(f"Computing UMAP on {n_proj} points...")
            try:
                umap_2d = _umap(X_sub)
                plot_cluster_scatter(
                    umap_2d, cluster_sub, f"UMAP colored by GMM cluster (K={gmm.K})", plots_dir / "15_umap_clusters.png"
                )
            except Exception as exc:
                logger.warn(f"UMAP failed: {exc}")

    with logger.step("Write final summary"):
        summary = {
            "embeddings_path": args.embeddings_path,
            "output_dir": str(output_dir),
            "cov_type": args.cov_type,
            "k_range": [args.k_min, args.k_max],
            "best_k_bic": sweep.best_k_bic,
            "best_k_aic": sweep.best_k_aic,
            "best_log_likelihood": gmm.history_.final_log_likelihood,
            "n_samples": ds.n_samples,
            "embedding_dim": ds.embedding_dim,
        }
        save_json(results_dir / "summary.json", summary)
        logger.table(
            headers=["metric", "value"],
            rows=[[k, str(v)] for k, v in summary.items()],
            title="GMM Analysis Summary",
        )

    logger.summary()
    logger.close()


if __name__ == "__main__":
    main()
