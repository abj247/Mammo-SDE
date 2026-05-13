#!/usr/bin/env python
"""Train a VAE (vanilla / σ-VAE / VampPrior) on extracted encoder embeddings.

Usage:
    python scripts/train_vae.py \\
        --embeddings-path outputs/embeddings/synthetic/dev10k/embeddings.h5 \\
        --variant vanilla \\
        --latent-dim 32 \\
        --epochs 50 \\
        --output-dir outputs/vae/synthetic_dev10k

    python scripts/train_vae.py \\
        --embeddings-path outputs/embeddings/synthetic/dev10k/embeddings.h5 \\
        --variant sigma \\
        --latent-dim 32 --epochs 50 \\
        --output-dir outputs/sigma_vae/synthetic_dev10k

    python scripts/train_vae.py \\
        --embeddings-path outputs/embeddings/synthetic/dev10k/embeddings.h5 \\
        --variant vamp \\
        --vamp-init-from-gmm outputs/gmm/synthetic_dev10k/checkpoints/best_gmm.pt \\
        --latent-dim 32 --epochs 50 \\
        --output-dir outputs/vamp_vae/synthetic_dev10k
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mammo_sde.analysis.vae_analysis import (
    active_dimensions,
    kl_per_dim,
    posterior_statistics,
    prior_sample_quality,
    reconstruction_metrics,
)
from mammo_sde.data.embedding_loader import make_dataloaders
from mammo_sde.models import build_vae, init_pseudoinputs_from_gmm
from mammo_sde.training.train_vae import save_vae_run, train_vae
from mammo_sde.utils.io import load_torch_checkpoint, save_json
from mammo_sde.utils.logger import StepLogger
from mammo_sde.utils.seed import set_seed
from mammo_sde.visualization.plots_vae import (
    plot_kl_per_dim,
    plot_posterior_stats,
    plot_real_vs_prior_samples,
    plot_recon_mse_hist,
    plot_reconstruction_scatter,
    plot_training_curves,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--variant", choices=["vanilla", "sigma", "vamp"], required=True)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--encoder-hidden", type=int, nargs="+", default=[512, 256])
    p.add_argument("--decoder-hidden", type=int, nargs="+", default=None)
    p.add_argument("--activation", choices=["relu", "gelu", "silu", "tanh"], default="gelu")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    # σ-VAE
    p.add_argument("--sigma-mode", choices=["learned", "optimal"], default="optimal")
    # VampPrior
    p.add_argument("--n-pseudoinputs", type=int, default=50)
    p.add_argument(
        "--vamp-init-from-gmm",
        type=str,
        default=None,
        help="Path to a GMM checkpoint; pseudoinputs initialized from its cluster centers.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger = StepLogger(
        total_steps=6,
        log_file=output_dir / "logs" / "train.log",
        title=f"Train {args.variant} VAE",
    )

    with logger.step("Load embeddings and create dataloaders"):
        train_loader, val_loader, test_loader, dataset = make_dataloaders(
            args.embeddings_path,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        input_dim = dataset.embedding_dim
        logger.metric("Input dim", input_dim)
        logger.metric("N train", len(train_loader.dataset))
        logger.metric("N val", len(val_loader.dataset))
        logger.metric("N test", len(test_loader.dataset))

    with logger.step(f"Build {args.variant} VAE model"):
        kwargs = dict(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            encoder_hidden=args.encoder_hidden,
            decoder_hidden=args.decoder_hidden,
            activation=args.activation,
        )
        if args.variant == "sigma":
            kwargs["sigma_mode"] = args.sigma_mode
        if args.variant == "vamp":
            kwargs["n_pseudoinputs"] = args.n_pseudoinputs
            if args.vamp_init_from_gmm:
                logger.info(f"Initializing pseudoinputs from GMM: {args.vamp_init_from_gmm}")
                gmm_state = load_torch_checkpoint(args.vamp_init_from_gmm)
                pseudoinit = init_pseudoinputs_from_gmm(gmm_state, n_pseudoinputs=args.n_pseudoinputs)
                kwargs["pseudoinput_init"] = pseudoinit
                kwargs["n_pseudoinputs"] = pseudoinit.shape[0]
                logger.metric("Pseudoinputs initialized from GMM", pseudoinit.shape)
        model = build_vae(args.variant, **kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        logger.metric("Model parameters", n_params)

    with logger.step(f"Train for {args.epochs} epochs"):
        history = train_vae(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            device=args.device,
            logger=logger,
            log_every=max(1, args.epochs // 10),
        )

    with logger.step("Save model + history"):
        config = {
            **kwargs,
            "variant": args.variant,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "pseudoinput_init": "<torch.Tensor>" if "pseudoinput_init" in kwargs else None,
        }
        # Strip non-serializable items
        config.pop("pseudoinput_init", None)
        save_vae_run(model, history, config, output_dir)
        logger.success(f"Saved checkpoint to {output_dir / 'checkpoints' / 'final_model.pt'}")

    with logger.step("Post-training analysis on test set"):
        recon_stats = reconstruction_metrics(model, test_loader, device=args.device)
        post_stats = posterior_statistics(model, test_loader, device=args.device)
        active = active_dimensions(model, test_loader, device=args.device)
        klpd = kl_per_dim(model, test_loader, device=args.device)
        # Sample from prior, compare to real
        real_arr = np.concatenate([x.numpy() for x, _ in [(batch[0], batch[1]) for batch in test_loader]])
        prior_q = prior_sample_quality(model, real_arr, n_samples=2000, device=args.device)
        save_json(output_dir / "results" / "reconstruction_metrics.json", recon_stats)
        save_json(output_dir / "results" / "posterior_statistics.json", post_stats)
        save_json(output_dir / "results" / "active_dimensions.json", active)
        save_json(output_dir / "results" / "kl_per_dim.json", klpd)
        save_json(output_dir / "results" / "prior_sample_quality.json", prior_q)
        logger.metric("Recon MSE (mean)", f"{recon_stats['mse_mean']:.4f}")
        logger.metric("Active dims", f"{active['n_active']}/{active['n_total']}")
        logger.metric("Total KL", f"{klpd['total_kl']:.3f}")

    with logger.step("Generate plots"):
        history_dict = {
            "train_total": history.train_total,
            "train_recon": history.train_recon,
            "train_kl": history.train_kl,
            "val_total": history.val_total,
            "val_recon": history.val_recon,
            "val_kl": history.val_kl,
        }
        plot_training_curves(history_dict, output_dir / "plots" / "01_training_curves.png")

        # Generate sample reconstructions
        model.eval()
        with torch.no_grad():
            batch_x = next(iter(test_loader))[0][:1000]
            x_recon = model.reconstruction(batch_x, use_mean=True).cpu().numpy()
            mse_per_sample = ((batch_x.numpy() - x_recon) ** 2).mean(axis=1)
        plot_reconstruction_scatter(batch_x.numpy(), x_recon, output_dir / "plots" / "02_reconstruction_scatter.png")
        plot_recon_mse_hist(mse_per_sample, output_dir / "plots" / "03_recon_mse_hist.png")

        # Re-extract posterior params from test_loader for plotting (need (N, D), not aggregated stats)
        all_mu = []
        all_lv = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(args.device)
                mu, lv = model.posterior_params(x)
                all_mu.append(mu.cpu().numpy())
                all_lv.append(lv.cpu().numpy())
        all_mu_np = np.concatenate(all_mu)
        all_lv_np = np.concatenate(all_lv)
        plot_posterior_stats(all_mu_np, all_lv_np, output_dir / "plots" / "04_posterior_stats.png")
        plot_kl_per_dim(np.array(klpd["kl_per_dim_mean"]), output_dir / "plots" / "05_kl_per_dim.png")

        with torch.no_grad():
            sampled = model.sample_prior(2000, device=args.device).cpu().numpy()
        plot_real_vs_prior_samples(real_arr[:2000], sampled, output_dir / "plots" / "06_real_vs_prior_samples.png")

        logger.success("6 plots saved to outputs/plots/")

    logger.summary()
    logger.close()


if __name__ == "__main__":
    main()
