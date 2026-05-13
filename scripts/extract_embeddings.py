#!/usr/bin/env python
"""Extract embeddings from a frozen encoder over a mammogram dataset.

Saves embeddings + metadata to a single HDF5 file at ``--output-dir/embeddings.h5``.

This is a TEMPLATE script. It supports three encoder types via --encoder-type:
    - mammo_clip : MammoCLIPEncoder (requires Mammo-CLIP checkpoint)
    - generic_vit : GenericViTEncoder (works for I-JEPA / HuggingFace ViTs)
    - synthetic : Generate fake embeddings for pipeline validation (no encoder needed)

For real encoders, the user must wire in the appropriate backbone factory inside
the encoder class (e.g., the ViT architecture matching the checkpoint).

Usage:
    python scripts/extract_embeddings.py \\
        --encoder-type synthetic \\
        --scope dev10k \\
        --output-dir outputs/embeddings/synthetic/dev10k

    python scripts/extract_embeddings.py \\
        --encoder-type mammo_clip \\
        --checkpoint-path /path/to/mammo_clip.pth \\
        --scope dev10k \\
        --output-dir outputs/embeddings/mammo_clip/dev10k
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mammo_sde.data.embedding_loader import generate_synthetic_embeddings
from mammo_sde.utils.io import save_embeddings_h5
from mammo_sde.utils.logger import StepLogger
from mammo_sde.utils.seed import set_seed

SCOPE_DEFAULTS = {
    "dev10k": 10_000,
    "longitudinal": 100_000,
    "full": 1_900_000,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract embeddings from a frozen encoder.")
    p.add_argument("--encoder-type", choices=["mammo_clip", "generic_vit", "synthetic"], required=True)
    p.add_argument(
        "--checkpoint-path", type=str, default=None, help="Path to encoder weights (not needed for synthetic)."
    )
    p.add_argument("--scope", choices=list(SCOPE_DEFAULTS.keys()), default="dev10k")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--embedding-dim", type=int, default=256, help="Output embedding dim (used by encoder head).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-clusters-synthetic", type=int, default=5, help="For --encoder-type synthetic.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = StepLogger(
        total_steps=3,
        log_file=output_dir / "extract.log",
        title=f"Extract embeddings: encoder={args.encoder_type}, scope={args.scope}",
    )

    n_samples = SCOPE_DEFAULTS[args.scope]
    out_path = output_dir / "embeddings.h5"

    with logger.step(f"Resolve encoder ({args.encoder_type})"):
        if args.encoder_type == "synthetic":
            logger.info(
                f"Generating synthetic embeddings: N={n_samples}, dim={args.embedding_dim}, "
                f"clusters={args.n_clusters_synthetic}"
            )
            embeddings, metadata = generate_synthetic_embeddings(
                n_samples=n_samples,
                embedding_dim=args.embedding_dim,
                n_clusters=args.n_clusters_synthetic,
                seed=args.seed,
            )
        elif args.encoder_type == "mammo_clip":
            logger.warn(
                "Mammo-CLIP encoder selected. This requires a checkpoint path and a "
                "wired-in backbone. Once you provide the path, edit the import and "
                "backbone factory in this script accordingly."
            )
            if not args.checkpoint_path:
                raise SystemExit("--checkpoint-path is required for --encoder-type mammo_clip")
            raise NotImplementedError(
                "Mammo-CLIP integration is a template. Wire in the actual backbone factory "
                "(e.g., from the batmanlab/Mammo-CLIP repo) inside this script before use."
            )
        elif args.encoder_type == "generic_vit":
            logger.warn("Generic ViT encoder selected. Wire in the actual backbone factory.")
            if not args.checkpoint_path:
                raise SystemExit("--checkpoint-path is required for --encoder-type generic_vit")
            raise NotImplementedError(
                "Generic ViT integration is a template. Wire in the actual backbone factory "
                "(e.g., timm.create_model) inside this script before use."
            )

    with logger.step("Save embeddings to HDF5"):
        attrs = {
            "encoder_type": args.encoder_type,
            "checkpoint_path": args.checkpoint_path or "synthetic",
            "scope": args.scope,
            "n_samples": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1],
            "seed": args.seed,
        }
        save_embeddings_h5(out_path, embeddings, metadata=metadata, attrs=attrs)
        logger.metric("Output file", out_path)
        logger.metric("Shape", embeddings.shape)
        logger.metric("Mean norm", f"{np.linalg.norm(embeddings, axis=1).mean():.3f}")

    with logger.step("Print summary"):
        logger.table(
            headers=["key", "value"],
            rows=[[k, v] for k, v in attrs.items()],
            title="Embedding extraction summary",
        )

    logger.summary()
    logger.close()


if __name__ == "__main__":
    main()
