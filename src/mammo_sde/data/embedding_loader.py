"""Convenience functions for loading embeddings and creating DataLoaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from mammo_sde.data.embedding_dataset import EmbeddingDataset


def make_dataloaders(
    h5_path: str | Path,
    batch_size: int = 256,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    num_workers: int = 0,
    normalize: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, EmbeddingDataset]:
    """Create train/val/test DataLoaders from a single HDF5 file.

    Returns
    -------
    train_loader, val_loader, test_loader, full_dataset
    """
    dataset = EmbeddingDataset(h5_path, return_metadata=True, normalize=normalize)
    n = len(dataset)
    n_test = int(test_fraction * n)
    n_val = int(val_fraction * n)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset


def generate_synthetic_embeddings(
    n_samples: int = 10_000,
    embedding_dim: int = 256,
    n_clusters: int = 5,
    cluster_separation: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate synthetic embeddings with K Gaussian clusters for smoke tests."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, cluster_separation, size=(n_clusters, embedding_dim))
    cluster_ids = rng.integers(0, n_clusters, size=n_samples)
    embeddings = centers[cluster_ids] + rng.normal(0.0, 1.0, size=(n_samples, embedding_dim))

    patient_ids = np.array([f"P{rng.integers(0, n_samples // 4):08d}" for _ in range(n_samples)])
    exam_times = rng.uniform(0.0, 10.0, size=n_samples).astype(np.float32)
    birads = rng.integers(1, 5, size=n_samples).astype(np.int32)
    age = rng.uniform(40.0, 75.0, size=n_samples).astype(np.float32)
    cancer_label = (rng.random(size=n_samples) < 0.05).astype(np.int32)

    metadata = {
        "patient_id": patient_ids,
        "exam_time": exam_times,
        "birads_density": birads,
        "age": age,
        "cancer_label": cancer_label,
        "synthetic_cluster_id": cluster_ids.astype(np.int32),
    }
    return embeddings.astype(np.float32), metadata
