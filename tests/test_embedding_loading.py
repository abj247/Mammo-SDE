"""Tests for HDF5 I/O of embeddings."""

from __future__ import annotations

import numpy as np
import torch

from mammo_sde.data.embedding_dataset import EmbeddingDataset
from mammo_sde.data.embedding_loader import generate_synthetic_embeddings, make_dataloaders
from mammo_sde.utils.io import load_embeddings_h5, save_embeddings_h5


def test_save_and_load_embeddings(tmp_path):
    embeddings = np.random.randn(100, 16).astype(np.float32)
    metadata = {
        "patient_id": np.array([f"P{i:04d}" for i in range(100)]),
        "age": np.random.uniform(40, 75, size=100).astype(np.float32),
        "birads_density": np.random.randint(1, 5, size=100).astype(np.int32),
    }
    attrs = {"encoder_type": "test", "scope": "test"}
    path = tmp_path / "test.h5"
    save_embeddings_h5(path, embeddings, metadata=metadata, attrs=attrs)

    loaded_emb, loaded_meta, loaded_attrs = load_embeddings_h5(path)
    assert loaded_emb.shape == embeddings.shape
    assert np.allclose(loaded_emb, embeddings)
    assert "patient_id" in loaded_meta
    assert loaded_attrs["encoder_type"] == "test"


def test_embedding_dataset(tmp_path):
    embeddings, metadata = generate_synthetic_embeddings(n_samples=200, embedding_dim=8, n_clusters=3, seed=1)
    path = tmp_path / "syn.h5"
    save_embeddings_h5(path, embeddings, metadata=metadata)
    ds = EmbeddingDataset(path, return_metadata=True)
    assert len(ds) == 200
    assert ds.embedding_dim == 8
    emb, meta = ds[0]
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (8,)
    assert "birads_density" in meta


def test_make_dataloaders(tmp_path):
    embeddings, metadata = generate_synthetic_embeddings(n_samples=300, embedding_dim=4, seed=2)
    path = tmp_path / "syn.h5"
    save_embeddings_h5(path, embeddings, metadata=metadata)
    train_l, val_l, test_l, full = make_dataloaders(path, batch_size=32, val_fraction=0.1, test_fraction=0.1)
    assert len(train_l.dataset) + len(val_l.dataset) + len(test_l.dataset) == 300
    batch = next(iter(train_l))
    assert batch[0].shape[1] == 4
