"""PyTorch Dataset wrapping pre-extracted exam-level embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mammo_sde.utils.io import load_embeddings_h5


@dataclass
class EmbeddingBatch:
    """One batch of embeddings + metadata."""

    embeddings: torch.Tensor  # (B, d)
    metadata: dict[str, torch.Tensor | list]


class EmbeddingDataset(Dataset):
    """In-memory dataset over a single HDF5 file of pre-extracted embeddings.

    Schema in HDF5:
        - dataset "embeddings": float32 array of shape (N, d)
        - group   "metadata":   each column is a 1-D array of length N
                                Common columns: patient_id, exam_time, birads_density,
                                age, cancer_label, study_date
        - root attrs:           encoder_name, scope, source_datalist, num_exams, etc.
    """

    def __init__(
        self,
        h5_path: str | Path,
        return_metadata: bool = True,
        normalize: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.h5_path = Path(h5_path)
        self.return_metadata = return_metadata
        self.dtype = dtype

        embeddings, metadata, attrs = load_embeddings_h5(self.h5_path)
        self.embeddings = embeddings.astype(np.float32)
        self.metadata = metadata
        self.attrs = attrs

        if normalize:
            mean = self.embeddings.mean(axis=0, keepdims=True)
            std = self.embeddings.std(axis=0, keepdims=True) + 1e-8
            self.embeddings = (self.embeddings - mean) / std
            self._norm_mean = mean
            self._norm_std = std
        else:
            self._norm_mean = None
            self._norm_std = None

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.embeddings.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        emb = torch.as_tensor(self.embeddings[idx], dtype=self.dtype)
        if not self.return_metadata:
            return emb, {}
        meta = {k: v[idx] for k, v in self.metadata.items()}
        return emb, meta

    def all_embeddings_tensor(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Return all embeddings as a single tensor — convenient for GMM EM."""
        t = torch.as_tensor(self.embeddings, dtype=self.dtype)
        if device is not None:
            t = t.to(device)
        return t

    def get_metadata_column(self, col: str) -> np.ndarray:
        """Get a single metadata column as numpy array."""
        if col not in self.metadata:
            raise KeyError(f"Metadata column {col!r} not found. Available: {list(self.metadata.keys())}")
        return self.metadata[col]
