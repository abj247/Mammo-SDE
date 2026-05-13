"""I/O helpers for embeddings, checkpoints, and results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml


def save_embeddings_h5(
    path: str | Path,
    embeddings: np.ndarray,
    metadata: dict[str, np.ndarray] | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Save embeddings + optional metadata columns to a single HDF5 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=4)
        if metadata is not None:
            grp = f.create_group("metadata")
            for k, v in metadata.items():
                if v.dtype.kind == "U" or v.dtype.kind == "O":
                    grp.create_dataset(k, data=np.array(v, dtype=h5py.string_dtype(encoding="utf-8")))
                else:
                    grp.create_dataset(k, data=v)
        if attrs is not None:
            for k, v in attrs.items():
                f.attrs[k] = v


def load_embeddings_h5(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Load embeddings + metadata + attrs from HDF5."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        embeddings = f["embeddings"][:]
        metadata: dict[str, np.ndarray] = {}
        if "metadata" in f:
            for k in f["metadata"].keys():
                arr = f["metadata"][k][:]
                if arr.dtype.kind == "O":
                    arr = np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in arr])
                metadata[k] = arr
        attrs = dict(f.attrs)
    return embeddings, metadata, attrs


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_yaml(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_torch_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_torch_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")
