"""Slide vectors: average the tile embeddings of one slide into one vector per model.

The tile embeddings are stored as float16. A float16 sum over ~35k tiles overflows (max 65504), so the mean
is always computed in float64 and stored as float32.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def mean_pool(features: np.ndarray) -> np.ndarray:
    """(N, D) tile features -> (D,) float32 mean, accumulated in float64."""
    if len(features) == 0:
        raise ValueError("slide has no tiles")
    return features.astype(np.float64).mean(axis=0).astype(np.float32)


def pool_slide(path: Path) -> tuple[np.ndarray, int, dict]:
    """Read one <model>/<WSI_ID>.h5 and return (mean vector, number of tiles, provenance attrs)."""
    with h5py.File(path, "r") as f:
        x = f["features"][:]
        attrs = {k: f.attrs[k] for k in ("model", "hf_id", "revision", "pooling", "dim")}
    v = mean_pool(x)
    if not np.isfinite(v).all():
        raise FloatingPointError(f"{path}: non-finite slide vector")
    return v, len(x), attrs


def save_slide_features(path: Path, wsi_ids: list[str], X: np.ndarray, n_tiles: np.ndarray, meta: dict) -> None:
    """One .npz per model: X (n_slides, D) float32, wsi_id, n_tiles, meta (JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), wsi_id=np.array(wsi_ids), n_tiles=n_tiles.astype(np.int64),
                        meta=json.dumps(meta))


def load_slide_features(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    z = np.load(path)
    return z["X"], z["wsi_id"].tolist(), z["n_tiles"], json.loads(str(z["meta"]))
