"""Tile cropping and the HDF5 embedding cache (one file per model per slide)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

from luad.data.slide_io import level_path, read_level, rescale_tiles, scaled_size


def load_tiles(wsi_id: str, slides_dir: Path, tiles_dir: Path, filt: str = "lanczos") -> tuple[np.ndarray, dict]:
    """Decode the 4x level, rescale to the tiling target mpp, and crop the saved tile coordinates.

    Returns (tiles uint8 (N, 224, 224, 3), tiling dict with coords and settings).
    """
    z = np.load(tiles_dir / f"{wsi_id}.npz")
    tiling = {k: z[k] for k in z.files if k != "settings"}
    settings = json.loads(str(z["settings"]))
    img, src_mpp = read_level(level_path(slides_dir, wsi_id))
    expect = scaled_size(img.shape[1], img.shape[0], src_mpp, settings["target_mpp"])
    if list(expect) != settings["scaled_size"]:
        raise ValueError(f"{wsi_id}: geometry changed since tiling ({expect} vs {settings['scaled_size']})")
    # per-tile resampling: never builds the full rescaled slide (a full rescale of the largest slides used >35 GB
    # RAM and the process was killed by the OOM killer on 2026-11-17)
    tiles = rescale_tiles(img, src_mpp, settings["target_mpp"], tiling["coords"], settings["tile_px"], filt)
    del img
    tiling["settings"] = settings
    return tiles, tiling


def h5_path(out_dir: Path, model: str, wsi_id: str) -> Path:
    return out_dir / model / f"{wsi_id}.h5"


def write_h5(path: Path, features: np.ndarray, coords: np.ndarray, attrs: dict) -> None:
    """Write atomically (tmp file + rename), so a crash never leaves a half file that looks finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("features", data=features, dtype="float16")
        f.create_dataset("coords", data=coords, dtype="int32")
        for k, v in attrs.items():
            # str(): some version strings are str subclasses (torch.__version__) that h5py cannot store
            f.attrs[k] = v if isinstance(v, (int, float)) else str(v) if isinstance(v, str) else json.dumps(v)
    os.replace(tmp, path)


def read_h5(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        return f["features"][:], f["coords"][:], dict(f.attrs)
