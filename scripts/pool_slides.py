"""Stage 5: mean-pool the tile embeddings into one vector per slide, per model.

Reads  artifacts/embeddings/<model>/<WSI_ID>.h5   (gitignored, 27 GB)
Writes artifacts/features/mean_<model>.npz        (small, committed: lets anyone rerun every model from here)

Rows follow artifacts/splits/cohort.csv order. Labels and splits are NOT stored here; they are joined by WSI_ID
from the split files, so the features cannot carry a stale label. n_tiles is stored for the tile-count shortcut
check only; it is never a model input.

Usage:
  uv run python scripts/pool_slides.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from luad.config import load_config, repo_path
from luad.embeddings.io import h5_path
from luad.embeddings.pool import pool_slide, save_slide_features


def main() -> None:
    ecfg, dcfg = load_config("embeddings"), load_config("data")
    emb_dir = repo_path(ecfg["paths"]["out_dir"])
    out_dir = repo_path("artifacts/features")
    wsi_ids = pd.read_csv(repo_path(dcfg["paths"]["splits_dir"]) / "cohort.csv")["WSI_ID"].tolist()
    counts = {}
    for model in ecfg["models"]:
        vecs, n_tiles, attrs = [], [], None
        for w in wsi_ids:
            v, n, a = pool_slide(h5_path(emb_dir, model, w))
            vecs.append(v)
            n_tiles.append(n)
            attrs = attrs or a
            if a["revision"] != attrs["revision"]:
                raise ValueError(f"{w}: {model} revision differs from the other slides")
        X = np.stack(vecs)
        meta = {**{k: (int(v) if k == "dim" else str(v)) for k, v in attrs.items()}, "aggregation": "mean",
                "accumulate": "float64", "n_slides": len(wsi_ids),
                "created": dt.datetime.now().isoformat(timespec="seconds")}
        save_slide_features(out_dir / f"mean_{model}.npz", wsi_ids, X, np.array(n_tiles), meta)
        counts[model] = np.array(n_tiles)
        print(f"{model}: X {X.shape}, |x| mean {np.linalg.norm(X, axis=1).mean():.2f}, "
              f"tiles {sum(n_tiles):,}", flush=True)
    first = next(iter(counts.values()))
    assert all((c == first).all() for c in counts.values()), "tile counts differ between models"


if __name__ == "__main__":
    main()
