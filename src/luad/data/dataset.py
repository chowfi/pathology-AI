"""Join slide features, labels, metadata and the fixed CV folds for the train+val patients.

The test patients are never loaded here. The final evaluation uses `load_test_set`, which writes to the
test access log.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from luad.config import load_config, repo_path
from luad.data.splits import load_split
from luad.embeddings.pool import load_slide_features

CLASSES = ["Acinar", "Cribriform", "In situ", "Lepidic", "Micropapillary", "Papillary", "Solid"]


@dataclass
class Dataset:
    wsi_ids: list[str]
    y: np.ndarray                                   # int class index into CLASSES
    streams: dict[str, np.ndarray]                  # model name -> (n, D) slide vectors
    meta: np.ndarray                                # (n, 2): age (years), sex (1 = female)
    n_tiles: np.ndarray                             # shortcut check only, never a model input by default
    folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)   # (train_idx, val_idx)
    fold_ids: list[tuple[int, int]] = field(default_factory=list)              # (repeat, fold)

    def subset(self, idx: np.ndarray, n_folds: int, n_repeats: int, seed: int) -> "Dataset":
        """The patients `idx` with NEW repeated stratified folds made only from them (for nested CV)."""
        from sklearn.model_selection import RepeatedStratifiedKFold
        sub = Dataset([self.wsi_ids[i] for i in idx], self.y[idx], {k: v[idx] for k, v in self.streams.items()},
                      self.meta[idx], self.n_tiles[idx])
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
        for k, (tr, va) in enumerate(rskf.split(np.zeros(len(idx)), sub.y)):
            sub.folds.append((tr, va))
            sub.fold_ids.append((k // n_folds, k % n_folds))
        return sub

    def get(self, name: str) -> np.ndarray:
        """A named input block: a model stream, 'meta' or 'tile_count' (log10 number of tiles)."""
        if name == "meta":
            return self.meta
        if name == "tile_count":
            return np.log10(self.n_tiles.astype(float))[:, None]
        return self.streams[name]


def _assemble(rows: pd.DataFrame, models: list[str], features_dir) -> Dataset:
    cohort = pd.read_csv(repo_path(load_config("data")["paths"]["splits_dir"]) / "cohort.csv",
                         dtype={"SampleNumber": str}).set_index("WSI_ID")
    ids = rows["WSI_ID"].tolist()
    streams, n_tiles = {}, None
    for m in models:
        X, fid, nt, _ = load_slide_features(repo_path(features_dir) / f"mean_{m}.npz")
        pos = {w: i for i, w in enumerate(fid)}
        idx = [pos[w] for w in ids]
        streams[m] = X[idx]
        n_tiles = nt[idx] if n_tiles is None else n_tiles
    c = cohort.loc[ids]
    if not (c["label"].values == rows["label"].values).all():
        raise ValueError("labels in cohort.csv and the split file disagree")
    meta = np.column_stack([c["Age"].astype(float), (c["Sex"] == "Female").astype(float)])
    y = np.array([CLASSES.index(lbl) for lbl in rows["label"]])
    return Dataset(ids, y, streams, meta, n_tiles)


def load_dev_set(models: list[str] | None = None) -> Dataset:
    """Train+val patients with the committed 15 CV folds."""
    dcfg, ecfg = load_config("data"), load_config("experiments")
    models = models or ecfg["models"]
    sdir = repo_path(dcfg["paths"]["splits_dir"])
    rows = load_split(sdir / f"split_seed{dcfg['seed']}.csv")              # test rows dropped
    ds = _assemble(rows, models, ecfg["features_dir"])
    folds = pd.read_csv(sdir / f"cv_folds_seed{dcfg['seed']}.csv", dtype={"SampleNumber": str})
    pos = {s: i for i, s in enumerate(rows["SampleNumber"])}
    if set(folds["SampleNumber"]) != set(pos):
        raise ValueError("CV folds and train+val patients differ")
    for (r, k), g in folds.groupby(["repeat", "fold"]):
        val = np.sort([pos[s] for s in g["SampleNumber"]])
        train = np.setdiff1d(np.arange(len(rows)), val)
        ds.folds.append((train, val))
        ds.fold_ids.append((int(r), int(k)))
    return ds


def load_test_set(reason: str, models: list[str] | None = None) -> tuple[Dataset, Dataset]:
    """(all train+val patients, test patients). Logged to the test access log; call once, at the end."""
    dcfg, ecfg = load_config("data"), load_config("experiments")
    models = models or ecfg["models"]
    split_csv = repo_path(dcfg["paths"]["splits_dir"]) / f"split_seed{dcfg['seed']}.csv"
    rows = load_split(split_csv, include_test=True, reason=reason,
                      test_access_log=repo_path(dcfg["paths"]["test_access_log"]))
    dev = _assemble(rows[rows["split"] != "test"].reset_index(drop=True), models, ecfg["features_dir"])
    test = _assemble(rows[rows["split"] == "test"].reset_index(drop=True), models, ecfg["features_dir"])
    return dev, test
