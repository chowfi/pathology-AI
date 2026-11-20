"""Join slide features, labels, metadata and the fixed CV folds for the train+val patients.

The test patients are never loaded here. The final evaluation uses `load_test_set`, which writes to the
test access log.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    # v5: supervised (ABMIL) slide vectors differ per fold: model -> (n_folds, n, D). Use for_fold(k).
    fold_streams: dict[str, np.ndarray] = field(default_factory=dict)

    def for_fold(self, k: int) -> "Dataset":
        """The dataset as seen by fold k: fold-specific streams filled in (cross-fitted inside fold k)."""
        if not self.fold_streams:
            return self
        return replace(self, streams={**self.streams, **{m: z[k] for m, z in self.fold_streams.items()}},
                       fold_streams={})

    def subset(self, idx: np.ndarray, n_folds: int, n_repeats: int, seed: int) -> "Dataset":
        """The patients `idx` with NEW repeated stratified folds made only from them (for nested CV).
        Fold-specific (ABMIL) streams are NOT carried over: they must be rebuilt for the new folds."""
        from sklearn.model_selection import RepeatedStratifiedKFold
        sub = Dataset([self.wsi_ids[i] for i in idx], self.y[idx], {k: v[idx] for k, v in self.streams.items()},
                      self.meta[idx], self.n_tiles[idx])
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
        for k, (tr, va) in enumerate(rskf.split(np.zeros(len(idx)), sub.y)):
            sub.folds.append((tr, va))
            sub.fold_ids.append((k // n_folds, k % n_folds))
        return sub

    def take(self, idx: np.ndarray) -> "Dataset":
        """The patients `idx` with their fixed streams only: no folds, no fold-specific streams."""
        idx = np.asarray(idx)
        return Dataset([self.wsi_ids[i] for i in idx], self.y[idx], {k: v[idx] for k, v in self.streams.items()},
                       self.meta[idx], self.n_tiles[idx])

    def get(self, name: str) -> np.ndarray:
        """A named input block: a model stream, 'meta' or 'tile_count' (log10 number of tiles)."""
        if name == "meta":
            return self.meta
        if name == "tile_count":
            return np.log10(self.n_tiles.astype(float))[:, None]
        if name not in self.streams and name in self.fold_streams:
            raise KeyError(f"{name!r} has fold-specific (cross-fitted) vectors: use ds.for_fold(k)")
        return self.streams[name]


FOLD_SPECIFIC = ("abmil", "joint_abmil")             # supervised aggregations: one cross-fitted vector set per fold


def aggregation(model: str, fusion: bool = False) -> str:
    """Tile -> slide aggregation of a model (conf/experiments.yaml: aggregation). fusion=True: the input streams
    of the FUSED model, which use `fusion_aggregation` when it is set (v7: 'joint_abmil', one attention over
    the tiles of all three models); the single-model rows always use `aggregation`."""
    cfg = load_config("experiments")
    if fusion and cfg.get("fusion_aggregation"):
        return cfg["fusion_aggregation"]
    return cfg.get("aggregation", {}).get(model, "mean")


def feature_file(model: str, fusion: bool = False) -> str:
    """Slide-vector file of a model under the configured aggregation. 'abmil' / 'joint_abmil': the cache of
    cross-fitted vectors of the 15 dev folds (scripts/abmil_features.py)."""
    agg = aggregation(model, fusion)
    if agg == "joint_abmil":
        return "joint_abmil_devfolds.npz"
    return f"abmil_{model}_devfolds.npz" if agg == "abmil" else f"{agg}_{model}.npz"


def _assemble(rows: pd.DataFrame, models: list[str], features_dir, fusion: bool = False) -> Dataset:
    cohort = pd.read_csv(repo_path(load_config("data")["paths"]["splits_dir"]) / "cohort.csv",
                         dtype={"SampleNumber": str}).set_index("WSI_ID")
    ids = rows["WSI_ID"].tolist()
    streams, n_tiles = {}, None
    for m in models:
        if aggregation(m, fusion) in FOLD_SPECIFIC:
            continue                                   # fold-specific: attached by load_dev_set
        X, fid, nt, _ = load_slide_features(repo_path(features_dir) / feature_file(m))
        pos = {w: i for i, w in enumerate(fid)}
        idx = [pos[w] for w in ids]
        streams[m] = X[idx]
        n_tiles = nt[idx] if n_tiles is None else n_tiles
    c = cohort.loc[ids]
    if not (c["label"].values == rows["label"].values).all():
        raise ValueError("labels in cohort.csv and the split file disagree")
    if n_tiles is None:                                # only ABMIL streams: take tile counts from a mean file
        _, fid, nt, _ = load_slide_features(repo_path(features_dir) / f"mean_{models[0]}.npz")
        pos = {w: i for i, w in enumerate(fid)}
        n_tiles = nt[[pos[w] for w in ids]]
    meta = np.column_stack([c["Age"].astype(float), (c["Sex"] == "Female").astype(float)])
    y = np.array([CLASSES.index(lbl) for lbl in rows["label"]])
    return Dataset(ids, y, streams, meta, n_tiles)


def load_fold_streams(ds: Dataset, model: str, path, key: str = "Z") -> np.ndarray:
    """Cross-fitted ABMIL vectors of the dev folds; refuses a cache made for other patients or other folds.
    key: the array in the file ('Z' for per-model ABMIL, 'Z_<model>' in the joint (v7) file)."""
    with np.load(path, allow_pickle=False) as z:
        if z["wsi_id"].tolist() != ds.wsi_ids:
            raise ValueError(f"{path}: patients differ from the dev set")
        val = [np.sort(v) for _, v in ds.folds]
        if len(val) != len(z[key]) or any(not np.array_equal(a, z[f"val_{k}"]) for k, a in enumerate(val)):
            raise ValueError(f"{path}: made for different CV folds; rebuild with scripts/abmil_features.py")
        return z[key]


def load_dev_set(models: list[str] | None = None, fusion: bool = False) -> Dataset:
    """Train+val patients with the committed 15 CV folds. fusion=True: the fused model's input streams
    (see aggregation())."""
    dcfg, ecfg = load_config("data"), load_config("experiments")
    models = models or ecfg["models"]
    sdir = repo_path(dcfg["paths"]["splits_dir"])
    rows = load_split(sdir / f"split_seed{dcfg['seed']}.csv")              # test rows dropped
    ds = _assemble(rows, models, ecfg["features_dir"], fusion)
    folds = pd.read_csv(sdir / f"cv_folds_seed{dcfg['seed']}.csv", dtype={"SampleNumber": str})
    pos = {s: i for i, s in enumerate(rows["SampleNumber"])}
    if set(folds["SampleNumber"]) != set(pos):
        raise ValueError("CV folds and train+val patients differ")
    for (r, k), g in folds.groupby(["repeat", "fold"]):
        val = np.sort([pos[s] for s in g["SampleNumber"]])
        train = np.setdiff1d(np.arange(len(rows)), val)
        ds.folds.append((train, val))
        ds.fold_ids.append((int(r), int(k)))
    for m in models:
        agg = aggregation(m, fusion)
        if agg in FOLD_SPECIFIC:
            ds.fold_streams[m] = load_fold_streams(ds, m, repo_path(ecfg["features_dir"]) / feature_file(m, fusion),
                                                   key="Z" if agg == "abmil" else f"Z_{m}")
    return ds


def concat(a: Dataset, b: Dataset) -> Dataset:
    """Patients of `a` then of `b` (no folds): the final fit uses the first part, prediction the second."""
    if set(a.streams) != set(b.streams):
        raise ValueError("the two sets have different streams")
    return Dataset(a.wsi_ids + b.wsi_ids, np.concatenate([a.y, b.y]),
                   {k: np.concatenate([a.streams[k], b.streams[k]]) for k in a.streams},
                   np.concatenate([a.meta, b.meta]), np.concatenate([a.n_tiles, b.n_tiles]))


def load_test_set(reason: str, models: list[str] | None = None) -> tuple[Dataset, Dataset]:
    """(all train+val patients, test patients). Logged to the test access log; call once, at the end.

    Only the fixed streams are filled in (e.g. Prism2). Supervised (ABMIL / joint ABMIL) vectors are NOT: the
    final test script cross-fits them on all train+val patients and attaches them (scripts/final_test.py)."""
    dcfg, ecfg = load_config("data"), load_config("experiments")
    models = models or ecfg["models"]
    split_csv = repo_path(dcfg["paths"]["splits_dir"]) / f"split_seed{dcfg['seed']}.csv"
    rows = load_split(split_csv, include_test=True, reason=reason,
                      test_access_log=repo_path(dcfg["paths"]["test_access_log"]))
    dev = _assemble(rows[rows["split"] != "test"].reset_index(drop=True), models, ecfg["features_dir"])
    test = _assemble(rows[rows["split"] == "test"].reset_index(drop=True), models, ecfg["features_dir"])
    return dev, test
