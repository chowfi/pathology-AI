"""Early fusion: transform each input block, join them, one logistic regression on top.

A single-model row is early fusion with one block, so every row of the table goes through this same code.
All transforms (L2 norm, standardising, PCA) are fitted on the training fold only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from luad.data.dataset import Dataset


@dataclass
class BlockTransform:
    l2_normalize: bool = False
    pca: int | None = None
    seed: int = 0
    scaler: StandardScaler = field(default=None, init=False)
    pca_: PCA = field(default=None, init=False)

    def _pre(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        if self.l2_normalize:
            X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.scaler = StandardScaler()
        Z = self.scaler.fit_transform(self._pre(X))
        if self.pca:
            k = min(self.pca, Z.shape[0] - 1, Z.shape[1])
            self.pca_ = PCA(n_components=k, svd_solver="full", random_state=self.seed)
            Z = self.pca_.fit_transform(Z)        # not whitened: strong components keep their larger variance
        return Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(self._pre(X))
        if self.pca_ is not None:
            Z = self.pca_.transform(Z)
        return Z


SMALL_BLOCKS = ("meta", "tile_count")          # 1-2 columns: standardise only, never PCA


class EarlyFusionLR:
    """Join the transformed blocks, then one logistic regression.

    config keys:
      blocks        input names, e.g. ["uni2h", "virchow2", "hoptimus1", "meta"]
      C             inverse L2 strength of the logistic regression
      l2_normalize  per-slide L2 norm of each embedding block before standardising
      pca           components per embedding block (fitted on the training fold), or None
      balance_blocks  scale each embedding block by sqrt(mean_columns / n_columns) so every model has the same
                    total variance (otherwise Virchow2's 2560 columns outweigh the 1536 of the others), while the
                    overall scale, and so the meaning of C, stays the same
      meta_weight   multiply the standardised age/sex columns; >1 means the L2 penalty shrinks them less
      block_cfg     optional {model: {l2_normalize, pca}}: per-model preprocessing instead of the shared one
      block_weights optional {model: w}: multiply that model's block by w after scaling. With an L2 penalty a
                    weight w acts like a separate inverse strength C * w^2 for that block; w = 0 drops it
    """

    def __init__(self, config: dict, clf_cfg: dict, seed: int = 0):
        self.config, self.clf_cfg, self.seed = config, clf_cfg, seed

    def _block_cfg(self, name: str) -> dict:
        if name in SMALL_BLOCKS:
            return {}
        own = self.config.get("block_cfg", {}).get(name)
        if own is not None:
            return {"l2_normalize": own.get("l2_normalize", False), "pca": own.get("pca")}
        return {"l2_normalize": self.config.get("l2_normalize", False), "pca": self.config.get("pca")}

    def _scale(self, name: str, Z: np.ndarray, weights: dict | None = None) -> np.ndarray:
        if name in SMALL_BLOCKS:
            return Z * self.config.get("meta_weight", 1.0) if name == "meta" else Z
        w = (weights if weights is not None else self.config.get("block_weights", {})).get(name, 1.0)
        if not self.config.get("balance_blocks", False):
            return Z if w == 1.0 else Z * w
        # equal total variance per model, normalised to the AVERAGE block size, so the overall feature scale
        # (and therefore the meaning of C) stays the same as without balancing. The first version divided by
        # sqrt(n_cols), which shrank every block ~40x and silently made C ~2000x stronger (found by the agent's
        # smoke run, 2026-11-18: balance_blocks + C=0.01 collapsed to AUROC 0.72).
        emb = [n for n in self._ncols if n not in SMALL_BLOCKS]
        mean_cols = np.mean([self._ncols[n] for n in emb])
        return Z * (np.sqrt(mean_cols / self._ncols[name]) * w)

    def _init_transforms(self) -> None:
        self.tf = {b: BlockTransform(seed=self.seed, **self._block_cfg(b)) for b in self.config["blocks"]}
        self._ncols = {}

    def _blocks(self, ds: Dataset, idx: np.ndarray, fit: bool) -> dict[str, np.ndarray]:
        Zs = {}
        for b in self.config["blocks"]:
            Zs[b] = self.tf[b].fit_transform(ds.get(b)[idx]) if fit else self.tf[b].transform(ds.get(b)[idx])
            if fit:
                self._ncols[b] = Zs[b].shape[1]
        return Zs

    def _join(self, Zs: dict[str, np.ndarray], weights: dict | None = None) -> np.ndarray:
        return np.hstack([self._scale(b, Z, weights) for b, Z in Zs.items()])

    def _design(self, ds: Dataset, idx: np.ndarray, fit: bool) -> np.ndarray:
        return self._join(self._blocks(ds, idx, fit))

    def fit(self, ds: Dataset, idx: np.ndarray) -> "EarlyFusionLR":
        self._init_transforms()
        Z = self._design(ds, idx, fit=True)
        self.clf = LogisticRegression(C=self.config["C"], random_state=self.seed, **self.clf_cfg)
        self.clf.fit(Z, ds.y[idx])
        if len(self.clf.classes_) != 7:
            raise ValueError("a class is missing from the training fold")
        return self

    def predict_proba(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._design(ds, idx, fit=False))


WEIGHT_VALUES = (0.0, 0.25, 0.5, 1.0)


def weight_grid(n: int) -> np.ndarray:
    """All block-weight vectors with values in WEIGHT_VALUES whose largest weight is 1 (the overall scale is
    left to C). For 3 models: 37 vectors, including (1, 0, 0) etc. = one model alone."""
    return np.array([w for w in itertools.product(WEIGHT_VALUES, repeat=n) if max(w) == 1.0])


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1))))


class EarlyTunedLR:
    """Early fusion of per-model tuned blocks with block weights learned inside the training fold.

    config keys:
      models, use_meta
      members      {model: tuned single-model + metadata baseline config}; each block keeps that config's
                   l2_normalize and pca (the members' C is not used: one C for the joint model)
      C            inverse L2 strength of the joint logistic regression
      meta_weight  as in EarlyFusionLR
      inner_folds  (default 5) inner CV used to choose the block weights (lowest out-of-fold log-loss over
                   weight_grid); the outer validation patients are never used
    After fit, `weights` = {model: chosen weight}.
    """

    def __init__(self, config: dict, clf_cfg: dict, seed: int = 0):
        if not config.get("members"):
            raise ValueError("early_tuned needs the tuned member configs ('members')")
        self.config, self.clf_cfg, self.seed = config, clf_cfg, seed
        self.models = list(config["models"])

    def _joint(self, weights: dict | None = None) -> EarlyFusionLR:
        c, mem = self.config, self.config["members"]
        return EarlyFusionLR({"blocks": self.models + (["meta"] if c.get("use_meta", True) else []),
                              "C": c["C"], "meta_weight": c.get("meta_weight", 1.0),
                              "block_cfg": {m: {"l2_normalize": mem[m].get("l2_normalize", False),
                                                "pca": mem[m].get("pca")} for m in self.models},
                              "block_weights": weights or {}}, self.clf_cfg, self.seed)

    def fit(self, ds: Dataset, idx: np.ndarray) -> "EarlyTunedLR":
        grid, y = weight_grid(len(self.models)), ds.y[idx]
        oof = np.zeros((len(grid), len(idx), 7))
        skf = StratifiedKFold(self.config.get("inner_folds", 5), shuffle=True, random_state=self.seed)
        for tr, va in skf.split(idx, y):
            e = self._joint()
            e._init_transforms()                              # transforms do not depend on the weights:
            Ztr, Zva = e._blocks(ds, idx[tr], True), e._blocks(ds, idx[va], False)   # fit them once per split
            for k, w in enumerate(grid):
                wd = dict(zip(self.models, w))
                clf = LogisticRegression(C=self.config["C"], random_state=self.seed, **self.clf_cfg)
                clf.fit(e._join(Ztr, wd), y[tr])
                if len(clf.classes_) != 7:
                    raise ValueError("a class is missing from an inner training fold")
                oof[k, va] = clf.predict_proba(e._join(Zva, wd))
        losses = [_log_loss(y, p) for p in oof]
        self.weights = dict(zip(self.models, grid[int(np.argmin(losses))].tolist()))
        self.model = self._joint(self.weights).fit(ds, idx)
        return self

    def predict_proba(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(ds, idx)
