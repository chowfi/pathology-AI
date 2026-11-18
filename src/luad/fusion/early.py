"""Early fusion: transform each input block, join them, one logistic regression on top.

A single-model row is early fusion with one block, so every row of the table goes through this same code.
All transforms (L2 norm, standardising, PCA) are fitted on the training fold only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
    """

    def __init__(self, config: dict, clf_cfg: dict, seed: int = 0):
        self.config, self.clf_cfg, self.seed = config, clf_cfg, seed

    def _block_cfg(self, name: str) -> dict:
        if name in SMALL_BLOCKS:
            return {}
        return {"l2_normalize": self.config.get("l2_normalize", False), "pca": self.config.get("pca")}

    def _scale(self, name: str, Z: np.ndarray) -> np.ndarray:
        if name in SMALL_BLOCKS:
            return Z * self.config.get("meta_weight", 1.0) if name == "meta" else Z
        if not self.config.get("balance_blocks", False):
            return Z
        # equal total variance per model, normalised to the AVERAGE block size, so the overall feature scale
        # (and therefore the meaning of C) stays the same as without balancing. The first version divided by
        # sqrt(n_cols), which shrank every block ~40x and silently made C ~2000x stronger (found by the agent's
        # smoke run, 2026-11-18: balance_blocks + C=0.01 collapsed to AUROC 0.72).
        emb = [n for n in self._ncols if n not in SMALL_BLOCKS]
        mean_cols = np.mean([self._ncols[n] for n in emb])
        return Z * np.sqrt(mean_cols / self._ncols[name])

    def _design(self, ds: Dataset, idx: np.ndarray, fit: bool) -> np.ndarray:
        Zs = {}
        for b in self.config["blocks"]:
            Zs[b] = self.tf[b].fit_transform(ds.get(b)[idx]) if fit else self.tf[b].transform(ds.get(b)[idx])
            if fit:
                self._ncols[b] = Zs[b].shape[1]
        return np.hstack([self._scale(b, Z) for b, Z in Zs.items()])

    def fit(self, ds: Dataset, idx: np.ndarray) -> "EarlyFusionLR":
        self.tf = {b: BlockTransform(seed=self.seed, **self._block_cfg(b)) for b in self.config["blocks"]}
        self._ncols = {}
        Z = self._design(ds, idx, fit=True)
        self.clf = LogisticRegression(C=self.config["C"], random_state=self.seed, **self.clf_cfg)
        self.clf.fit(Z, ds.y[idx])
        if len(self.clf.classes_) != 7:
            raise ValueError("a class is missing from the training fold")
        return self

    def predict_proba(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._design(ds, idx, fit=False))
