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


class EarlyFusionLR:
    """config: blocks (list of input names), C, l2_normalize, pca (applied to embedding blocks, not to 'meta')."""

    def __init__(self, config: dict, clf_cfg: dict, seed: int = 0):
        self.config, self.clf_cfg, self.seed = config, clf_cfg, seed

    def _block_cfg(self, name: str) -> dict:
        if name in ("meta", "tile_count"):          # 1-2 columns: standardise only
            return {}
        return {"l2_normalize": self.config.get("l2_normalize", False), "pca": self.config.get("pca")}

    def fit(self, ds: Dataset, idx: np.ndarray) -> "EarlyFusionLR":
        self.tf = {b: BlockTransform(seed=self.seed, **self._block_cfg(b)) for b in self.config["blocks"]}
        Z = np.hstack([self.tf[b].fit_transform(ds.get(b)[idx]) for b in self.config["blocks"]])
        self.clf = LogisticRegression(C=self.config["C"], random_state=self.seed, **self.clf_cfg)
        self.clf.fit(Z, ds.y[idx])
        if len(self.clf.classes_) != 7:
            raise ValueError("a class is missing from the training fold")
        return self

    def predict_proba(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        Z = np.hstack([self.tf[b].transform(ds.get(b)[idx]) for b in self.config["blocks"]])
        return self.clf.predict_proba(Z)
