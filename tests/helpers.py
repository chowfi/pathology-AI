"""Synthetic data shared by the search and fusion tests."""
from __future__ import annotations

import numpy as np

from luad.data.dataset import Dataset

CLF = {"class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000, "tol": 1e-4}


def _toy(n=140, d=40, signal=2.0, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 7
    X = rng.standard_normal((n, d))
    X[np.arange(n), y] += signal                       # class k lights up column k
    meta = np.column_stack([rng.normal(55, 10, n), rng.integers(0, 2, n)])
    ds = Dataset([f"WSI-{i}" for i in range(n)], y, {"a": X, "noise": rng.standard_normal((n, d))}, meta,
                 rng.integers(3000, 30000, n))
    for r in range(2):
        groups = np.random.default_rng(r).permutation(n // 7)[np.arange(n) // 7]   # 7 patients = one per class
        for k in range(5):
            val = np.flatnonzero(groups % 5 == k)                                    # stratified
            ds.folds.append((np.setdiff1d(np.arange(n), val), val))
            ds.fold_ids.append((r, k))
    return ds
