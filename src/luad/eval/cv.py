"""Score one configuration on the 15 fixed CV folds (train+val patients only)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from joblib import Parallel, delayed

from luad.data.dataset import Dataset
from luad.eval.metrics import per_class_auroc, score


def _one_fold(make_model: Callable, ds: Dataset, train: np.ndarray, val: np.ndarray) -> tuple[dict, np.ndarray]:
    m = make_model().fit(ds, train)
    p = m.predict_proba(ds, val)
    s = score(ds.y[val], p)
    s["per_class_auroc"] = per_class_auroc(ds.y[val], p).tolist()
    return s, p


def cross_validate(make_model: Callable, ds: Dataset, n_jobs: int = 1) -> dict:
    """Returns per-fold scores, their mean/sd, and out-of-fold probabilities per repeat (n_repeats, n, 7)."""
    out = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(make_model, ds, tr, va) for tr, va in ds.folds)
    n_rep = max(r for r, _ in ds.fold_ids) + 1
    oof = np.full((n_rep, len(ds.y), 7), np.nan)
    for (r, _), (tr, va), (_, p) in zip(ds.fold_ids, ds.folds, out):
        oof[r, va] = p
    folds = [s for s, _ in out]
    res = {"folds": folds, "oof": oof}
    for k in ("macro_auroc", "balanced_accuracy"):
        v = np.array([f[k] for f in folds])
        res[k] = v
        res[f"{k}_mean"], res[f"{k}_sd"] = float(v.mean()), float(v.std(ddof=1))
    return res


def corrected_se(diff: np.ndarray, n_train: int, n_val: int) -> float:
    """Nadeau & Bengio (2003) corrected standard error of a mean over overlapping CV folds."""
    return float(np.sqrt((1 / len(diff) + n_val / n_train) * diff.var(ddof=1)))
