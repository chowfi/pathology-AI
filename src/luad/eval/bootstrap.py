"""Test-set metrics with bootstrap uncertainty (scripts/final_test.py).

Resamples are drawn within each class, so every class keeps its size and every one-vs-rest AUROC stays defined
(the 42 test patients include classes with only a few patients). Paired differences use the SAME resamples for
both models, so the shared patient-level noise cancels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from luad.eval.metrics import N_CLASSES, balanced_accuracy, macro_auroc

METRICS = ["macro_auroc", "balanced_accuracy", "accuracy", "macro_f1"]


def metrics(y: np.ndarray, P: np.ndarray) -> dict[str, float]:
    pred = P.argmax(1)
    return {"macro_auroc": macro_auroc(y, P), "balanced_accuracy": balanced_accuracy(y, P),
            "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=range(N_CLASSES), zero_division=0))}


def stratified_resamples(y: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    """n bootstrap resamples of the patient indices, drawn with replacement within each class."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    return [np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(n)]


def evaluate(y: np.ndarray, probs: dict[str, np.ndarray], n_boot: int, seed: int,
             ref: str = "fused") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every row x metric with percentile 95% CIs, `ref` - row paired differences with 95% CIs).

    share_resamples_ref_not_better: fraction of resamples in which `ref` does not beat the row (a bootstrap
    one-sided check, not a p-value with a multiple-comparison correction)."""
    point = {r: metrics(y, P) for r, P in probs.items()}
    boot: dict[str, list[dict]] = {r: [] for r in probs}
    for idx in stratified_resamples(y, n_boot, seed):
        for r, P in probs.items():
            boot[r].append(metrics(y[idx], P[idx]))
    rows, paired = [], []
    for r in probs:
        for k in METRICS:
            v = np.array([b[k] for b in boot[r]])
            rows.append({"row": r, "metric": k, "value": round(point[r][k], 4),
                         "ci_low": round(float(np.percentile(v, 2.5)), 4),
                         "ci_high": round(float(np.percentile(v, 97.5)), 4)})
            if r != ref:
                d = np.array([f[k] - b[k] for f, b in zip(boot[ref], boot[r])])
                paired.append({"row": r, "metric": k, f"{ref}_minus_row": round(point[ref][k] - point[r][k], 4),
                               "ci_low": round(float(np.percentile(d, 2.5)), 4),
                               "ci_high": round(float(np.percentile(d, 97.5)), 4),
                               f"share_resamples_{ref}_not_better": round(float((d <= 0).mean()), 4)})
    return pd.DataFrame(rows), pd.DataFrame(paired)
