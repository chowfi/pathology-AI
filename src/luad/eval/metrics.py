"""Primary metrics: macro one-vs-rest AUROC (plus per class) and balanced accuracy."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

N_CLASSES = 7


def macro_auroc(y: np.ndarray, proba: np.ndarray) -> float:
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=np.arange(N_CLASSES)))


def per_class_auroc(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """AUROC of class k vs the rest; NaN if class k is absent from y."""
    out = np.full(proba.shape[1], np.nan)
    for k in range(proba.shape[1]):
        pos = y == k
        if 0 < pos.sum() < len(y):
            out[k] = roc_auc_score(pos, proba[:, k])
    return out


def balanced_accuracy(y: np.ndarray, proba: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, proba.argmax(1)))


def score(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {"macro_auroc": macro_auroc(y, proba), "balanced_accuracy": balanced_accuracy(y, proba)}
