"""Late fusion: one logistic regression per foundation model, then combine their class probabilities.

Combiners:
  mean_prob       average of the probabilities
  mean_logit      average of the log-probabilities, then softmax (geometric mean)
  learned_weights weighted average; weights learned inside the training fold
  weighted_vote   each model votes for its top class with its learned weight (ties broken by mean probability)
  stack           a second logistic regression on the base models' log-probabilities (+ age/sex)

Weights and the stacker are learned from INNER cross-validation on the training fold only (out-of-fold
probabilities of the base models), so they never see the outer validation patients.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from luad.data.dataset import Dataset
from luad.fusion.early import EarlyFusionLR

COMBINERS = ("mean_prob", "mean_logit", "learned_weights", "weighted_vote", "stack")
EPS = 1e-6


def simplex_grid(n: int, step: float = 0.1) -> np.ndarray:
    k = round(1 / step)
    pts = [c for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]
    return np.array(pts, float) / k


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


class LateFusion:
    """config: models, use_meta, combine, C, l2_normalize, pca, meta_weight, [stack_C], [inner_folds]."""

    def __init__(self, config: dict, clf_cfg: dict, seed: int = 0):
        if config["combine"] not in COMBINERS:
            raise ValueError(f"unknown combiner {config['combine']}")
        self.config, self.clf_cfg, self.seed = config, clf_cfg, seed

    def _base(self, model: str) -> EarlyFusionLR:
        c = self.config
        blocks = [model] + (["meta"] if c.get("use_meta", True) else [])
        return EarlyFusionLR({"blocks": blocks, "C": c["C"], "l2_normalize": c.get("l2_normalize", False),
                              "pca": c.get("pca"), "meta_weight": c.get("meta_weight", 1.0)},
                             self.clf_cfg, self.seed)

    def _inner_oof(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        """(n_models, len(idx), 7) out-of-fold probabilities inside the training fold."""
        models = self.config["models"]
        oof = np.zeros((len(models), len(idx), 7))
        skf = StratifiedKFold(self.config.get("inner_folds", 5), shuffle=True, random_state=self.seed)
        for tr, va in skf.split(idx, ds.y[idx]):
            for i, m in enumerate(models):
                oof[i, va] = self._base(m).fit(ds, idx[tr]).predict_proba(ds, idx[va])
        return oof

    def _stack_features(self, P: np.ndarray, meta: np.ndarray | None) -> np.ndarray:
        F = np.log(np.clip(P, EPS, 1)).transpose(1, 0, 2).reshape(P.shape[1], -1)
        return F if meta is None else np.hstack([F, meta])

    def fit(self, ds: Dataset, idx: np.ndarray) -> "LateFusion":
        c, y = self.config, ds.y[idx]
        comb = c["combine"]
        self.weights = np.full(len(c["models"]), 1 / len(c["models"]))
        if comb in ("learned_weights", "weighted_vote", "stack"):
            oof = self._inner_oof(ds, idx)
            if comb == "stack":
                meta = None
                if c.get("use_meta", True):
                    self.meta_scaler = StandardScaler().fit(ds.meta[idx])
                    meta = self.meta_scaler.transform(ds.meta[idx])
                self.stacker = LogisticRegression(C=c.get("stack_C", 1.0), random_state=self.seed, **self.clf_cfg)
                self.stacker.fit(self._stack_features(oof, meta), y)
            else:
                grid = simplex_grid(len(c["models"]))
                losses = [log_loss(y, np.tensordot(w, oof, axes=1)) for w in grid]
                self.weights = grid[int(np.argmin(losses))]
        self.bases = [self._base(m).fit(ds, idx) for m in c["models"]]
        return self

    def predict_proba(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        P = np.stack([b.predict_proba(ds, idx) for b in self.bases])     # (M, n, 7)
        comb = self.config["combine"]
        if comb == "mean_prob":
            return P.mean(0)
        if comb == "mean_logit":
            L = np.log(np.clip(P, EPS, 1)).mean(0)
            E = np.exp(L - L.max(1, keepdims=True))
            return E / E.sum(1, keepdims=True)
        if comb == "learned_weights":
            return np.tensordot(self.weights, P, axes=1)
        if comb == "weighted_vote":
            votes = np.zeros(P.shape[1:])
            for w, p in zip(self.weights, P):
                votes[np.arange(p.shape[0]), p.argmax(1)] += w
            out = votes + 1e-3 * P.mean(0)                  # tiny tie-breaker keeps a ranking for AUROC
            return out / out.sum(1, keepdims=True)
        meta = self.meta_scaler.transform(ds.meta[idx]) if self.config.get("use_meta", True) else None
        return self.stacker.predict_proba(self._stack_features(P, meta))
