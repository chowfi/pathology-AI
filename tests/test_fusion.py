"""Fusion methods on synthetic data: valid probabilities, no leakage, sensible learned weights."""
from __future__ import annotations

import numpy as np
import pytest

from luad.eval.cv import cross_validate
from luad.fusion.late import COMBINERS, simplex_grid
from luad.fusion.registry import build_model

from helpers import CLF, _toy

BASE = {"models": ["a", "noise"], "use_meta": True, "C": 1.0, "pca": 16}


@pytest.mark.parametrize("cfg", [{"method": "early"}, {"method": "early", "balance_blocks": True, "meta_weight": 3},
                                 *[{"method": "late", "combine": c} for c in COMBINERS],
                                 {"method": "gated", "epochs": 60, "n_seeds": 2, "hidden": 16}])
def test_every_method_gives_valid_probabilities(cfg):
    ds = _toy()
    tr, va = ds.folds[0]
    p = build_model({**BASE, **cfg}, CLF, seed=0).fit(ds, tr).predict_proba(ds, va)
    assert p.shape == (len(va), 7) and np.allclose(p.sum(1), 1, atol=1e-5) and (p >= 0).all()


def test_learned_weights_prefer_the_informative_model():
    ds = _toy(signal=3.0)
    tr, _ = ds.folds[0]
    m = build_model({**BASE, "method": "late", "combine": "learned_weights"}, CLF).fit(ds, tr)
    assert m.weights[0] >= 0.7                              # stream "a" carries the signal


def test_late_fusion_never_uses_validation_labels():
    ds = _toy()
    tr, va = ds.folds[0]
    cfg = {**BASE, "method": "late", "combine": "stack"}
    p1 = build_model(cfg, CLF).fit(ds, tr).predict_proba(ds, va)
    ds.y = ds.y.copy()
    ds.y[va] = (ds.y[va] + 3) % 7                           # scramble validation labels only
    p2 = build_model(cfg, CLF).fit(ds, tr).predict_proba(ds, va)
    assert np.allclose(p1, p2)


def test_gated_is_deterministic_and_gates_sum_to_one():
    ds = _toy()
    tr, va = ds.folds[0]
    cfg = {**BASE, "method": "gated", "epochs": 40, "n_seeds": 1, "hidden": 16}
    p1, g = build_model(cfg, CLF, seed=3).fit(ds, tr).predict_proba(ds, va, return_gates=True)
    p2 = build_model(cfg, CLF, seed=3).fit(ds, tr).predict_proba(ds, va)
    assert np.allclose(p1, p2) and g.shape == (len(va), 2) and np.allclose(g.sum(1), 1, atol=1e-5)


def test_fusion_with_signal_beats_chance_in_cv():
    ds = _toy(signal=3.0)
    r = cross_validate(lambda: build_model({**BASE, "method": "late", "combine": "mean_prob"}, CLF), ds)
    assert r["macro_auroc_mean"] > 0.7


def test_simplex_grid():
    g = simplex_grid(3)
    assert len(g) == 66 and np.allclose(g.sum(1), 1)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        build_model({**BASE, "method": "magic"}, CLF)
