"""CV, early fusion and the search loop on synthetic data (no real features needed)."""
from __future__ import annotations

import numpy as np

from luad.agent.search import GridProposer, run_search
from luad.eval.cv import corrected_se, cross_validate
from luad.fusion.early import EarlyFusionLR

from helpers import CLF, _toy

def test_signal_beats_noise_and_oof_is_complete():
    ds = _toy()
    a = cross_validate(lambda: EarlyFusionLR({"blocks": ["a"], "C": 1.0}, CLF), ds)
    z = cross_validate(lambda: EarlyFusionLR({"blocks": ["noise"], "C": 1.0}, CLF), ds)
    assert a["macro_auroc_mean"] > 0.75 and abs(z["macro_auroc_mean"] - 0.5) < 0.1
    assert not np.isnan(a["oof"]).any() and len(a["folds"]) == 10


def test_transforms_are_fitted_on_the_training_fold_only():
    ds = _toy()
    m = EarlyFusionLR({"blocks": ["a", "meta"], "C": 1.0, "pca": 8, "l2_normalize": True}, CLF)
    tr, va = ds.folds[0]
    m.fit(ds, tr)
    assert np.allclose(m.tf["a"].scaler.mean_, (ds.streams["a"][tr] /
                       np.linalg.norm(ds.streams["a"][tr], axis=1, keepdims=True)).mean(0))
    assert m.tf["a"].pca_.n_components_ == 8 and m.tf["meta"].pca_ is None
    assert m.predict_proba(ds, va).shape == (len(va), 7)


def test_corrected_se_is_larger_than_naive():
    d = np.random.default_rng(0).normal(0.01, 0.02, 15)
    naive = d.std(ddof=1) / np.sqrt(15)
    assert corrected_se(d, 134, 33) > 2 * naive


def test_search_respects_budget_patience_and_acceptance(tmp_path):
    ds = _toy()
    space = {"C": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]}
    log = tmp_path / "s.jsonl"
    r = run_search(ds, lambda c: EarlyFusionLR(c, CLF), GridProposer(space, 0), {"blocks": ["a"]},
                   budget=4, patience=10, accept_k=1.0, n_jobs=1, log_path=log)
    assert len(r.history) == 4 and r.stop_reason.startswith("budget")
    assert len(log.read_text().splitlines()) == 4
    for h in r.history[1:]:
        assert (h["decision"] == "accept") == (h["gain"] > h["threshold"])
    r2 = run_search(ds, lambda c: EarlyFusionLR(c, CLF), GridProposer(space, 0), {"blocks": ["a"]},
                    budget=60, patience=60, accept_k=1.0, n_jobs=1)
    assert len(r2.history) == 6 and r2.stop_reason == "search space exhausted"
    r3 = run_search(ds, lambda c: EarlyFusionLR(c, CLF), GridProposer(space, 0), {"blocks": ["noise"]},
                    budget=60, patience=2, accept_k=1e6, n_jobs=1)
    assert len(r3.history) == 3 and "without an accepted" in r3.stop_reason
