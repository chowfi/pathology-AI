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


def test_exploration_trials_do_not_count_toward_patience():
    ds = _toy()
    space = {"C": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]}
    r = run_search(ds, lambda c: EarlyFusionLR(c, CLF), GridProposer(space, 0), {"blocks": ["noise"]},
                   budget=60, patience=2, accept_k=1e6, n_jobs=1, explore=4)
    assert len(r.history) == 6 and "without an accepted" in r.stop_reason    # 4 free trials + 2 counted


def test_best_explored_incumbent_removes_the_first_trial_advantage():
    """v4b rule: nothing is accepted while exploring; at the end the best explored trial becomes the best."""
    ds = _toy()
    space = {"C": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]}
    mk = lambda c: EarlyFusionLR(c, CLF)                                        # noqa: E731
    r = run_search(ds, mk, GridProposer(space, 0), {"blocks": ["a"]}, budget=60, patience=60, accept_k=1.0,
                   n_jobs=1, explore=4, incumbent="best_explored")
    explored = r.history[:4]
    assert all(h["decision"] == "explore" and h["gain"] is None for h in explored)
    top = max(explored, key=lambda h: h["macro_auroc_mean"])
    assert explored[-1]["best_trial"] == top["trial"] and all(h["best_trial"] is None for h in explored[:-1])
    later = r.history[4:]
    assert later and all(h["decision"] in ("accept", "reject") for h in later)
    first_later = later[0]
    d = np.array(first_later["fold_macro_auroc"]) - np.array(top["fold_macro_auroc"])
    assert np.isclose(first_later["gain"], d.mean(), atol=1e-4)                  # compared with the best explored
    # the search ends inside the exploration phase: the best explored trial is still the result
    r2 = run_search(ds, mk, GridProposer(space, 0), {"blocks": ["a"]}, budget=3, patience=60, accept_k=1.0,
                    n_jobs=1, explore=4, incumbent="best_explored")
    assert r2.best["trial"] == max(r2.history, key=lambda h: h["macro_auroc_mean"])["trial"]


def test_fold_specific_streams_reach_each_fold_and_plain_access_is_refused():
    import pytest
    ds = _toy(signal=3.0)
    sig = ds.streams.pop("a")
    k = len(ds.folds)
    # fold 0 sees the informative vectors, every other fold sees pure noise
    noise = np.random.default_rng(9).standard_normal(sig.shape)
    ds.fold_streams["a"] = np.stack([sig if j == 0 else noise for j in range(k)])
    with pytest.raises(KeyError):
        ds.get("a")
    r = cross_validate(lambda: EarlyFusionLR({"blocks": ["a"], "C": 1.0}, CLF), ds)
    assert r["macro_auroc"][0] > 0.9 and r["macro_auroc"][1:].mean() < 0.7
    assert np.array_equal(ds.for_fold(0).get("a"), sig) and not ds.for_fold(0).fold_streams
    assert not ds.subset(np.arange(70), 5, 1, 0).fold_streams        # new folds: must be rebuilt


def test_fold_stream_cache_is_checked_and_read_under_its_own_key(tmp_path):
    import pytest
    from luad.data.dataset import load_fold_streams
    ds = _toy()
    Z = np.random.default_rng(0).standard_normal((len(ds.folds), len(ds.y), 4))
    val = {f"val_{k}": np.sort(v) for k, (_, v) in enumerate(ds.folds)}
    np.savez(tmp_path / "joint.npz", Z_a=Z, wsi_id=np.array(ds.wsi_ids), **val)       # v7 joint file: Z_<model>
    assert np.array_equal(load_fold_streams(ds, "a", tmp_path / "joint.npz", key="Z_a"), Z)
    val["val_0"] = val["val_0"][1:]                                                     # made for other folds
    np.savez(tmp_path / "bad.npz", Z=Z, wsi_id=np.array(ds.wsi_ids), **val)
    with pytest.raises(ValueError):
        load_fold_streams(ds, "a", tmp_path / "bad.npz")
