"""Final-pick rules (v6): 'auroc' keeps the search's incumbent; 'one_se_ba' takes the best balanced accuracy among
the trials whose AUROC ties with the best (gap within 1 corrected SE)."""
from __future__ import annotations

import numpy as np
import pytest

from luad.agent.search import GridProposer, run_search
from luad.agent.selection import incumbent_from_log, select, tied_with_best
from luad.fusion.early import EarlyFusionLR

from helpers import CLF, _toy

N_TRAIN, N_VAL = 134, 33


def _trial(t, folds, ba, best_trial=None):
    folds = np.asarray(folds, float)
    return {"trial": t, "fold_macro_auroc": folds.round(5).tolist(), "macro_auroc_mean": float(folds.mean()),
            "balanced_accuracy_mean": ba, "best_trial": best_trial}


def _history():
    base = 0.88 + np.random.default_rng(0).normal(0, 0.02, 15)
    wobble = np.array([0.004, -0.006] * 7 + [0.0])               # mean gap 0.0009, SE of the gap ~0.003
    return [
        _trial(1, base, 0.48),                                   # best AUROC
        _trial(2, base + wobble, 0.50),                          # gap within noise: ties
        _trial(3, base - 0.03, 0.55),                            # clearly worse AUROC, best balanced accuracy
    ]


def test_one_se_ba_picks_best_balanced_accuracy_among_auroc_ties_only():
    h = _history()
    assert [t["trial"] for t in tied_with_best(h, N_TRAIN, N_VAL)] == [1, 2]
    assert select(h, "one_se_ba", h[0], N_TRAIN, N_VAL)["trial"] == 2   # trial 3 has the best BA but is out


def test_auroc_rule_keeps_the_incumbent_and_unknown_rule_fails():
    h = _history()
    assert select(h, "auroc", h[0], N_TRAIN, N_VAL) is h[0]
    with pytest.raises(ValueError):
        select(h, "highest_sum", h[0], N_TRAIN, N_VAL)


def test_equal_balanced_accuracy_falls_back_to_auroc():
    h = _history()
    h[1]["balanced_accuracy_mean"] = h[0]["balanced_accuracy_mean"]
    assert select(h, "one_se_ba", h[0], N_TRAIN, N_VAL)["trial"] == 1


def test_run_search_applies_the_rule_and_the_log_gives_the_same_pick(tmp_path):
    import json
    ds = _toy()
    space = {"C": [1e-3, 1e-2, 0.1, 1.0, 10.0]}
    make = lambda c: EarlyFusionLR(c, CLF)
    r_auc = run_search(ds, make, GridProposer(space, 0), {"blocks": ["a"]}, budget=5, patience=5, accept_k=1.0,
                       n_jobs=1, log_path=tmp_path / "a.jsonl")
    r_ba = run_search(ds, make, GridProposer(space, 0), {"blocks": ["a"]}, budget=5, patience=5, accept_k=1.0,
                      n_jobs=1, selection="one_se_ba")
    log = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert incumbent_from_log(log)["trial"] == r_auc.best["trial"]          # default rule unchanged
    n_val = int(np.mean([len(v) for _, v in ds.folds]))
    assert r_ba.best["trial"] == select(log, "one_se_ba", incumbent_from_log(log), len(ds.y) - n_val, n_val)["trial"]
    with pytest.raises(ValueError):
        run_search(ds, make, GridProposer(space, 0), {"blocks": ["a"]}, budget=1, patience=1, accept_k=1.0,
                   n_jobs=1, selection="nope")
