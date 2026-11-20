"""Failure-analysis helpers on synthetic predictions (no slides, no test set)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from luad.data.dataset import CLASSES
from luad.eval.failure import any_right_table, confusion, held_out_fold, majority, pick_examples, recall_table

from helpers import _toy


def _oof(y, right_mask, wrong_class=0, reps=3):
    P = np.full((reps, len(y), 7), 0.01)
    for i, c in enumerate(y):
        P[:, i, c if right_mask[i] else (c + 1 + wrong_class) % 7] = 0.9
    return P


def test_recall_confusion_and_any_right_agree_with_hand_counts():
    y = np.repeat(np.arange(7), 4)                          # 4 patients per class
    a_right = np.tile([True, True, False, False], 7)        # model A: 2 of 4 right in every class
    b_right = np.tile([True, False, True, False], 7)        # model B: a different 2 of 4
    oof = {"a": _oof(y, a_right), "b": _oof(y, b_right)}
    t = recall_table(y, oof)
    assert np.allclose(t.loc["a", CLASSES], 0.5) and np.isclose(t.loc["a", "balanced accuracy"], 0.5)
    cm = confusion(y, oof["a"])
    assert np.allclose(np.diag(cm), 0.5) and np.allclose(cm.sum(1), 1)
    assert np.isclose(any_right_table(y, oof, ["a", "b"])["balanced accuracy"], 0.75)


def test_majority_vote_over_repeats():
    P = np.zeros((3, 1, 7))
    P[0, 0, 2] = P[1, 0, 2] = P[2, 0, 5] = 1.0
    assert majority(P)[0] == 2


def test_examples_are_the_most_confident_errors_then_correct_references():
    cases = pd.DataFrame({"wsi_id": list("abcdef"), "true": ["Lepidic"] * 4 + ["In situ"] * 2,
                          "fused_pred": ["In situ", "In situ", "Lepidic", "In situ", "In situ", "Lepidic"],
                          "margin": [0.1, 0.5, -0.3, 0.3, -0.4, 0.2], "p_true": [0.2, 0.1, 0.6, 0.2, 0.7, 0.2]})
    assert pick_examples(cases, "Lepidic", "In situ") == ["b", "d", "c", "e"]


def test_held_out_fold_holds_the_patient_out():
    ds = _toy()
    for i in (0, 17, 99):
        j = held_out_fold(ds, i)
        assert ds.fold_ids[j][0] == 0 and i in ds.folds[j][1] and i not in ds.folds[j][0]
