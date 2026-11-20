"""Final test evaluation helpers: stratified bootstrap, paired differences, and the take / concat datasets."""
from __future__ import annotations

import numpy as np
import pytest

from luad.data.dataset import Dataset, concat
from luad.eval.bootstrap import evaluate, metrics, stratified_resamples


def _probs(y, noise, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((len(y), 7)) * noise
    P[np.arange(len(y)), y] += 1.0
    return P / P.sum(1, keepdims=True)


def test_resamples_keep_every_class_and_its_size():
    y = np.repeat(np.arange(7), [3, 4, 9, 5, 6, 8, 7])
    for idx in stratified_resamples(y, 50, seed=0):
        assert np.array_equal(np.bincount(y[idx], minlength=7), np.bincount(y, minlength=7))


def test_resamples_are_reproducible():
    y = np.repeat(np.arange(7), 6)
    a, b = stratified_resamples(y, 5, seed=3), stratified_resamples(y, 5, seed=3)
    assert all(np.array_equal(i, j) for i, j in zip(a, b))


def test_metrics_of_a_perfect_model():
    y = np.repeat(np.arange(7), 6)
    m = metrics(y, _probs(y, 0.1, 0))
    assert m == pytest.approx({"macro_auroc": 1.0, "balanced_accuracy": 1.0, "accuracy": 1.0, "macro_f1": 1.0})


def test_paired_difference_of_a_model_with_itself_is_zero_and_cis_contain_the_point():
    y = np.repeat(np.arange(7), 6)
    good, bad = _probs(y, 3.0, 1), _probs(y, 30.0, 2)
    m, paired = evaluate(y, {"row": bad, "same": good, "fused": good}, n_boot=200, seed=0)
    same = paired[paired.row == "same"]
    assert (same.fused_minus_row == 0).all() and (same.ci_low == 0).all() and (same.ci_high == 0).all()
    assert ((m.ci_low <= m.value + 1e-9) & (m.value <= m.ci_high + 1e-9)).all()
    better = paired[(paired.row == "row") & (paired.metric == "macro_auroc")].iloc[0]
    assert better.fused_minus_row > 0 and better.share_resamples_fused_not_better < 0.05


def test_take_and_concat_keep_patients_streams_and_order():
    rng = np.random.default_rng(0)
    ds = Dataset([f"W{i}" for i in range(10)], np.arange(10) % 7, {"a": rng.random((10, 3))}, rng.random((10, 2)),
                 np.arange(10) + 100, folds=[(np.arange(5), np.arange(5, 10))], fold_ids=[(0, 0)],
                 fold_streams={"b": rng.random((1, 10, 4))})
    tr, va = np.array([0, 2, 4, 6, 8]), np.array([9, 1])
    a, b = ds.take(tr), ds.take(va)
    assert a.folds == [] and a.fold_streams == {} and a.wsi_ids == ["W0", "W2", "W4", "W6", "W8"]
    c = concat(a, b)
    order = np.concatenate([tr, va])
    assert c.wsi_ids == [ds.wsi_ids[i] for i in order]
    assert np.array_equal(c.y, ds.y[order]) and np.array_equal(c.streams["a"], ds.streams["a"][order])
    assert np.array_equal(c.meta, ds.meta[order]) and np.array_equal(c.n_tiles, ds.n_tiles[order])
    with pytest.raises(ValueError):
        concat(a, Dataset(b.wsi_ids, b.y, {}, b.meta, b.n_tiles))
