"""ABMIL attention pooling on synthetic slides (CPU): cross-fitting never leaks labels, pooled vectors are
weighted means of the slide's own tiles, results are deterministic, and attention finds sparse signal tiles."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from luad.embeddings.abmil import TileBank, crossfit, fold_features, train_abmil, pool

CFG = {"hidden": 8, "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4, "epochs": 15, "max_tiles": 64,
       "n_parts": 3}


def _slides(n=42, tiles=60, d=12, frac=0.1, signal=4.0, seed=0):
    """Each slide: mostly background tiles; a small fraction of tiles carries the class (column = class)."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 7
    feats = []
    for i in range(n):
        x = rng.standard_normal((tiles, d)).astype(np.float32)
        k = max(1, int(frac * tiles))
        x[:k, y[i]] += signal                                   # the first k tiles are the informative ones
        feats.append(x)
    return TileBank.from_arrays(feats, device="cpu"), y, int(max(1, frac * tiles))


def test_pooled_vector_is_a_weighted_mean_of_the_slides_own_tiles():
    bank, y, _ = _slides()
    net, stats = train_abmil(bank, np.arange(30), y[:30], CFG, seed=0)
    Z, att = pool(bank, net, stats, np.arange(30, 42), return_attention=True)
    for j, i in enumerate(range(30, 42)):
        x = bank.slide(i).numpy()
        assert np.isclose(att[j].sum(), 1, atol=1e-5) and (att[j] >= 0).all()
        assert np.allclose(Z[j], att[j] @ x, atol=1e-4)


def test_training_is_deterministic():
    bank, y, _ = _slides()
    a = crossfit(bank, np.arange(35), np.arange(35, 42), y[:35], CFG, seed=3)
    b = crossfit(bank, np.arange(35), np.arange(35, 42), y[:35], CFG, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_crossfit_vector_of_a_training_patient_ignores_its_own_label():
    from sklearn.model_selection import StratifiedKFold
    bank, y, _ = _slides()
    tr, ap = np.arange(35), np.arange(35, 42)
    parts = list(StratifiedKFold(3, shuffle=True, random_state=1).split(tr, y[tr]))   # same split in both runs
    z1, _ = crossfit(bank, tr, ap, y[tr], CFG, seed=1, parts=parts)
    y2 = y[tr].copy()
    y2[4] = (y2[4] + 3) % 7                                    # change the label of training patient 4 only
    z2, _ = crossfit(bank, tr, ap, y2, CFG, seed=1, parts=parts)
    assert np.allclose(z1[4], z2[4], atol=1e-6)                # its vector came from a net that never saw it
    assert not np.allclose(z1, z2)                             # (the other nets did see the change)


def test_attention_finds_the_sparse_informative_tiles():
    bank, y, k = _slides(n=70, frac=0.1)
    net, stats = train_abmil(bank, np.arange(70), y, {**CFG, "epochs": 40}, seed=0)
    _, att = pool(bank, net, stats, np.arange(70), return_attention=True)
    mass = np.mean([a[:k].sum() for a in att])
    assert mass > 3 * (k / 60)                                 # well above the 10% a mean would give them


def test_fold_features_rows_follow_the_fold():
    bank, y, _ = _slides()
    idx = np.arange(42)
    folds = [(np.arange(0, 35), np.arange(35, 42)), (np.arange(7, 42), np.arange(0, 7))]
    Z = fold_features(bank, idx, y, folds, CFG, seed=0)
    assert Z.shape == (2, 42, 12) and np.isfinite(Z).all()
    tr, va = folds[0]
    z_tr, z_va = crossfit(bank, tr, va, y[tr], CFG, seed=0)
    assert np.allclose(Z[0, tr], z_tr) and np.allclose(Z[0, va], z_va)


def test_missing_class_in_training_parts_is_an_error():
    bank, y, _ = _slides()
    with pytest.raises(ValueError):
        crossfit(bank, np.arange(10), np.arange(10, 20), y[:10], {**CFG, "n_parts": 5}, seed=0)


# ---- v7: tile-level fusion (one attention over several views of the same tiles) ------------------------------
from luad.embeddings.abmil import MultiViewBank, pool_joint, split_views, train_joint_abmil  # noqa: E402


def _views(n=42, tiles=60, frac=0.1, signal=4.0, seed=0):
    """Two views of the same tiles: classes 0-3 show in view A (12-d), classes 4-6 only in view B (8-d)."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 7
    A, B = [], []
    k = max(1, int(frac * tiles))
    for i in range(n):
        a = rng.standard_normal((tiles, 12)).astype(np.float32)
        b = rng.standard_normal((tiles, 8)).astype(np.float32)
        if y[i] < 4:
            a[:k, y[i]] += signal
        else:
            b[:k, y[i]] += signal
        A.append(a), B.append(b)
    return MultiViewBank([TileBank.from_arrays(A), TileBank.from_arrays(B)]), y, k


JCFG = {**CFG, "proj": 8}


def test_joint_pooling_uses_one_attention_for_all_views():
    bank, y, _ = _views()
    net, stats = train_joint_abmil(bank, np.arange(30), y[:30], JCFG, seed=0)
    Z, att = pool_joint(bank, net, stats, np.arange(30, 42), return_attention=True)
    za, zb = split_views(Z, bank.dims)
    for j, i in enumerate(range(30, 42)):
        xa, xb = (x.numpy() for x in bank.slide(i))
        assert np.isclose(att[j].sum(), 1, atol=1e-5)
        assert np.allclose(za[j], att[j] @ xa, atol=1e-4) and np.allclose(zb[j], att[j] @ xb, atol=1e-4)


def test_joint_crossfit_is_deterministic_and_does_not_leak_labels():
    from sklearn.model_selection import StratifiedKFold
    bank, y, _ = _views()
    tr, ap = np.arange(35), np.arange(35, 42)
    parts = list(StratifiedKFold(3, shuffle=True, random_state=1).split(tr, y[tr]))
    kw = dict(train_fn=train_joint_abmil, pool_fn=pool_joint, parts=parts)
    z1, a1 = crossfit(bank, tr, ap, y[tr], JCFG, seed=1, **kw)
    z1b, a1b = crossfit(bank, tr, ap, y[tr], JCFG, seed=1, **kw)
    assert np.array_equal(z1, z1b) and np.array_equal(a1, a1b)
    y2 = y[tr].copy()
    y2[4] = (y2[4] + 3) % 7
    z2, _ = crossfit(bank, tr, ap, y2, JCFG, seed=1, **kw)
    assert np.allclose(z1[4], z2[4], atol=1e-6) and not np.allclose(z1, z2)


def test_joint_attention_finds_signal_that_lives_in_either_view():
    bank, y, k = _views(n=70)
    net, stats = train_joint_abmil(bank, np.arange(70), y, {**JCFG, "epochs": 40}, seed=0)
    _, att = pool_joint(bank, net, stats, np.arange(70), return_attention=True)
    for group in (y < 4, y >= 4):                              # signal in view A / in view B
        assert np.mean([att[i][:k].sum() for i in np.flatnonzero(group)]) > 3 * (k / 60)


def test_views_must_hold_the_same_tiles():
    a = TileBank.from_arrays([np.zeros((5, 3), np.float32), np.zeros((4, 3), np.float32)])
    b = TileBank.from_arrays([np.zeros((5, 2), np.float32), np.zeros((6, 2), np.float32)])
    with pytest.raises(ValueError):
        MultiViewBank([a, b])


def test_fit_then_apply_on_a_second_bank_equals_crossfit():
    """The final test script trains on a bank of the dev slides and pools the test slides from another bank."""
    from luad.embeddings.abmil import apply_parts, fit_parts
    bank, y, _ = _slides()
    tr, ap = np.arange(35), np.arange(35, 42)
    z_tr, z_ap = crossfit(bank, tr, ap, y[tr], CFG, seed=5)
    dev = TileBank.from_arrays([bank.slide(i).numpy() for i in tr], device="cpu")
    test = TileBank.from_arrays([bank.slide(i).numpy() for i in ap], device="cpu")
    f_tr, nets = fit_parts(dev, np.arange(len(tr)), y[tr], CFG, seed=5)
    f_ap = apply_parts(test, nets, np.arange(len(ap)))
    assert np.array_equal(z_tr, f_tr) and np.array_equal(z_ap, f_ap)
