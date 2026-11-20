"""Slide-level mean pooling."""
from __future__ import annotations

import numpy as np

from luad.embeddings.io import write_h5
from luad.embeddings.pool import load_slide_features, mean_pool, pool_slide, save_slide_features


def test_mean_pool_does_not_overflow_float16():
    x = np.full((40_000, 4), 60.0, np.float16)        # float16 sum would be 2.4e6 > 65504 = inf
    v = mean_pool(x)
    assert v.dtype == np.float32 and np.allclose(v, 60.0)


def test_pool_slide_and_round_trip(tmp_path):
    f = np.random.default_rng(0).standard_normal((7, 5)).astype(np.float16)
    p = tmp_path / "m" / "WSI-1.h5"
    write_h5(p, f, np.zeros((7, 2), np.int32), {"model": "m", "hf_id": "x/y", "revision": "a" * 40,
                                                 "pooling": "model_output", "dim": 5})
    v, n, a = pool_slide(p)
    assert n == 7 and a["dim"] == 5 and np.allclose(v, f.astype(np.float64).mean(0), atol=1e-6)
    out = tmp_path / "mean_m.npz"
    save_slide_features(out, ["WSI-1"], v[None], np.array([n]), {"model": "m"})
    X, ids, nt, meta = load_slide_features(out)
    assert X.shape == (1, 5) and ids == ["WSI-1"] and nt[0] == 7 and meta["model"] == "m"


def test_aggregation_config_picks_the_feature_file():
    from luad.data.dataset import aggregation, feature_file
    assert feature_file("virchow2") == "prism2_virchow2.npz"
    for m in ["uni2h", "hoptimus1"]:                     # v3-v4b: mean pooling; v5: cross-fitted ABMIL cache
        agg = aggregation(m)
        assert feature_file(m) == (f"abmil_{m}_devfolds.npz" if agg == "abmil" else f"{agg}_{m}.npz")
