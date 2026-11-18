"""Embedding pipeline pieces that run without a GPU or model weights."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
import torch

from luad.config import load_config
from luad.data.slide_io import rescale, rescale_tiles, scaled_size
from luad.embeddings.io import load_tiles, read_h5, write_h5
from luad.embeddings.models import Encoder, pool_fn

CFG = load_config("embeddings")


def test_virchow2_pooling_skips_register_tokens():
    o = torch.zeros(2, 261, 1280)
    o[:, 0] = 1.0                    # CLS
    o[:, 1:5] = 100.0                # registers: must NOT enter the mean
    o[:, 5:] = 3.0                   # patch tokens
    v = pool_fn(CFG["models"]["virchow2"])(o)
    assert v.shape == (2, 2560)
    assert torch.all(v[:, :1280] == 1.0) and torch.all(v[:, 1280:] == 3.0)


def test_declared_dims_and_normalisation():
    m = CFG["models"]
    assert (m["uni2h"]["dim"], m["virchow2"]["dim"], m["hoptimus1"]["dim"]) == (1536, 2560, 1536)
    assert m["uni2h"]["mean"] == m["virchow2"]["mean"] == [0.485, 0.456, 0.406]
    assert m["hoptimus1"]["mean"] == [0.707223, 0.578729, 0.703617]
    assert all(len(s["revision"]) == 40 for s in m.values())          # pinned to exact commits


def test_encoder_normalises_before_the_model():
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x
            return x.mean(dim=(2, 3))

    spec = CFG["models"]["hoptimus1"]
    mean = torch.tensor(spec["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(spec["std"]).view(1, 3, 1, 1)
    enc = Encoder("hoptimus1", spec, Spy(), mean, std, pool_fn(spec))
    x01 = torch.rand(2, 3, 8, 8)
    out = enc.embed(x01)
    assert out.dtype == torch.float16
    assert torch.allclose(seen["x"], (x01 - mean) / std)


def test_h5_round_trip(tmp_path):
    f = np.random.rand(5, 16).astype(np.float16)
    c = np.arange(10, dtype=np.int32).reshape(5, 2)
    p = tmp_path / "m" / "WSI-1.h5"
    write_h5(p, f, c, {"model": "m", "tiling": {"tile_px": 224}, "torch": torch.__version__})
    f2, c2, a = read_h5(p)
    assert (f2 == f).all() and (c2 == c).all() and json.loads(a["tiling"])["tile_px"] == 224
    assert not p.with_suffix(".h5.tmp").exists()


def _fake_slide(tmp_path, mpp=0.41533, w=1300, h=900):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    slides, tiles = tmp_path / "slides", tmp_path / "tiles"
    slides.mkdir()
    tiles.mkdir()
    ppcm = 1e4 / mpp
    tifffile.imwrite(slides / "WSI-9_4x.tif", img, tile=(256, 256), resolution=(ppcm, ppcm),
                     resolutionunit="CENTIMETER", photometric="rgb")
    sw, sh = scaled_size(w, h, mpp, 0.5)
    coords = np.array([[0, 0], [224, 448], [448, 224]], np.int32)
    settings = {"target_mpp": 0.5, "scaled_size": [sw, sh], "tile_px": 224}
    np.savez(tiles / "WSI-9.npz", coords=coords, settings=json.dumps(settings))
    return img, slides, tiles, coords


def test_load_tiles_crops_the_rescaled_image(tmp_path):
    img, slides, tiles_dir, coords = _fake_slide(tmp_path)
    tiles, tiling = load_tiles("WSI-9", slides, tiles_dir)
    big = rescale(img, 0.41533, 0.5)
    assert tiles.shape == (3, 224, 224, 3)
    for t, (x, y) in zip(tiles, coords):
        d = np.abs(t.astype(int) - big[y:y + 224, x:x + 224].astype(int))
        assert d.max() <= 2 and (d > 0).mean() < 5e-3      # per-tile resampling == full rescale (float rounding)


def test_rescale_tiles_matches_full_rescale_at_borders():
    img = np.random.default_rng(1).integers(0, 255, (700, 1000, 3), dtype=np.uint8)
    sw, sh = scaled_size(1000, 700, 0.41533, 0.5)
    coords = np.array([[0, 0], [sw - 224, sh - 224], [224, 0]], np.int32)   # corners touch the image border
    a = rescale_tiles(img, 0.41533, 0.5, coords, 224)
    big = rescale(img, 0.41533, 0.5)
    for t, (x, y) in zip(a, coords):
        d = np.abs(t.astype(int) - big[y:y + 224, x:x + 224].astype(int))
        assert d.max() <= 2 and (d > 0).mean() < 5e-3


def test_load_tiles_rejects_changed_geometry(tmp_path):
    _, slides, tiles_dir, coords = _fake_slide(tmp_path)
    np.savez(tiles_dir / "WSI-9.npz", coords=coords,
             settings=json.dumps({"target_mpp": 0.5, "scaled_size": [1, 1], "tile_px": 224}))
    with pytest.raises(ValueError):
        load_tiles("WSI-9", slides, tiles_dir)
