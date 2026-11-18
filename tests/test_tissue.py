"""Tissue mask and grid on a synthetic thumbnail: pink tissue with air holes, grey dust, white glass."""
from __future__ import annotations

import numpy as np

from luad.data.slide_io import rescale, scaled_size, thumbnail
from luad.data.tissue import block_fraction, select_tiles, tissue_mask

MPP = 4.0          # thumbnail um/px
MASK_KW = dict(median_filter_px=5, min_saturation=0.05, closing_diameter_um=250, min_object_area_mm2=0.05)


def _thumb():
    img = np.full((400, 600, 3), 235, np.uint8)                    # glass
    img[100:300, 100:400] = (200, 120, 190)                        # H&E-like pink tissue block
    yy, xx = np.mgrid[:400, :600]
    for cy, cx in [(150, 150), (200, 250), (250, 350)]:            # alveolar air spaces, ~100 um wide
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < 12 ** 2] = 235
    img[50:54, 500:504] = (90, 90, 90)                             # dust speck (grey: low saturation)
    img[350:353, 500:503] = (200, 120, 190)                        # tiny coloured speck (< 0.05 mm^2)
    return img


def test_holes_filled_specks_removed_background_kept():
    m = tissue_mask(_thumb(), MPP, **MASK_KW)
    assert not m.raw[150, 150] and m.closed[150, 150]              # air space filled by closing
    assert not m.closed[50, 502] and not m.closed[351, 501]        # dust and speck gone
    assert not m.closed[20, 20] and not m.closed[350, 50]          # glass stays background
    assert m.closed[100:300, 100:400].mean() > 0.99


def test_grid_selection_and_coordinates():
    m = tissue_mask(_thumb(), MPP, **MASK_KW)
    # thumbnail is 8x smaller than the 0.5 um/px image -> scaled image 4800 x 3200, tiles 224 px = 28 thumb px
    sel = select_tiles(m.closed, m.raw, 4800, 3200, 224, 224, 8, 0.5)
    assert tuple(sel["grid_shape"]) == (3200 // 224, 4800 // 224)
    assert (sel["coords"] % 224 == 0).all()
    x, y = sel["coords"].T // 8
    assert ((x >= 100 - 28) & (x < 400) & (y >= 100 - 28) & (y < 300)).all()   # all kept tiles touch the tissue
    assert (sel["frac_closed"] >= 0.5).all()
    assert 50 < len(sel["coords"]) < 90                            # tissue is ~ 300x200 px -> ~ 10.7 x 7.1 tiles


def test_raw_fraction_rule_drops_empty_tiles():
    closed = np.ones((56, 56), bool)                               # 2x2 tiles, all "tissue" after closing
    raw = np.zeros((56, 56), bool)
    raw[:28, :28] = True                                           # only the top-left tile has real tissue
    raw[28:31, 28:56] = True                                       # bottom-right: a thin wall, ~11% tissue
    sel = select_tiles(closed, raw, 448, 448, 224, 224, 8, 0.5, min_raw_fraction=0.10)
    assert sorted(map(tuple, sel["coords"].tolist())) == [(0, 0), (224, 224)]


def test_block_fraction():
    m = np.zeros((10, 10), bool)
    m[:5, :5] = True
    assert block_fraction(m, 5, 2, 2).tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert block_fraction(m, 4, 3, 3).shape == (3, 3)              # grid larger than mask -> zero padding


def test_geometry_shared_by_thumbnail_and_rescale():
    img = np.random.default_rng(0).integers(0, 255, (1000, 1600, 3), dtype=np.uint8)
    w, h = scaled_size(1600, 1000, 0.41533, 0.5)
    assert (w, h) == (1329, 831)
    assert rescale(img, 0.41533, 0.5).shape == (h, w, 3)
    th = thumbnail(img, 0.41533, 0.5, 8)
    assert th.shape == (round(h / 8), round(w / 8), 3)
