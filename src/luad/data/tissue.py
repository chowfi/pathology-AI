"""Tissue mask on a low-resolution thumbnail, and the tile grid it selects.

Steps (all at the thumbnail scale, 4 um/px by default):
  1. saturation channel (H&E tissue is coloured; glass background is grey/white, dust is grey/black)
  2. median filter, then Otsu threshold (with a floor) -> raw mask
  3. morphological closing with a ~250 um disk -> fills alveolar air spaces inside the tissue
     (lepidic / in situ tumours line air spaces; without this their tiles would be dropped)
  4. remove objects smaller than ~0.05 mm^2 -> drops dust and specks
  5. keep a grid tile when >= 50% of it is tissue on the closed mask AND >= 10% is tissue on the raw mask
     (the second rule drops near-empty tiles: air spaces just below the closing size; checked visually)

No stain normalisation: the foundation models were trained on varied staining.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects


@dataclass
class MaskResult:
    raw: np.ndarray            # bool, after Otsu
    closed: np.ndarray         # bool, after closing + speck removal (used for tile selection)
    otsu: float                # Otsu threshold on 0-1 saturation
    threshold: float           # threshold actually used (Otsu, floored)


def saturation(thumb_rgb: np.ndarray) -> np.ndarray:
    """HSV saturation in [0, 1]."""
    return np.asarray(Image.fromarray(thumb_rgb).convert("HSV"))[..., 1].astype(np.float32) / 255


def tissue_mask(thumb_rgb: np.ndarray, thumb_mpp: float, median_filter_px: int, min_saturation: float,
                closing_diameter_um: float, min_object_area_mm2: float) -> MaskResult:
    sat = ndimage.median_filter(saturation(thumb_rgb), size=median_filter_px)
    otsu = float(threshold_otsu(sat))
    thr = max(otsu, min_saturation)
    raw = sat > thr
    radius = max(1, round(closing_diameter_um / thumb_mpp / 2))
    # pad so closing does not glue tissue to the image border
    closed = binary_closing(np.pad(raw, radius), disk(radius))[radius:-radius, radius:-radius]
    min_px = round(min_object_area_mm2 * 1e6 / thumb_mpp ** 2)
    closed = remove_small_objects(closed, min_size=min_px)
    return MaskResult(raw=raw, closed=closed, otsu=otsu, threshold=thr)


def block_fraction(mask: np.ndarray, block: int, nx: int, ny: int) -> np.ndarray:
    """Mean of `mask` over each block x block cell of an ny x nx grid (zero-padded at the border)."""
    need_h, need_w = ny * block, nx * block
    m = np.zeros((need_h, need_w), np.float32)
    h, w = min(need_h, mask.shape[0]), min(need_w, mask.shape[1])
    m[:h, :w] = mask[:h, :w]
    return m.reshape(ny, block, nx, block).mean(axis=(1, 3))


def select_tiles(closed: np.ndarray, raw: np.ndarray, scaled_w: int, scaled_h: int, tile_px: int,
                 stride_px: int, thumb_factor: int, min_tissue_fraction: float,
                 min_raw_fraction: float = 0.0) -> dict[str, np.ndarray]:
    """Grid of full tiles over the scaled image; returns kept coords (x, y top-left, scaled pixels) + fractions."""
    if stride_px != tile_px or tile_px % thumb_factor:
        raise NotImplementedError("grid assumes stride == tile size and tile size divisible by thumb_factor")
    nx, ny = scaled_w // tile_px, scaled_h // tile_px
    block = tile_px // thumb_factor
    frac_closed = block_fraction(closed, block, nx, ny)
    frac_raw = block_fraction(raw, block, nx, ny)
    iy, ix = np.nonzero((frac_closed >= min_tissue_fraction) & (frac_raw >= min_raw_fraction))
    return {"coords": np.stack([ix * tile_px, iy * tile_px], axis=1).astype(np.int32),
            "frac_closed": frac_closed[iy, ix].astype(np.float32),
            "frac_raw": frac_raw[iy, ix].astype(np.float32),
            "grid_shape": np.array([ny, nx], np.int32)}
