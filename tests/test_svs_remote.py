"""The level copy must decode to exactly the same pixels as the source tiles. No network: synthetic tiles."""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest
import tifffile
from PIL import Image

from luad.data.svs_remote import (DATA_START, Ifd, Tag, bigtiff_header, build_ifd, file_reader, find_level,
                                  parse_ifds, tile_span)

TILE = 32


def _jpeg_tile(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = (rng.random((TILE, TILE, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)   # YCbCr JPEG, self-contained, like Aperio tiles
    return buf.getvalue()


def _tag(code, typ, values):
    fmt = {3: "H", 4: "I", 16: "Q"}[typ]
    return Tag(code, typ, len(values), struct.pack(f"<{len(values)}{fmt}", *values))


def _source(tmp_path, w=70, h=40):
    """A fake source file: some junk bytes, then the tiles back to back, like a .svs level."""
    tiles = [_jpeg_tile(i) for i in range(-(-w // TILE) * -(-h // TILE))]
    lo = 1000
    offs, pos = [], lo
    for t in tiles:
        offs.append(pos)
        pos += len(t)
    blob = b"\x00" * lo + b"".join(tiles)
    ifd = Ifd(index=3, tags={
        256: _tag(256, 4, [w]), 257: _tag(257, 4, [h]), 258: _tag(258, 3, [8, 8, 8]), 259: _tag(259, 3, [7]),
        262: _tag(262, 3, [6]), 277: _tag(277, 3, [3]), 284: _tag(284, 3, [1]), 322: _tag(322, 3, [TILE]),
        323: _tag(323, 3, [TILE]), 324: _tag(324, 16, offs), 325: _tag(325, 16, [len(t) for t in tiles]),
        530: _tag(530, 3, [2, 2])})
    return blob, ifd, tiles


def _write_copy(tmp_path, blob, ifd):
    lo, hi = tile_span(ifd)
    out = tmp_path / "copy.tif"
    ifd_off = DATA_START + (hi - lo) + (-(DATA_START + hi - lo) % 8)
    with out.open("wb") as f:
        f.write(bigtiff_header(ifd_off))
        f.write(blob[lo:hi])
        f.write(b"\x00" * (ifd_off - DATA_START - (hi - lo)))
        f.write(build_ifd(ifd, DATA_START - lo, 0.41533, "test", ifd_off))
    return out


def test_copy_decodes_to_same_pixels(tmp_path):
    blob, ifd, tiles = _source(tmp_path)
    out = _write_copy(tmp_path, blob, ifd)
    img = tifffile.imread(out)
    assert img.shape == (40, 70, 3)
    ref = np.asarray(Image.open(io.BytesIO(tiles[0])).convert("RGB"))
    # tifffile/libjpeg vs PIL/libjpeg can differ by rounding in the colour conversion; content must match
    assert np.abs(img[:TILE, :TILE].astype(int) - ref.astype(int)).max() <= 2


def test_copy_metadata_round_trip(tmp_path):
    blob, ifd, _ = _source(tmp_path)
    out = _write_copy(tmp_path, blob, ifd)
    back = parse_ifds(file_reader(out))
    assert len(back) == 1 and back[0].width == 70 and back[0].height == 40
    with tifffile.TiffFile(out) as t:
        page = t.pages[0]
        assert page.compression == 7 and page.tilewidth == TILE
        x_res = page.tags["XResolution"].value
        assert abs(1e4 / (x_res[0] / x_res[1]) - 0.41533) < 1e-4       # pixels per cm -> um/px


def test_non_contiguous_tiles_rejected(tmp_path):
    _, ifd, tiles = _source(tmp_path)
    offs = list(ifd.get(324))
    offs[1] += 5
    ifd.tags[324] = _tag(324, 16, offs)
    with pytest.raises(ValueError):
        tile_span(ifd)


def test_find_level_by_downsample():
    def lvl(i, w):
        return Ifd(index=i, tags={256: _tag(256, 4, [w]), 257: _tag(257, 4, [w]), 322: _tag(322, 3, [240])})
    ifds = [lvl(0, 1600), Ifd(index=1, tags={256: _tag(256, 4, [300])}), lvl(2, 800), lvl(3, 400)]
    assert find_level(ifds, 4).index == 3
    with pytest.raises(LookupError):
        find_level(ifds, 8)
