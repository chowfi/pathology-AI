"""Fetch a few tissue tiles from pyramid level 3 (4x, ~0.415 um/px) of one remote .svs and rescale to 0.5 um/px.

Only downloads the needed tile bytes (HTTP Range). Saves 224x224 PNG crops for preprocessing checks.
Usage: uv run python scripts/explore/fetch_sample_tiles.py WSI-1.svs artifacts/sample_tiles [n_tiles]
"""
from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from slide_scale_scan import SVS_URL, fetch  # noqa: E402

TARGET_MPP = 0.5


def tiled_levels(url: str) -> list[dict]:
    """All tiled IFDs of a little-endian BigTIFF with the tags needed to read tiles."""
    off = struct.unpack("<Q", fetch(url, 0, 16)[8:16])[0]
    out = []
    while off:
        b = fetch(url, off, 8 + 20 * 64)
        n = struct.unpack("<Q", b[:8])[0]
        d: dict = {}
        for i in range(n):
            e = b[8 + 20 * i: 28 + 20 * i]
            t, ty = struct.unpack("<HH", e[:4])
            c = struct.unpack("<Q", e[4:12])[0]
            w = {3: 2, 4: 4, 16: 8}.get(ty)
            if t in (256, 257, 322, 323, 324, 325) and w:
                raw = e[12:12 + w * c] if w * c <= 8 else fetch(url, struct.unpack("<Q", e[12:20])[0], w * c)
                d[t] = struct.unpack("<%d%s" % (c, {3: "H", 4: "I", 16: "Q"}[ty]), raw)
            elif t == 347:  # shared JPEG tables, if any
                d[t] = fetch(url, struct.unpack("<Q", e[12:20])[0], c) if c > 8 else e[12:12 + c]
        if 322 in d:
            out.append(d)
        off = struct.unpack("<Q", fetch(url, off + 8 + 20 * n, 8))[0]
    return out


def main(name: str, outdir: str, n_tiles: int = 16) -> None:
    url, out = SVS_URL.format(name), Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    levels = tiled_levels(url)
    l0w = levels[0][256][0]
    lv = next(L for L in levels if abs(l0w / L[256][0] - 4) < 0.05)
    W, H, tw, th = lv[256][0], lv[257][0], lv[322][0], lv[323][0]
    mpp = 0.103832 * l0w / W
    ntx = -(-W // tw)
    tables = lv.get(347)
    scale = mpp / TARGET_MPP
    print(f"{name}: 4x level {W}x{H}, tile {tw}, {mpp:.4f} um/px -> resize x{scale:.4f} to {TARGET_MPP} um/px")
    kept = 0
    for ty in range(2, -(-H // th) - 2, 4):
        for tx in range(2, ntx - 2, 4):
            i = ty * ntx + tx
            data = fetch(url, lv[324][i], lv[325][i])
            im = Image.open(io.BytesIO(tables[:-2] + data[2:] if tables else data)).convert("RGB")
            if (np.asarray(im.convert("HSV"))[..., 1] > 30).mean() < 0.6:
                continue
            im = im.resize((round(tw * scale), round(th * scale)), Image.LANCZOS)
            for k, (x, y) in enumerate([(0, 0), (224, 224)]):
                im.crop((x, y, x + 224, y + 224)).save(out / f"{name[:-4]}_t{ty}_{tx}_{k}.png")
            kept += 1
            if kept >= n_tiles:
                print(f"saved {2 * kept} crops to {out}")
                return
    print(f"saved {2 * kept} crops to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)
