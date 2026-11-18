"""Measure the true pixel size of every slide, without downloading the images.

Reads only file headers via HTTP Range requests:
  * original .svs on the authors' server (Aperio header: MPP, AppMag, level-0 size, pyramid levels)
  * Hugging Face .jpg copy (JPEG SOF marker: pixel size), if it exists

Output: artifacts/metadata/slide_scale.csv
Usage:  uv run python scripts/explore/slide_scale_scan.py
"""
from __future__ import annotations

import csv
import re
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SVS_URL = "https://leelab.kmmu.edu.cn/PathologyRepository/api/download?file={}"
JPG_URL = "https://huggingface.co/datasets/kmmuleelab/Lung_Pathology_Image_JPG/resolve/main/{}"
OUT = Path("artifacts/metadata/slide_scale.csv")
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url: str, start: int, n: int, tries: int = 6) -> bytes:
    """Download bytes [start, start+n) with retries. Raises FileNotFoundError on HTTP 404."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{start + n - 1}"})
            return urllib.request.urlopen(req, timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from e
            time.sleep(2 + 2 * k)
        except Exception:
            time.sleep(2 + 2 * k)
    raise ConnectionError(url)


def svs_levels(name: str) -> tuple[list[dict], str]:
    """Return (tiled pyramid levels as [{w,h,tile}], first-IFD ImageDescription) of a BigTIFF/TIFF .svs."""
    url = SVS_URL.format(name)
    head = fetch(url, 0, 16)
    e = "<" if head[:2] == b"II" else ">"
    big = struct.unpack(e + "H", head[2:4])[0] == 43
    off = struct.unpack(e + "Q", head[8:16])[0] if big else struct.unpack(e + "I", head[4:8])[0]
    ent, cnt_fmt, ptr = (20, "Q", "Q") if big else (12, "H", "I")
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 7: 1, 16: 8}
    levels, desc = [], ""
    while off:
        hdr = 8 if big else 2
        b = fetch(url, off, hdr + ent * 64)
        n = struct.unpack(e + cnt_fmt, b[:hdr])[0]
        if hdr + ent * n > len(b):
            b = fetch(url, off, hdr + ent * n + 8)
        tags = {}
        for i in range(n):
            x = b[hdr + i * ent: hdr + (i + 1) * ent]
            tag, typ = struct.unpack(e + "HH", x[:4])
            count = struct.unpack(e + ("Q" if big else "I"), x[4:12] if big else x[4:8])[0]
            val = x[12:20] if big else x[8:12]
            if tag in (256, 257, 322) and typ in (3, 4, 16):
                tags[tag] = struct.unpack(e + {3: "H", 4: "I", 16: "Q"}[typ], val[: sizes[typ]])[0]
            elif tag == 270 and not desc:
                nbytes = count * sizes.get(typ, 1)
                raw = val[:nbytes] if nbytes <= len(val) else fetch(url, struct.unpack(e + ptr, val)[0], nbytes)
                desc = raw.decode("latin1", "replace").strip("\x00")
        if 322 in tags:  # tiled = pyramid level (strip images are thumbnail/label/macro)
            levels.append({"w": tags[256], "h": tags[257], "tile": tags[322]})
        nxt = fetch(url, off + hdr + ent * n, 8 if big else 4)
        off = struct.unpack(e + ptr, nxt)[0]
    return levels, desc


def jpg_size(name: str) -> tuple[int, int]:
    """Read width/height from the JPEG start-of-frame marker of the Hugging Face copy."""
    url, n = JPG_URL.format(name), 65536
    while n <= 1 << 21:
        b, i = fetch(url, 0, n), 2
        while i < len(b) - 9:
            if b[i] != 0xFF:
                i += 1
                continue
            m = b[i + 1]
            if m in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", b[i + 5: i + 9])
                return w, h
            i += 2 + struct.unpack(">H", b[i + 2: i + 4])[0]
        n *= 4
    raise ValueError(f"no SOF marker in {name}")


def scan(k: int) -> dict:
    row: dict = {"WSI_ID": f"WSI-{k}"}
    try:
        levels, desc = svs_levels(f"WSI-{k}.svs")
        mpp = float(re.search(r"MPP\s*=\s*([\d.]+)", desc).group(1))
        l0 = levels[0]
        row.update(svs_w=l0["w"], svs_h=l0["h"], svs_mpp=mpp,
                   appmag=re.search(r"AppMag\s*=\s*(\d+)", desc).group(1),
                   n_levels=len(levels),
                   level_downsamples=";".join(f"{l0['w'] / L['w']:.2f}" for L in levels))
        l4x = [L for L in levels if abs(l0["w"] / L["w"] - 4) < 0.05]
        row["has_4x_level"] = bool(l4x)
        if l4x:
            row.update(l4x_w=l4x[0]["w"], l4x_h=l4x[0]["h"], l4x_mpp=round(mpp * l0["w"] / l4x[0]["w"], 5))
    except Exception as ex:
        row["error"] = f"svs: {type(ex).__name__} {str(ex)[:80]}"
        return row
    try:
        w, h = jpg_size(f"WSI-{k}.jpg")
        row.update(jpg_on_hf=True, jpg_w=w, jpg_h=h,
                   jpg_mpp=round(mpp * (row["svs_w"] / w + row["svs_h"] / h) / 2, 4))
    except FileNotFoundError:
        row["jpg_on_hf"] = False
    except Exception as ex:
        row["error"] = f"jpg: {type(ex).__name__} {str(ex)[:80]}"
    return row


def main() -> None:
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(scan, range(1, 409)))
    keys = ["WSI_ID", "svs_w", "svs_h", "svs_mpp", "appmag", "n_levels", "level_downsamples", "has_4x_level",
            "l4x_w", "l4x_h", "l4x_mpp", "jpg_on_hf", "jpg_w", "jpg_h", "jpg_mpp", "error"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    bad = [r["WSI_ID"] for r in rows if r.get("error")]
    print(f"wrote {OUT} ({len(rows)} rows); errors: {len(bad)} {bad[:20]}")


if __name__ == "__main__":
    main()
