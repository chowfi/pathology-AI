"""Stage 2: download the 4x pyramid level of each patient's selected slide into a single-level tiled TIFF.

For each WSI_ID in artifacts/splits/cohort.csv (209 slides, one per patient):
  1. read the .svs directory tags over HTTP Range (no pixel data of label/macro images)
  2. find the 4x level and its contiguous tile byte span
  3. stream that span into <out_dir>/<WSI_ID>_4x.tif.part (resumes from the current size after a crash)
  4. check every tile is a complete JPEG, then append a BigTIFF directory and rename to <WSI_ID>_4x.tif
  5. append provenance + timing to the manifest CSV (committed to git)

Safe to stop and restart at any time. Finished slides are skipped.

Usage:
  uv run python scripts/download_slides.py                  # all slides
  uv run python scripts/download_slides.py --only WSI-1     # one slide (testing)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import mmap
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from luad.config import load_config, repo_path
from luad.data.svs_remote import (DATA_START, T_DESC, T_TILE_COUNTS, T_TILE_H, T_TILE_OFFSETS, T_TILE_W,
                                  RangeReader, bigtiff_header, build_ifd, find_level, parse_ifds, tile_span)

MANIFEST_COLS = ["WSI_ID", "source_ifd", "base_width", "base_height", "base_mpp", "app_mag", "downsample",
                 "width", "height", "mpp", "tile_w", "tile_h", "n_tiles", "tile_bytes", "tile_sha256",
                 "seconds", "mb_per_s", "downloaded_at"]
_lock = threading.Lock()


def log(msg: str, log_file: Path) -> None:
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with _lock, log_file.open("a") as f:
        f.write(line + "\n")


def done_ids(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    return set(pd.read_csv(manifest)["WSI_ID"])


def append_manifest(manifest: Path, row: dict) -> None:
    with _lock:
        new = not manifest.exists()
        with manifest.open("a", newline="") as f:
            w = csv.DictWriter(f, MANIFEST_COLS)
            if new:
                w.writeheader()
            w.writerow(row)


def download_one(wsi_id: str, cfg: dict, out_dir: Path, log_file: Path) -> dict:
    t0 = time.time()
    reader = RangeReader(cfg["url_template"].format(wsi_id=wsi_id))
    ifds = parse_ifds(reader)
    base = next(d for d in ifds if d.tiled)
    desc0 = base.get(T_DESC)[0] if T_DESC in base.tags else ""
    base_mpp = float(re.search(r"MPP\s*=\s*([\d.]+)", desc0).group(1))
    app_mag = re.search(r"AppMag\s*=\s*(\d+)", desc0)
    lv = find_level(ifds, cfg["downsample"])
    lo, hi = tile_span(lv)
    span = hi - lo
    ds = base.width / lv.width
    mpp = base_mpp * ds
    offs, cnts = lv.get(T_TILE_OFFSETS), lv.get(T_TILE_COUNTS)

    final = out_dir / f"{wsi_id}_4x.tif"
    part = out_dir / f"{wsi_id}_4x.tif.part"
    have = max(0, part.stat().st_size - DATA_START) if part.exists() else 0
    if have > span:                                   # stale / corrupt partial file
        part.unlink()
        have = 0
    chunk = cfg["chunk_mb"] << 20
    fetched = span - have
    with part.open("r+b" if part.exists() else "wb") as f:
        if have == 0:
            f.write(bigtiff_header(0))                 # placeholder; patched once the IFD is written
        f.seek(DATA_START + have)
        f.truncate()
        while have < span:
            n = min(chunk, span - have)
            f.write(reader(lo + have, n))
            have += n
        f.flush()
        os.fsync(f.fileno())

    # every tile must be a complete JPEG, and the tile data is hashed for provenance
    delta = DATA_START - lo
    with part.open("r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        bad = [i for i, (o, c) in enumerate(zip(offs, cnts))
               if mm[o + delta: o + delta + 2] != b"\xff\xd8"
               or b"\xff\xd9" not in mm[o + delta + max(0, c - 16): o + delta + c]]
        sha = hashlib.sha256(mm[DATA_START: DATA_START + span]).hexdigest()
        mm.close()
        if bad:
            part.unlink()                                # force a clean re-download next run
            raise ValueError(f"{len(bad)} broken JPEG tiles (first {bad[:5]}); partial file deleted")
        ifd_off = DATA_START + span + (-(DATA_START + span) % 8)
        text = (f"luad-fm-fusion copy of {wsi_id}.svs level IFD {lv.index}: downsample {ds:.4f} of base "
                f"{base.width}x{base.height} at {base_mpp} um/px; this level {mpp:.5f} um/px; JPEG tiles copied "
                f"byte for byte")
        f.seek(DATA_START + span)
        f.write(b"\x00" * (ifd_off - DATA_START - span))
        f.write(build_ifd(lv, delta, mpp, text, ifd_off))
        f.seek(0)
        f.write(bigtiff_header(ifd_off))
        f.flush()
        os.fsync(f.fileno())
    part.rename(final)

    secs = time.time() - t0
    row = {"WSI_ID": wsi_id, "source_ifd": lv.index, "base_width": base.width, "base_height": base.height,
           "base_mpp": base_mpp, "app_mag": app_mag.group(1) if app_mag else "", "downsample": round(ds, 4),
           "width": lv.width, "height": lv.height, "mpp": round(mpp, 5), "tile_w": lv.get(T_TILE_W)[0],
           "tile_h": lv.get(T_TILE_H)[0], "n_tiles": len(offs), "tile_bytes": span, "tile_sha256": sha,
           "seconds": round(secs, 1), "mb_per_s": round(fetched / 1e6 / secs, 2),
           "downloaded_at": dt.datetime.now().isoformat(timespec="seconds")}
    log(f"done {wsi_id}: {lv.width}x{lv.height} {len(offs)} tiles {span / 1e6:.0f} MB "
        f"in {secs:.0f}s ({row['mb_per_s']} MB/s)", log_file)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="WSI_IDs to download (default: all in cohort.csv)")
    ap.add_argument("--workers", type=int, help="override conf download.workers")
    args = ap.parse_args()

    cfg_all = load_config("data")
    cfg = cfg_all["download"]
    out_dir, manifest, log_file = (repo_path(cfg[k]) for k in ("out_dir", "manifest_csv", "log_file"))
    for p in (out_dir, manifest.parent, log_file.parent):
        p.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(repo_path(cfg_all["paths"]["splits_dir"]) / "cohort.csv", dtype={"SampleNumber": str})
    wanted = args.only or cohort["WSI_ID"].tolist()
    finished = done_ids(manifest)
    todo = [w for w in wanted if not (w in finished and (out_dir / f"{w}_4x.tif").exists())]
    log(f"start: {len(wanted)} wanted, {len(wanted) - len(todo)} already done, {len(todo)} to download", log_file)

    failed = []
    with ThreadPoolExecutor(args.workers or cfg["workers"]) as pool:
        futs = {pool.submit(download_one, w, cfg, out_dir, log_file): w for w in todo}
        for fut in as_completed(futs):
            w = futs[fut]
            try:
                append_manifest(manifest, fut.result())
            except Exception as e:
                failed.append(w)
                log(f"FAILED {w}: {type(e).__name__}: {e}", log_file)
    log(f"finished: {len(todo) - len(failed)} downloaded, {len(failed)} failed {failed}", log_file)
    if failed:
        raise SystemExit("some slides failed; run the script again to retry them (it resumes)")


if __name__ == "__main__":
    main()
