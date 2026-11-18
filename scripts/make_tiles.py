"""Stage 3: tissue mask + tile grid for every downloaded slide.

For each artifacts/slides/<WSI_ID>_4x.tif:
  * decode the 4x level, make a 4 um/px thumbnail (8x smaller than the 0.5 um/px image)
  * tissue mask (Otsu on saturation, closing, speck removal), keep tiles with >= 50% tissue
  * write artifacts/tiles/<WSI_ID>.npz      coords (x, y) in 0.5 um/px pixels + settings
          artifacts/masks/<WSI_ID>_mask.png closed mask
          artifacts/masks/<WSI_ID>_overlay.jpg  thumbnail with mask outline (red) and kept tiles (green) for QC
  * append a row to artifacts/metadata/tiling_summary.csv (tile counts, tissue area, thresholds, time)

No pixels are rescaled to full size here; the embedding step does that in memory with the same geometry.
Skips slides already done. Safe to run while the download is still going.

Usage:
  uv run python scripts/make_tiles.py                 # all downloaded slides
  uv run python scripts/make_tiles.py --only WSI-1 --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from luad.config import load_config, repo_path
from luad.data.slide_io import level_path, read_level, scaled_size, thumbnail
from luad.data.tissue import select_tiles, tissue_mask


def overlay(thumb: np.ndarray, closed: np.ndarray, coords: np.ndarray, factor: int, tile_px: int,
            max_w: int = 1600) -> Image.Image:
    """Thumbnail with kept tiles tinted green and the closed-mask outline in red."""
    img = thumb.astype(np.float32).copy()
    kept = np.zeros(closed.shape, bool)
    b = tile_px // factor
    for x, y in coords // factor:
        kept[y:y + b, x:x + b] = True
    img[kept] = 0.65 * img[kept] + 0.35 * np.array([0, 200, 0])
    edge = closed & ~ndimage.binary_erosion(closed, iterations=2)
    img[edge] = [230, 0, 0]
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    if out.width > max_w:
        out = out.resize((max_w, round(out.height * max_w / out.width)), Image.BOX)
    return out


def process(wsi_id: str, cfg: dict) -> dict:
    t0 = time.time()
    p, m = cfg["paths"], cfg["mask"]
    img, src_mpp = read_level(level_path(repo_path(p["slides_dir"]), wsi_id))
    sw, sh = scaled_size(img.shape[1], img.shape[0], src_mpp, cfg["target_mpp"])
    f = cfg["thumb_factor"]
    thumb = thumbnail(img, src_mpp, cfg["target_mpp"], f)
    del img
    thumb_mpp = cfg["target_mpp"] * f
    mk = tissue_mask(thumb, thumb_mpp, m["median_filter_px"], m["min_saturation"], m["closing_diameter_um"],
                     m["min_object_area_mm2"])
    sel = select_tiles(mk.closed, mk.raw, sw, sh, cfg["tile_px"], cfg["stride_px"], f, m["min_tissue_fraction"],
                       m["min_raw_fraction"])

    settings = {"wsi_id": wsi_id, "src_mpp": src_mpp,
                "target_mpp": cfg["target_mpp"], "scaled_size": [sw, sh], "tile_px": cfg["tile_px"],
                "stride_px": cfg["stride_px"], "thumb_factor": f, "mask": m, "otsu": mk.otsu,
                "threshold": mk.threshold, "created": dt.datetime.now().isoformat(timespec="seconds")}
    tiles_dir, masks_dir = repo_path(p["tiles_dir"]), repo_path(p["masks_dir"])
    np.savez_compressed(tiles_dir / f"{wsi_id}.npz", **sel, settings=json.dumps(settings))
    Image.fromarray(mk.closed.astype(np.uint8) * 255).save(masks_dir / f"{wsi_id}_mask.png", optimize=True)
    overlay(thumb, mk.closed, sel["coords"], f, cfg["tile_px"]).save(masks_dir / f"{wsi_id}_overlay.jpg",
                                                                    quality=85)
    px_mm2 = thumb_mpp ** 2 / 1e6
    return {"WSI_ID": wsi_id, "scaled_w": sw, "scaled_h": sh, "n_grid": int(np.prod(sel["grid_shape"])),
            "n_tiles": len(sel["coords"]), "tissue_mm2_closed": round(mk.closed.sum() * px_mm2, 2),
            "tissue_mm2_raw": round(mk.raw.sum() * px_mm2, 2),
            "median_tile_frac_raw": round(float(np.median(sel["frac_raw"])) if len(sel["coords"]) else 0, 3),
            "otsu": round(mk.otsu, 4), "threshold": round(mk.threshold, 4), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true", help="redo slides that already have tiles")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()

    cfg = load_config("tiling")
    p = cfg["paths"]
    for k in ("tiles_dir", "masks_dir"):
        repo_path(p[k]).mkdir(parents=True, exist_ok=True)
    summary_csv, log_file = repo_path(p["summary_csv"]), repo_path(p["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(repo_path(load_config("data")["paths"]["splits_dir"]) / "cohort.csv")
    wanted = args.only or cohort["WSI_ID"].tolist()
    ready = [w for w in wanted if level_path(repo_path(p["slides_dir"]), w).exists()]
    todo = [w for w in ready if args.force or not (repo_path(p["tiles_dir"]) / f"{w}.npz").exists()]
    print(f"{len(wanted)} wanted, {len(ready)} downloaded, {len(todo)} to process", flush=True)

    rows = []
    with ProcessPoolExecutor(args.workers or cfg["workers"]) as pool:
        futs = {pool.submit(process, w, cfg): w for w in todo}
        for fut in as_completed(futs):
            w = futs[fut]
            try:
                r = fut.result()
                rows.append(r)
                msg = f"done {w}: {r['n_tiles']} tiles, tissue {r['tissue_mm2_closed']} mm2, {r['seconds']}s"
            except Exception as e:
                msg = f"FAILED {w}: {type(e).__name__}: {e}"
            line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
            print(line, flush=True)
            with log_file.open("a") as fh:
                fh.write(line + "\n")

    if rows:
        new = pd.DataFrame(rows)
        if summary_csv.exists():
            old = pd.read_csv(summary_csv)
            new = pd.concat([old[~old["WSI_ID"].isin(new["WSI_ID"])], new])
        new = new.drop(columns=["label"], errors="ignore").merge(cohort[["WSI_ID", "label"]], on="WSI_ID")
        new.sort_values("WSI_ID", key=lambda s: s.str[4:].astype(int)).to_csv(summary_csv, index=False)
        s = pd.read_csv(summary_csv)
        print(f"\ntiles per slide by subtype ({len(s)} slides so far):")
        print(s.groupby("label")["n_tiles"].describe()[["count", "min", "50%", "max"]].round(0).to_string())


if __name__ == "__main__":
    main()
