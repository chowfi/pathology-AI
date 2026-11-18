"""Stage 4: tile embeddings for UNI2-h, Virchow2 and H-optimus-1, in one pass per slide.

For each slide that has tiles (artifacts/tiles/<WSI_ID>.npz):
  * CPU thread: decode the 4x level, rescale to 0.5 um/px (same geometry as tiling), crop the tiles
  * GPU: each batch is copied to the GPU once and run through all three models (each with its own
    normalisation and pooling, fp16 autocast)
  * write artifacts/embeddings/<model>/<WSI_ID>.h5: features (N, dim) float16, coords (N, 2), provenance attrs
  * append per-slide timing to artifacts/logs/timing.csv

The CPU prepares the next slide while the GPU works, so the rescale adds almost no wall-clock time.
Refuses to start if the GPU power limit is above conf run.max_power_w. Skips slides already done; safe to stop.

Usage:
  uv run python scripts/extract_embeddings.py                      # all slides with tiles
  uv run python scripts/extract_embeddings.py --only WSI-1 WSI-55  # a few slides
  uv run python scripts/extract_embeddings.py --models virchow2    # one model
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import queue
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import timm
import torch

from luad.config import load_config, repo_path
from luad.embeddings.io import h5_path, load_tiles, write_h5
from luad.embeddings.models import load_encoder
from luad.utils.gpu import require_power_limit, start_gpu_logger
from luad.utils.seed import seed_everything

TIMING_COLS = ["WSI_ID", "n_tiles", "load_s", "gpu_s", "uni2h_s", "virchow2_s", "hoptimus1_s", "wall_s",
               "tiles_per_s", "finished_at"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path("."), text=True).strip()
    except Exception:
        return "unknown"


def log(msg: str, path) -> None:
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def loader(todo: list[str], q: queue.Queue, slides_dir, tiles_dir, filt: str) -> None:
    """Background thread: prepare slides in order; put (wsi_id, tiles, tiling, seconds or exception)."""
    for w in todo:
        t0 = time.time()
        try:
            tiles, tiling = load_tiles(w, slides_dir, tiles_dir, filt)
            q.put((w, tiles, tiling, time.time() - t0))
        except Exception as e:
            q.put((w, None, None, e))
    q.put(None)


def warm_up(encoders, batch: int) -> None:
    """Ramp the load up in steps instead of jumping straight to full power."""
    x = torch.rand(batch, 3, 224, 224, device="cuda")
    for bs in (4, 16, batch // 2, batch):
        for e in encoders:
            e.embed(x[:bs])
        torch.cuda.synchronize()
        time.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--models", nargs="*", help="subset of conf models (default: all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg, tcfg, dcfg = load_config("embeddings"), load_config("tiling"), load_config("data")
    seed_everything(dcfg["seed"])
    run, paths = cfg["run"], cfg["paths"]
    names = args.models or list(cfg["models"])
    out_dir = repo_path(paths["out_dir"])
    slides_dir, tiles_dir = repo_path(tcfg["paths"]["slides_dir"]), repo_path(tcfg["paths"]["tiles_dir"])
    log_file, timing_csv = repo_path(paths["log_file"]), repo_path(paths["timing_csv"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(repo_path(dcfg["paths"]["splits_dir"]) / "cohort.csv")
    wanted = args.only or cohort["WSI_ID"].tolist()
    ready = [w for w in wanted if (tiles_dir / f"{w}.npz").exists()]
    todo = [w for w in ready if args.force or not all(h5_path(out_dir, n, w).exists() for n in names)]
    limit = require_power_limit(run["max_power_w"])
    log(f"start: models={names} wanted={len(wanted)} tiled={len(ready)} todo={len(todo)} "
        f"power_limit={limit:.0f}W", log_file)
    if not todo:
        return

    gpu_logger = start_gpu_logger(repo_path(paths["gpu_log"]))
    q: queue.Queue = queue.Queue(maxsize=run["prefetch_slides"])
    th = threading.Thread(target=loader, args=(todo, q, slides_dir, tiles_dir, tcfg["rescale_filter"]),
                          daemon=True)
    try:
        encoders = [load_encoder(n, cfg["models"][n]) for n in names]
        warm_up(encoders, run["batch_size"])
        common = {"git_commit": git_commit(), "torch": torch.__version__, "timm": timm.__version__,
                  "gpu": torch.cuda.get_device_name(0), "precision": run["precision"]}
        th.start()
        bs = run["batch_size"]
        while (item := q.get()) is not None:
            w, tiles, tiling, load_s = item
            if tiles is None:
                log(f"FAILED {w}: loading: {type(load_s).__name__}: {load_s}", log_file)
                continue
            t_wall = time.time()
            feats = {e.name: [] for e in encoders}
            per_model = {e.name: 0.0 for e in encoders}
            for i in range(0, len(tiles), bs):     # no whole-slide pinned copy: keeps RAM at ~1 slide
                x = torch.from_numpy(tiles[i:i + bs]).to("cuda").permute(0, 3, 1, 2).float().div_(255)
                for e in encoders:
                    t = time.time()
                    feats[e.name].append(e.embed(x))
                    torch.cuda.synchronize()
                    per_model[e.name] += time.time() - t
            gpu_s = time.time() - t_wall
            for e in encoders:
                f = torch.cat(feats[e.name]).cpu().numpy()
                if not np.isfinite(f).all():
                    raise FloatingPointError(f"{w} {e.name}: NaN/Inf in embeddings")
                s = e.spec
                write_h5(h5_path(out_dir, e.name, w), f, tiling["coords"], {
                    "wsi_id": w, "model": e.name, "hf_id": s["hf_id"], "revision": s["revision"],
                    "pooling": s["pooling"], "dim": int(f.shape[1]), "mean": s["mean"], "std": s["std"],
                    "tiling": tiling["settings"], "created": dt.datetime.now().isoformat(timespec="seconds"),
                    **common})
            del tiles
            wall = time.time() - t_wall
            row = {"WSI_ID": w, "n_tiles": len(tiling["coords"]), "load_s": round(load_s, 1),
                   "gpu_s": round(gpu_s, 1), **{f"{k}_s": round(v, 1) for k, v in per_model.items()},
                   "wall_s": round(wall, 1), "tiles_per_s": round(len(tiling["coords"]) / gpu_s, 1),
                   "finished_at": dt.datetime.now().isoformat(timespec="seconds")}
            new = not timing_csv.exists()
            with timing_csv.open("a", newline="") as fh:
                wr = csv.DictWriter(fh, TIMING_COLS, extrasaction="ignore")
                if new:
                    wr.writeheader()
                wr.writerow(row)
            log(f"done {w}: {row['n_tiles']} tiles, load {row['load_s']}s (CPU, overlapped), "
                f"GPU {row['gpu_s']}s = {row['tiles_per_s']} tiles/s through {len(encoders)} models", log_file)
    finally:
        gpu_logger.terminate()
    log("finished", log_file)


if __name__ == "__main__":
    main()
