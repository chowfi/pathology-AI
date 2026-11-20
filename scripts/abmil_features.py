"""Stage 5c: cross-fitted attention-pooled slide vectors for the 15 dev CV folds (train+val patients only).

Default (v5): for each model with aggregation 'abmil' (conf/experiments.yaml), per-model gated-attention MIL.
--joint (v7): tile-level fusion for the FUSED model's streams (fusion_aggregation: joint_abmil): one attention
    network sees all three models' embeddings of each tile; its shared attention pools each model's raw tiles.
--models M (ablation): per-model ABMIL for the named models, whatever their configured aggregation.

For each CV fold: 5 networks are trained on parts of the fold's training patients (cross-fitting), so every
training patient gets a vector from a network that never saw its label, and every validation patient gets the
mean over the 5 networks. Settings: conf/experiments.yaml `abmil` (+ `joint_abmil`), fixed in the round's
PLAN.md. The test set is not touched.

Writes artifacts/features/<file>.npz (Z or Z_<model>: (15, n, d), wsi_id, val_<k> fold check, config) and
<file>_log.csv (one row per network: training loss at the first / last epoch).

Usage:
  uv run python scripts/abmil_features.py              # v5 per-model vectors
  uv run python scripts/abmil_features.py --joint      # v7 tile-level fusion vectors
  uv run python scripts/abmil_features.py --models virchow2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS (before CUDA starts)

import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
import torch                                                       # noqa: E402

from luad.config import load_config, repo_path                     # noqa: E402
from luad.data.dataset import FOLD_SPECIFIC, aggregation, feature_file, load_dev_set   # noqa: E402
from luad.embeddings.abmil import (MultiViewBank, TileBank, fold_features, pool_joint,  # noqa: E402
                                   split_views, train_joint_abmil)
from luad.utils.gpu import require_power_limit, start_gpu_logger   # noqa: E402
from luad.utils.seed import seed_everything                        # noqa: E402


def _save(out, arrays: dict, ds, meta: dict, log: list[dict]) -> None:
    np.savez_compressed(out, wsi_id=np.array(ds.wsi_ids), meta=json.dumps(meta),
                        **arrays, **{f"val_{k}": np.sort(v) for k, (_, v) in enumerate(ds.folds)})
    lf = pd.DataFrame(log)
    lf.to_csv(out.with_name(out.stem + "_log.csv"), index=False)
    print(f"wrote {out.name} {[a.shape for a in arrays.values()]} in {meta['seconds']}s; {len(lf)} networks, "
          f"training loss first epoch {lf.loss_first.mean():.2f} -> last {lf.loss_last.mean():.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", action="store_true", help="v7 tile-level fusion vectors for the fused model")
    ap.add_argument("--models", nargs="*", help="per-model ABMIL for these models (ablation)")
    args = ap.parse_args()
    ecfg, emb = load_config("experiments"), load_config("embeddings")
    acfg = ecfg["abmil"]
    seed_everything(acfg["seed"])
    torch.use_deterministic_algorithms(True)
    require_power_limit(emb["run"]["max_power_w"])
    fixed = [m for m in ecfg["models"] if aggregation(m) not in FOLD_SPECIFIC]
    ds = load_dev_set(fixed) if fixed else load_dev_set()           # folds + labels; caches not needed here
    emb_dir, feat_dir = repo_path(emb["paths"]["out_dir"]), repo_path(ecfg["features_dir"])
    gpu_logger = start_gpu_logger(repo_path(emb["paths"]["gpu_log"]))
    try:
        if args.joint:
            if ecfg.get("fusion_aggregation") != "joint_abmil":
                raise ValueError("conf/experiments.yaml: fusion_aggregation is not 'joint_abmil'")
            cfg, t0, log = {**acfg, **ecfg.get("joint_abmil", {})}, time.time(), []
            bank = MultiViewBank.from_h5(ecfg["models"], ds.wsi_ids, emb_dir)
            Z = fold_features(bank, np.arange(len(ds.y)), ds.y, ds.folds, cfg, cfg["seed"], log=log,
                              train_fn=train_joint_abmil, pool_fn=pool_joint)
            if not np.isfinite(Z).all():
                raise FloatingPointError("non-finite joint ABMIL vectors")
            parts = dict(zip([f"Z_{m}" for m in ecfg["models"]], split_views(Z, bank.dims)))
            meta = {"models": ecfg["models"], "aggregation": "joint_abmil", "config": cfg, "n_folds": len(ds.folds),
                    "seconds": round(time.time() - t0, 1), "created": dt.datetime.now().isoformat(timespec="seconds")}
            _save(feat_dir / feature_file(ecfg["models"][0], fusion=True), parts, ds, meta, log)
            return
        models = args.models or [m for m in ecfg["models"] if aggregation(m) == "abmil"]
        for m in models:
            t0, log = time.time(), []
            bank = TileBank.from_h5(m, ds.wsi_ids, emb_dir)
            Z = fold_features(bank, np.arange(len(ds.y)), ds.y, ds.folds, acfg, acfg["seed"], log=log)
            if not np.isfinite(Z).all():
                raise FloatingPointError(f"{m}: non-finite ABMIL vectors")
            meta = {"model": m, "aggregation": "abmil", "config": acfg, "n_folds": len(ds.folds),
                    "seconds": round(time.time() - t0, 1), "created": dt.datetime.now().isoformat(timespec="seconds")}
            _save(feat_dir / f"abmil_{m}_devfolds.npz", {"Z": Z}, ds, meta, log)
            del bank
            torch.cuda.empty_cache()
    finally:
        gpu_logger.terminate()


if __name__ == "__main__":
    main()
