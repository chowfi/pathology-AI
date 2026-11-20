"""Stage 8: nested CV - how much of the selected CV score is selection luck ("selection optimism")?

Outer loop: the 5 folds of repeat 0 of the committed CV folds (train+val patients only; test untouched).
For each outer fold:
  1. inner data = the other 4/5 of the patients, with new 5x3 stratified folds made only from them
  2. run the WHOLE selection on the inner data: every baseline search (same grids as scripts/run_baselines.py),
     then the fusion agent (deterministic bandit, same budget/patience/acceptance/final-pick rule); the members of
     the *_tuned fusion methods are the inner <model>+meta picks; "best single model" and "best single model +
     metadata" are also chosen inside, by inner CV
  3. refit each selected config on all inner patients and score it ONCE on the held-out outer fold
Optimism = inner CV score of the selected config - its outer score. The outer scores also give an honest,
selection-free estimate of fused vs best single model (paired over the 5 outer folds).

Supervised aggregations are rebuilt for the inner folds from inner patients only (cross-fitted, cached in
artifacts/features/abmil_nested/): per-model ABMIL (v5) for the single-model streams, and tile-level fusion (v7,
fusion_aggregation: joint_abmil) for the fused model's streams.

--reuse-baselines DIR (v7): do not search the baseline rows again; take their inner picks from the logs in DIR
    (an earlier round's nested_cv folder) with the configured final-pick rule. Valid only when the baseline rows
    did not change.

Usage:
  uv run python scripts/nested_cv.py
  uv run python scripts/nested_cv.py --reuse-baselines artifacts/results/v5/nested_cv
"""
from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS for ABMIL (v5)

import numpy as np
import pandas as pd

from luad.agent.bandit import BanditProposer
from luad.agent.search import GridProposer, run_search
from luad.agent.selection import incumbent_from_log, select
from luad.agent.space import FusionSpace
from luad.config import load_config, repo_path
from luad.data.dataset import aggregation, load_dev_set
from luad.eval.baselines import rows_for, space_for
from luad.eval.metrics import score
from luad.fusion.early import EarlyFusionLR
from luad.fusion.registry import build_model
from luad.utils.seed import seed_everything


def inner_streams(ds, sub, tr: np.ndarray, k: int, banks: dict, joint: bool = False) -> None:
    """Fold-specific vectors for the inner folds of outer fold k, cross-fitted on inner patients only (cached).
    joint=False: per-model ABMIL for the models with aggregation 'abmil'; joint=True: tile-level fusion (v7)."""
    import torch

    from luad.embeddings.abmil import (MultiViewBank, TileBank, fold_features, pool_joint, split_views,
                                       train_joint_abmil)
    ecfg, emb = load_config("experiments"), load_config("embeddings")
    acfg = ecfg["abmil"]
    cache_dir = repo_path(ecfg["features_dir"]) / "abmil_nested"
    cache_dir.mkdir(parents=True, exist_ok=True)
    val = [np.sort(v) for _, v in sub.folds]
    emb_dir, seed = repo_path(emb["paths"]["out_dir"]), acfg["seed"] + 100_000 * (k + 1)

    def cached(path, keys):
        if not path.exists():
            return None
        with np.load(path) as z:
            if z["wsi_id"].tolist() == sub.wsi_ids and all(np.array_equal(a, z[f"val_{j}"]) for j, a in enumerate(val)):
                return {key: z[key] for key in keys}
        return None

    def save(path, arrays):
        np.savez_compressed(path, wsi_id=np.array(sub.wsi_ids), **arrays, **{f"val_{j}": a for j, a in enumerate(val)})

    torch.use_deterministic_algorithms(True)
    if joint:
        models = ecfg["models"]
        path, keys = cache_dir / f"joint_outer{k}.npz", [f"Z_{m}" for m in models]
        got = cached(path, keys)
        if got is None:
            if "joint" not in banks:
                for key in [b for b in banks if b != "joint"]:   # free per-model banks: the joint bank needs the room
                    del banks[key]
                torch.cuda.empty_cache()
                banks["joint"] = MultiViewBank.from_h5(models, ds.wsi_ids, emb_dir)
            Z = fold_features(banks["joint"], tr, sub.y, sub.folds, {**acfg, **ecfg.get("joint_abmil", {})}, seed,
                              train_fn=train_joint_abmil, pool_fn=pool_joint)
            got = dict(zip(keys, split_views(Z, banks["joint"].dims)))
            save(path, got)
            print(f"outer {k}: built inner joint (tile-level fusion) vectors {Z.shape}", flush=True)
        for m in models:
            sub.fold_streams[m] = got[f"Z_{m}"]
        return
    for m in [m for m in ecfg["models"] if aggregation(m) == "abmil"]:
        path = cache_dir / f"{m}_outer{k}.npz"
        got = cached(path, ["Z"])
        if got is None:
            if m not in banks:
                banks[m] = TileBank.from_h5(m, ds.wsi_ids, emb_dir)
            got = {"Z": fold_features(banks[m], tr, sub.y, sub.folds, acfg, seed)}
            save(path, got)
            print(f"outer {k}: built inner ABMIL vectors for {m} {got['Z'].shape}", flush=True)
        sub.fold_streams[m] = got["Z"]


def _row_model(config: dict, blocks: list[str], clf: dict, seed: int):
    return EarlyFusionLR({"blocks": blocks, **config}, clf, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse-baselines", help="earlier round's nested_cv folder: take the baseline picks from its logs")
    args = ap.parse_args()
    ecfg, acfg = load_config("experiments"), load_config("agent")
    s = ecfg["search"]
    rule = s.get("selection", "auroc")
    seed_everything(s["seed"])
    ds = load_dev_set()
    joint = ecfg.get("fusion_aggregation") == "joint_abmil"
    ds_f = load_dev_set(fusion=True) if ecfg.get("fusion_aggregation") else ds
    if ds.fold_streams or ds_f.fold_streams:
        from luad.utils.gpu import require_power_limit
        require_power_limit(load_config("embeddings")["run"]["max_power_w"])
    outer = [(j, tr, va) for j, ((r, _), (tr, va)) in enumerate(zip(ds.fold_ids, ds.folds)) if r == 0]
    out_dir = repo_path(ecfg["results_dir"]) / "nested_cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    clf, seed = ecfg["classifier"], s["seed"]
    search_kw = dict(explore=s.get("explore", 0), incumbent=s.get("incumbent", "first"), selection=rule)
    rows, banks = [], {}
    for k, (j, tr, va) in enumerate(outer):
        sub = ds.subset(tr, 5, 3, seed + k)
        n_val_in = int(np.mean([len(v) for _, v in sub.folds]))
        n_tr_in = len(sub.y) - n_val_in
        ds_k = ds.for_fold(j)                  # outer refit: vectors cross-fitted on the outer training patients
        sel = {}
        for row, blocks in rows_for(ecfg["models"]).items():
            if row == "tile_count":
                continue
            if args.reuse_baselines:
                h = [json.loads(x) for x in (Path(args.reuse_baselines) / f"outer{k}_{row}.jsonl").read_text().splitlines()]
                best = select(h, rule, incumbent_from_log(h), n_tr_in, n_val_in)
            else:
                if ds.fold_streams and not sub.fold_streams:
                    inner_streams(ds, sub, tr, k, banks)
                best = run_search(sub, lambda c, b=blocks: _row_model(c, b, clf, seed), GridProposer(space_for(blocks, ecfg), seed),
                                  {"blocks": blocks}, s["budget"], s["patience"], s["accept_k"], s["n_jobs"],
                                  log_path=out_dir / f"outer{k}_{row}.jsonl", **search_kw).best
            sel[row] = (best, partial(_row_model, best["config"], blocks, clf, seed), ds_k)
        members = {m: sel[f"{m}+meta"][0]["config"] for m in acfg["models"]}   # chosen inside: no outer info
        fixed = {"models": acfg["models"], "use_meta": acfg["use_meta"], "members": members}
        if ds_f is ds:
            sub_f, ds_fk = sub, ds_k
            if ds.fold_streams and not sub.fold_streams:
                inner_streams(ds, sub, tr, k, banks)
        else:                                  # the fused model's own streams (v7: tile-level fusion)
            sub_f, ds_fk = ds_f.subset(tr, 5, 3, seed + k), ds_f.for_fold(j)
            if joint:
                inner_streams(ds_f, sub_f, tr, k, banks, joint=True)
        r = run_search(sub_f, lambda c: build_model(c, clf, seed=seed), BanditProposer(FusionSpace(acfg["space"]), seed,
                       acfg["proposer"]["ucb_c"]), fixed, s["budget"], s["patience"], s["accept_k"], s["n_jobs"],
                       log_path=out_dir / f"outer{k}_fused_bandit.jsonl", **search_kw)
        sel["fused (agent, bandit)"] = (r.best, partial(build_model, {**fixed, **r.best["config"]}, clf, seed), ds_fk)
        singles = list(ecfg["models"])
        best_single = max(singles, key=lambda m: sel[m][0]["macro_auroc_mean"])
        best_single_meta = max([f"{m}+meta" for m in singles], key=lambda m: sel[m][0]["macro_auroc_mean"])
        sel[f"best single ({best_single})"] = sel[best_single]
        sel[f"best single + meta ({best_single_meta})"] = sel[best_single_meta]
        for name, (best, make, data) in sel.items():
            o = score(ds.y[va], make().fit(data, tr).predict_proba(data, va))
            rows.append({"outer_fold": k, "row": name.split(" (")[0] if name.startswith("best") else name,
                         "chosen": name, "trial": best["trial"], "config": json.dumps(best["config"]),
                         "inner_cv_auroc": round(best["macro_auroc_mean"], 4),
                         "inner_cv_ba": round(best["balanced_accuracy_mean"], 4),
                         "outer_auroc": round(o["macro_auroc"], 4), "outer_ba": round(o["balanced_accuracy"], 4)})
        f = [x for x in rows if x["outer_fold"] == k]
        fused = next(x for x in f if x["row"].startswith("fused"))
        bsm = next(x for x in f if x["row"] == "best single + meta")
        print(f"outer {k}: fused inner {fused['inner_cv_auroc']:.3f} -> outer {fused['outer_auroc']:.3f} "
              f"(BA {fused['outer_ba']:.3f}) | {bsm['chosen']} inner {bsm['inner_cv_auroc']:.3f} -> outer "
              f"{bsm['outer_auroc']:.3f} (BA {bsm['outer_ba']:.3f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_fold.csv", index=False)
    df["optimism"] = df["inner_cv_auroc"] - df["outer_auroc"]
    summ = df.groupby("row").agg(inner_cv_auroc=("inner_cv_auroc", "mean"), outer_auroc=("outer_auroc", "mean"),
                                 outer_auroc_sd=("outer_auroc", "std"), outer_ba=("outer_ba", "mean"),
                                 outer_ba_sd=("outer_ba", "std"), optimism=("optimism", "mean"),
                                 optimism_sd=("optimism", "std")).round(4)
    summ.to_csv(out_dir / "summary.csv")
    print(summ.to_string())
    for metric in ("outer_auroc", "outer_ba"):
        piv = df.pivot(index="outer_fold", columns="row", values=metric)
        d = piv["fused (agent, bandit)"] - piv["best single + meta"]
        print(f"fused - best single + meta, {metric} (paired over 5 outer folds): {d.mean():+.4f} ± {d.std():.4f} "
              f"(fused better in {(d > 0).sum()}/5 folds)")


if __name__ == "__main__":
    main()
