"""Stage 8: the ONE-TIME test evaluation of the locked rows (README results table).

Every row is refit on all train+val patients with its locked pick and scored once on the test patients:
  rows   <final.round>/baselines/summary.csv   (single models and metadata, logistic regression)
  fused  <final.round>/agent/<final.fused_run>/summary.json   (the main Claude run's pick)
Supervised slide vectors are cross-fitted exactly as in CV: 5 networks on parts of the training patients; each
training patient gets the vector of the network that did not see it, each test patient the mean of the 5.
  UNI2-h, H-optimus-1 rows: per-model ABMIL      Virchow2 rows: Prism2 (fixed vectors)
  fused model: joint (tile-level) ABMIL for all three models (v7)
The networks are trained on a GPU bank of the training slides only; the bank is freed, and the test slides are
loaded into a second bank for pooling (all 209 slides of the joint bank do not fit on the GPU at once).

Modes:
  default (dry run)  No test patient is read. Training = the training patients of CV fold 0 (repeat 0), "test" =
                     its validation patients, cross-fit seed index 0. Then every row must reproduce its logged
                     fold-0 CV score and the cached fold-0 ABMIL vectors: an end-to-end check of this script.
  --test             The real run, once. Refuses if the test access log already has an entry, if the output
                     directory exists, or if tracked files have uncommitted changes (--allow-dirty overrides
                     the last check; recorded in summary.json).

Outputs (<final.out_dir>, or <final.out_dir>_dryrun):
  table.md             the README table: test AUROC / balanced accuracy with 95% CIs + CV and nested-CV columns
  metrics.csv          every row x metric (macro AUROC, balanced accuracy, accuracy, macro F1) with 95% CIs
  paired.csv           fused - row, paired over the same bootstrap resamples, with 95% CIs
  per_class_auroc.csv  one-vs-rest AUROC per class and row
  confusion.csv        counts per row, true class and predicted class
  predictions.csv      probabilities per patient and row
  vectors.npz, abmil_log.csv, summary.json (picks, seeds, file hashes, git commit, timing)

Usage:
  uv run python scripts/final_test.py            # dry run (dev patients only)
  uv run python scripts/final_test.py --test     # ONE TIME, only after the user's approval
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from dataclasses import replace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS (before CUDA starts)

import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
import torch                                                       # noqa: E402
from sklearn.metrics import confusion_matrix                       # noqa: E402

from luad.config import load_config, repo_path                     # noqa: E402
from luad.data.dataset import (CLASSES, FOLD_SPECIFIC, Dataset, aggregation, concat, load_dev_set,  # noqa: E402
                               load_test_set)
from luad.embeddings.abmil import (MultiViewBank, TileBank, apply_parts, fit_parts, pool_joint,  # noqa: E402
                                   split_views, train_joint_abmil)
from luad.eval.baselines import rows_for                           # noqa: E402
from luad.eval.bootstrap import evaluate, metrics                 # noqa: E402
from luad.eval.metrics import per_class_auroc                      # noqa: E402
from luad.fusion.early import EarlyFusionLR                        # noqa: E402
from luad.fusion.registry import build_model                       # noqa: E402
from luad.utils.gpu import require_power_limit, start_gpu_logger   # noqa: E402
from luad.utils.seed import seed_everything                        # noqa: E402

ROWS = ["meta", "uni2h", "virchow2", "hoptimus1", "uni2h+meta", "virchow2+meta", "hoptimus1+meta"]
LABEL = {"uni2h": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
AGG_NAME = {"abmil": "ABMIL", "prism2": "Prism2", "mean": "mean pooling"}


# ---- cross-fitted slide vectors ----------------------------------------------------------------------------------
def crossfit_two_banks(make_bank, dev_ids: list[str], test_ids: list[str], y_dev: np.ndarray, cfg: dict,
                       seed: int, log: list, tag: str, **fns) -> tuple[np.ndarray, np.ndarray]:
    """Train the part networks on a bank of the dev slides, free it, pool the test slides from a second bank."""
    bank = make_bank(dev_ids)
    rows: list[dict] = []
    z_dev, nets = fit_parts(bank, np.arange(len(dev_ids)), y_dev, cfg, seed, log=rows, **fns)
    dims = getattr(bank, "dims", [bank.dim])
    del bank
    torch.cuda.empty_cache()
    bank = make_bank(test_ids)
    z_test = apply_parts(bank, nets, np.arange(len(test_ids)), pool_fn=fns.get("pool_fn"))
    del bank, nets
    torch.cuda.empty_cache()
    if not (np.isfinite(z_dev).all() and np.isfinite(z_test).all()):
        raise FloatingPointError(f"{tag}: non-finite vectors")
    log.extend({"vectors": tag, **r} for r in rows)
    return np.concatenate([z_dev, z_test]), np.array(dims)


def slide_vectors(dev: Dataset, test: Dataset, seed: int, log: list) -> tuple[dict, dict]:
    """({model: (n_dev + n_test, d)} per-model ABMIL streams, {model: ...} joint streams of the fused model)."""
    ecfg, emb = load_config("experiments"), load_config("embeddings")
    acfg, models, emb_dir = ecfg["abmil"], ecfg["models"], repo_path(emb["paths"]["out_dir"])
    single = {}
    for m in [m for m in models if aggregation(m) == "abmil"]:
        t0 = time.time()
        single[m], _ = crossfit_two_banks(lambda ids, m=m: TileBank.from_h5(m, ids, emb_dir), dev.wsi_ids,
                                          test.wsi_ids, dev.y, acfg, seed, log, m)
        print(f"  {m} ABMIL vectors {single[m].shape} in {time.time() - t0:.0f}s", flush=True)
    joint = {}
    if ecfg.get("fusion_aggregation") == "joint_abmil":
        t0 = time.time()
        Z, dims = crossfit_two_banks(lambda ids: MultiViewBank.from_h5(models, ids, emb_dir), dev.wsi_ids,
                                     test.wsi_ids, dev.y, {**acfg, **ecfg.get("joint_abmil", {})}, seed, log,
                                     "joint", train_fn=train_joint_abmil, pool_fn=pool_joint)
        joint = dict(zip(models, split_views(Z, list(dims))))
        print(f"  joint ABMIL vectors {Z.shape} in {time.time() - t0:.0f}s", flush=True)
    return single, joint


# ---- helpers -----------------------------------------------------------------------------------------------------
def sha256(path) -> str:
    return hashlib.sha256(repo_path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=repo_path(".")).stdout.strip()


def logged_trial(log_path, n: int) -> dict:
    for line in repo_path(log_path).read_text().splitlines():
        if line.strip() and (h := json.loads(line))["trial"] == n:
            return h
    raise KeyError(f"trial {n} not in {log_path}")


def refuse_unless_first_run(out_dir, allow_dirty: bool) -> bool:
    """Guards of the real run. Returns True if tracked files have uncommitted changes (allowed by flag)."""
    log = repo_path(load_config("data")["paths"]["test_access_log"])
    if log.exists() and log.read_text().strip():
        raise SystemExit(f"{log} already has an entry: the test set was opened before. Not running again.")
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: the final evaluation was run before. Not running again.")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not allow_dirty:
        raise SystemExit("Tracked files have uncommitted changes. Commit the locked code and configs first, so the "
                         "test run maps to one commit (or pass --allow-dirty; it is recorded).")
    return dirty


def table_md(m: pd.DataFrame, best_meta: str, picks: pd.DataFrame, nested: pd.DataFrame | None,
             fused_cv: dict) -> str:
    def cell(row, k):
        r = m[(m.row == row) & (m.metric == k)].iloc[0]
        return f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"

    def cv(row):
        if row == "fused":
            return f"{fused_cv['macro_auroc_mean']:.3f} ± {fused_cv['macro_auroc_sd']:.3f}"
        return f"{picks.loc[row, 'macro_auroc_mean']:.3f} ± {picks.loc[row, 'macro_auroc_sd']:.3f}"

    def nest(row):
        key = {"fused": "fused (agent, bandit)", best_meta: "best single + meta"}.get(row, row)
        return f"{nested.loc[key, 'outer_auroc']:.3f}" if nested is not None and key in nested.index else "—"

    names = {"meta": "Metadata only (age + sex)",
             **{m_: f"{LABEL[m_]} alone ({AGG_NAME.get(aggregation(m_), aggregation(m_))})" for m_ in LABEL},
             best_meta: f"Best single + meta ({LABEL[best_meta.split('+')[0]]} + meta)",
             "fused": "**Fused (agent) + meta**"}
    lines = ["| Configuration | Test macro AUROC [95% CI] | Test balanced acc. [95% CI] | CV AUROC (mean ± SD) "
             "| Nested-CV AUROC |", "|---|---|---|---|---|"]
    for row in ["meta", "uni2h", "virchow2", "hoptimus1", best_meta, "fused"]:
        lines.append(f"| {names[row]} | {cell(row, 'macro_auroc')} | {cell(row, 'balanced_accuracy')} | {cv(row)} "
                     f"| {nest(row)} |")
    return "\n".join(lines) + "\n"


# ---- main --------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="the real, one-time run on the test patients")
    ap.add_argument("--allow-dirty", action="store_true", help="--test with uncommitted tracked changes (recorded)")
    args = ap.parse_args()
    ecfg, emb, fcfg = load_config("experiments"), load_config("embeddings"), load_config("experiments")["final"]
    clf, seed = ecfg["classifier"], ecfg["search"]["seed"]
    rd = repo_path(fcfg["round"])
    out = repo_path(fcfg["out_dir"] + ("" if args.test else "_dryrun"))
    dirty = refuse_unless_first_run(out, args.allow_dirty) if args.test else None
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    require_power_limit(emb["run"]["max_power_w"])

    # locked picks (read before any test data) ---------------------------------------------------------------------
    picks = pd.read_csv(rd / "baselines" / "summary.csv").set_index("row")
    fused_summary = json.loads((rd / "agent" / fcfg["fused_run"] / "summary.json").read_text())
    fused_cfg = fused_summary["selected_config"]
    meta_rows = [r for r in ROWS if r.endswith("+meta")]
    best_meta = max(meta_rows, key=lambda r: picks.loc[r, "macro_auroc_mean"])     # by CV AUROC, as in the plan
    nested_csv = rd / "nested_cv" / "summary.csv"
    nested = pd.read_csv(nested_csv).set_index("row") if nested_csv.exists() else None
    fixed = [m for m in ecfg["models"] if aggregation(m) not in FOLD_SPECIFIC]

    # data --------------------------------------------------------------------------------------------------------
    if args.test:
        dev, test = load_test_set("final evaluation of the locked v7 rows and fused model (scripts/final_test.py)",
                                  fixed)
        seed_index = fcfg["crossfit_seed_index"]
    else:
        full = load_dev_set()
        tr0, va0 = full.folds[0]
        assert full.fold_ids[0] == (0, 0)
        dev, test = full.take(tr0), full.take(va0)
        seed_index = 0
    ds = concat(dev, test)
    dev_idx, test_idx = np.arange(len(dev.y)), np.arange(len(dev.y), len(ds.y))
    print(f"{'TEST' if args.test else 'DRY RUN'}: fit on {len(dev_idx)} patients, score {len(test_idx)}", flush=True)

    # cross-fitted vectors ----------------------------------------------------------------------------------------
    gpu_logger = start_gpu_logger(repo_path(emb["paths"]["gpu_log"]))
    t0, ablog = time.time(), []
    try:
        single, joint = slide_vectors(dev, test, ecfg["abmil"]["seed"] + 1000 * seed_index, ablog)
    finally:
        gpu_logger.terminate()
    ds = replace(ds, streams={**ds.streams, **single})
    ds_f = replace(ds, streams=joint) if joint else ds

    # refit every row on the training patients, score the test patients once -----------------------------------------
    blocks, probs = rows_for(ecfg["models"]), {}
    for row in ROWS:
        c = {**json.loads(picks.loc[row, "config"]), "blocks": blocks[row]}
        probs[row] = EarlyFusionLR(c, clf, seed=seed).fit(ds, dev_idx).predict_proba(ds, test_idx)
    probs["fused"] = build_model(fused_cfg, clf, seed=seed).fit(ds_f, dev_idx).predict_proba(ds_f, test_idx)
    y = ds.y[test_idx]

    m, paired = evaluate(y, probs, fcfg["bootstrap"], fcfg["bootstrap_seed"])
    out.mkdir(parents=True, exist_ok=False if args.test else True)
    m.to_csv(out / "metrics.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    pd.DataFrame({r: per_class_auroc(y, P) for r, P in probs.items()}, index=CLASSES).T.round(4) \
        .to_csv(out / "per_class_auroc.csv")
    pd.DataFrame([{"row": r, "true": CLASSES[i], "pred": CLASSES[j], "count": int(n)}
                  for r, P in probs.items()
                  for (i, j), n in np.ndenumerate(confusion_matrix(y, P.argmax(1), labels=range(len(CLASSES))))]) \
        .to_csv(out / "confusion.csv", index=False)
    pd.DataFrame([{"wsi_id": ds.wsi_ids[i], "true": CLASSES[ds.y[i]], "row": r, "pred": CLASSES[P[j].argmax()],
                   **{f"p_{c}": round(float(p), 5) for c, p in zip(CLASSES, P[j])}}
                  for r, P in probs.items() for j, i in enumerate(test_idx)]).to_csv(out / "predictions.csv", index=False)
    np.savez_compressed(out / "vectors.npz", wsi_id=np.array(ds.wsi_ids), n_dev=len(dev_idx),
                        **{f"abmil_{k}": v for k, v in single.items()}, **{f"joint_{k}": v for k, v in joint.items()})
    pd.DataFrame(ablog).to_csv(out / "abmil_log.csv", index=False)
    table = table_md(m, best_meta, picks, nested, fused_summary)
    (out / "table.md").write_text(table)

    # dry run: this script must reproduce the logged fold-0 CV scores and the cached fold-0 vectors ---------------
    check = None
    if not args.test:
        check = []
        for row in ROWS + ["fused"]:
            h = (logged_trial(rd / "agent" / fcfg["fused_run"] / "trials.jsonl", fused_summary["selected_trial"])
                 if row == "fused" else logged_trial(picks.loc[row, "source_log"], int(picks.loc[row, "selected_trial"])))
            got = metrics(y, probs[row])
            check.append({"row": row, "auroc_logged": h["fold_macro_auroc"][0], "auroc_here": round(got["macro_auroc"], 5),
                          "ba_logged": h["fold_balanced_accuracy"][0], "ba_here": round(got["balanced_accuracy"], 5)})
        full_order = np.concatenate([tr0, va0])
        for name, z in [*[(f"abmil_{k}", v) for k, v in single.items()], *[(f"joint_{k}", v) for k, v in joint.items()]]:
            f = repo_path(ecfg["features_dir"]) / ("joint_abmil_devfolds.npz" if name.startswith("joint")
                                                   else f"{name}_devfolds.npz")
            key = f"Z_{name.split('_', 1)[1]}" if name.startswith("joint") else "Z"
            with np.load(f) as c:
                check.append({"row": f"vectors {name}", "max_abs_diff": float(np.abs(c[key][0][full_order] - z).max())})
        check = pd.DataFrame(check)
        check.to_csv(out / "dry_run_check.csv", index=False)

    summary = {"mode": "test" if args.test else "dry_run", "finished": dt.datetime.now().isoformat(timespec="seconds"),
               "seconds": round(time.time() - t0, 1), "n_fit": len(dev_idx), "n_scored": len(test_idx),
               "crossfit_seed": ecfg["abmil"]["seed"] + 1000 * seed_index, "classifier_seed": seed,
               "bootstrap": fcfg["bootstrap"], "bootstrap_seed": fcfg["bootstrap_seed"], "best_single_meta": best_meta,
               "fused": {"run": fcfg["fused_run"], "trial": fused_summary["selected_trial"], "config": fused_cfg},
               "rows": {r: {"trial": int(picks.loc[r, "selected_trial"]), "config": json.loads(picks.loc[r, "config"])}
                        for r in ROWS},
               "git_commit": git("rev-parse", "HEAD"), "uncommitted_tracked_changes": dirty,
               "sha256": {p: sha256(p) for p in ["conf/experiments.yaml", "conf/agent.yaml",
                                                 f"{fcfg['round']}/baselines/summary.csv",
                                                 f"{fcfg['round']}/agent/{fcfg['fused_run']}/summary.json"]}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print("\n" + table)
    print(paired[paired.metric.isin(["macro_auroc", "balanced_accuracy"])].to_string(index=False))
    if check is not None:
        print("\ndry-run check (logged fold-0 CV score vs this script):\n" + check.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
