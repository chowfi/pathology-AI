"""Stage 6: the baseline rows of the comparison table, selected on the 15 CV folds (train+val only).

Rows (each one gets the same classifier, folds, budget, patience and acceptance rule):
  meta                  age + sex only
  <model>               one foundation model alone (x3)
  <model>+meta          one foundation model + age + sex (x3); the best by CV is "best single model + metadata"
  tile_count            shortcut check (log10 tile count only); supplementary, not in the main table

Writes artifacts/results/baselines/<row>.jsonl (one line per trial: config, fold scores, decision, reason)
and artifacts/results/baselines/summary.csv. The test set is not touched.

Usage:
  uv run python scripts/run_baselines.py
  uv run python scripts/run_baselines.py --rows meta uni2h
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from luad.agent.search import GridProposer, run_search
from luad.config import load_config, repo_path
from luad.data.dataset import CLASSES, load_dev_set
from luad.eval.baselines import rows_for, space_for
from luad.fusion.early import EarlyFusionLR
from luad.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="*")
    args = ap.parse_args()
    cfg = load_config("experiments")
    s = cfg["search"]
    seed_everything(s["seed"])
    ds = load_dev_set()
    out = repo_path(cfg["results_dir"]) / "baselines"
    all_rows = rows_for(cfg["models"])
    chosen = args.rows or list(all_rows)
    summary = []
    for row in chosen:
        blocks = all_rows[row]
        space = space_for(blocks, cfg)
        res = run_search(ds, lambda c: EarlyFusionLR(c, cfg["classifier"], seed=s["seed"]),
                         GridProposer(space, s["seed"]), {"blocks": blocks}, s["budget"], s["patience"],
                         s["accept_k"], s["n_jobs"], log_path=out / f"{row}.jsonl", explore=s.get("explore", 0),
                     incumbent=s.get("incumbent", "first"), selection=s.get("selection", "auroc"))
        b = res.best
        top = max(res.history, key=lambda h: h["macro_auroc_mean"])
        summary.append({"row": row, "blocks": "+".join(blocks), "selected_trial": b["trial"],
                        "config": json.dumps(b["config"]),
                        "macro_auroc_mean": round(b["macro_auroc_mean"], 4),
                        "macro_auroc_sd": round(b["macro_auroc_sd"], 4),
                        "balanced_accuracy_mean": round(b["balanced_accuracy_mean"], 4),
                        "balanced_accuracy_sd": round(b["balanced_accuracy_sd"], 4),
                        "n_trials": len(res.history), "stop_reason": res.stop_reason,
                        "raw_max_trial": top["trial"], "raw_max_macro_auroc": round(top["macro_auroc_mean"], 4),
                        **{f"auroc_{c}": v for c, v in zip(CLASSES, b["per_class_auroc_mean"])}})
        r = summary[-1]
        print(f"{row:16s} AUROC {r['macro_auroc_mean']:.3f} ± {r['macro_auroc_sd']:.3f}  "
              f"BA {r['balanced_accuracy_mean']:.3f} ± {r['balanced_accuracy_sd']:.3f}  "
              f"trials {r['n_trials']:2d} ({res.stop_reason})  {r['config']}", flush=True)
    df = pd.DataFrame(summary)
    path = out / "summary.csv"
    if path.exists() and args.rows:                       # partial rerun: replace only those rows
        old = pd.read_csv(path)
        df = pd.concat([old[~old["row"].isin(df["row"])], df], ignore_index=True)
    df.to_csv(path, index=False)
    fm = df[df["row"].str.endswith("+meta") & (df["row"] != "meta")]
    if len(fm):
        best = fm.loc[fm["macro_auroc_mean"].idxmax()]
        print(f"best single model + metadata (by CV): {best['row']} {best['macro_auroc_mean']:.3f}")


if __name__ == "__main__":
    main()
