"""v7 ablation (CV only, not a table row): Virchow2 tiles with per-model ABMIL instead of the Prism2 slide encoder.

The v7 fused model pools Virchow2 tiles with (joint) attention, while the Virchow2 table row uses Prism2 (fixed in
v3). This ablation separates "tile-level fusion helps" from "Virchow2 simply does better with attention pooling".
Same grid, search loop and final-pick rule as the single-model rows (scripts/run_baselines.py). Needs
artifacts/features/abmil_virchow2_devfolds.npz (scripts/abmil_features.py --models virchow2).

Usage:
  uv run python scripts/ablation_virchow2_abmil.py
"""
from __future__ import annotations

import json

import pandas as pd

from luad.agent.search import GridProposer, run_search
from luad.config import load_config, repo_path
from luad.data.dataset import CLASSES, load_dev_set, load_fold_streams
from luad.eval.baselines import space_for
from luad.fusion.early import EarlyFusionLR
from luad.utils.seed import seed_everything

NAME = "virchow2_abmil"


def main() -> None:
    cfg = load_config("experiments")
    s, clf = cfg["search"], cfg["classifier"]
    seed_everything(s["seed"])
    ds = load_dev_set()
    ds.fold_streams[NAME] = load_fold_streams(ds, "virchow2", repo_path(cfg["features_dir"]) / "abmil_virchow2_devfolds.npz")
    out = repo_path(cfg["results_dir"]) / "ablation"
    summary = []
    for row, blocks, grid_blocks in [(NAME, [NAME], ["virchow2"]), (f"{NAME}+meta", [NAME, "meta"], ["virchow2", "meta"])]:
        res = run_search(ds, lambda c: EarlyFusionLR(c, clf, seed=s["seed"]), GridProposer(space_for(grid_blocks, cfg), s["seed"]),
                         {"blocks": blocks}, s["budget"], s["patience"], s["accept_k"], s["n_jobs"],
                         log_path=out / f"{row}.jsonl", explore=s.get("explore", 0),
                         incumbent=s.get("incumbent", "first"), selection=s.get("selection", "auroc"))
        b = res.best
        summary.append({"row": row, "selected_trial": b["trial"], "config": json.dumps(b["config"]),
                        "macro_auroc_mean": round(b["macro_auroc_mean"], 4), "macro_auroc_sd": round(b["macro_auroc_sd"], 4),
                        "balanced_accuracy_mean": round(b["balanced_accuracy_mean"], 4),
                        "balanced_accuracy_sd": round(b["balanced_accuracy_sd"], 4), "n_trials": len(res.history),
                        **{f"auroc_{c}": v for c, v in zip(CLASSES, b["per_class_auroc_mean"])}})
        print(f"{row:22s} AUROC {b['macro_auroc_mean']:.4f}  BA {b['balanced_accuracy_mean']:.4f}  {b['config']}", flush=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
