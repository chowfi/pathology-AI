"""CV comparison for the current round: the official fused pick (main Claude run) vs every single-model row.

Paired over the 15 fixed CV folds (dev patients only; the test set is not touched). Reads the row picks from
<results_dir>/baselines/summary.csv (selected trial in its source log) and the fused pick from
<results_dir>/agent/<run>/summary.json + trials.jsonl. Writes <results_dir>/comparison_cv.csv.

Usage:
  uv run python scripts/compare_cv.py                 # run = main_llm
  uv run python scripts/compare_cv.py --run main_bandit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from luad.config import load_config, repo_path
from luad.data.dataset import load_dev_set
from luad.eval.cv import corrected_se

ROWS = ["meta", "uni2h", "virchow2", "hoptimus1", "uni2h+meta", "virchow2+meta", "hoptimus1+meta"]
METRICS = [("macro_auroc", "fold_macro_auroc"), ("balanced_accuracy", "fold_balanced_accuracy")]


def trial(log: Path, n: int) -> dict:
    for line in log.read_text().splitlines():
        if line.strip() and (h := json.loads(line))["trial"] == n:
            return h
    raise KeyError(f"trial {n} not in {log}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main_llm", help="agent run whose pick is the fused model")
    args = ap.parse_args()
    out = repo_path(load_config("experiments")["results_dir"])
    ds = load_dev_set()
    n_val = int(np.mean([len(v) for _, v in ds.folds]))  # as in run_search
    n_tr = len(ds.y) - n_val

    run = out / "agent" / args.run
    f = trial(run / "trials.jsonl", json.loads((run / "summary.json").read_text())["selected_trial"])
    b = pd.read_csv(out / "baselines" / "summary.csv").set_index("row")
    cmp = []
    for row in ROWS:
        r = trial(repo_path(b.loc[row, "source_log"]), int(b.loc[row, "selected_trial"]))
        for metric, key in METRICS:
            d = np.array(f[key]) - np.array(r[key])
            cmp.append({"row": row, "metric": metric, "row_mean": round(float(np.mean(r[key])), 4),
                        "fused_mean": round(float(np.mean(f[key])), 4), "fused_minus_row": round(float(d.mean()), 4),
                        "corrected_se": round(corrected_se(d, n_tr, n_val), 4),
                        "folds_fused_better": int((d > 0).sum())})
    cdf = pd.DataFrame(cmp)
    cdf.to_csv(out / "comparison_cv.csv", index=False)
    print(f"fused pick: {args.run} trial {f['trial']} (AUROC {f['macro_auroc_mean']:.4f}, "
          f"BA {f['balanced_accuracy_mean']:.4f}); paired over {len(f['fold_macro_auroc'])} folds:")
    print(cdf.to_string(index=False))
    print(f"wrote {out / 'comparison_cv.csv'}")


if __name__ == "__main__":
    main()
