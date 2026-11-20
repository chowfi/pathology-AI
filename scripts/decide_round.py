"""Choose the final round between two by the nested-CV rule written in their PLAN.md files.

Rule (luad.eval.decision): the new round wins if its fused outer balanced accuracy is higher AND its fused outer
AUROC is not lower by more than 1 corrected SE of the paired outer differences. Writes <new>/decision.json.

Usage:
  uv run python scripts/decide_round.py --old artifacts/results/v6 --new artifacts/results/v7
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from luad.config import repo_path
from luad.data.dataset import load_dev_set
from luad.eval.decision import round_decision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    old, new = repo_path(args.old), repo_path(args.new)
    ds = load_dev_set(["virchow2"])                     # only the fold sizes are needed
    tr, va = next(f for f, (r, _) in zip(ds.folds, ds.fold_ids) if r == 0)
    dec = round_decision(pd.read_csv(old / "nested_cv" / "per_fold.csv"), pd.read_csv(new / "nested_cv" / "per_fold.csv"),
                         len(tr), len(va), old_name=old.name, new_name=new.name)
    (new / "decision.json").write_text(json.dumps(dec, indent=1))
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
