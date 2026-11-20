"""Choosing between two search rounds from their nested-CV outer scores (the rule in the rounds' PLAN.md files).

Rule (v6 and v7 plans): use the NEW round if, for the fused row, its mean outer balanced accuracy is higher than
the old round's AND its mean outer AUROC is not lower than the old round's by more than 1 corrected SE of the
paired outer differences. Otherwise the old round stays final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from luad.eval.cv import corrected_se

FUSED_ROW = "fused (agent, bandit)"
RULE_TEXT = ("use the new round if nested-CV fused outer balanced accuracy is higher than the old round's AND outer "
             "AUROC is not lower than the old round's by more than 1 corrected SE of the paired outer differences")


def round_decision(old: pd.DataFrame, new: pd.DataFrame, n_train: int, n_val: int, old_name: str, new_name: str,
                   row: str = FUSED_ROW) -> dict:
    """old/new: nested-CV per_fold.csv tables (columns outer_fold, row, outer_auroc, outer_ba)."""
    a = old[old.row == row].set_index("outer_fold").sort_index()
    b = new[new.row == row].set_index("outer_fold").sort_index()
    if list(a.index) != list(b.index):
        raise ValueError("the two rounds have different outer folds")
    d_auc, d_ba = (b.outer_auroc - a.outer_auroc).values, (b.outer_ba - a.outer_ba).values
    se = corrected_se(d_auc, n_train, n_val) if np.any(d_auc != 0) else 0.0
    ba_up, auc_ok = bool(d_ba.mean() > 0), bool(d_auc.mean() >= -se)
    return {"rule": RULE_TEXT, "old_round": old_name, "new_round": new_name,
            "old_outer_auroc": round(float(a.outer_auroc.mean()), 4), "new_outer_auroc": round(float(b.outer_auroc.mean()), 4),
            "old_outer_ba": round(float(a.outer_ba.mean()), 4), "new_outer_ba": round(float(b.outer_ba.mean()), 4),
            "auroc_diff": round(float(d_auc.mean()), 4), "auroc_diff_corrected_se": round(se, 4),
            "ba_diff": round(float(d_ba.mean()), 4), "folds_ba_better": int((d_ba > 0).sum()),
            "folds_auroc_better": int((d_auc > 0).sum()),
            "balanced_accuracy_higher": ba_up, "auroc_not_clearly_worse": auc_ok,
            "final_round": new_name if (ba_up and auc_ok) else old_name}
