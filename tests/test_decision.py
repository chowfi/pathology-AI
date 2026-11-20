"""Round decision (v6/v7 plans): the new round needs higher outer balanced accuracy and no clear AUROC loss."""
from __future__ import annotations

import pandas as pd

from luad.eval.decision import FUSED_ROW, round_decision


def _table(auc, ba):
    return pd.DataFrame({"outer_fold": range(5), "row": FUSED_ROW, "outer_auroc": auc, "outer_ba": ba})


def test_new_round_needs_higher_ba_and_no_clear_auroc_loss():
    old = _table([0.86, 0.87, 0.85, 0.88, 0.86], [0.45, 0.47, 0.44, 0.48, 0.46])
    better_ba = _table([0.861, 0.868, 0.851, 0.879, 0.861], [0.47, 0.48, 0.45, 0.49, 0.46])
    assert round_decision(old, better_ba, 134, 33, "v6", "v7")["final_round"] == "v7"
    clear_auc_loss = _table([0.83, 0.84, 0.82, 0.85, 0.83], [0.47, 0.48, 0.45, 0.49, 0.46])
    assert round_decision(old, clear_auc_loss, 134, 33, "v6", "v7")["final_round"] == "v6"
    lower_ba = _table([0.87, 0.88, 0.86, 0.89, 0.87], [0.44, 0.46, 0.43, 0.47, 0.45])
    assert round_decision(old, lower_ba, 134, 33, "v6", "v7")["final_round"] == "v6"
