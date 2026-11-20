"""The final pick from a finished search (round v6).

"auroc" (v1-v5): the search's own incumbent, i.e. the accepted best by mean macro AUROC.
"one_se_ba" (v6): the one-standard-error rule with a balanced-accuracy tie-break. A trial ties with the best
    trial when its mean paired per-fold AUROC gap is at most 1 corrected SE (the same SE as the acceptance rule):
    the gap is noise. Among the best trial and its ties, the highest mean balanced accuracy wins.

The README lists macro AUROC and balanced accuracy as primary metrics. AUROC is threshold-free and much less noisy
(per-fold SD ~0.025 vs ~0.07 here), so it stays in charge; balanced accuracy only decides between trials that are
statistically tied on AUROC. The rule only reads finished trial records, so it never changes the search itself.
"""
from __future__ import annotations

import numpy as np

from luad.eval.cv import corrected_se

RULES = ("auroc", "one_se_ba")


def tied_with_best(history: list[dict], n_train: int, n_val: int) -> list[dict]:
    """The trial with the highest mean AUROC (equal: earlier) plus every trial within 1 corrected SE of it."""
    top = max(history, key=lambda h: (h["macro_auroc_mean"], -h["trial"]))
    f = np.array(top["fold_macro_auroc"])
    out = []
    for h in history:
        d = f - np.array(h["fold_macro_auroc"])
        if h is top or d.mean() <= corrected_se(d, n_train, n_val):
            out.append(h)
    return out


def select(history: list[dict], rule: str, incumbent: dict, n_train: int, n_val: int) -> dict:
    """The final trial record of a search under `rule`. `incumbent` = the search's accepted best."""
    if rule == "auroc":
        return incumbent
    if rule == "one_se_ba":
        tied = tied_with_best(history, n_train, n_val)
        return max(tied, key=lambda h: (h["balanced_accuracy_mean"], h["macro_auroc_mean"], -h["trial"]))
    raise ValueError(f"unknown selection rule {rule!r}; known: {RULES}")


def incumbent_from_log(history: list[dict]) -> dict:
    """The 'auroc' pick of a logged search: the current best after its last trial (run_search's own result)."""
    t = history[-1].get("best_trial")
    if t is None:                                   # stopped inside the exploration phase
        return max(history, key=lambda h: h["macro_auroc_mean"])
    return next(h for h in history if h["trial"] == t)
