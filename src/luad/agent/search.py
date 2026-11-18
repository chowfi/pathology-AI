"""The search loop shared by every row: propose -> evaluate on 15 CV folds -> accept or reject -> log.

Defences against overfitting the selection (README guardrail):
  * a hard trial budget, the same for every row
  * early stop after `patience` trials in a row without an accepted improvement; the first `explore` trials do
    not count toward patience (exploration phase, added in v2: in v1 trial 1 anchored every run)
  * noise-aware acceptance: a trial replaces the current best only if its mean paired per-fold gain in macro
    AUROC is larger than accept_k x the corrected standard error of that gain

Proposers only choose the next config; the loop, the scoring and the log are identical for all of them.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from luad.data.dataset import Dataset
from luad.eval.cv import corrected_se, cross_validate


@dataclass
class Proposal:
    config: dict
    reason: str                    # written justification, logged verbatim
    proposer: str                  # who actually proposed it (e.g. "llm" or "bandit (fallback: ...)")
    info: dict = field(default_factory=dict)   # e.g. tokens, cost, request id


class Proposer(Protocol):
    name: str

    def propose(self, history: list[dict], ctx: dict) -> Proposal | None:
        """Next proposal, or None when there is nothing left to try. ctx: budget, patience, trials_left,
        streak, best_trial."""


class GridProposer:
    """Deterministic fallback: every grid point once, in a seeded random order."""
    name = "seeded_grid"

    def __init__(self, space: dict[str, list], seed: int):
        keys = list(space)
        grid = [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]
        order = np.random.default_rng(seed).permutation(len(grid))
        self.queue = [grid[i] for i in order]

    def propose(self, history, ctx=None):
        tried = {json.dumps(h["config"], sort_keys=True) for h in history}
        for c in self.queue:
            if json.dumps(c, sort_keys=True) not in tried:
                return Proposal(c, f"next untried grid point in seeded order ({len(tried) + 1}/{len(self.queue)})",
                                self.name)
        return None


@dataclass
class SearchResult:
    best: dict                     # trial record of the accepted best
    history: list[dict]
    stop_reason: str


def run_search(ds: Dataset, make_model: Callable[[dict], object], proposer: Proposer, fixed: dict,
               budget: int, patience: int, accept_k: float, n_jobs: int, log_path: Path | None = None,
               explore: int = 0) -> SearchResult:
    """`fixed` is merged into every proposed config (e.g. the input blocks of the row)."""
    n_val = int(np.mean([len(v) for _, v in ds.folds]))
    n_train = len(ds.y) - n_val
    history, best, streak, stop = [], None, 0, f"budget of {budget} trials used"
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
    for t in range(1, budget + 1):
        ctx = {"budget": budget, "patience": patience, "trials_left": budget - t + 1, "streak": streak,
               "best_trial": best["trial"] if best else None, "accept_k": accept_k, "explore": explore,
               "exploring": t <= explore}
        prop = proposer.propose(history, ctx)
        if prop is None:
            stop = "search space exhausted"
            break
        cfg = prop.config
        full = {**fixed, **cfg}
        res = cross_validate(lambda: make_model(full), ds, n_jobs=n_jobs)
        rec = {"trial": t, "config": cfg, "proposer": prop.proposer, "proposal_reason": prop.reason,
               "proposal_info": prop.info,
               "macro_auroc_mean": res["macro_auroc_mean"], "macro_auroc_sd": res["macro_auroc_sd"],
               "balanced_accuracy_mean": res["balanced_accuracy_mean"],
               "balanced_accuracy_sd": res["balanced_accuracy_sd"],
               "fold_macro_auroc": res["macro_auroc"].round(5).tolist(),
               "fold_balanced_accuracy": res["balanced_accuracy"].round(5).tolist(),
               "per_class_auroc_mean": np.nanmean([f["per_class_auroc"] for f in res["folds"]], 0)
               .round(4).tolist()}
        if best is None:
            rec.update(gain=None, se=None, threshold=None, decision="accept", decision_reason="first trial")
        else:
            d = res["macro_auroc"] - np.array(best["fold_macro_auroc"])
            se = corrected_se(d, n_train, n_val)
            gain, thr = float(d.mean()), accept_k * se
            ok = gain > thr
            rec.update(gain=round(gain, 5), se=round(se, 5), threshold=round(thr, 5),
                       decision="accept" if ok else "reject",
                       decision_reason=(f"gain {gain:+.4f} > {accept_k} x SE {se:.4f}" if ok else
                                        f"gain {gain:+.4f} <= {accept_k} x SE {se:.4f} (within noise)"))
        rec["time"] = dt.datetime.now().isoformat(timespec="seconds")
        if rec["decision"] == "accept":
            best, streak = rec, 0
        elif t > explore:                       # exploration-phase trials do not count toward patience
            streak += 1
        rec["best_trial"] = best["trial"]
        history.append(rec)
        if log_path:
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        if streak >= patience:
            stop = f"{patience} trials in a row without an accepted improvement"
            break
    return SearchResult(best, history, stop)
