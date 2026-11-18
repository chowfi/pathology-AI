"""Stage 8: nested CV - how much of the selected CV score is selection luck ("selection optimism")?

Outer loop: the 5 folds of repeat 0 of the committed CV folds (train+val patients only; test untouched).
For each outer fold:
  1. inner data = the other 4/5 of the patients, with new 5x3 stratified folds made only from them
  2. run the WHOLE selection on the inner data: the fusion agent (deterministic bandit, same budget/patience/
     acceptance) and every baseline search (same grids as scripts/run_baselines.py); "best single model" and
     "best single model + metadata" are also chosen inside, by inner CV
  3. refit each selected config on all inner patients and score it ONCE on the held-out outer fold
Optimism = inner CV score of the selected config - its outer score. The outer scores also give an honest,
selection-free estimate of fused vs best single model (paired over the 5 outer folds).

Usage:
  uv run python scripts/nested_cv.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from luad.agent.bandit import BanditProposer
from luad.agent.search import GridProposer, run_search
from luad.agent.space import FusionSpace
from luad.config import load_config, repo_path
from luad.data.dataset import load_dev_set
from luad.eval.baselines import rows_for, space_for
from luad.eval.metrics import score
from luad.fusion.early import EarlyFusionLR
from luad.fusion.registry import build_model
from luad.utils.seed import seed_everything


def main() -> None:
    ecfg, acfg = load_config("experiments"), load_config("agent")
    s = ecfg["search"]
    seed_everything(s["seed"])
    ds = load_dev_set()
    outer = [(tr, va) for (r, _), (tr, va) in zip(ds.fold_ids, ds.folds) if r == 0]
    out_dir = repo_path(ecfg["results_dir"]) / "nested_cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    clf, seed = ecfg["classifier"], s["seed"]
    rows = []
    for k, (tr, va) in enumerate(outer):
        sub = ds.subset(tr, 5, 3, seed + k)
        sel = {}
        for row, blocks in rows_for(ecfg["models"]).items():
            if row == "tile_count":
                continue
            r = run_search(sub, lambda c: EarlyFusionLR(c, clf, seed=seed), GridProposer(space_for(blocks, ecfg), seed),
                           {"blocks": blocks}, s["budget"], s["patience"], s["accept_k"], s["n_jobs"],
                           log_path=out_dir / f"outer{k}_{row}.jsonl")
            sel[row] = (r.best, lambda c=r.best["config"], b=blocks: EarlyFusionLR({"blocks": b, **c}, clf, seed=seed))
        space = FusionSpace(acfg["space"])
        fixed = {"models": acfg["models"], "use_meta": acfg["use_meta"]}
        r = run_search(sub, lambda c: build_model(c, clf, seed=seed), BanditProposer(space, seed, acfg["proposer"]["ucb_c"]),
                       fixed, s["budget"], s["patience"], s["accept_k"], s["n_jobs"],
                       log_path=out_dir / f"outer{k}_fused_bandit.jsonl")
        sel["fused (agent, bandit)"] = (r.best, lambda c={**fixed, **r.best["config"]}: build_model(c, clf, seed=seed))
        singles = [m for m in ecfg["models"]]
        best_single = max(singles, key=lambda m: sel[m][0]["macro_auroc_mean"])
        best_single_meta = max([f"{m}+meta" for m in singles], key=lambda m: sel[m][0]["macro_auroc_mean"])
        sel[f"best single ({best_single})"] = sel[best_single]
        sel[f"best single + meta ({best_single_meta})"] = sel[best_single_meta]
        for name, (best, make) in sel.items():
            p = make().fit(ds, tr).predict_proba(ds, va)
            o = score(ds.y[va], p)
            rows.append({"outer_fold": k, "row": name.split(" (")[0] if name.startswith("best") else name,
                         "chosen": name, "config": json.dumps(best["config"]),
                         "inner_cv_auroc": round(best["macro_auroc_mean"], 4),
                         "outer_auroc": round(o["macro_auroc"], 4), "outer_ba": round(o["balanced_accuracy"], 4)})
        f = [x for x in rows if x["outer_fold"] == k]
        fused = next(x for x in f if x["row"].startswith("fused"))
        bsm = next(x for x in f if x["row"] == "best single + meta")
        print(f"outer {k}: fused inner {fused['inner_cv_auroc']:.3f} -> outer {fused['outer_auroc']:.3f} | "
              f"{bsm['chosen']} inner {bsm['inner_cv_auroc']:.3f} -> outer {bsm['outer_auroc']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_fold.csv", index=False)
    df["optimism"] = df["inner_cv_auroc"] - df["outer_auroc"]
    summ = df.groupby("row").agg(inner_cv_auroc=("inner_cv_auroc", "mean"), outer_auroc=("outer_auroc", "mean"),
                                 outer_auroc_sd=("outer_auroc", "std"), outer_ba=("outer_ba", "mean"),
                                 optimism=("optimism", "mean"), optimism_sd=("optimism", "std")).round(4)
    summ.to_csv(out_dir / "summary.csv")
    print(summ.to_string())
    piv = df.pivot(index="outer_fold", columns="row", values="outer_auroc")
    d = piv["fused (agent, bandit)"] - piv["best single + meta"]
    print(f"fused - best single + meta (outer, paired over 5 folds): {d.mean():+.4f} ± {d.std():.4f} "
          f"(fused better in {(d > 0).sum()}/5 folds)")


if __name__ == "__main__":
    main()
