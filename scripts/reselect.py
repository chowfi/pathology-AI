"""Round v6: apply the final-pick rule of conf/experiments.yaml (search.selection) to FINISHED trial logs.

No new searches and no Claude calls: the rule only changes which logged trial is picked (artifacts/results/v6/PLAN.md).
  1. baseline rows   <source>/baselines/<row>.jsonl            -> <results_dir>/baselines/summary.csv
  2. agent runs      <source>/agent/<run>/trials.jsonl          -> <results_dir>/agent/<run>/summary.json
  3. CV comparison   official fused pick vs every row, paired over the 15 folds -> comparison_cv.csv
  4. nested CV       <source>/nested_cv/outer<k>_<row>.jsonl: pick with the rule inside each outer fold, refit on
                     the outer training patients, score once on the held-out outer fold -> nested_cv/
  5. decision        v5 vs v6 by the rule in the v6 plan -> decision.json
The members of the *_tuned fusion methods stay as they were during the searches (the source rule's picks),
because the logged trial scores were computed with them. The test set is not touched.

Usage:
  uv run python scripts/reselect.py --source artifacts/results/v5
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from luad.agent.selection import incumbent_from_log, select
from luad.config import load_config, repo_path
from luad.data.dataset import CLASSES, load_dev_set
from luad.eval.baselines import rows_for
from luad.eval.cv import corrected_se
from luad.eval.decision import round_decision
from luad.eval.metrics import score
from luad.fusion.early import EarlyFusionLR
from luad.fusion.registry import build_model
from luad.utils.seed import seed_everything

SINGLE_ROWS = ["meta", "uni2h", "virchow2", "hoptimus1", "uni2h+meta", "virchow2+meta", "hoptimus1+meta"]


def load_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sizes(ds) -> tuple[int, int]:
    """(n_train, n_val) exactly as run_search computes them."""
    n_val = int(np.mean([len(v) for _, v in ds.folds]))
    return len(ds.y) - n_val, n_val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="finished round to re-pick from, e.g. artifacts/results/v5")
    args = ap.parse_args()
    ecfg, acfg = load_config("experiments"), load_config("agent")
    s, clf = ecfg["search"], ecfg["classifier"]
    rule, seed = s["selection"], s["seed"]
    seed_everything(seed)
    src, out = repo_path(args.source), repo_path(ecfg["results_dir"])
    ds = load_dev_set()
    n_tr, n_va = sizes(ds)

    # 1. baseline rows --------------------------------------------------------------------------------------------
    b5 = pd.read_csv(src / "baselines" / "summary.csv").set_index("row")
    rows, picks = [], {}
    for row in b5.index:
        h = load_log(src / "baselines" / f"{row}.jsonl")
        inc = incumbent_from_log(h)
        assert inc["trial"] == b5.loc[row, "selected_trial"], f"{row}: log incumbent != source summary"
        p = picks[row] = select(h, rule, inc, n_tr, n_va)
        rows.append({"row": row, "blocks": b5.loc[row, "blocks"], "selection": rule, "selected_trial": p["trial"],
                     "source_selected_trial": inc["trial"], "config": json.dumps(p["config"]),
                     "macro_auroc_mean": round(p["macro_auroc_mean"], 4), "macro_auroc_sd": round(p["macro_auroc_sd"], 4),
                     "balanced_accuracy_mean": round(p["balanced_accuracy_mean"], 4),
                     "balanced_accuracy_sd": round(p["balanced_accuracy_sd"], 4), "n_trials": len(h),
                     "source_log": str((src / "baselines" / f"{row}.jsonl").relative_to(repo_path("."))),
                     **{f"auroc_{c}": v for c, v in zip(CLASSES, p["per_class_auroc_mean"])}})
    (out / "baselines").mkdir(parents=True, exist_ok=True)
    bdf = pd.DataFrame(rows)
    bdf.to_csv(out / "baselines" / "summary.csv", index=False)
    print("baseline rows (source pick -> v6 pick):")
    for r in rows:
        o = b5.loc[r["row"]]
        print(f"  {r['row']:15s} #{r['source_selected_trial']:2d} {o['macro_auroc_mean']:.3f}/{o['balanced_accuracy_mean']:.3f}"
              f" -> #{r['selected_trial']:2d} {r['macro_auroc_mean']:.3f}/{r['balanced_accuracy_mean']:.3f}")

    # 2. agent runs -----------------------------------------------------------------------------------------------
    print("agent runs (source pick -> v6 pick):")
    runs = {}
    for d in sorted((src / "agent").iterdir()):
        if not (d / "summary.json").exists():
            continue
        h, s5 = load_log(d / "trials.jsonl"), json.loads((d / "summary.json").read_text())
        inc = incumbent_from_log(h)
        assert inc["trial"] == s5["selected_trial"], f"{d.name}: log incumbent != source summary"
        p = select(h, rule, inc, n_tr, n_va)
        fixed = {k: s5["selected_config"][k] for k in ("models", "use_meta", "members") if k in s5["selected_config"]}
        s6 = {**s5, "selection": rule, "source_selected_trial": inc["trial"], "selected_trial": p["trial"],
              "selected_config": {**fixed, **p["config"]}, "macro_auroc_mean": p["macro_auroc_mean"],
              "macro_auroc_sd": p["macro_auroc_sd"], "balanced_accuracy_mean": p["balanced_accuracy_mean"],
              "balanced_accuracy_sd": p["balanced_accuracy_sd"], "per_class_auroc_mean": p["per_class_auroc_mean"],
              "source_log": str((d / "trials.jsonl").relative_to(repo_path("."))),
              "llm_calls_this_round": 0, "llm_cost_this_round_usd": 0.0}
        (out / "agent" / d.name).mkdir(parents=True, exist_ok=True)
        (out / "agent" / d.name / "summary.json").write_text(json.dumps(s6, indent=1))
        runs[d.name] = (p, h)
        print(f"  {d.name:13s} #{inc['trial']:2d} {inc['macro_auroc_mean']:.3f}/{inc['balanced_accuracy_mean']:.3f}"
              f" -> #{p['trial']:2d} {p['macro_auroc_mean']:.3f}/{p['balanced_accuracy_mean']:.3f}  {p['config']}")

    # 3. CV comparison: official fused pick (main Claude run) vs every row, paired over the 15 folds ---------------
    f, cmp = runs["main_llm"][0], []
    for row in SINGLE_ROWS:
        r = picks[row]
        for metric, key in [("macro_auroc", "fold_macro_auroc"), ("balanced_accuracy", "fold_balanced_accuracy")]:
            dlt = np.array(f[key]) - np.array(r[key])
            cmp.append({"row": row, "metric": metric, "row_mean": round(float(np.mean(r[key])), 4),
                        "fused_mean": round(float(np.mean(f[key])), 4), "fused_minus_row": round(float(dlt.mean()), 4),
                        "corrected_se": round(corrected_se(dlt, n_tr, n_va), 4), "folds_fused_better": int((dlt > 0).sum())})
    cdf = pd.DataFrame(cmp)
    cdf.to_csv(out / "comparison_cv.csv", index=False)
    print(f"\nofficial fused pick: main_llm trial {f['trial']} (AUROC {f['macro_auroc_mean']:.4f}, BA "
          f"{f['balanced_accuracy_mean']:.4f}); paired over 15 folds:")
    print(cdf.pivot(index="row", columns="metric", values="fused_minus_row").loc[SINGLE_ROWS].to_string())

    # 4. nested CV ------------------------------------------------------------------------------------------------
    nsrc = src / "nested_cv"
    outer = [(j, tr, va) for j, ((r, _), (tr, va)) in enumerate(zip(ds.fold_ids, ds.folds)) if r == 0]
    prev = pd.read_csv(nsrc / "per_fold.csv")
    nrows = []
    for k, (j, tr, va) in enumerate(outer):
        sub = ds.subset(tr, 5, 3, seed + k)                  # the same inner folds as the source run (sizes only)
        itr, iva = sizes(sub)
        ds_k = ds.for_fold(j)
        sel, inc = {}, {}
        for row, blocks in rows_for(ecfg["models"]).items():
            if row == "tile_count":
                continue
            h = load_log(nsrc / f"outer{k}_{row}.jsonl")
            inc[row] = incumbent_from_log(h)
            p = select(h, rule, inc[row], itr, iva)
            sel[row] = (p, lambda c=p["config"], b=blocks: EarlyFusionLR({"blocks": b, **c}, clf, seed=seed))
        members = {m: inc[f"{m}+meta"]["config"] for m in acfg["models"]}     # as used in the inner fused search
        fixed = {"models": acfg["models"], "use_meta": acfg["use_meta"], "members": members}
        h = load_log(nsrc / f"outer{k}_fused_bandit.jsonl")
        inc_f = incumbent_from_log(h)
        p = select(h, rule, inc_f, itr, iva)
        sel["fused (agent, bandit)"] = (p, lambda c={**fixed, **p["config"]}: build_model(c, clf, seed=seed))
        # reproduction check: the source pick refitted here must give the source run's outer score
        o_src = score(ds.y[va], build_model({**fixed, **inc_f["config"]}, clf, seed=seed).fit(ds_k, tr).predict_proba(ds_k, va))
        want = prev[(prev.outer_fold == k) & (prev.row == "fused (agent, bandit)")].iloc[0]
        assert abs(o_src["macro_auroc"] - want.outer_auroc) < 1e-3, f"outer {k}: refit does not reproduce the source"
        singles = list(ecfg["models"])
        bs = max(singles, key=lambda m: sel[m][0]["macro_auroc_mean"])
        bsm = max([f"{m}+meta" for m in singles], key=lambda m: sel[m][0]["macro_auroc_mean"])
        sel[f"best single ({bs})"], sel[f"best single + meta ({bsm})"] = sel[bs], sel[bsm]
        for name, (best, make) in sel.items():
            o = score(ds.y[va], make().fit(ds_k, tr).predict_proba(ds_k, va))
            nrows.append({"outer_fold": k, "row": name.split(" (")[0] if name.startswith("best") else name,
                          "chosen": name, "trial": best["trial"], "config": json.dumps(best["config"]),
                          "inner_cv_auroc": round(best["macro_auroc_mean"], 4),
                          "inner_cv_ba": round(best["balanced_accuracy_mean"], 4),
                          "outer_auroc": round(o["macro_auroc"], 4), "outer_ba": round(o["balanced_accuracy"], 4)})
        fr = next(x for x in nrows if x["outer_fold"] == k and x["row"].startswith("fused"))
        print(f"outer {k}: fused trial {inc_f['trial']} -> {fr['trial']}  outer AUROC {want.outer_auroc:.3f} -> "
              f"{fr['outer_auroc']:.3f}, BA {want.outer_ba:.3f} -> {fr['outer_ba']:.3f}", flush=True)
    ndf = pd.DataFrame(nrows)
    (out / "nested_cv").mkdir(parents=True, exist_ok=True)
    ndf.to_csv(out / "nested_cv" / "per_fold.csv", index=False)
    ndf["optimism"] = ndf["inner_cv_auroc"] - ndf["outer_auroc"]
    summ = ndf.groupby("row").agg(inner_cv_auroc=("inner_cv_auroc", "mean"), outer_auroc=("outer_auroc", "mean"),
                                  outer_auroc_sd=("outer_auroc", "std"), outer_ba=("outer_ba", "mean"),
                                  outer_ba_sd=("outer_ba", "std"), optimism=("optimism", "mean")).round(4)
    summ.to_csv(out / "nested_cv" / "summary.csv")
    print(summ.to_string())

    # 5. decision: old vs new round (rule written in the v6 plan before this step was run) ----------------------
    dec = round_decision(prev, ndf, len(outer[0][1]), len(outer[0][2]), old_name=src.name, new_name=out.name)
    (out / "decision.json").write_text(json.dumps(dec, indent=1))
    print("\nDECISION:", json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
