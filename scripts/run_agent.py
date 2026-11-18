"""Stage 7: the fusion-strategy search agent (train+val CV only; the test set is not touched).

Loop: proposer (Claude, or the deterministic bandit) -> 15-fold CV -> noise-aware accept/reject -> log -> repeat.
Budget, patience and acceptance rule are the same as for the baseline rows (conf/experiments.yaml).

Writes artifacts/results/agent/<run>/trials.jsonl (one line per trial, with the proposer's written reasoning,
tokens and cost) and summary.json.

Usage:
  uv run python scripts/run_agent.py --run main                # Claude proposer (key from .env), bandit fallback
  uv run python scripts/run_agent.py --run bandit --no-llm     # deterministic, no key needed
  uv run python scripts/run_agent.py --run smoke --budget 5    # short test run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd
from dotenv import load_dotenv

from luad.agent.bandit import BanditProposer
from luad.agent.llm import LLMProposer
from luad.agent.search import run_search
from luad.agent.space import FusionSpace
from luad.config import load_config, repo_path
from luad.data.dataset import load_dev_set
from luad.fusion.registry import build_model
from luad.utils.seed import seed_everything


def baselines_text() -> str:
    p = repo_path(load_config("experiments")["results_dir"]) / "baselines" / "summary.csv"
    df = pd.read_csv(p)
    df = df[df["row"] != "tile_count"]
    return "\n".join(f"- {r.row}: AUROC {r.macro_auroc_mean:.4f} ± {r.macro_auroc_sd:.4f}, "
                     f"BA {r.balanced_accuracy_mean:.3f}" for r in df.itertuples())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name (output folder)")
    ap.add_argument("--no-llm", action="store_true", help="use the deterministic bandit only")
    ap.add_argument("--budget", type=int, help="override the trial budget (for short test runs only)")
    ap.add_argument("--seed", type=int, help="override the bandit seed")
    args = ap.parse_args()

    load_dotenv(repo_path(".env"))                  # ANTHROPIC_API_KEY; the file is gitignored
    ecfg, acfg = load_config("experiments"), load_config("agent")
    s = dict(ecfg["search"])
    if args.budget:
        s["budget"] = args.budget
    seed = args.seed if args.seed is not None else acfg["proposer"]["seed"]
    seed_everything(seed)
    space = FusionSpace(acfg["space"])
    bandit = BanditProposer(space, seed, acfg["proposer"]["ucb_c"])
    use_llm = acfg["proposer"]["kind"] == "llm" and not args.no_llm
    proposer = LLMProposer(space, acfg["llm"], bandit, baselines_text(), s) if use_llm else bandit
    if use_llm and proposer.disabled:
        print(f"LLM proposer disabled ({proposer.disabled}); using the deterministic bandit", flush=True)

    ds = load_dev_set(acfg["models"])
    out = repo_path(acfg["results_dir"]) / args.run
    fixed = {"models": acfg["models"], "use_meta": acfg["use_meta"]}
    t0 = dt.datetime.now()
    res = run_search(ds, lambda c: build_model(c, ecfg["classifier"], seed=ecfg["search"]["seed"]), proposer,
                     fixed, s["budget"], s["patience"], s["accept_k"], s["n_jobs"], log_path=out / "trials.jsonl",
                     explore=s.get("explore", 0))
    b = res.best
    summary = {"run": args.run, "proposer": proposer.name if not (use_llm and proposer.disabled) else "bandit",
               "seed": seed, "budget": s["budget"], "patience": s["patience"], "explore": s.get("explore", 0), "accept_k": s["accept_k"],
               "n_trials": len(res.history), "stop_reason": res.stop_reason,
               "selected_trial": b["trial"], "selected_config": {**fixed, **b["config"]},
               "macro_auroc_mean": b["macro_auroc_mean"], "macro_auroc_sd": b["macro_auroc_sd"],
               "balanced_accuracy_mean": b["balanced_accuracy_mean"],
               "balanced_accuracy_sd": b["balanced_accuracy_sd"],
               "per_class_auroc_mean": b["per_class_auroc_mean"],
               "llm_calls": getattr(proposer, "calls", 0), "llm_cost_usd": round(getattr(proposer, "cost", 0.0), 4),
               "proposers_used": pd.Series([h["proposer"] for h in res.history]).value_counts().to_dict(),
               "started": t0.isoformat(timespec="seconds"),
               "finished": dt.datetime.now().isoformat(timespec="seconds")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for h in res.history:
        print(f"#{h['trial']:2d} {h['decision']:6s} AUROC {h['macro_auroc_mean']:.4f}  {h['proposer'][:40]:40s} "
              f"{json.dumps(h['config'])}", flush=True)
    print(f"selected trial {b['trial']}: AUROC {b['macro_auroc_mean']:.4f} ± {b['macro_auroc_sd']:.4f}, "
          f"BA {b['balanced_accuracy_mean']:.3f}; {res.stop_reason}; LLM calls {summary['llm_calls']}, "
          f"cost ${summary['llm_cost_usd']}")


if __name__ == "__main__":
    main()
