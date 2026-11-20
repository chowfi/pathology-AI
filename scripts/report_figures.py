"""Figures for the report, built only from saved results (dev data; the test set is not read).

  rounds.png   fused model vs best single model + metadata, per development round: CV and nested-CV AUROC

Usage:
  uv run --group notebooks python scripts/report_figures.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from luad.config import repo_path  # noqa: E402

ROUNDS = [("v1", "first agent\nmean pooling"), ("v2", "exploration\nphase"), ("v3", "Prism2 for\nVirchow2"),
          ("v4", "tuned\nmembers"), ("v4b", "incumbent\nfix"), ("v5", "ABMIL for\nUNI2-h, H-opt-1"),
          ("v6", "BA\ntie-break"), ("v7", "tile-level\nfusion")]
SURF, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def round_scores(v: str) -> dict | None:
    d = repo_path(f"artifacts/results/{v}")
    main, nested, base = d / "agent/main_llm/summary.json", d / "nested_cv/summary.csv", d / "baselines/summary.csv"
    if not (main.exists() and nested.exists() and base.exists()):
        return None
    b = pd.read_csv(base).set_index("row")
    meta_rows = [r for r in b.index if r.endswith("+meta") and r != "meta"]
    n = pd.read_csv(nested).set_index("row")
    return {"fused_cv": json.loads(main.read_text())["macro_auroc_mean"],
            "fused_nested": n.loc["fused (agent, bandit)", "outer_auroc"],
            "single_cv": b.loc[meta_rows, "macro_auroc_mean"].max(),
            "single_nested": n.loc["best single + meta", "outer_auroc"]}


def rounds_figure(out) -> None:
    rows = [(v, lbl, s) for v, lbl in ROUNDS if (s := round_scores(v))]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    series = [("fused_cv", BLUE, "-", "o", "full"), ("fused_nested", BLUE, "--", "o", "none"),
              ("single_cv", ORANGE, "-", "s", "full"), ("single_nested", ORANGE, "--", "s", "none")]
    names = {"fused_cv": "Fused + meta, CV", "fused_nested": "Fused + meta, nested CV (unseen patients)",
             "single_cv": "Best single + meta, CV", "single_nested": "Best single + meta, nested CV"}
    for key, col, ls, mk, fill in series:
        ys = [s[key] for _, _, s in rows]
        ax.plot(x, ys, ls=ls, color=col, lw=2, marker=mk, ms=7, fillstyle=fill, mfc=SURF if fill == "none" else col,
                mew=1.8, label=names[key])
    ax.set_xticks(list(x), [f"{v}\n{lbl}" for v, lbl, _ in rows], fontsize=8, color=INK2)
    ax.set_ylabel("Macro AUROC (dev patients)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title("Development rounds: the fused model vs the best single model + metadata", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    out = repo_path("artifacts/figures/report")
    out.mkdir(parents=True, exist_ok=True)
    rounds_figure(out / "rounds.png")
    print(f"wrote {out / 'rounds.png'}")


if __name__ == "__main__":
    main()
