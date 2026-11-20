"""Failure analysis of the ONE test run, from its saved predictions (no model is refit; the test set is not reloaded).

Reads artifacts/results/final/predictions.csv and the cohort facts, and writes:
  artifacts/results/final/test_errors.csv     one row per test patient: fused prediction, which rows were right,
                                              diagnosis, conflicting labels, slides, tiles
  artifacts/results/final/test_recall.csv     recall per class and row
  artifacts/figures/report/test_confusion.png fused + meta vs best single + meta, counts

Usage:
  uv run --group notebooks python scripts/test_analysis.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from luad.config import load_config, repo_path  # noqa: E402
from luad.data.dataset import CLASSES  # noqa: E402

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SINGLE = ["uni2h", "virchow2", "hoptimus1"]


def main() -> None:
    final = repo_path(load_config("experiments")["final"]["out_dir"])
    best = json.loads((final / "summary.json").read_text())["best_single_meta"]
    p = pd.read_csv(final / "predictions.csv")
    wide = p.pivot(index="wsi_id", columns="row", values="pred")
    true = p.drop_duplicates("wsi_id").set_index("wsi_id")["true"].loc[wide.index]
    fused = p[p.row == "fused"].set_index("wsi_id").loc[wide.index]
    cohort = pd.read_csv(repo_path(load_config("data")["paths"]["splits_dir"]) / "cohort.csv",
                         dtype={"SampleNumber": str}).set_index("WSI_ID").loc[wide.index]
    tiles = pd.read_csv(repo_path("artifacts/metadata/tiling_summary.csv")).set_index("WSI_ID").loc[wide.index]

    t = pd.DataFrame({"true": true, "fused_pred": wide["fused"],
                      "p_true": [fused.loc[w, f"p_{c}"] for w, c in true.items()],
                      "p_pred": [fused.loc[w, f"p_{c}"] for w, c in wide["fused"].items()],
                      "diagnosis": cohort["diagnosis"].str.extract(r"\((\w+)\)")[0],
                      "n_slides": cohort["n_slides"], "all_labels": cohort["all_labels"],
                      "conflicting_labels": cohort["conflicting_labels"], "n_tiles": tiles["n_tiles"]})
    for r in SINGLE + [best]:
        t[f"{r}_right"] = wide[r] == true
    t["fused_right"] = t.fused_pred == t.true
    t["n_single_right"] = t[[f"{m}_right" for m in SINGLE]].sum(axis=1)
    t = t.sort_values(["fused_right", "true"]).round(3)
    t.to_csv(final / "test_errors.csv")

    rec = pd.DataFrame({r: [(wide[r][true == c] == c).mean() for c in CLASSES] for r in wide.columns},
                       index=CLASSES).T.round(2)
    rec["n_right"] = [(wide[r] == true).sum() for r in rec.index]
    rec.to_csv(final / "test_recall.csv")

    wrong, right = t[~t.fused_right], t[t.fused_right]
    print(f"test patients per class: {true.value_counts().reindex(CLASSES).to_dict()}")
    print(f"\nrecall per class and patients right (of {len(t)}):\n{rec.to_string()}")
    print(f"\nfused wrong on {len(wrong)} patients; how many of the 3 single image models got them right:\n"
          f"{wrong.n_single_right.value_counts().sort_index().to_dict()}")
    print(f"fused right but {best} wrong: {(t.fused_right & ~t[f'{best}_right']).sum()}; "
          f"{best} right but fused wrong: {(~t.fused_right & t[f'{best}_right']).sum()}")
    for col in ["diagnosis", "conflicting_labels"]:
        print(f"\nfused recall by {col}:\n{t.groupby(col).fused_right.agg(['mean', 'size']).round(2).to_string()}")
    print(f"\nfused recall, 1 slide vs several: "
          f"{t.groupby(t.n_slides > 1).fused_right.mean().round(2).to_dict()}")
    print(f"\nfused errors:\n{wrong[['true', 'fused_pred', 'p_true', 'p_pred', 'diagnosis', 'all_labels', 'n_slides', 'n_single_right', f'{best}_right']].to_string()}")
    print(f"median fused probability of the predicted class: right {right.p_pred.median():.2f}, "
          f"wrong {wrong.p_pred.median():.2f}")

    cm = pd.read_csv(final / "confusion.csv")
    label = {"uni2h": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
    names = {"fused": "Fused (agent) + meta", best: f"Best single + meta ({label[best.split('+')[0]]} + meta)"}
    m = pd.read_csv(final / "metrics.csv")
    ba = m[m.metric == "balanced_accuracy"].set_index("row")["value"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURF)
    for ax, r in zip(axes, ["fused", best]):
        M = cm[cm.row == r].pivot(index="true", columns="pred", values="count").reindex(index=CLASSES, columns=CLASSES)
        n = M.sum(axis=1).values
        ax.imshow(M.values / n[:, None], cmap=BLUES, vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                if M.values[i, j]:
                    ax.text(j, i, int(M.values[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if M.values[i, j] / n[i] > 0.62 else INK)
        ax.set_xticks(range(7), CLASSES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(7), [f"{c} ({k})" for c, k in zip(CLASSES, n)], fontsize=8)
        ax.set_xlabel("Predicted", color=INK2)
        ax.set_ylabel("True (patients)", color=INK2)
        ax.tick_params(length=0, colors=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{names[r]}\n{int(np.trace(M.values))} of {int(n.sum())} patients right, balanced accuracy {ba[r]:.3f}",
                     loc="left", fontsize=10, color=INK)
    fig.suptitle("Test set (42 patients, one run): counts; colour = share of the true class", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    out = repo_path("artifacts/figures/report/test_confusion.png")
    fig.savefig(out, dpi=200)
    print(f"\nwrote {out}, {final / 'test_errors.csv'}, {final / 'test_recall.csv'}")


if __name__ == "__main__":
    main()
