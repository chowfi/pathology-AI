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
ANY1 = "at least 1 of 3 image models right"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1-2 of the reference palette


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
                       index=CLASSES).T
    rec["n_right"] = [(wide[r] == true).sum() for r in rec.index]
    # upper bound for picking the right model per patient: at least one of the 3 image-only models is right
    any1 = t[[f"{m}_right" for m in SINGLE]].any(axis=1)
    rec.loc[ANY1] = [any1[t.true == c].mean() for c in CLASSES] + [any1.sum()]
    rec.insert(len(CLASSES), "BA", rec[CLASSES].mean(axis=1))
    rec = rec.round(2)
    rec.to_csv(final / "test_recall.csv")

    prof = pd.DataFrame({"patients": true.value_counts(), "fused_recall": t.groupby("true").fused_right.mean(),
                         "share_with_more_slides": (t.n_slides > 1).groupby(t.true).mean(),
                         "median_tiles": t.groupby("true").n_tiles.median(),
                         "median_age": cohort["Age"].groupby(true).median(),
                         "share_female": (cohort["Sex"] == "Female").groupby(true).mean()}).reindex(CLASSES).round(2)
    prof.index.name = "true"
    prof.to_csv(final / "test_class_profile.csv")
    print(f"\nper-class profile:\n{prof.to_string()}")

    bins = pd.cut(t.p_pred, [0, 0.3, 0.4, 1.0], right=False, labels=["< 0.30", "0.30-0.40", ">= 0.40"])
    conf = t.groupby(bins, observed=False).fused_right.agg(patients="size", accuracy="mean").round(2)
    conf.index.name = "fused_p_pred"
    conf.to_csv(final / "test_confidence.csv")

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
    print(f"\nfused accuracy by confidence (probability of the predicted class):\n{conf.to_string()}")
    print(f"\nfused errors, true -> predicted:\n"
          f"{wrong.groupby(['true', 'fused_pred']).size().sort_values(ascending=False).to_string()}")

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

    # recall per class: best single + meta vs fused, with the "at least one image model right" upper bound
    n_cls = true.value_counts().reindex(CLASSES)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF, gridspec_kw={"width_ratios": [3.2, 1]})
    x, w = np.arange(len(CLASSES)), 0.36
    for k, (r, col, lab) in enumerate([(best, ORANGE, names[best]), ("fused", BLUE, names["fused"])]):
        v = rec.loc[r, CLASSES].astype(float).values
        ax.bar(x + (k - 0.5) * w, v, w - 0.03, color=col, label=f"{lab}: BA {ba[r]:.3f}", zorder=2)
    ub = rec.loc[ANY1, CLASSES].astype(float).values
    ax.scatter(x, ub, marker="_", s=900, linewidths=2.2, color=INK, zorder=3,
               label=f"At least 1 of 3 image models right: BA {rec.loc[ANY1, 'BA']:.2f}")
    ax.set_xticks(x, [f"{c}\n({k})" for c, k in zip(CLASSES, n_cls)], fontsize=8, color=INK2)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Recall (share of the class right)", color=INK2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=1, frameon=False, fontsize=8.5, labelcolor=INK, markerscale=0.45)
    ax.set_title("Recall per class (test patients in brackets)", loc="left", fontsize=10, color=INK)
    bx.bar(range(len(conf)), conf.accuracy.values, 0.6, color=BLUE, zorder=2)
    for i, (n_, a_) in enumerate(zip(conf.patients, conf.accuracy)):
        bx.text(i, a_ + 0.03, f"{a_:.2f}\nn={n_}", ha="center", va="bottom", fontsize=8, color=INK)
    bx.set_xticks(range(len(conf)), conf.index.astype(str), fontsize=8, color=INK2)
    bx.set_xlabel("Fused probability of the predicted class", color=INK2)
    bx.set_ylim(0, 1.25); bx.set_ylabel("Fused accuracy", color=INK2)
    bx.set_title("Accuracy by confidence", loc="left", fontsize=10, color=INK)
    for a in (ax, bx):
        a.set_facecolor(SURF); a.tick_params(length=0, colors=INK2)
        a.yaxis.grid(True, color="#e4e3df", linewidth=0.8, zorder=0)
        for s in a.spines.values():
            s.set_visible(False)
    fig.suptitle("Test set (42 patients, one run, saved predictions)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    out2 = repo_path("artifacts/figures/report/test_recall.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}, {out2}, {final / 'test_errors.csv'}, {final / 'test_recall.csv'}, "
          f"{final / 'test_confidence.csv'}")


if __name__ == "__main__":
    main()
