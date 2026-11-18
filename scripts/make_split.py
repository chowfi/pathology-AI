"""Stage 1: build the 209-patient cohort and the fixed patient-level split.

Writes (all committed to git):
  artifacts/splits/cohort.csv                 one row per patient: selected slide, label, age, sex, quirk flags
  artifacts/splits/split_seed<seed>.csv       SampleNumber -> WSI_ID, label, split
  artifacts/splits/cv_folds_seed<seed>.csv    repeated stratified K-fold over train+val patients
  artifacts/splits/split_seed<seed>.json      seed, counts, input checksum, policies

Usage: uv run python scripts/make_split.py
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess

from luad.config import load_config, repo_path
from luad.data.cohort import build_cohort, load_slides
from luad.data.splits import make_cv_folds, make_split, sha256, validate_split
from luad.utils.seed import seed_everything


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path("."), text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    cfg = load_config("data")
    seed = cfg["seed"]
    seed_everything(seed)
    out_dir = repo_path(cfg["paths"]["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = repo_path(cfg["paths"]["metadata_csv"])
    slides = load_slides(str(meta_csv), cfg["cohort"]["label_column"])
    cohort = build_cohort(slides)
    split = make_split(cohort, cfg["split"]["fractions"], seed)
    summary = validate_split(split, slides, cfg["cohort"]["expected_n_patients"])
    folds = make_cv_folds(split, cfg["split"]["cv_folds"], cfg["split"]["cv_repeats"], seed)

    cohort.to_csv(out_dir / "cohort.csv", index=False)
    split_cols = ["SampleNumber", "WSI_ID", "label", "split", "conflicting_labels"]
    split[split_cols].sort_values(["split", "label", "SampleNumber"]).to_csv(
        out_dir / f"split_seed{seed}.csv", index=False)
    folds.to_csv(out_dir / f"cv_folds_seed{seed}.csv", index=False)
    meta = {
        "seed": seed,
        "created": dt.date.today().isoformat(),
        "code_commit": git_commit(),
        "input_metadata_csv": cfg["paths"]["metadata_csv"],
        "input_metadata_sha256": sha256(meta_csv),
        "split_unit": "patient (SampleNumber)",
        "stratified_by": "subtype label",
        "fractions": cfg["split"]["fractions"],
        "cv": {"folds": cfg["split"]["cv_folds"], "repeats": cfg["split"]["cv_repeats"], "patients": "train+val"},
        "policies": {k: cfg["cohort"][k] for k in ("slide_selection", "conflicting_label_policy",
                                                    "forbidden_feature_columns")},
        "flagged_patients": {
            "conflicting_labels": cohort.loc[cohort["conflicting_labels"], "SampleNumber"].tolist(),
            "age_mismatch_across_slides": cohort.loc[cohort["age_mismatch_across_slides"], "SampleNumber"].tolist(),
        },
        **summary,
    }
    (out_dir / f"split_seed{seed}.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"seed={seed}  patients={summary['n_patients']}")
    print(f"{'subtype':15s} {'train':>5s} {'val':>5s} {'test':>5s}")
    for lab, c in summary["class_counts"].items():
        print(f"{lab:15s} {c['train']:5d} {c['val']:5d} {c['test']:5d}")
    print(f"flags: {meta['flagged_patients']}")
    print(f"wrote {out_dir}/cohort.csv, split_seed{seed}.csv/.json, cv_folds_seed{seed}.csv")


if __name__ == "__main__":
    main()
