"""Patient-level stratified train/val/test split, fixed CV folds, and a guarded test-set loader.

Rules enforced here (README "Mandatory"):
  * The split unit is the patient (SampleNumber). The cohort has one row per patient, so no patient can be in
    two splits; `validate_split` checks this again against ALL slides in CLWD.csv.
  * Stratified by subtype at both stages, so every class appears in train, val and test.
  * Seeded and saved to artifacts/splits/ as a versioned, committed artifact.
  * Test labels are only returned by `load_split(..., include_test=True)`, which appends to a log, so every
    test-set access is recorded.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

SPLITS = ("train", "val", "test")


def make_split(cohort: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Return the cohort with a `split` column. Two stratified stages: test off first, then val from the rest."""
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"fractions must sum to 1: {fractions}")
    if cohort["SampleNumber"].duplicated().any():
        raise ValueError("cohort must have exactly one row per patient")
    rest, test = train_test_split(cohort, test_size=fractions["test"], stratify=cohort["label"],
                                  random_state=seed)
    val_share = fractions["val"] / (fractions["train"] + fractions["val"])
    train, val = train_test_split(rest, test_size=val_share, stratify=rest["label"], random_state=seed)
    out = cohort.copy()
    out["split"] = ""
    for name, part in (("train", train), ("val", val), ("test", test)):
        out.loc[part.index, "split"] = name
    return out


def make_cv_folds(split_df: pd.DataFrame, n_folds: int, n_repeats: int, seed: int) -> pd.DataFrame:
    """Fixed repeated stratified K-fold over train+val patients (never test).

    Returns long format: SampleNumber, repeat, fold (the fold in which the patient is held out).
    Every agent trial and every tuning run uses these exact folds.
    """
    dev = split_df[split_df["split"].isin(["train", "val"])].reset_index(drop=True)
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    rows = []
    for k, (_, held_out) in enumerate(rskf.split(dev, dev["label"])):
        for i in held_out:
            rows.append({"SampleNumber": dev.at[i, "SampleNumber"], "repeat": k // n_folds, "fold": k % n_folds})
    return pd.DataFrame(rows).sort_values(["repeat", "fold", "SampleNumber"]).reset_index(drop=True)


def validate_split(split_df: pd.DataFrame, all_slides: pd.DataFrame, expected_n_patients: int) -> dict:
    """Raise if any mandatory rule is broken; return a summary for the metadata file."""
    if len(split_df) != expected_n_patients:
        raise AssertionError(f"expected {expected_n_patients} patients, got {len(split_df)}")
    if set(split_df["split"]) != set(SPLITS):
        raise AssertionError(f"splits present: {set(split_df['split'])}")
    # every slide in CLWD.csv maps to exactly one split through its patient -> no patient spans two splits
    slide_split = all_slides.merge(split_df[["SampleNumber", "split"]], on="SampleNumber", how="left")
    if slide_split["split"].isna().any():
        raise AssertionError("some slides belong to no patient in the split")
    spans = slide_split.groupby("SampleNumber")["split"].nunique()
    if (spans > 1).any():
        raise AssertionError(f"patients in more than one split: {list(spans[spans > 1].index)}")
    counts = pd.crosstab(split_df["label"], split_df["split"])[list(SPLITS)]
    if (counts == 0).any().any():
        raise AssertionError(f"a class is missing from a split:\n{counts}")
    return {"n_patients": {s: int((split_df["split"] == s).sum()) for s in SPLITS},
            "class_counts": {lab: {s: int(counts.at[lab, s]) for s in SPLITS} for lab in counts.index}}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(split_csv: str | Path, include_test: bool = False, test_access_log: str | Path | None = None,
               reason: str = "") -> pd.DataFrame:
    """Load the committed split. Test rows are dropped unless include_test=True, which is logged.

    Use include_test=True exactly once, for the final evaluation, with a reason string.
    """
    df = pd.read_csv(split_csv, dtype={"SampleNumber": str})
    if not include_test:
        return df[df["split"] != "test"].reset_index(drop=True)
    if not reason:
        raise ValueError("include_test=True needs a reason, e.g. 'final evaluation of agent-selected config'")
    if test_access_log is not None:
        log = Path(test_access_log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"),
                                "split_file": str(split_csv),
                                "reason": reason}) + "\n")
    return df
