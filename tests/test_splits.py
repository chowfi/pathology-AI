"""Tests for the cohort and the split. They run on the real CLWD.csv and on the committed split files."""
from __future__ import annotations

import pandas as pd
import pytest

from luad.config import load_config, repo_path
from luad.data.cohort import build_cohort, load_slides, wsi_number
from luad.data.splits import load_split, make_cv_folds, make_split, validate_split

CFG = load_config("data")
SEED = CFG["seed"]


@pytest.fixture(scope="module")
def slides() -> pd.DataFrame:
    return load_slides(str(repo_path(CFG["paths"]["metadata_csv"])), CFG["cohort"]["label_column"])


@pytest.fixture(scope="module")
def cohort(slides) -> pd.DataFrame:
    return build_cohort(slides)


@pytest.fixture(scope="module")
def split(cohort) -> pd.DataFrame:
    return make_split(cohort, CFG["split"]["fractions"], SEED)


def test_wsi_number_sorts_numerically():
    assert sorted(["WSI-103", "WSI-35", "WSI-4"], key=wsi_number) == ["WSI-4", "WSI-35", "WSI-103"]


def test_cohort_one_row_per_patient(cohort):
    assert len(cohort) == 209 and cohort["SampleNumber"].is_unique


def test_duplicate_id_is_one_patient_with_lowest_slide(cohort):
    row = cohort.set_index("SampleNumber").loc["8377886"]
    assert row["WSI_ID"] == "WSI-35" and row["Age"] == 68 and row["age_mismatch_across_slides"]


def test_selected_slide_is_lowest_number(slides, cohort):
    lowest = slides.groupby("SampleNumber")["wsi_number"].min()
    assert (cohort.set_index("SampleNumber")["WSI_ID"].map(wsi_number) == lowest).all()


def test_conflicting_patients_use_selected_slide_label(slides, cohort):
    flagged = cohort[cohort["conflicting_labels"]]
    assert sorted(flagged["SampleNumber"]) == ["8225322", "8240634", "8245975", "8248415", "8248805"]
    slide_label = slides.set_index("WSI_ID")["label"]
    assert (flagged["label"] == flagged["WSI_ID"].map(slide_label)).all()


def test_split_rules(split, slides):
    summary = validate_split(split, slides, 209)       # raises on any broken rule
    n = summary["n_patients"]
    assert (n["train"], n["val"], n["test"]) == (146, 21, 42)


def test_no_patient_in_two_splits_at_slide_level(split, slides):
    per_slide = slides.merge(split[["SampleNumber", "split"]], on="SampleNumber")
    assert per_slide.groupby("SampleNumber")["split"].nunique().max() == 1


def test_split_is_stratified(split):
    overall = split["label"].value_counts(normalize=True)
    for s in ("train", "val", "test"):
        part = split.loc[split["split"] == s, "label"].value_counts(normalize=True)
        assert set(part.index) == set(overall.index)                 # every class present
        assert (part - overall).abs().max() < 0.06                   # proportions close to overall


def test_split_is_deterministic(cohort, split):
    again = make_split(cohort, CFG["split"]["fractions"], SEED)
    assert (again.sort_values("SampleNumber")["split"].values == split.sort_values("SampleNumber")["split"].values).all()


def test_cv_folds_exclude_test_and_cover_dev(split):
    folds = make_cv_folds(split, CFG["split"]["cv_folds"], CFG["split"]["cv_repeats"], SEED)
    test_ids = set(split.loc[split["split"] == "test", "SampleNumber"])
    dev_ids = set(split.loc[split["split"] != "test", "SampleNumber"])
    assert not (set(folds["SampleNumber"]) & test_ids)
    for _, rep in folds.groupby("repeat"):                          # each patient held out once per repeat
        assert set(rep["SampleNumber"]) == dev_ids and rep["SampleNumber"].is_unique


def test_committed_split_matches_code(split):
    committed = pd.read_csv(repo_path(CFG["paths"]["splits_dir"]) / f"split_seed{SEED}.csv",
                            dtype={"SampleNumber": str}).set_index("SampleNumber")["split"]
    fresh = split.set_index("SampleNumber")["split"]
    assert committed.sort_index().equals(fresh.sort_index())


def test_test_rows_hidden_by_default(tmp_path):
    path = repo_path(CFG["paths"]["splits_dir"]) / f"split_seed{SEED}.csv"
    assert "test" not in set(load_split(path)["split"])
    with pytest.raises(ValueError):
        load_split(path, include_test=True)                          # no reason given
    log = tmp_path / "access.log"
    df = load_split(path, include_test=True, test_access_log=log, reason="unit test")
    assert "test" in set(df["split"]) and log.read_text().count("unit test") == 1
