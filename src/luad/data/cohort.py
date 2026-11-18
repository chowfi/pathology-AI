"""Build the patient-level cohort: one slide per patient, one label per patient, quirks flagged.

Decisions (also in conf/data.yaml and the report):
  * Patient key = SampleNumber. ID 8377886 (ages 68/69, otherwise identical) is ONE patient.
  * One slide per patient = lowest WSI number, compared as an integer.
  * Label = label of the selected slide (image and label always match). Patients whose slides disagree are
    flagged `conflicting_labels` for a sensitivity analysis.
  * Age and sex come from the selected slide. Pathological_Diagnosis is kept for analysis only and is never a
    model input (AIS == In situ exactly).
"""
from __future__ import annotations

import pandas as pd

SUBTYPES = ("Acinar", "Cribriform", "In situ", "Lepidic", "Micropapillary", "Papillary", "Solid")


def wsi_number(wsi_id: str) -> int:
    """'WSI-35' -> 35. Sorting on this avoids the text-sort trap ('WSI-103' < 'WSI-35')."""
    return int(wsi_id.split("-")[1])


def load_slides(metadata_csv: str, label_column: str = "Tumor Subtype") -> pd.DataFrame:
    """Read CLWD.csv into a tidy slide table with stable column names."""
    df = pd.read_csv(metadata_csv, dtype={"SampleNumber": str})
    df = df.rename(columns={label_column: "label", "Pathological_Diagnosis": "diagnosis"})
    df["wsi_number"] = df["WSI_ID"].map(wsi_number)
    unknown = set(df["label"]) - set(SUBTYPES)
    if unknown:
        raise ValueError(f"unexpected subtype labels: {unknown}")
    return df


def build_cohort(slides: pd.DataFrame) -> pd.DataFrame:
    """One row per patient (SampleNumber) with the selected slide, label, metadata and quirk flags."""
    per_patient = slides.groupby("SampleNumber")
    selected = slides.loc[per_patient["wsi_number"].idxmin()].copy()
    stats = per_patient.agg(
        n_slides=("WSI_ID", "size"),
        all_slides=("WSI_ID", lambda s: ";".join(sorted(s, key=wsi_number))),
        n_labels=("label", "nunique"),
        all_labels=("label", lambda s: ";".join(sorted(set(s)))),
        n_ages=("Age", "nunique"),
    )
    cohort = selected.merge(stats, left_on="SampleNumber", right_index=True)
    cohort["conflicting_labels"] = cohort["n_labels"] > 1
    cohort["age_mismatch_across_slides"] = cohort["n_ages"] > 1
    cols = ["SampleNumber", "WSI_ID", "label", "Age", "Sex", "diagnosis", "n_slides", "all_slides",
            "all_labels", "conflicting_labels", "age_mismatch_across_slides"]
    return cohort[cols].sort_values("SampleNumber").reset_index(drop=True)
