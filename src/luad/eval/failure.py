"""Failure analysis on the DEV patients (train+val, out-of-fold). The test set is never read here.

Used by notebooks/failure_analysis.ipynb:
  * out-of-fold probabilities of a round's selected rows (single models, fused) on the 15 CV folds
  * per-class recall, confusion matrices, "at least one model right", and a per-patient case table
  * attention maps: for a patient, ONE network with the round's recipe is trained on the training patients of the
    repeat-0 CV fold that holds the patient out, so the network never saw that patient's label
  * slide thumbnails, attention heat maps and the top-attended tiles exactly as the foundation models saw them
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from luad.config import load_config, repo_path
from luad.data.dataset import CLASSES, Dataset
from luad.eval.baselines import rows_for
from luad.eval.cv import cross_validate
from luad.fusion.early import EarlyFusionLR
from luad.fusion.registry import build_model

SINGLE_ROWS = ["uni2h", "virchow2", "hoptimus1", "uni2h+meta", "virchow2+meta", "hoptimus1+meta"]


# ---- predictions ------------------------------------------------------------------------------------------------
def round_oof(round_dir: str | Path, ds_single: Dataset, ds_fused: Dataset, n_jobs: int = 15) -> dict[str, np.ndarray]:
    """{row: (n_repeats, n, 7) out-of-fold probabilities} for the round's selected single rows and fused model."""
    cfg = load_config("experiments")
    clf, seed, rd = cfg["classifier"], cfg["search"]["seed"], repo_path(round_dir)
    picks = pd.read_csv(rd / "baselines" / "summary.csv").set_index("row")
    blocks = rows_for(cfg["models"])
    out = {}
    for row in SINGLE_ROWS:
        c = {**json.loads(picks.loc[row, "config"]), "blocks": blocks[row]}
        out[row] = cross_validate(lambda c=c: EarlyFusionLR(c, clf, seed=seed), ds_single, n_jobs=n_jobs)["oof"]
    fused = json.loads((rd / "agent" / "main_llm" / "summary.json").read_text())["selected_config"]
    out["fused"] = cross_validate(lambda: build_model(fused, clf, seed=seed), ds_fused, n_jobs=n_jobs)["oof"]
    return out


def majority(P: np.ndarray) -> np.ndarray:
    """Per patient, the class predicted most often over the CV repeats (ties: highest mean probability)."""
    votes = np.stack([np.bincount(P[:, i].argmax(1), minlength=P.shape[2]) for i in range(P.shape[1])])
    return np.argmax(votes + 1e-3 * P.mean(0), axis=1)


def recall_table(y: np.ndarray, oof: dict[str, np.ndarray]) -> pd.DataFrame:
    """Recall per class (mean over repeats) + balanced accuracy, one row per model."""
    rows = {name: np.mean([[(P[k].argmax(1)[y == c] == c).mean() for c in range(len(CLASSES))]
                           for k in range(P.shape[0])], axis=0) for name, P in oof.items()}
    t = pd.DataFrame(rows, index=CLASSES).T
    t["balanced accuracy"] = t[CLASSES].mean(axis=1)
    return t


def confusion(y: np.ndarray, P: np.ndarray) -> pd.DataFrame:
    """Row-normalised confusion matrix (rows = true class), pooled over the CV repeats."""
    cm = np.zeros((len(CLASSES), len(CLASSES)))
    for k in range(P.shape[0]):
        np.add.at(cm, (y, P[k].argmax(1)), 1)
    return pd.DataFrame(cm / cm.sum(1, keepdims=True), index=CLASSES, columns=CLASSES)


def any_right_table(y: np.ndarray, oof: dict[str, np.ndarray], models: list[str]) -> pd.Series:
    """Recall per class if a patient counts as right when AT LEAST ONE of `models` is right (an upper bound)."""
    right = [np.stack([oof[m][k].argmax(1) == y for k in range(oof[m].shape[0])]) for m in models]
    anyr = np.logical_or.reduce(right)
    s = pd.Series([anyr[:, y == c].mean() for c in range(len(CLASSES))], index=CLASSES)
    s["balanced accuracy"] = s.mean()
    return s


def case_table(ds: Dataset, y: np.ndarray, oof: dict[str, np.ndarray], fused: str = "fused") -> pd.DataFrame:
    """One row per dev patient: true class, fused prediction and probabilities, which single models were right,
    and the clinical / slide facts that can explain an error (diagnosis, conflicting labels, slides, tissue)."""
    cohort = pd.read_csv(repo_path(load_config("data")["paths"]["splits_dir"]) / "cohort.csv",
                         dtype={"SampleNumber": str}).set_index("WSI_ID").loc[ds.wsi_ids]
    tiles = pd.read_csv(repo_path("artifacts/metadata/tiling_summary.csv")).set_index("WSI_ID").loc[ds.wsi_ids]
    P = oof[fused].mean(0)
    pred = majority(oof[fused])
    t = pd.DataFrame({"wsi_id": ds.wsi_ids, "true": [CLASSES[i] for i in y], "fused_pred": [CLASSES[i] for i in pred],
                      "p_true": P[np.arange(len(y)), y].round(3), "p_pred": P[np.arange(len(y)), pred].round(3),
                      "diagnosis": cohort["diagnosis"].str.extract(r"\((\w+)\)")[0].values,
                      "age": cohort["Age"].values, "sex": cohort["Sex"].values,
                      "n_slides": cohort["n_slides"].values, "all_labels": cohort["all_labels"].values,
                      "conflicting_labels": cohort["conflicting_labels"].values,
                      "n_tiles": tiles["n_tiles"].values, "tissue_mm2": tiles["tissue_mm2_raw"].values})
    for m in [r for r in SINGLE_ROWS if "+" not in r]:
        t[f"{m}_right"] = majority(oof[m]) == y
    t["margin"] = (t["p_pred"] - t["p_true"]).round(3)
    return t


def pick_examples(cases: pd.DataFrame, true: str, pred: str, n_wrong: int = 2) -> list[str]:
    """The n_wrong patients of class `true` that the fused model most confidently calls `pred`, then the most
    confidently correct patient of `true` and of `pred` (for comparison)."""
    wrong = cases[(cases.true == true) & (cases.fused_pred == pred)].sort_values("margin", ascending=False)
    ok_true = cases[(cases.true == true) & (cases.fused_pred == true)].sort_values("p_true", ascending=False)
    ok_pred = cases[(cases.true == pred) & (cases.fused_pred == pred)].sort_values("p_true", ascending=False)
    return wrong.wsi_id.head(n_wrong).tolist() + ok_true.wsi_id.head(1).tolist() + ok_pred.wsi_id.head(1).tolist()


# ---- attention ---------------------------------------------------------------------------------------------------
def held_out_fold(ds: Dataset, i: int) -> int:
    """Index of the repeat-0 CV fold whose validation part holds patient i."""
    return next(j for j, ((r, _), (_, va)) in enumerate(zip(ds.fold_ids, ds.folds)) if r == 0 and i in va)


def attention_maps(ds: Dataset, wsi_ids: list[str], bank, kind: str) -> dict[str, np.ndarray]:
    """{wsi_id: attention over the slide's tiles}. kind: a model name (per-model ABMIL; `bank` = MultiViewBank,
    its view of that model is used) or 'joint' (tile-level fusion over all views). One network per held-out fold,
    trained with the configured recipe on that fold's training patients only."""
    import torch

    from luad.embeddings.abmil import pool, pool_joint, train_abmil, train_joint_abmil
    cfg = load_config("experiments")
    acfg = cfg["abmil"]
    idx = {w: ds.wsi_ids.index(w) for w in wsi_ids}
    by_fold: dict[int, list[str]] = {}
    for w, i in idx.items():
        by_fold.setdefault(held_out_fold(ds, i), []).append(w)
    out = {}
    torch.use_deterministic_algorithms(True)
    for j, ws in by_fold.items():
        tr = ds.folds[j][0]
        if kind == "joint":
            net, st = train_joint_abmil(bank, tr, ds.y[tr], {**acfg, **cfg.get("joint_abmil", {})}, acfg["seed"] + 1000 * j)
            _, att = pool_joint(bank, net, st, [idx[w] for w in ws], return_attention=True)
        else:
            view = bank.views[cfg["models"].index(kind)]
            net, st = train_abmil(view, tr, ds.y[tr], acfg, acfg["seed"] + 1000 * j)
            _, att = pool(view, net, st, [idx[w] for w in ws], return_attention=True)
        out |= dict(zip(ws, att))
    return out


# ---- slide pictures ----------------------------------------------------------------------------------------------
def slide_pictures(wsi_id: str, attention: dict[str, np.ndarray], top: int = 6, factor: int = 16):
    """(thumbnail, {kind: heat map on the thumbnail grid}, {kind: top-attended tiles (top, 224, 224, 3)}).
    Tiles are cut from the 4x level and resampled to 0.5 um/px exactly as for the foundation models."""
    from luad.data.slide_io import level_path, read_level, rescale_tiles, thumbnail
    tcfg = load_config("tiling")
    coords = np.load(repo_path(tcfg["paths"]["tiles_dir"]) / f"{wsi_id}.npz")["coords"]
    img, mpp = read_level(level_path(repo_path(tcfg["paths"]["slides_dir"]), wsi_id))
    thumb = thumbnail(img, mpp, tcfg["target_mpp"], factor)
    cell = tcfg["tile_px"] // factor                           # one tile = `cell` thumbnail pixels
    heats, tiles = {}, {}
    for kind, a in attention.items():
        h = np.full(thumb.shape[:2], np.nan, np.float32)
        scale = a / np.percentile(a, 99.5)                     # robust scale: the top 0.5% of tiles = 1
        for (x, y), v in zip(coords // factor, np.clip(scale, 0, 1)):
            h[y:y + cell, x:x + cell] = v
        heats[kind] = h
        tiles[kind] = rescale_tiles(img, mpp, tcfg["target_mpp"], coords[np.argsort(a)[::-1][:top]], tcfg["tile_px"],
                                    tcfg.get("rescale_filter", "lanczos"))
    del img
    return thumb, heats, tiles
