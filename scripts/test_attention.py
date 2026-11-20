"""Attention maps of the FINAL networks on example test slides: visualization only, after the one test run.

The final test (scripts/final_test.py) did not save its ABMIL networks. This script trains them again with the
same recipe and the same cross-fit seed (abmil.seed + 1000 x final.crossfit_seed_index) on the same 167 dev
patients, and checks that they reproduce the saved vectors in artifacts/results/final/vectors.npz (dev and test,
every model). Only if every check passes does it draw the maps. No classifier is fit and no metric is computed;
the examples are chosen from the saved predictions (test_errors.csv). The test access is logged with its reason.

Writes:
  artifacts/results/final/test_attention_check.csv   max |difference| to the saved vectors, per model and split
  artifacts/results/final/test_attention.npz         attention per example slide (mean over the 5 part networks)
  artifacts/figures/failure_test/<case>_<WSI>.png    thumbnail, attention maps, top-attended tiles

Usage:
  uv run --group notebooks python scripts/test_attention.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS (before CUDA starts)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from luad.config import load_config, repo_path  # noqa: E402
from luad.data.dataset import FOLD_SPECIFIC, aggregation, load_test_set  # noqa: E402
from luad.embeddings.abmil import (MultiViewBank, TileBank, apply_parts, fit_parts, pool, pool_joint,  # noqa: E402
                                   train_joint_abmil)
from luad.eval import failure as fa  # noqa: E402
from luad.utils.gpu import require_power_limit, start_gpu_logger  # noqa: E402
from luad.utils.seed import seed_everything  # noqa: E402

PAIRS = [("Lepidic", "In situ"), ("Papillary", "Micropapillary"), ("Micropapillary", "Solid")]  # main test confusions
N_WRONG = 2
TOL = 1e-6
NAMES = {"uni2h": "UNI2-h attention", "hoptimus1": "H-optimus-1 attention", "joint": "tile-level fusion attention"}
SURF, INK = "#fcfcfb", "#0b0b0b"
HEAT = LinearSegmentedColormap.from_list("heat", ["#cde2fb00", "#3987e5aa", "#0d366bff"])


def examples(t: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Same rule as the dev notebook: the most confident errors of each pair, then the most confident correct
    patient of the true class and of the predicted class."""
    cases = t.assign(wsi_id=t.index, margin=t.p_pred - t.p_true)
    return {pair: fa.pick_examples(cases, *pair, n_wrong=N_WRONG) for pair in PAIRS}


def tag(t: pd.DataFrame, wsi: str) -> str:
    r = t.loc[wsi]
    return f"{r.true} correct" if r.fused_right else f"{r.true} called {r.fused_pred}"


def attention_mean(bank, nets, idx, pool_fn) -> list[np.ndarray]:
    """Attention per slide, averaged over the part networks (the test vector is the mean of their pooled vectors)."""
    out = [np.zeros(int(n), np.float64) for n in bank.n_tiles(idx)]
    for net, st in nets:
        _, att = pool_fn(bank, net, st, idx, return_attention=True)
        for k, a in enumerate(att):
            out[k] += a / len(nets)
    return out


def main() -> None:
    ecfg, emb, fcfg = load_config("experiments"), load_config("embeddings"), load_config("experiments")["final"]
    final = repo_path(fcfg["out_dir"])
    saved = np.load(final / "vectors.npz", allow_pickle=True)
    n_dev = int(saved["n_dev"])
    t = pd.read_csv(final / "test_errors.csv", index_col=0)
    picks = examples(t)
    wanted = sorted({w for ws in picks.values() for w in ws})
    print("examples:", {f"{a}->{b}": ws for (a, b), ws in picks.items()}, flush=True)

    seed_everything(ecfg["search"]["seed"])
    torch.use_deterministic_algorithms(True)
    require_power_limit(emb["run"]["max_power_w"])
    dev, test = load_test_set("visualization only: attention maps of the final networks, retrained with the final "
                              "cross-fit seed and checked against vectors.npz; no classifier fit, no metric "
                              "(scripts/test_attention.py)",
                              [m for m in ecfg["models"] if aggregation(m) not in FOLD_SPECIFIC])
    assert dev.wsi_ids + test.wsi_ids == list(saved["wsi_id"]), "patient order differs from the final test run"
    acfg, models, emb_dir = ecfg["abmil"], ecfg["models"], repo_path(emb["paths"]["out_dir"])
    seed = acfg["seed"] + 1000 * fcfg["crossfit_seed_index"]
    tidx = [test.wsi_ids.index(w) for w in wanted]

    runs = [(m, lambda ids, m=m: TileBank.from_h5(m, ids, emb_dir), acfg, {}, pool, saved[f"abmil_{m}"])
            for m in models if f"abmil_{m}" in saved.files]
    runs.append(("joint", lambda ids: MultiViewBank.from_h5(models, ids, emb_dir),
                 {**acfg, **ecfg.get("joint_abmil", {})}, {"train_fn": train_joint_abmil, "pool_fn": pool_joint},
                 pool_joint, np.concatenate([saved[f"joint_{m}"] for m in models], axis=1)))
    checks, att = [], {}
    gpu_logger = start_gpu_logger(repo_path(emb["paths"]["gpu_log"]))
    try:
        for kind, make_bank, cfg, fns, pool_fn, ref in runs:
            bank = make_bank(dev.wsi_ids)
            z_dev, nets = fit_parts(bank, np.arange(n_dev), dev.y, cfg, seed, **fns)
            del bank
            torch.cuda.empty_cache()
            bank = make_bank(test.wsi_ids)
            z_test = apply_parts(bank, nets, np.arange(len(test.wsi_ids)), pool_fn=pool_fn)
            for split, z, r in [("dev", z_dev, ref[:n_dev]), ("test", z_test, ref[n_dev:])]:
                checks.append({"model": kind, "split": split, "max_abs_diff": float(np.abs(z - r).max())})
                print(f"  check {kind} {split}: max |diff| {checks[-1]['max_abs_diff']:.2e}", flush=True)
            if max(c["max_abs_diff"] for c in checks[-2:]) <= TOL:
                att[kind] = dict(zip(wanted, attention_mean(bank, nets, tidx, pool_fn)))
            del bank, nets
            torch.cuda.empty_cache()
    finally:
        gpu_logger.terminate()
    pd.DataFrame(checks).to_csv(final / "test_attention_check.csv", index=False)
    if any(c["max_abs_diff"] > TOL for c in checks):
        raise SystemExit(f"check failed (tolerance {TOL}): the retrained networks are not the final ones; no maps")
    np.savez_compressed(final / "test_attention.npz", wsi_id=np.array(wanted),
                        **{f"{k}__{w}": v[w].astype(np.float32) for k, v in att.items() for w in wanted})

    out = repo_path("artifacts/figures/failure_test")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9, "text.color": INK})
    cohort = pd.read_csv(repo_path(load_config("data")["paths"]["splits_dir"]) / "cohort.csv").set_index("WSI_ID")
    for wsi in wanted:
        name, r, c = tag(t, wsi), t.loc[wsi], cohort.loc[wsi]
        thumb, heats, tiles = fa.slide_pictures(wsi, {k: att[k][wsi] for k in att})
        fig = plt.figure(figsize=(14, 6.4))
        gs = fig.add_gridspec(2, 12, height_ratios=[2.2, 1])
        ax = fig.add_subplot(gs[0, 0:3]); ax.imshow(thumb); ax.axis("off")
        ax.set_title("H&E (0.5 um/px, 16x smaller)", loc="left")
        for n, kind in enumerate(["uni2h", "hoptimus1", "joint"]):
            ax = fig.add_subplot(gs[0, 3 * (n + 1):3 * (n + 2)])
            ax.imshow(thumb, alpha=0.55); ax.imshow(heats[kind], cmap=HEAT, vmin=0, vmax=1); ax.axis("off")
            ax.set_title(NAMES[kind], loc="left")
        for n in range(6):
            ax = fig.add_subplot(gs[1, 2 * n:2 * n + 2]); ax.imshow(tiles["joint"][n]); ax.axis("off")
            if n == 0:
                ax.set_title("top-attended tiles, fusion attention (112 um each)", loc="left")
        lbl = f" | slides: {r.all_labels}" if r.conflicting_labels else ""
        fig.suptitle(f"TEST {name}: {wsi} | true {r.true} ({r.diagnosis}, age {c.Age}, {c.Sex}) -> fused "
                     f"{r.fused_pred} (p {r.p_pred:.2f}; p true {r.p_true:.2f}){lbl}", x=0.01, ha="left",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{name.replace(' ', '_')}_{wsi}.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {name}: {wsi}", flush=True)
    print(f"wrote {final / 'test_attention_check.csv'}, {final / 'test_attention.npz'}, {out}")


if __name__ == "__main__":
    main()
