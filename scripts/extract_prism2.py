"""Stage 5b: Prism2 slide embeddings for the Virchow2 stream.

Prism2 (paige-ai/Prism2, pinned revision) is a slide encoder: a perceiver that cross-attends from learned latent
queries to all tile embeddings of a slide. Its input is the Virchow2 CLS token of every tissue tile (224 px at
0.5 um/px) - exactly the first 1280 numbers of our saved Virchow2 tile features, so no tile is re-encoded.
Output: get_base_embedding -> one 2560-d vector per slide.

Recipe as verified in scripts/explore/check_prism2_shim.py (artifacts/logs/check_prism2_shim.txt): PyTorch SDPA
shim in place of flash-attn (max diff 2e-6 vs reference attention), bf16 autocast, one slide per call (no
padding). Refuses to start above the configured GPU power limit.

Writes artifacts/features/prism2_virchow2.npz (same format as the mean-pooled files).

Usage:
  uv run python scripts/extract_prism2.py
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
import torch

from luad.config import load_config, repo_path
from luad.embeddings import flash_attn_shim
from luad.embeddings.io import h5_path, read_h5
from luad.embeddings.pool import save_slide_features
from luad.utils.gpu import require_power_limit, start_gpu_logger
from luad.utils.seed import seed_everything

PRISM2_ID = "paige-ai/Prism2"
PRISM2_REV = "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a"     # pinned: the remote code version that was inspected
N_CLS = 1280                                                 # Virchow2 CLS token = first 1280 of our 2560-d features


def main() -> None:
    ecfg, dcfg = load_config("embeddings"), load_config("data")
    seed_everything(dcfg["seed"])
    require_power_limit(ecfg["run"]["max_power_w"])
    gpu_logger = start_gpu_logger(repo_path(ecfg["paths"]["gpu_log"]))
    try:
        flash_attn_shim.install()
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(PRISM2_ID, revision=PRISM2_REV, trust_remote_code=True,
                                          torch_dtype="auto").to("cuda").eval()
        proc = AutoProcessor.from_pretrained(PRISM2_ID, revision=PRISM2_REV, trust_remote_code=True)
        wsi_ids = pd.read_csv(repo_path(dcfg["paths"]["splits_dir"]) / "cohort.csv")["WSI_ID"].tolist()
        emb_dir = repo_path(ecfg["paths"]["out_dir"])
        vecs, n_tiles, t0 = [], [], time.time()
        for i, w in enumerate(wsi_ids, 1):
            f, _, attrs = read_h5(h5_path(emb_dir, "virchow2", w))
            if attrs["pooling"] != "cls_plus_mean_patch":
                raise ValueError(f"{w}: unexpected Virchow2 pooling {attrs['pooling']}")
            cls = torch.from_numpy(f[:, :N_CLS].astype(np.float32))
            b = proc(tile_embeddings=[cls]).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                v = model.get_base_embedding(**b).float().cpu().numpy()[0]
            if not np.isfinite(v).all():
                raise FloatingPointError(f"{w}: non-finite Prism2 embedding")
            vecs.append(v)
            n_tiles.append(len(f))
            if i % 20 == 0 or i == len(wsi_ids):
                print(f"{i}/{len(wsi_ids)} slides, {time.time() - t0:.0f}s, last {w}: {len(f)} tiles, "
                      f"|v|={np.linalg.norm(v):.2f}", flush=True)
        X = np.stack(vecs)
        meta = {"model": "prism2", "hf_id": PRISM2_ID, "revision": PRISM2_REV, "input": "virchow2 CLS (1280)",
                "virchow2_revision": str(attrs["revision"]), "output": "get_base_embedding", "dim": int(X.shape[1]),
                "aggregation": "prism2", "precision": "bf16 autocast", "attention": "SDPA shim for flash-attn",
                "n_slides": len(wsi_ids), "seconds": round(time.time() - t0, 1),
                "created": dt.datetime.now().isoformat(timespec="seconds")}
        out = repo_path("artifacts/features/prism2_virchow2.npz")
        save_slide_features(out, wsi_ids, X, np.array(n_tiles), meta)
        print(f"wrote {out}: X {X.shape}, {meta['seconds']}s", flush=True)
    finally:
        gpu_logger.terminate()


if __name__ == "__main__":
    main()
