"""End-to-end check of saved embeddings: recompute random tiles with each model card's own recipe
(one image at a time, card transform, card precision) and compare with the .h5 rows.

Usage: uv run python scripts/checks/verify_embeddings.py WSI-55 [--n 16]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms as T

from luad.config import load_config, repo_path
from luad.embeddings.io import h5_path, load_tiles, read_h5
from luad.embeddings.models import load_encoder
from luad.utils.gpu import require_power_limit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wsi_id")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    cfg, tcfg = load_config("embeddings"), load_config("tiling")
    require_power_limit(cfg["run"]["max_power_w"])
    tiles, tiling = load_tiles(args.wsi_id, repo_path(tcfg["paths"]["slides_dir"]),
                               repo_path(tcfg["paths"]["tiles_dir"]))
    idx = np.random.default_rng(0).choice(len(tiles), args.n, replace=False)
    imgs = [Image.fromarray(tiles[i]) for i in idx]
    for name, spec in cfg["models"].items():
        feats, coords, attrs = read_h5(h5_path(repo_path(cfg["paths"]["out_dir"]), name, args.wsi_id))
        assert (coords == tiling["coords"]).all(), "coords differ from tiling"
        enc = load_encoder(name, spec)
        if name == "hoptimus1":   # card: ToTensor + its own Normalize, fp16 autocast
            tf, autocast = T.Compose([T.ToTensor(), T.Normalize(spec["mean"], spec["std"])]), True
        else:                     # card: timm transform from the model's pretrained config
            tf = create_transform(**resolve_data_config(enc.model.pretrained_cfg, model=enc.model))
            autocast = name == "virchow2"
        ref = []
        with torch.inference_mode():
            for im in imgs:
                x = tf(im).unsqueeze(0).cuda()
                with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                    ref.append(enc.pool(enc.model(x)).float().cpu())
        ref = torch.cat(ref)
        got = torch.from_numpy(feats[idx].astype(np.float32))
        c = torch.nn.functional.cosine_similarity(ref, got, dim=1)
        rnd = torch.nn.functional.cosine_similarity(got, got.roll(1, 0), dim=1)
        print(f"{name:10s} h5 {feats.shape} {attrs['revision'][:8]}  card-vs-saved cosine min={c.min():.5f} "
              f"mean={c.mean():.5f}  | different tiles mean={rnd.mean():.3f}")
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
