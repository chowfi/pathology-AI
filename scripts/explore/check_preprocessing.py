"""Check per-model preprocessing against each model card, on real tiles.

Checks per model:
  [1] card-exact pipeline (one image at a time, card transform) vs our batched pipeline -> cosine ~1
  [2] wrong normalization -> cosine clearly < 1 (shows the trap is real)
  [3] flipped tile vs different tiles -> embeddings follow image content
  [4] throughput (tiles/s)
Virchow2 extra: output token layout, CLS-only slice for Prism2, register-token trap.

Power safety: refuses to run if the GPU power limit is above MAX_W; logs power/temp/clock every 0.5 s to
artifacts/logs/gpu_log.csv (flushed each line, so it survives a sudden power loss).

Usage: uv run python scripts/explore/check_preprocessing.py artifacts/sample_tiles [--models uni2 virchow2 hopt1]
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import time
from pathlib import Path

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms as T

MAX_W = 450
BATCH = 64
DEV = "cuda"
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HOPT = ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))
LOG = Path("artifacts/logs/gpu_log.csv")


def gpu_query(fields: str) -> list[str]:
    out = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True)
    return [x.strip() for x in out.strip().split(",")]


def start_gpu_logger() -> subprocess.Popen:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    f = LOG.open("a", buffering=1)
    f.write(f"# run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw,temperature.gpu,clocks.sm,utilization.gpu,memory.used",
         "--format=csv,noheader", "-lms", "500"], stdout=f, stderr=subprocess.STDOUT)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


def ours(model, mean, std, post, imgs) -> torch.Tensor:
    """Our pipeline: tiles already 224 px at 0.5 um/px -> ToTensor + per-model Normalize, batched, fp16 autocast."""
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    outs = []
    for i in range(0, len(imgs), BATCH):
        x = torch.stack([tf(im) for im in imgs[i:i + BATCH]]).to(DEV)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outs.append(post(model(x)).float())
    return torch.cat(outs)


def card(model, tf, post, imgs, autocast: bool) -> torch.Tensor:
    """Card-exact: one image at a time with the card's own transform and precision."""
    outs = []
    for im in imgs:
        x = tf(im).unsqueeze(0).to(DEV)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            outs.append(post(model(x)).float())
    return torch.cat(outs)


def speed(model, post, n: int = 20) -> float:
    x = torch.randn(BATCH, 3, 224, 224, device=DEV)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for bs in (4, 16, BATCH):  # gentle ramp-up instead of an instant jump to full load
            post(model(x[:bs]))
            torch.cuda.synchronize()
            time.sleep(0.5)
        t = time.time()
        for _ in range(n):
            post(model(x))
        torch.cuda.synchronize()
    return BATCH * n / (time.time() - t)


def report(name, model, mean, std, post, card_tf, card_autocast, wrong, tiles) -> torch.Tensor:
    cfg = model.pretrained_cfg
    print(f"\n=== {name}  params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"  timm cfg: mean={cfg.get('mean')} std={cfg.get('std')} input={cfg.get('input_size')}")
    print(f"  card transform steps: {[type(t).__name__ for t in card_tf.transforms]}")
    e_card = card(model, card_tf, post, tiles, card_autocast)
    e = ours(model, mean, std, post, tiles)
    c = cos(e_card, e)
    norms = e.norm(dim=1)
    print(f"  dim={e.shape[1]}  NaN={torch.isnan(e).any().item()}  norm mean={norms.mean():.2f} sd={norms.std():.2f}")
    print(f"  [1] card vs ours cosine: min={c.min():.5f} mean={c.mean():.5f}")
    cw = cos(e, ours(model, *wrong, post, tiles))
    print(f"  [2] wrong normalization vs correct: cosine mean={cw.mean():.3f} min={cw.min():.3f}")
    flip = ours(model, mean, std, post, [im.transpose(Image.FLIP_LEFT_RIGHT) for im in tiles])
    n = len(tiles)
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    print(f"  [3] same tile flipped cosine={cos(e, flip).mean():.3f}  vs different tiles={cos(e[:, None], e[None])[off].mean():.3f}")
    print(f"  [4] speed fp16 batch {BATCH}: {speed(model, post):.0f} tiles/s")
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles")
    ap.add_argument("--models", nargs="+", default=["uni2", "virchow2", "hopt1"])
    args = ap.parse_args()

    limit = float(gpu_query("power.limit")[0])
    if limit > MAX_W:
        raise SystemExit(f"GPU power limit is {limit:.0f} W (> {MAX_W} W). Run: sudo nvidia-smi -pl {MAX_W}")
    tiles = [Image.open(f).convert("RGB") for f in sorted(glob.glob(f"{args.tiles}/*.png"))]
    print(f"{len(tiles)} tiles; GPU power limit {limit:.0f} W; logging to {LOG}")
    logger = start_gpu_logger()
    try:
        if "uni2" in args.models:
            kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                      mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                      act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
            m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw).eval().to(DEV)
            tf = create_transform(**resolve_data_config(m.pretrained_cfg, model=m))
            report("UNI2-h", m, *IMNET, lambda o: o, tf, False, HOPT, tiles)
            del m
            torch.cuda.empty_cache()
        if "virchow2" in args.models:
            m = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked,
                                  act_layer=torch.nn.SiLU).eval().to(DEV)
            with torch.inference_mode():
                shape = tuple(m(torch.randn(1, 3, 224, 224, device=DEV)).shape)
            print(f"\nVirchow2 raw output {shape} (card: 1 x 261 x 1280 = CLS + 4 register + 256 patch)")
            v2 = lambda o: torch.cat([o[:, 0], o[:, 5:].mean(1)], dim=-1)  # noqa: E731
            tf = create_transform(**resolve_data_config(m.pretrained_cfg, model=m))
            e = report("Virchow2", m, *IMNET, v2, tf, True, HOPT, tiles)
            cls = ours(m, *IMNET, lambda o: o[:, 0], tiles)
            print(f"  [5] Prism2 input (CLS only) == first 1280 of our 2560 vector? max diff={(e[:, :1280] - cls).abs().max():.2e}")
            bad = ours(m, *IMNET, lambda o: torch.cat([o[:, 0], o[:, 1:].mean(1)], dim=-1), tiles)
            print(f"  [6] trap: register tokens in the mean -> cosine vs correct {cos(e, bad).mean():.4f}")
            del m
            torch.cuda.empty_cache()
        if "hopt1" in args.models:
            m = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5,
                                  dynamic_img_size=False).eval().to(DEV)
            tf = T.Compose([T.ToTensor(), T.Normalize(*HOPT)])
            report("H-optimus-1", m, *HOPT, lambda o: o, tf, True, IMNET, tiles)
    finally:
        logger.terminate()
    print(f"\nGPU log: {LOG}")


if __name__ == "__main__":
    main()
