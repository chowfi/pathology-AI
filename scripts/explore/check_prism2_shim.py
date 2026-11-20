"""Verify the flash-attn shim, then smoke-test Prism2 on sample tiles.

  [A] shim vs plain reference attention (explicit softmax, fp32) on random variable-length data with GQA
  [B] Prism2 loads and runs with the shim; base embedding shape, no NaN
  [C] batch invariance: each slide embedded alone == embedded inside a padded batch
  [D] different slides -> different embeddings

Usage: uv run python scripts/explore/check_prism2_shim.py artifacts/sample_tiles
"""
from __future__ import annotations

import glob
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms as T

from luad.embeddings import flash_attn_shim  # moved into the package (used by scripts/extract_prism2.py)

DEV = "cuda"
PRISM2_REV = "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a"  # pinned: the code version that was inspected


def reference(q, k, v, cq, ck):
    """Plain attention per sequence in fp32: softmax(q k^T / sqrt(d)) v, with GQA via repeat_interleave."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    g = q.shape[1] // k.shape[1]
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]].float()
        ki = k[ck[i]:ck[i + 1]].float().repeat_interleave(g, dim=1)
        vi = v[ck[i]:ck[i + 1]].float().repeat_interleave(g, dim=1)
        att = torch.einsum("qhd,khd->hqk", qi, ki) / qi.shape[-1] ** 0.5
        out[cq[i]:cq[i + 1]] = torch.einsum("hqk,khd->qhd", att.softmax(-1), vi)
    return out


def test_shim() -> None:
    torch.manual_seed(0)
    lq, hq, hkv, d = 256, 20, 10, 128          # Prism2: 256 latents, 20 q heads, 10 kv heads, 2560/20 = 128
    lens_k = [37, 1500, 6000]                  # very different tile counts per slide
    cq = torch.arange(0, lq * len(lens_k) + 1, lq, dtype=torch.int32, device=DEV)
    ck = torch.tensor([0] + torch.tensor(lens_k).cumsum(0).tolist(), dtype=torch.int32, device=DEV)
    for dtype in (torch.float32, torch.bfloat16):
        q = torch.randn(int(cq[-1]), hq, d, device=DEV, dtype=dtype)
        k = torch.randn(int(ck[-1]), hkv, d, device=DEV, dtype=dtype)
        v = torch.randn(int(ck[-1]), hkv, d, device=DEV, dtype=dtype)
        ref = reference(q, k, v, cq.tolist(), ck.tolist())
        got = flash_attn_shim.flash_attn_varlen_func(q, k, v, cq, ck, lq, max(lens_k)).float()
        print(f"  [A] {str(dtype):15s} max abs diff vs reference = {(got - ref).abs().max():.2e}  "
              f"(output scale ~{ref.abs().mean():.2e})")


def main() -> None:
    limit = float(subprocess.check_output(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"], text=True))
    if limit > 450:
        raise SystemExit(f"GPU power limit {limit:.0f} W > 450 W. Run: sudo nvidia-smi -pl 450")
    print("[A] shim vs reference attention")
    test_shim()

    # Virchow2 CLS-only tile embeddings (Prism2's required input), grouped by slide
    tiles = defaultdict(list)
    for f in sorted(glob.glob(f"{sys.argv[1]}/*.png")):
        tiles[Path(f).name.split("_")[0]].append(Image.open(f).convert("RGB"))
    v2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked,
                           act_layer=torch.nn.SiLU).eval().to(DEV)
    tf = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    emb = {}
    for s, ims in tiles.items():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            emb[s] = v2(torch.stack([tf(i) for i in ims]).to(DEV))[:, 0].float()   # CLS only, (N, 1280)
    del v2
    torch.cuda.empty_cache()

    flash_attn_shim.install()
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained("paige-ai/Prism2", revision=PRISM2_REV, trust_remote_code=True, torch_dtype="auto").to(DEV).eval()
    proc = AutoProcessor.from_pretrained("paige-ai/Prism2", revision=PRISM2_REV, trust_remote_code=True)
    names = list(emb)
    # make slide lengths differ so padding is exercised
    slides = [emb[n][: 16 - 4 * i] for i, n in enumerate(names)]
    print(f"\n[B] Prism2 loaded; slides {names} with {[len(s) for s in slides]} tiles")

    def base(batch_list):
        b = proc(tile_embeddings=batch_list).to(DEV)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model.get_base_embedding(**b).float()

    batched = base(slides)
    print(f"  shape={tuple(batched.shape)}  NaN={torch.isnan(batched).any().item()}  "
          f"norm={batched.norm(dim=1).tolist()}")
    alone = torch.cat([base([s]) for s in slides])
    c = torch.nn.functional.cosine_similarity(alone, batched, dim=-1)
    print(f"[C] alone vs in padded batch: cosine {[round(x, 5) for x in c.tolist()]}")
    n = torch.nn.functional.normalize(batched, dim=-1)
    print(f"[D] cosine between different slides:\n{(n @ n.T).cpu().numpy().round(3)}")


if __name__ == "__main__":
    main()
