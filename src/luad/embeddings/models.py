"""Load the tile encoders from conf/embeddings.yaml and turn their outputs into one vector per tile.

Each model gets its own normalisation and pooling. The settings were checked against the model cards
(scripts/explore/check_preprocessing.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

LAYERS = {"SwiGLUPacked": ("timm.layers", "SwiGLUPacked"), "SiLU": ("torch.nn", "SiLU")}


def _resolve(kwargs: dict[str, Any]) -> dict[str, Any]:
    """YAML strings like 'SwiGLUPacked' -> the class timm expects."""
    import importlib
    out = dict(kwargs)
    for k in ("mlp_layer", "act_layer"):
        if isinstance(out.get(k), str):
            mod, name = LAYERS[out[k]]
            out[k] = getattr(importlib.import_module(mod), name)
    return out


def pool_fn(spec: dict) -> Callable[[torch.Tensor], torch.Tensor]:
    """Model output -> one vector per tile."""
    if spec["pooling"] == "model_output":
        return lambda o: o
    if spec["pooling"] == "cls_plus_mean_patch":
        k = spec["n_prefix_tokens"]
        return lambda o: torch.cat([o[:, 0], o[:, k:].mean(1)], dim=-1)
    raise ValueError(f"unknown pooling {spec['pooling']}")


@dataclass
class Encoder:
    name: str
    spec: dict
    model: torch.nn.Module
    mean: torch.Tensor          # (1, 3, 1, 1) on device
    std: torch.Tensor
    pool: Callable[[torch.Tensor], torch.Tensor]

    @torch.inference_mode()
    def embed(self, x01: torch.Tensor, autocast: bool = True) -> torch.Tensor:
        """x01: (B, 3, 224, 224) float in [0, 1] on the model's device -> (B, dim) float16."""
        x = (x01 - self.mean) / self.std
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=autocast and x.device.type == "cuda"):
            return self.pool(self.model(x)).to(torch.float16)


def load_encoder(name: str, spec: dict, device: str = "cuda") -> Encoder:
    import timm
    model = timm.create_model(f"hf-hub:{spec['hf_id']}@{spec['revision']}", pretrained=True,
                              **_resolve(spec["timm_kwargs"]))
    model = model.eval().to(device)
    mean = torch.tensor(spec["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec["std"], device=device).view(1, 3, 1, 1)
    return Encoder(name, spec, model, mean, std, pool_fn(spec))
