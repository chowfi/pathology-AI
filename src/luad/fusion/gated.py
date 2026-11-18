"""Gated fusion (learned, intermediate): the network decides per patient how much to trust each model.

  each model:  standardise (+ PCA) -> dropout -> Linear(d_m, h) -> LayerNorm      = z_m      (fitted per fold)
  gate:        softmax(Linear(concat z_1..z_M))                                  = g (M weights, sum 1)
  fused:       sum_m g_m z_m
  head:        Linear(fused + age/sex, 7) -> softmax     (the same linear-softmax family as every other row)

Trained full-batch on the training fold with class-weighted cross-entropy, AdamW and a FIXED number of
epochs (no early stopping, because there is no separate data inside the fold to stop on). The predictions are
averaged over `n_seeds` networks to reduce seed noise. CPU, one thread, deterministic.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from luad.data.dataset import Dataset
from luad.fusion.early import BlockTransform


class GatedNet(nn.Module):
    def __init__(self, dims: list[int], hidden: int, n_meta: int, n_classes: int, dropout: float):
        super().__init__()
        self.proj = nn.ModuleList(nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.LayerNorm(hidden))
                                  for d in dims)
        self.gate = nn.Linear(hidden * len(dims), len(dims))
        self.head = nn.Linear(hidden + n_meta, n_classes)

    def forward(self, xs: list[torch.Tensor], meta: torch.Tensor | None):
        z = torch.stack([p(x) for p, x in zip(self.proj, xs)], 1)          # (B, M, h)
        g = torch.softmax(self.gate(z.flatten(1)), -1)                     # (B, M)
        f = (g.unsqueeze(-1) * z).sum(1)
        if meta is not None:
            f = torch.cat([f, meta], 1)
        return self.head(f), g


class GatedFusion:
    """config: models, use_meta, l2_normalize, pca, meta_weight, hidden, dropout, weight_decay, lr, epochs, n_seeds."""

    def __init__(self, config: dict, clf_cfg: dict | None = None, seed: int = 0):
        self.config, self.seed = config, seed

    def _inputs(self, ds: Dataset, idx: np.ndarray, fit: bool):
        xs = []
        for m in self.config["models"]:
            X = ds.get(m)[idx]
            xs.append(torch.tensor(self.tf[m].fit_transform(X) if fit else self.tf[m].transform(X),
                                   dtype=torch.float32))
        meta = None
        if self.config.get("use_meta", True):
            M = self.tf["meta"].fit_transform(ds.meta[idx]) if fit else self.tf["meta"].transform(ds.meta[idx])
            meta = torch.tensor(M * self.config.get("meta_weight", 1.0), dtype=torch.float32)
        return xs, meta

    def fit(self, ds: Dataset, idx: np.ndarray) -> "GatedFusion":
        c = self.config
        torch.set_num_threads(1)
        self.tf = {m: BlockTransform(c.get("l2_normalize", False), c.get("pca"), self.seed) for m in c["models"]}
        self.tf["meta"] = BlockTransform()
        xs, meta = self._inputs(ds, idx, fit=True)
        y = torch.tensor(ds.y[idx])
        counts = torch.bincount(y, minlength=7).float()
        if (counts == 0).any():
            raise ValueError("a class is missing from the training fold")
        cw = counts.sum() / (7 * counts)                                    # same as class_weight="balanced"
        self.nets = []
        for s in range(c.get("n_seeds", 3)):
            torch.manual_seed(self.seed * 1000 + s)
            net = GatedNet([x.shape[1] for x in xs], c.get("hidden", 64), 0 if meta is None else meta.shape[1],
                           7, c.get("dropout", 0.25))
            opt = torch.optim.AdamW(net.parameters(), lr=c.get("lr", 1e-3), weight_decay=c.get("weight_decay", 1e-2))
            loss_fn = nn.CrossEntropyLoss(weight=cw)
            net.train()
            for _ in range(c.get("epochs", 200)):
                opt.zero_grad()
                loss = loss_fn(net(xs, meta)[0], y)
                loss.backward()
                opt.step()
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def predict_proba(self, ds: Dataset, idx: np.ndarray, return_gates: bool = False):
        xs, meta = self._inputs(ds, idx, fit=False)
        outs = [net(xs, meta) for net in self.nets]
        p = torch.stack([torch.softmax(o[0], -1) for o in outs]).mean(0).numpy()
        if return_gates:
            return p, torch.stack([o[1] for o in outs]).mean(0).numpy()
        return p
