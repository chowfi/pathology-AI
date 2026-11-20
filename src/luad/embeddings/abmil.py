"""Gated-attention MIL pooling (Ilse et al., 2018) as a supervised tile -> slide aggregation (round v5).

The network learns WHERE to look: attention weights over the tiles of a slide. The exported slide vector is the
attention-weighted mean of the raw tile embeddings, sum_i a_i h_i, so it stays in the foundation model's own
space (like the mean-pooled vector) and goes into the same logistic-regression pipeline as every other row.

ABMIL learns from labels, so a slide vector must never come from a network trained on that slide's label.
`crossfit` makes this impossible by construction: training patients get out-of-fold vectors (from the network
trained without their part), patients to score get the mean over all part networks.

Settings are fixed in artifacts/results/v5/PLAN.md (not tuned on this data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn

N_CLASSES = 7


class TileBank:
    """All tile embeddings of a set of slides in one tensor (on the GPU when available) + per-slide offsets."""

    def __init__(self, feats: torch.Tensor, offsets: np.ndarray):
        self.feats, self.offsets = feats, offsets
        self.device, self.dim = feats.device, feats.shape[1]

    @classmethod
    def from_arrays(cls, arrays: list[np.ndarray], device: str = "cpu") -> "TileBank":
        offsets = np.concatenate([[0], np.cumsum([len(a) for a in arrays])])
        return cls(torch.from_numpy(np.concatenate(arrays).astype(np.float32)).to(device), offsets)

    @classmethod
    def from_h5(cls, model: str, wsi_ids: list[str], emb_dir: Path, device: str = "cuda") -> "TileBank":
        import h5py

        from luad.embeddings.io import h5_path
        paths = [h5_path(emb_dir, model, w) for w in wsi_ids]
        sizes = []
        for p in paths:
            with h5py.File(p, "r") as f:
                sizes.append(f["features"].shape)
        offsets = np.concatenate([[0], np.cumsum([s[0] for s in sizes])])
        feats = torch.empty((int(offsets[-1]), sizes[0][1]), dtype=torch.float16, device=device)
        for i, p in enumerate(paths):
            with h5py.File(p, "r") as f:
                feats[offsets[i]:offsets[i + 1]] = torch.from_numpy(f["features"][:]).to(device)
        return cls(feats, offsets)

    def slide(self, i: int) -> torch.Tensor:
        return self.feats[self.offsets[i]:self.offsets[i + 1]]

    def n_tiles(self, idx) -> np.ndarray:
        return np.diff(self.offsets)[idx]


class GatedABMIL(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float, n_classes: int = N_CLASSES):
        super().__init__()
        self.drop, self.head_drop = nn.Dropout(dropout), nn.Dropout(dropout)
        self.V, self.U, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)
        self.head = nn.Linear(dim, n_classes)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        s = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
        return torch.softmax(s, dim=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.drop(x)
        a = self.attention(x)
        return self.head(self.head_drop(a @ x)), a


def _tile_stats(bank: TileBank, idx) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean / SD over all tiles of the slides `idx` (float64 sums: no overflow)."""
    s = torch.zeros(bank.dim, dtype=torch.float64, device=bank.device)
    s2, n = torch.zeros_like(s), 0
    for i in idx:
        x = bank.slide(i).double()
        s += x.sum(0)
        s2 += (x * x).sum(0)
        n += len(x)
    mu = s / n
    sd = (s2 / n - mu * mu).clamp_min(1e-12).sqrt()
    return mu.float(), sd.float()


def train_abmil(bank: TileBank, idx: np.ndarray, y: np.ndarray, cfg: dict, seed: int):
    """Train one network on the slides `idx` (labels y). Returns (network in eval mode, standardisation stats)."""
    y = np.asarray(y)
    counts = np.bincount(y, minlength=N_CLASSES)
    if (counts == 0).any():
        raise ValueError(f"ABMIL training set misses classes {np.flatnonzero(counts == 0).tolist()}")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = bank.device
    mu, sd = _tile_stats(bank, idx)
    net = GatedABMIL(bank.dim, cfg["hidden"], cfg["dropout"]).to(dev)
    w = torch.tensor(len(y) / (N_CLASSES * counts), dtype=torch.float32, device=dev)   # class-balanced
    yt = torch.as_tensor(y, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    net.train()
    losses = []
    for _ in range(cfg["epochs"]):
        tot = torch.zeros((), device=dev)
        for j in torch.randperm(len(idx), generator=gen).tolist():
            x = bank.slide(idx[j])
            if len(x) > cfg["max_tiles"]:
                x = x[torch.randperm(len(x), generator=gen)[:cfg["max_tiles"]].to(dev)]
            logits, _ = net((x.float() - mu) / sd)
            # weight per sample by hand: with one slide per step, `weight=` + mean reduction would cancel out
            loss = F.cross_entropy(logits[None], yt[j:j + 1]) * w[yt[j]]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach()
        losses.append(float(tot) / len(idx))
    net.eval()
    net.train_losses = losses
    return net, (mu, sd)


@torch.no_grad()
def pool(bank: TileBank, net: GatedABMIL, stats, idx, return_attention: bool = False):
    """Attention-weighted mean of the RAW tile embeddings for each slide in `idx`: (len(idx), dim) float32."""
    mu, sd = stats
    Z, att = np.zeros((len(idx), bank.dim), np.float32), []
    for j, i in enumerate(idx):
        x = bank.slide(i).float()
        a = net.attention((x - mu) / sd)
        Z[j] = (a @ x).cpu().numpy()
        if return_attention:
            att.append(a.cpu().numpy())
    return (Z, att) if return_attention else Z


# ---- v7: tile-level fusion -------------------------------------------------------------------------------------
class MultiViewBank:
    """Several tile banks of the SAME tiles, one per foundation model (all views share the per-slide offsets)."""

    def __init__(self, views: list[TileBank]):
        if any(not np.array_equal(v.offsets, views[0].offsets) for v in views):
            raise ValueError("views differ in the number of tiles per slide")
        self.views, self.offsets = views, views[0].offsets
        self.dims = [v.dim for v in views]
        self.dim, self.device = sum(self.dims), views[0].device

    @classmethod
    def from_h5(cls, models: list[str], wsi_ids: list[str], emb_dir: Path, device: str = "cuda") -> "MultiViewBank":
        """Loads each model's tiles; refuses slides whose tile coordinates differ between the models."""
        import h5py

        from luad.embeddings.io import h5_path
        for w in wsi_ids:
            coords = []
            for m in models:
                with h5py.File(h5_path(emb_dir, m, w), "r") as f:
                    coords.append(f["coords"][:])
            if any(not np.array_equal(c, coords[0]) for c in coords[1:]):
                raise ValueError(f"{w}: the models embedded different tiles")
        return cls([TileBank.from_h5(m, wsi_ids, emb_dir, device) for m in models])

    def slide(self, i: int) -> list[torch.Tensor]:
        return [v.slide(i) for v in self.views]

    def n_tiles(self, idx) -> np.ndarray:
        return np.diff(self.offsets)[idx]


class JointABMIL(nn.Module):
    """Per view: dropout -> Linear(d -> proj) -> ReLU; concatenated tile vector t_i; ONE gated attention a_i over
    the tiles (shared by all views); linear head on sum_i a_i t_i."""

    def __init__(self, dims: list[int], proj: int, hidden: int, dropout: float, n_classes: int = N_CLASSES):
        super().__init__()
        self.proj = nn.ModuleList(nn.Sequential(nn.Dropout(dropout), nn.Linear(d, proj), nn.ReLU()) for d in dims)
        width = proj * len(dims)
        self.V, self.U, self.w = nn.Linear(width, hidden), nn.Linear(width, hidden), nn.Linear(hidden, 1)
        self.head, self.head_drop = nn.Linear(width, n_classes), nn.Dropout(dropout)

    def tiles(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([p(x) for p, x in zip(self.proj, xs)], dim=1)

    def _attend(self, t: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.w(torch.tanh(self.V(t)) * torch.sigmoid(self.U(t))).squeeze(-1), dim=0)

    def attention(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return self._attend(self.tiles(xs))

    def forward(self, xs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.tiles(xs)
        a = self._attend(t)
        return self.head(self.head_drop(a @ t)), a


def train_joint_abmil(bank: MultiViewBank, idx: np.ndarray, y: np.ndarray, cfg: dict, seed: int):
    """train_abmil for several views of the same tiles: the same random tiles are used in every view."""
    y = np.asarray(y)
    counts = np.bincount(y, minlength=N_CLASSES)
    if (counts == 0).any():
        raise ValueError(f"ABMIL training set misses classes {np.flatnonzero(counts == 0).tolist()}")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = bank.device
    stats = [_tile_stats(v, idx) for v in bank.views]
    net = JointABMIL(bank.dims, cfg.get("proj", 128), cfg["hidden"], cfg["dropout"]).to(dev)
    w = torch.tensor(len(y) / (N_CLASSES * counts), dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    net.train()
    losses = []
    for _ in range(cfg["epochs"]):
        tot = torch.zeros((), device=dev)
        for j in torch.randperm(len(idx), generator=gen).tolist():
            xs = bank.slide(idx[j])
            if len(xs[0]) > cfg["max_tiles"]:
                sel = torch.randperm(len(xs[0]), generator=gen)[:cfg["max_tiles"]].to(dev)
                xs = [x[sel] for x in xs]
            logits, _ = net([(x.float() - mu) / sd for x, (mu, sd) in zip(xs, stats)])
            loss = F.cross_entropy(logits[None], yt[j:j + 1]) * w[yt[j]]     # per-sample class weight
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach()
        losses.append(float(tot) / len(idx))
    net.eval()
    net.train_losses = losses
    return net, stats


@torch.no_grad()
def pool_joint(bank: MultiViewBank, net: JointABMIL, stats, idx, return_attention: bool = False):
    """For each slide, every view's RAW tile embeddings pooled with the ONE shared attention, concatenated:
    (len(idx), sum(dims)) float32. split_views() cuts it back into one vector per model."""
    Z, att = np.zeros((len(idx), bank.dim), np.float32), []
    for j, i in enumerate(idx):
        xs = [x.float() for x in bank.slide(i)]
        a = net.attention([(x - mu) / sd for x, (mu, sd) in zip(xs, stats)])
        Z[j] = torch.cat([a @ x for x in xs]).cpu().numpy()
        if return_attention:
            att.append(a.cpu().numpy())
    return (Z, att) if return_attention else Z


def split_views(Z: np.ndarray, dims: list[int]) -> list[np.ndarray]:
    """Cut concatenated pooled vectors (..., sum(dims)) into one array per view."""
    return np.split(Z, np.cumsum(dims)[:-1], axis=-1)


def crossfit(bank, train_idx: np.ndarray, apply_idx: np.ndarray, y_train: np.ndarray, cfg: dict,
             seed: int, parts: list | None = None, log: list | None = None, train_fn=None,
             pool_fn=None) -> tuple[np.ndarray, np.ndarray]:
    """(out-of-fold vectors for train_idx, mean-over-part-networks vectors for apply_idx).

    The vector of a training patient comes only from the network trained WITHOUT its part, so its own label is
    never used for it. Labels of apply_idx are not even an input. train_fn / pool_fn default to the per-model
    ABMIL (v5); pass train_joint_abmil / pool_joint with a MultiViewBank for tile-level fusion (v7).
    """
    z_tr, nets = fit_parts(bank, train_idx, y_train, cfg, seed, parts=parts, log=log, train_fn=train_fn,
                           pool_fn=pool_fn)
    return z_tr, apply_parts(bank, nets, apply_idx, pool_fn=pool_fn)


def fit_parts(bank, train_idx: np.ndarray, y_train: np.ndarray, cfg: dict, seed: int, parts: list | None = None,
              log: list | None = None, train_fn=None, pool_fn=None) -> tuple[np.ndarray, list]:
    """Training half of crossfit: one network per part of train_idx.

    Returns (out-of-fold vectors for train_idx, [(network, stats)] for apply_parts). Split from crossfit so the
    final test script can train on a bank of the dev slides only and pool the test slides from a second bank
    (all 209 slides of the joint bank do not fit on the GPU at once). Pooling is deterministic and draws no
    random numbers, so the split gives the same vectors as crossfit.
    """
    train_fn, pool_fn = train_fn or train_abmil, pool_fn or pool
    train_idx, y_train = np.asarray(train_idx), np.asarray(y_train)
    if parts is None:
        parts = list(StratifiedKFold(cfg.get("n_parts", 5), shuffle=True, random_state=seed)
                     .split(train_idx, y_train))
    z_tr, nets = np.zeros((len(train_idx), bank.dim), np.float32), []
    for p, (a, b) in enumerate(parts):
        net, st = train_fn(bank, train_idx[a], y_train[a], cfg, seed + p)
        z_tr[b] = pool_fn(bank, net, st, train_idx[b])
        nets.append((net, st))
        if log is not None:
            log.append({"part": p, "seed": seed + p, "n_train": len(a), "loss_first": net.train_losses[0],
                        "loss_last": net.train_losses[-1]})
    return z_tr, nets


def apply_parts(bank, nets: list, idx: np.ndarray, pool_fn=None) -> np.ndarray:
    """Apply half of crossfit: mean over the part networks of the pooled vectors of the slides `idx` of `bank`
    (which may be a different bank from the training one). Labels are not an input."""
    pool_fn, idx = pool_fn or pool, np.asarray(idx)
    z = np.zeros((len(idx), bank.dim), np.float32)
    if len(idx):
        for net, st in nets:
            z += pool_fn(bank, net, st, idx) / len(nets)
    return z


def fold_features(bank, bank_idx: np.ndarray, y: np.ndarray, folds: list, cfg: dict, seed: int,
                  log: list | None = None, **fit_fns) -> np.ndarray:
    """(n_folds, n, dim): for fold k, training rows = out-of-fold vectors, validation rows = part-network mean.

    bank_idx maps the dataset's patient order to slides in the bank; folds use dataset (local) indices.
    fit_fns: optional train_fn / pool_fn, passed to crossfit (v7 tile-level fusion).
    """
    bank_idx = np.asarray(bank_idx)
    Z = np.full((len(folds), len(y), bank.dim), np.nan, np.float32)
    for k, (tr, va) in enumerate(folds):
        fl = [] if log is not None else None
        Z[k, tr], Z[k, va] = crossfit(bank, bank_idx[tr], bank_idx[va], y[tr], cfg, seed + 1000 * k, log=fl,
                                      **fit_fns)
        if log is not None:
            log.extend({"fold": k, **r} for r in fl)
    return Z
