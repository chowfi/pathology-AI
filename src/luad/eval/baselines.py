"""Baseline rows and their search spaces, shared by run_baselines.py and the nested-CV check."""
from __future__ import annotations


def rows_for(models: list[str]) -> dict[str, list[str]]:
    rows = {"meta": ["meta"]}
    rows |= {m: [m] for m in models}
    rows |= {f"{m}+meta": [m, "meta"] for m in models}
    rows["tile_count"] = ["tile_count"]
    return rows


def space_for(blocks: list[str], cfg: dict) -> dict[str, list]:
    """Embedding rows search the full single-model space; metadata/tile-count rows only C. meta_weight only
    applies when a row has both an embedding and age/sex."""
    embedding = any(b in cfg["models"] for b in blocks)
    space = dict(cfg["space"]) if embedding else {"C": cfg["space"]["C"]}
    if not (embedding and "meta" in blocks):
        space.pop("meta_weight", None)
    return space
