"""Fusion strategies registered by name. A config dict with a 'method' key -> a model with fit/predict_proba.

Every method ends in a linear softmax classifier (multinomial logistic regression), so the classifier family is
the same in every row of the comparison table; only the way the three models are combined changes.
"""
from __future__ import annotations

from luad.fusion.early import EarlyFusionLR, EarlyTunedLR
from luad.fusion.gated import GatedFusion
from luad.fusion.late import LateFusion


def _early(config, clf_cfg, seed):
    blocks = list(config["models"]) + (["meta"] if config.get("use_meta", True) else [])
    return EarlyFusionLR({**config, "blocks": blocks}, clf_cfg, seed)


REGISTRY = {
    "early": _early,
    "late": lambda c, clf, s: LateFusion(c, clf, s),
    "gated": lambda c, clf, s: GatedFusion(c, clf, s),
    # v4: members = each model's own tuned single-model + metadata baseline config
    "late_tuned": lambda c, clf, s: LateFusion(c, clf, s),
    "early_tuned": lambda c, clf, s: EarlyTunedLR(c, clf, s),
}


def build_model(config: dict, clf_cfg: dict, seed: int = 0):
    if config["method"] not in REGISTRY:
        raise ValueError(f"unknown fusion method {config['method']!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[config["method"]](config, clf_cfg, seed)
