"""Deterministic proposer: a UCB bandit over fusion methods, then a local or random config inside the method.

It reads the same trial history as the LLM proposer, so it is also a feedback loop:
  1. every method is tried once first, in the order of conf/agent.yaml
  2. then the method with the highest UCB score is chosen:
       mean scaled reward of its trials + ucb_c * sqrt(2 ln N / n_method)
     (rewards = CV macro AUROC, min-max scaled over all trials so far)
  3. inside the method: half the time change one knob of that method's best trial (local refinement),
     otherwise draw an untried config at random (exploration). Seeded, so a rerun gives the same trials.
"""
from __future__ import annotations

import numpy as np

from luad.agent.search import Proposal
from luad.agent.space import FusionSpace


class BanditProposer:
    name = "bandit"

    def __init__(self, space: FusionSpace, seed: int, ucb_c: float = 1.0):
        self.space, self.rng, self.c = space, np.random.default_rng(seed), ucb_c

    def _untried(self, draw, tried: set[str], tries: int = 500) -> dict | None:
        for _ in range(tries):
            cfg = draw()
            if self.space.key(cfg) not in tried:
                return cfg
        return None

    def propose(self, history: list[dict], ctx: dict | None = None) -> Proposal | None:
        tried = {self.space.key(h["config"]) for h in history}
        methods = [m for m in self.space.methods
                   if sum(h["config"].get("method") == m for h in history) < self.space.size(m)]
        if not methods:
            return None
        by_m = {m: [h for h in history if h["config"].get("method") == m] for m in methods}
        new = [m for m in methods if not by_m[m]]
        if new:
            m, why = new[0], f"method '{new[0]}' not tried yet; every method is tried once first"
        else:
            scores = np.array([h["macro_auroc_mean"] for h in history])
            lo, hi = scores.min(), scores.max()
            scale = (lambda x: (x - lo) / (hi - lo)) if hi > lo else (lambda x: 0.5)
            n = len(history)
            ucb = {m: np.mean([scale(h["macro_auroc_mean"]) for h in by_m[m]])
                   + self.c * np.sqrt(2 * np.log(n) / len(by_m[m])) for m in methods}
            m = max(ucb, key=ucb.get)
            why = "UCB scores " + ", ".join(f"{k} {v:.2f}" for k, v in ucb.items()) + f" -> '{m}'"
        best = max(by_m[m], key=lambda h: h["macro_auroc_mean"]) if by_m[m] else None
        cfg = None
        if best is not None and self.rng.random() < 0.5:
            cfg = self._untried(lambda: self.space.mutate(best["config"], self.rng), tried, tries=50)
            if cfg is not None:
                changed = [k for k in cfg if cfg[k] != best["config"].get(k)]
                why += f"; local step: change {changed} of trial {best['trial']} (best '{m}' so far)"
        if cfg is None:
            cfg = self._untried(lambda: self.space.sample(m, self.rng), tried)
            why += "; random untried config in this method"
        if cfg is None:
            return None
        return Proposal(cfg, why, self.name)
