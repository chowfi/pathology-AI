"""The fusion search space (conf/agent.yaml): validation, canonical form, sampling, JSON schema for the LLM."""
from __future__ import annotations

import json
import math

import numpy as np


class FusionSpace:
    def __init__(self, space_cfg: dict):
        self.common = space_cfg["common"]
        self.methods = {m: v for m, v in space_cfg.items() if m != "common"}

    def knobs(self, method: str, combine: str | None = None) -> dict[str, list]:
        """The knobs that matter for this method (stack_C only for combine=stack)."""
        k = {**self.common, **self.methods[method]}
        if method == "late" and combine != "stack":
            k.pop("stack_C", None)
        if method == "gated":
            k.pop("C", None)
        return k

    @staticmethod
    def _match(value, allowed: list):
        """Return the allowed value equal to `value` (floats compared with a relative tolerance)."""
        for a in allowed:
            if a is None or value is None or isinstance(a, (bool, str)) or isinstance(value, (bool, str)):
                if a == value and type(a) is type(value):
                    return a
            elif math.isclose(float(a), float(value), rel_tol=1e-9):
                return a
        raise ValueError(f"value {value!r} not in {allowed}")

    def canonical(self, config: dict) -> dict:
        """Validate and keep only the knobs the method uses, in a fixed order. Raises ValueError."""
        method = config.get("method")
        if method not in self.methods:
            raise ValueError(f"unknown method {method!r}; choose from {list(self.methods)}")
        knobs = self.knobs(method, config.get("combine"))
        out = {"method": method}
        for k, allowed in knobs.items():
            if k not in config:
                raise ValueError(f"missing knob {k!r} for method {method}")
            try:
                out[k] = self._match(config[k], allowed)
            except ValueError as e:
                raise ValueError(f"{k}: {e}") from None
        return out

    @staticmethod
    def key(config: dict) -> str:
        return json.dumps(config, sort_keys=True)

    def size(self, method: str) -> int:
        if method == "late":
            k = self.knobs("late", "mean_prob")
            n_other = math.prod(len(v) for kk, v in k.items() if kk != "combine")
            n_comb = len(self.methods["late"]["combine"]) - 1 + len(self.methods["late"]["stack_C"])
            return n_other * n_comb
        return math.prod(len(v) for v in self.knobs(method).values())

    def sample(self, method: str, rng: np.random.Generator) -> dict:
        cfg = {"method": method}
        for k, v in self.knobs(method, "stack").items():
            cfg[k] = v[rng.integers(len(v))]
        return self.canonical(cfg)

    def mutate(self, config: dict, rng: np.random.Generator) -> dict:
        """Change one knob of `config` to a different allowed value."""
        knobs = {k: v for k, v in self.knobs(config["method"], "stack").items() if len(v) > 1}
        k = list(knobs)[rng.integers(len(knobs))]
        choices = [v for v in knobs[k] if v != config.get(k)]
        new = {**config, k: choices[rng.integers(len(choices))]}
        if new["method"] == "late" and "stack_C" not in new:
            new["stack_C"] = self.methods["late"]["stack_C"][0]
        return self.canonical(new)

    def json_schema(self) -> dict:
        """Structured-output schema for one proposal. Knobs a method does not use must be null."""
        props = {"analysis": {"type": "string", "description": "What the trials so far show (2-4 sentences)."},
                 "reasoning": {"type": "string", "description": "Why this configuration is the best next trial."},
                 "expected_effect": {"type": "string", "description": "What you expect to see, and what you will "
                                                                      "conclude if it does not happen."},
                 "method": {"type": "string", "enum": list(self.methods)}}
        all_knobs: dict[str, list] = dict(self.common)
        for v in self.methods.values():
            for k, vals in v.items():
                all_knobs.setdefault(k, vals)
        for k, vals in all_knobs.items():
            props[k] = {"enum": list(vals) + ([None] if None not in vals else [])}
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    def describe(self) -> str:
        return json.dumps({"common (all methods)": self.common, **self.methods}, indent=1)
