"""Agent: search space, bandit fallback and LLM proposer (fake client: no key, no network, no cost)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest

from luad.agent.bandit import BanditProposer
from luad.agent.llm import LLMProposer
from luad.agent.search import run_search
from luad.agent.space import FusionSpace
from luad.config import load_config
from luad.fusion.registry import build_model

from helpers import CLF, _toy

ACFG = load_config("agent")
SPACE = FusionSpace(ACFG["space"])
SEARCH = {"budget": 10, "patience": 5, "accept_k": 1.0}
CTX = {"budget": 10, "patience": 5, "trials_left": 10, "streak": 0, "best_trial": None, "accept_k": 1.0}
NULLS = {k: None for k in SPACE.json_schema()["properties"] if k not in ("analysis", "reasoning", "expected_effect")}


def test_canonical_keeps_only_used_knobs_and_rejects_bad_values():
    c = SPACE.canonical({**NULLS, "method": "late", "combine": "mean_prob", "C": 0.1, "l2_normalize": True,
                         "pca": 64, "meta_weight": 3.0, "stack_C": 0.01, "hidden": 64})
    assert c == {"method": "late", "l2_normalize": True, "pca": 64, "meta_weight": 3.0, "C": 0.1,
                 "combine": "mean_prob"}
    s = SPACE.canonical({**c, "combine": "stack", "stack_C": 0.01})
    assert s["stack_C"] == 0.01
    with pytest.raises(ValueError):
        SPACE.canonical({**c, "C": 0.5})                    # not in the grid
    with pytest.raises(ValueError):
        SPACE.canonical({**c, "method": "magic"})
    with pytest.raises(ValueError):
        SPACE.canonical({**c, "l2_normalize": 1})           # int is not a bool


def test_tuned_methods_have_only_their_own_knobs():
    lt = SPACE.canonical({**NULLS, "method": "late_tuned", "combine": "learned_weights", "C": 0.1, "pca": 64,
                          "stack_C": 0.01})
    assert lt == {"method": "late_tuned", "combine": "learned_weights"}          # members fix C, L2, PCA
    assert SPACE.canonical({**lt, "combine": "stack", "stack_C": 0.01})["stack_C"] == 0.01
    et = SPACE.canonical({**NULLS, "method": "early_tuned", "C": 0.001, "meta_weight": 3.0, "pca": 64})
    assert et == {"method": "early_tuned", "C": 0.001, "meta_weight": 3.0}
    assert SPACE.size("late_tuned") == 4 + 4 and SPACE.size("early_tuned") == 7 * 3


def test_bandit_mutation_of_a_late_tuned_trial_stays_valid():
    rng = np.random.default_rng(0)
    c = SPACE.canonical({"method": "late_tuned", "combine": "mean_prob"})
    for _ in range(20):
        m = SPACE.mutate(c, rng)
        assert SPACE.canonical(m) == m and m != c


def test_schema_is_strict():
    sch = SPACE.json_schema()
    assert sch["additionalProperties"] is False and set(sch["required"]) == set(sch["properties"])
    assert None in sch["properties"]["C"]["enum"]


def _fake_history(cfgs, scores):
    return [{"trial": i + 1, "config": c, "macro_auroc_mean": s, "macro_auroc_sd": 0.04,
             "balanced_accuracy_mean": 0.4, "per_class_auroc_mean": [0.8] * 7, "decision": "reject",
             "gain": 0.0, "threshold": 0.01} for i, (c, s) in enumerate(zip(cfgs, scores))]


def test_bandit_tries_every_method_then_is_deterministic_and_never_repeats():
    b1, b2 = BanditProposer(SPACE, 0), BanditProposer(SPACE, 0)
    hist, seen = [], set()
    for t in range(12):
        p1, p2 = b1.propose(hist, CTX), b2.propose(hist, CTX)
        assert p1.config == p2.config                       # same seed + same history = same proposal
        if t < len(SPACE.methods):
            assert p1.config["method"] == list(SPACE.methods)[t]
        k = SPACE.key(p1.config)
        assert k not in seen and SPACE.canonical(p1.config) == p1.config
        seen.add(k)
        hist = _fake_history([*(h["config"] for h in hist), p1.config], [0.8] * (t + 1))


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        self.last = kw
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(model="claude-opus-5-5", stop_reason="end_turn", usage=usage,
                               content=[SimpleNamespace(type="text", text=json.dumps(a))])


def _answer(**cfg):
    return {**NULLS, "analysis": "a", "reasoning": "r", "expected_effect": "e", **cfg}


EARLY = dict(method="early", C=0.1, l2_normalize=True, pca=16, meta_weight=3.0, balance_blocks=True)


def _llm(answers, **llm_over):
    cl = FakeClient(answers)
    p = LLMProposer(SPACE, {**ACFG["llm"], **llm_over}, BanditProposer(SPACE, 0), "- base: AUROC 0.80", SEARCH,
                    client=cl)
    return p, cl


def test_llm_valid_proposal_is_used_with_reasoning_and_cost():
    p, cl = _llm([_answer(**EARLY)])
    prop = p.propose([], CTX)
    assert prop.proposer == "llm" and prop.config["method"] == "early" and "REASONING: r" in prop.reason
    assert prop.info["cost_usd"] == pytest.approx((1000 * 4 + 500 * 20) / 1e6)
    assert cl.last["output_config"]["format"]["type"] == "json_schema"
    assert "test set" in cl.last["system"][0]["text"]


@pytest.mark.parametrize("bad, why", [(_answer(**{**EARLY, "C": 0.5}), "invalid proposal"),
                                      (anthropic.APIError.__new__(anthropic.APIError), "API error")])
def test_llm_failures_fall_back_to_the_bandit(bad, why):
    p, _ = _llm([bad])
    prop = p.propose([], CTX)
    assert prop.proposer.startswith("bandit (fallback: " + why)


def test_llm_duplicate_falls_back_and_caps_are_enforced():
    p, cl = _llm([_answer(**EARLY)] * 3, max_calls_per_run=2)
    hist = _fake_history([SPACE.canonical(_answer(**EARLY))], [0.8])
    assert "duplicate" in p.propose(hist, CTX).proposer
    p.propose([], CTX)
    assert "call cap" in p.propose([], CTX).proposer and cl.calls == 2


def test_no_key_means_deterministic_bandit(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    p = LLMProposer(SPACE, ACFG["llm"], BanditProposer(SPACE, 0), "", SEARCH)
    assert p.disabled and p.propose([], CTX).proposer.startswith("bandit (fallback: no ANTHROPIC_API_KEY")


def test_agent_loop_end_to_end_on_toy_data(tmp_path):
    ds = _toy()
    answers = [_answer(**EARLY), _answer(method="late", C=1.0, l2_normalize=False, pca=16, meta_weight=1.0,
                                         combine="mean_logit")]
    p, _ = _llm(answers)
    p.cfg = {**p.cfg, "max_calls_per_run": 2}
    clf = {**CLF}
    r = run_search(ds, lambda c: build_model(c, clf, seed=0), p, {"models": ["a", "noise"], "use_meta": True},
                   budget=3, patience=10, accept_k=1.0, n_jobs=1, log_path=tmp_path / "t.jsonl")
    rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert [r_["proposer"][:6] for r_ in rows] == ["llm", "llm", "bandit"]
    assert all(np.isfinite(r_["macro_auroc_mean"]) for r_ in rows) and r.best is not None


def test_llm_prompt_lists_the_tuned_members():
    mem = {"virchow2": {"C": 0.001, "l2_normalize": False, "pca": 16, "meta_weight": 1.0}}
    p = LLMProposer(SPACE, ACFG["llm"], BanditProposer(SPACE, 0), "", SEARCH, client=FakeClient([]), members=mem)
    assert json.dumps(mem) in p.system and "late_tuned" in p.system and "early_tuned" in p.system


def test_tuned_members_come_from_the_meta_rows(tmp_path):
    from luad.eval.baselines import tuned_members
    (tmp_path / "baselines").mkdir()
    (tmp_path / "baselines" / "summary.csv").write_text(
        'row,config\nvirchow2,"{""C"": 0.0001}"\nvirchow2+meta,"{""C"": 0.001, ""pca"": 16}"\n')
    assert tuned_members(["virchow2"], tmp_path) == {"virchow2": {"C": 0.001, "pca": 16}}
    with pytest.raises(FileNotFoundError):
        tuned_members(["uni2h"], tmp_path)


def test_llm_prompt_and_table_describe_the_best_explored_rule():
    from luad.agent.llm import history_table
    p = LLMProposer(SPACE, ACFG["llm"], BanditProposer(SPACE, 0), "",
                    {**SEARCH, "explore": 10, "incumbent": "best_explored"}, client=FakeClient([]))
    assert "Nothing is accepted during the exploration phase" in p.system
    h = _fake_history([{"method": "late_tuned", "combine": "mean_prob"}], [0.85])[0]
    h.update(decision="explore", gain=None, threshold=None)
    assert "explore" in history_table([h]) and "first trial" not in history_table([h])


def test_llm_prompt_describes_the_configured_aggregation():
    from luad.agent.llm import AGG_TEXT
    from luad.data.dataset import aggregation
    p = LLMProposer(SPACE, ACFG["llm"], BanditProposer(SPACE, 0), "", SEARCH, client=FakeClient([]))
    for m in ["uni2h", "virchow2", "hoptimus1"]:            # the proposer is told about the FUSED model's inputs
        assert AGG_TEXT[aggregation(m, fusion=True)] in p.system
