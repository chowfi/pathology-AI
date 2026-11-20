"""LLM proposer: Claude reads the full trial log and proposes the next fusion configuration, with its reasoning.

Safety and cost rules:
  * the API key comes from the environment (.env, gitignored); it is never logged
  * the answer is forced into a JSON schema (structured outputs) and then validated against the search space;
    an invalid, duplicate or failed proposal is replaced by the bandit for that trial, and the reason is logged
  * hard caps on calls and on cost per run; after a cap the bandit proposes
  * every call logs model, request id, tokens and cost
Claude never sees the test set: the log contains CV results on train+val only.
"""
from __future__ import annotations

import json
import os

from luad.agent.search import Proposal
from luad.agent.space import FusionSpace

SYSTEM = """You are the search policy of an agent that looks for the best way to FUSE three pathology foundation \
models for 7-class lung adenocarcinoma growth-pattern classification (Acinar, Cribriform, In situ, Lepidic, \
Micropapillary, Papillary, Solid).

Data. 167 train+val patients (class counts: Acinar 26, Cribriform 12, In situ 36, Lepidic 17, Micropapillary 21, \
Papillary 32, Solid 23). One slide per patient. Each slide is one embedding per model: {streams}, plus age and \
sex. The aggregation is fixed; it is not part of the search. The test set is locked away and is never used during \
the search.

Evaluation of one trial. 15 fixed folds (5-fold stratified CV repeated 3 times). Score = mean macro one-vs-rest \
AUROC over the 15 folds; balanced accuracy is also reported. All preprocessing (L2 norm, standardising, PCA) is \
fitted inside each training fold. Class imbalance is handled by class-balanced weights. Every method ends in a \
linear softmax classifier, so the classifier family is fixed; only the fusion changes.

Methods and knobs.
- early: standardise each model (optional per-slide L2 norm first, optional PCA per model), concatenate with \
age/sex, one multinomial logistic regression with inverse L2 strength C. balance_blocks rescales the model blocks \
to equal total variance (overall scale, and so the meaning of C, unchanged) so no model dominates by its number of \
columns. meta_weight multiplies the standardised age/sex columns (higher = less shrinkage of metadata).
- late: one logistic regression per model (each with age/sex, same C / l2_normalize / pca / meta_weight), then \
combine probabilities: mean_prob, mean_logit (geometric mean), learned_weights (simplex weights learned by inner CV \
in the training fold), weighted_vote, stack (a second logistic regression with inverse strength stack_C on the base \
log-probabilities + age/sex, trained on inner-CV predictions).
- gated: per-model projection (dropout, Linear to `hidden`, LayerNorm), a softmax gate over the 3 models per \
patient, weighted sum, + age/sex, linear softmax head. AdamW (lr, weight_decay), fixed epochs, averaged over \
n_seeds networks. Only 134 training patients per fold, so it overfits easily.
- late_tuned: like late, but each per-model logistic regression is that model's OWN tuned single-model + age/sex \
baseline (configs below), so every member runs at the setting that suited it alone instead of one shared C / PCA / \
L2. Knobs: combine (and stack_C for stack). learned_weights may give one model weight 1, so the best single model \
is a special case.
- early_tuned: each model block keeps the L2 / PCA of its own tuned baseline (configs below); the blocks are \
joined with age/sex (meta_weight) in one logistic regression with inverse strength C. A weight per model block \
(0, 0.25, 0.5 or 1; the largest is 1) is learned by inner CV inside the training fold; with the L2 penalty a block \
weight w acts like a separate inverse strength C x w^2 for that block, and 0 drops the block.
Tuned member configs (the selected <model>+meta baseline rows, same folds): {members}

Search space (JSON; knobs a method does not use must be null):
{space}

Search rules (enforced by code, not by you).
- Budget {budget} trials. The first {explore} trials are an exploration phase: they never count toward the stop \
rule, so use them to map the space (every method, and the knobs most likely to matter).{explore_rule} After \
that, the search stops after {patience} trials in a row without an accepted improvement.
- A trial replaces the current best only if its mean paired per-fold AUROC gain over the best is larger than \
{accept_k} x the corrected standard error of that gain (Nadeau-Bengio correction for overlapping folds). The \
threshold is usually 0.005-0.02 AUROC. Gains below it are treated as noise. So chasing tiny differences wastes \
budget; propose trials that could plausibly give a clear gain, or that answer an open question about the space.
- A configuration that was already tried is rejected.

Your goal: find a fusion configuration that is ACCEPTED as better than the best single-model baselines, using few \
trials, while avoiding selection on noise. Use the evidence in the log: per-class AUROC, fold SDs, which knobs \
helped, which methods are untested. Reason like a careful scientist: state what the evidence shows, choose the \
next trial, and say what you expect and what you would conclude if it fails. Be concise."""


MODEL_NAMES = {"uni2h": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
AGG_TEXT = {
    "mean": "mean of tile embeddings",
    "prism2": "Prism2 slide embedding (Paige's slide encoder applied to the Virchow2 tile CLS tokens)",
    "abmil": ("attention-weighted mean of tile embeddings; the gated-attention MIL weights are learned inside each "
              "training fold with cross-fitting, so no slide vector comes from a network that saw its own label"),
    "joint_abmil": ("attention-weighted mean of tile embeddings, where ONE gated-attention network sees all three "
                    "models' embeddings of each tile and its attention weights are shared by the three models "
                    "(tile-level fusion); learned inside each training fold with cross-fitting"),
}


def streams_text(models: list[str]) -> str:
    from luad.data.dataset import aggregation
    dims = {"uni2h": 1536, "virchow2": 2560, "hoptimus1": 1536}
    return ", ".join(f"{MODEL_NAMES.get(m, m)} ({dims.get(m, '?')}-d, {AGG_TEXT[aggregation(m, fusion=True)]})"
                     for m in models)


EXPLORE_RULE = {
    "first": " The first trial is accepted as the starting best.",
    "best_explored": (" Nothing is accepted during the exploration phase; at its end, the explored trial with the "
                      "highest mean AUROC becomes the current best."),
}


def history_table(history: list[dict]) -> str:
    if not history:
        return "(no trials yet)"
    lines = []
    for h in history:
        c = h["config"]
        knobs = " ".join(f"{k}={v}" for k, v in c.items() if k != "method")
        pc = " ".join(f"{n[:4]} {v:.2f}" for n, v in zip(
            ["Acinar", "Cribriform", "In situ", "Lepidic", "Micropapillary", "Papillary", "Solid"],
            h.get("per_class_auroc_mean", [])))
        if h["gain"] is not None:
            dec = h["decision"] + f" (gain {h['gain']:+.4f}, threshold {h['threshold']:.4f})"
        else:
            dec = h["decision"] + (" (first trial)" if h["decision"] == "accept" else "")
        lines.append(f"#{h['trial']} {c['method']} {knobs}\n   AUROC {h['macro_auroc_mean']:.4f} ± "
                     f"{h['macro_auroc_sd']:.4f} | BA {h['balanced_accuracy_mean']:.3f} | per-class {pc} | {dec}")
    return "\n".join(lines)


class LLMProposer:
    name = "llm"

    def __init__(self, space: FusionSpace, llm_cfg: dict, fallback, baselines_text: str, search_cfg: dict,
                 client=None, members: dict | None = None, models: list[str] | None = None):
        self.space, self.cfg, self.fallback, self.baselines = space, llm_cfg, fallback, baselines_text
        self.system = SYSTEM.format(space=space.describe(), budget=search_cfg["budget"],
                                    patience=search_cfg["patience"], accept_k=search_cfg["accept_k"],
                                    explore=search_cfg.get("explore", 0),
                                    explore_rule=EXPLORE_RULE[search_cfg.get("incumbent", "first")],
                                    members=json.dumps(members) if members else "(not used)",
                                    streams=streams_text(models or ["uni2h", "virchow2", "hoptimus1"]))
        self.schema = space.json_schema()
        self.calls, self.cost = 0, 0.0
        self.client = client
        self.disabled = None
        if self.client is None:
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                self.disabled = "no ANTHROPIC_API_KEY in the environment"
            else:
                import anthropic
                self.client = anthropic.Anthropic()

    def _fallback(self, history, ctx, why: str, info: dict | None = None) -> Proposal | None:
        p = self.fallback.propose(history, ctx)
        if p is not None:
            p.proposer = f"bandit (fallback: {why})"
            p.info = {**(info or {}), **p.info}
        return p

    def _cost(self, u) -> float:
        pr = self.cfg["price_per_mtok"]
        return ((u.input_tokens or 0) * pr["input"] + (u.output_tokens or 0) * pr["output"]
                + (getattr(u, "cache_read_input_tokens", 0) or 0) * pr["cache_read"]
                + (getattr(u, "cache_creation_input_tokens", 0) or 0) * pr["cache_write"]) / 1e6

    def user_message(self, history, ctx) -> str:
        best = f"trial #{ctx['best_trial']}" if ctx["best_trial"] else "none yet"
        if not ctx["best_trial"] and ctx.get("exploring"):
            best += " (set at the end of the exploration phase)"
        if ctx.get("exploring"):
            phase = (f"Exploration phase: trial {ctx['budget'] - ctx['trials_left'] + 1} of {ctx['explore']} "
                     "(does not count toward the stop rule).")
        else:
            phase = (f"Trials in a row without an accepted improvement: {ctx['streak']} "
                     f"(stop at {ctx['patience']}).")
        return (f"Single-model baselines (same folds, same budget and rules; CV on train+val):\n{self.baselines}\n\n"
                f"Fusion trials so far:\n{history_table(history)}\n\n"
                f"Current accepted best: {best}. Trials left: {ctx['trials_left']}. {phase}\n"
                "Propose the next trial.")

    def propose(self, history: list[dict], ctx: dict) -> Proposal | None:
        if self.disabled:
            return self._fallback(history, ctx, self.disabled)
        if self.calls >= self.cfg["max_calls_per_run"]:
            return self._fallback(history, ctx, f"call cap of {self.cfg['max_calls_per_run']} reached")
        if self.cost >= self.cfg["max_cost_usd_per_run"]:
            return self._fallback(history, ctx, f"cost cap ${self.cfg['max_cost_usd_per_run']} reached")
        import anthropic
        self.calls += 1
        try:
            r = self.client.beta.messages.create(
                model=self.cfg["model"], max_tokens=self.cfg["max_tokens"],
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": self.user_message(history, ctx)}],
                output_config={"effort": self.cfg["effort"],
                               "format": {"type": "json_schema", "schema": self.schema}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        except anthropic.APIError as e:
            return self._fallback(history, ctx, f"API error {type(e).__name__}")
        cost = self._cost(r.usage)
        self.cost += cost
        info = {"model": r.model, "request_id": getattr(r, "_request_id", None), "stop_reason": r.stop_reason,
                "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
                "cache_read_tokens": getattr(r.usage, "cache_read_input_tokens", 0),
                "cache_write_tokens": getattr(r.usage, "cache_creation_input_tokens", 0),
                "cost_usd": round(cost, 5), "run_cost_usd": round(self.cost, 4)}
        if r.stop_reason != "end_turn":
            return self._fallback(history, ctx, f"stop_reason {r.stop_reason}", info)
        try:
            text = next(b.text for b in r.content if b.type == "text")
            ans = json.loads(text)
            cfg = self.space.canonical(ans)
        except (StopIteration, json.JSONDecodeError, ValueError) as e:
            return self._fallback(history, ctx, f"invalid proposal: {str(e)[:120]}", info)
        if self.space.key(cfg) in {self.space.key(h["config"]) for h in history}:
            return self._fallback(history, ctx, "duplicate of an earlier trial", info)
        reason = (f"ANALYSIS: {ans['analysis']}\nREASONING: {ans['reasoning']}\n"
                  f"EXPECTED: {ans['expected_effect']}")
        return Proposal(cfg, reason, self.name, info)
