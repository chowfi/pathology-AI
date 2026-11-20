# Search round v6: final selection with a balanced-accuracy tie-break

Written 2026-11-19, after the v5 dev diagnostics and before the rule was applied to the nested-CV logs.
The test set has not been touched.

## Why

- The README lists macro AUROC and balanced accuracy as the primary metrics. Up to v5, every search picked its
  final trial on macro AUROC only.
- v5 dev diagnostics (artifacts/figures/v5_diagnostics/): the selected fused model (main Claude run, trial 8,
  early fusion, C 1e-4) wins on CV AUROC (0.885 vs 0.870 for the best single model + metadata) but not on
  balanced accuracy (0.480 vs 0.496). Its recall is 0.22 on Lepidic and 0.22 on Papillary. 19 of the 74
  explored fused configs beat the best single model on both metrics in CV, mostly late fusion.
- Known before this plan was written (disclosed): the rule below, applied to the v5 CV logs, picks main-run
  trial 19 (late_tuned, mean_logit; CV AUROC 0.881, balanced accuracy 0.502). Its nested-CV result is not known.

## The rule (one-standard-error rule with a balanced-accuracy tie-break)

For one search (one baseline row or one agent run), using only that search's own trials:

1. best = the trial with the highest mean macro AUROC over the CV folds (equal means: the earlier trial)
2. a trial ties with best if its mean paired per-fold AUROC gap to best is at most 1 corrected SE of that gap
   (the same Nadeau-Bengio SE as the acceptance rule; n_train = n - n_val)
3. among best and its ties, pick the highest mean balanced accuracy (equal: higher AUROC, then earlier trial)

In short: take the trials with about the best AUROC; from those, pick the best balanced accuracy.
AUROC (the less noisy metric) stays in charge; balanced accuracy only decides between trials that tie on AUROC.

## What changes and what does not

- Changes: only the final pick from each finished trial log, for every row: all single-model and metadata
  rows, the 4 agent runs, and every row inside every outer fold of nested CV.
- No new searches and no Claude calls. Proposals, acceptance (gain > 1 x corrected SE), exploration (10),
  patience (15), budget (60), folds, seeds and classifier are unchanged, so the v5 trial logs stay valid.
- The members of the *_tuned fusion methods stay as they were during the searches (picked by the AUROC rule),
  because the trial scores were computed with them.
- "Best single model (+ metadata)" = the row whose picked trial has the highest mean AUROC (as in v5).
- Official fused model = the main Claude run's pick under this rule. Baselines = the v6 picks.

## Decision rule: v5 or v6 (agreed with the user before the nested-CV check)

Use v6 if, in nested CV (5 outer folds, held-out patients), the fused model's mean outer balanced accuracy is
higher than in v5, AND its mean outer AUROC is not lower than in v5 by more than 1 corrected SE of the paired
outer differences. Otherwise v5 stays the final round. Both rounds are reported.

## Code and outputs

- src/luad/agent/selection.py: the rule; `run_search(selection=...)` applies it to new searches.
- scripts/reselect.py: applies the rule to the v5 logs and refits the nested-CV picks on the outer folds.
- Outputs: artifacts/results/v6/{baselines/summary.csv, agent/<run>/summary.json, nested_cv/, decision.json}.

## Outcome (recorded 2026-11-19, after scripts/reselect.py, before any test-set access)

Nested CV, fused (bandit procedure): outer AUROC v5 0.867 -> v6 0.872 (+0.005, corrected SE 0.003); outer balanced
accuracy v5 0.466 -> v6 0.467 (+0.001; better in 1 of 5 outer folds, worse in 1, equal in 3). By the rule above,
**v6 is the final round** (balanced accuracy higher, AUROC not worse). The balanced-accuracy gain on held-out
patients is negligible: the rule makes the CV pick beat every row on both metrics, but nested CV does not show
a real balanced-accuracy gain on unseen patients.

Final fused model = main Claude run trial 19: late_tuned, mean_logit, members = the v5 '<model>+meta' picks
(as used in the search); CV AUROC 0.881, balanced accuracy 0.502. Baselines = artifacts/results/v6/baselines.
