# Search round v4: plan (written 2026-11-19, before any v4 code or run)

## Why a v4 round

v3 result (CV on the 167 train+val patients, test set untouched): the fused model ties the best single model
(Prism2 on Virchow2 tiles, 0.860 macro AUROC); it clearly beats UNI2-h and H-optimus-1. Two facts in the v3
logs point to a specific weakness of the v1-v3 fusion methods:

1. The three models want very different settings. Selected single-model configs: Prism2 C=1e-4 + PCA 16,
   UNI2-h C=1e-2 + PCA 128, H-optimus-1 C=1e-2. At C=1e-4, UNI2-h drops to 0.766 and H-optimus-1 to 0.775.
   In early and late fusion all three models share ONE C, one PCA and one L2 setting, so the fused model
   can never use every member at its own best setting.
2. Prism2 is best or tied on every class. The fused model gains on Micropapillary and Papillary but loses on
   Cribriform and Lepidic, where the two mean-pooled models are weak.

## What changes (idea 1 only)

Two new fusion methods are added to the agent's search space. The old methods (early, late, gated) stay.

- `late_tuned`: each member is that model's own tuned single-model + metadata baseline (its selected C,
  L2, PCA, meta_weight), then the same combiners as `late` (mean_prob, mean_logit, learned_weights,
  weighted_vote, stack). learned_weights includes weight (1, 0, 0), so "one model alone" is a special case.
- `early_tuned`: each model block keeps the L2 / PCA setting of its own tuned single-model + metadata
  baseline; the blocks are joined with age/sex in one logistic regression; a weight per model block (grid
  0, 0.25, 0.5, 1, the largest weight = 1) is learned by inner CV inside the training fold (log-loss).
  In an L2-penalised logistic regression a block weight w acts like a separate C (C x w^2) for that block,
  and weight 0 removes the block, so "one model alone" is again a special case.

The member configs are fixed inputs (not searched by the agent):
- main search: taken from the v4 baseline rows `<model>+meta` (selected on the same 15 folds, like every
  baseline row)
- nested CV: taken from the baseline searches run inside each outer fold (so no outer-fold information)

## What does NOT change

Features and aggregation (UNI2-h mean, Virchow2 -> Prism2, H-optimus-1 mean), the 15 folds, the classifier
family (multinomial logistic regression), the budget (60), patience (15), exploration phase (10), the
acceptance rule (gain > 1 x corrected SE), the baseline rows and their grids, the bandit seed, the Claude
model and caps. The fused model for the final table is still the selection of the MAIN Claude run.

## Rules for this round

1. This is the last search round. If v4 still ties the best single model, the tie is reported.
2. The baselines are rerun with the v4 code; they must be identical to v3 (determinism check).
3. Nested CV is rerun with the new methods in the bandit's space.
4. The test set is used once, at the end, after the configs are locked.

## Amendment 1 (2026-11-19, after the v4 results, before v4b): selection rule

What v4 showed: all three Claude runs kept trial 1 (late_tuned mean_prob, CV 0.854) as the selected model.
Trial 2 (late_tuned learned_weights, 0.864) missed the acceptance threshold by 0.001 (gain +0.0099 vs
1 x corrected SE 0.0110), and no later trial beat trial 1 by more than 1 SE, because every good configuration
lies inside a plateau about 0.01 wide. The same issue was seen in v1. The v2 exploration phase only changed the
stop counter, not which trial is the current best.

Diagnosis: the noise bar (gain > 1 x corrected SE) works as intended; the flaw is the tie-break. Trial 1 is the
current best without any test, so inside a plateau whichever trial comes first wins. Which trial comes first is
arbitrary (Claude's first guess, or the seeded grid order for the baselines).

Change (`search.incumbent: best_explored`): nothing is accepted during the 10 exploration trials; at the end of
the exploration phase the explored trial with the highest mean AUROC becomes the current best; afterwards the
acceptance rule and the stop rule are unchanged. It applies to EVERY row (the baselines use the same loop).

v4b reruns, with the same features, folds, space, budget and caps: all baseline rows (so the tuned members may
change), the 3 Claude runs, the bandit run and nested CV. Results go to artifacts/results/v4b; v4 is kept.
The fused model is still the selection of the main Claude run. The best of 10 explored trials carries some
selection optimism; nested CV measures it. The test set is still unused.
This amends rule 1 above ("last search round"): v4b is a rerun of v4 under a corrected selection rule, with no
new methods or features. Any further change (for example attention pooling for UNI2-h / H-optimus-1) would be a
new, separately documented round.
