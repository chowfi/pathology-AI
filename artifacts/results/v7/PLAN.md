# Search round v7: tile-level fusion (one attention over all three foundation models)

Written 2026-11-19, before any v7 code was run on the data. The test set has not been touched.

## Why

- v6 dev analysis (out-of-fold, 167 dev patients): the three image-only single models disagree on 58% of
  patients. "At least one of the three is right" reaches a mean recall of 0.70; each model alone and the fused
  model reach 0.48-0.49. The information is spread across the models.
- Every fusion method so far combines the models AFTER each one has summarised the slide on its own (its own
  attention or slide encoder). The hard classes depend on small regions (Lepidic vs In situ: a small invasive
  focus; Papillary vs Acinar), which a single model's summary can under-weight.
- Idea: fuse at the tile level, before the summary. One attention network sees all three embeddings of each tile
  and decides where to look. All three models embedded exactly the same tiles (same coordinates).

## Method (fixed in advance, not tuned on this data)

- Input per tile: UNI2-h (1536-d), Virchow2 (2560-d, the saved tile embedding: CLS + mean patch token),
  H-optimus-1 (1536-d); each standardised per feature with the mean / SD of the training tiles.
- Per model: dropout 0.25 -> Linear(d -> 128) -> ReLU (CLAM-style projection); concatenated -> 384-d per tile.
- Gated attention as in v5: V, U: 384 -> 128; ONE attention weight per tile, shared by the three models.
- Head: Linear 384 -> 7 on sum_i a_i t_i (dropout 0.25).
- Training identical to v5: AdamW lr 2e-4, weight decay 1e-4, 30 epochs, one slide per step, at most 4096 random
  tiles per step (the same tiles in all three views), class-balanced loss weighted per sample, fp32,
  deterministic GPU algorithms, the v5 seed scheme.
- Exported: for each model m, sum_i a_i h_i^m of its RAW tile embeddings with the shared attention: three slide
  vectors in each model's own space (1536, 2560, 1536). They replace the fused model's input streams; the
  fusion methods, the search space and every downstream step are unchanged.
- Cross-fitting exactly as in v5 (5 parts per training set): for each CV fold, for each inner fold of nested
  CV, and at the end for the test set (trained on all 167 dev patients).

## What changes and what does not

- Changes: only the input streams of the FUSED model.
- The single-model rows keep their own aggregation (UNI2-h ABMIL, H-optimus-1 ABMIL, Virchow2 Prism2) and their
  v6 picks. They are not rerun: v7/baselines/summary.csv is a copy of v6/baselines/summary.csv.
- Members of the *_tuned fusion methods = the v6 '<model>+meta' picks.
- Search unchanged: space, budget 60, patience 15, exploration 10, acceptance 1 x corrected SE, and the v6 final
  pick rule (one_se_ba).
- Order: (1) cross-fitted joint vectors for the 15 CV folds; (2) bandit search run (free); (3) nested CV: in each
  outer fold, inner cross-fitted joint vectors + the fused bandit search; the inner baseline picks are reused from
  the v5 logs under the v6 rule (the baseline searches are unchanged). (4) The decision below. (5) Only if v7 is
  chosen: the 3 Claude-proposed runs (main + 2 repeats, about $2, after the user approves the cost); the official
  fused model is the main Claude run's pick.
- Ablation (reported, not a table row): Virchow2 tiles with per-model ABMIL (v5 settings), CV only. It separates
  "joint attention" from "Virchow2 now gets attention pooling instead of Prism2".

## Decision rule: v6 or v7 (agreed with the user before running)

Use v7 if, in nested CV (5 outer folds, held-out patients), the fused model's mean outer balanced accuracy is
higher than in v6, AND its mean outer AUROC is not lower than in v6 by more than 1 corrected SE of the paired
outer differences. Otherwise v6 stays the final round. Both rounds are reported.

## Checks allowed before the real runs

Only a smoke test that the training loss goes down on one training set. If a technical problem forces a change,
it is recorded here first.

## Outcome (recorded 2026-11-19, before any test-set access; Claude runs not yet made)

- Nested CV, fused (bandit procedure): outer AUROC v6 0.872 -> v7 0.867 (-0.005, corrected SE 0.013); outer
  balanced accuracy v6 0.467 -> v7 0.479 (+0.012; better in 2 of 5 outer folds). By the rule above, **v7 is
  chosen** (balanced accuracy higher, AUROC not clearly worse). Both changes are within noise: tile-level fusion
  gave no clear gain on held-out patients.
- v7 bandit run (CV): trial 18, AUROC 0.876, balanced accuracy 0.514.
- Ablation (CV): Virchow2 tiles with per-model ABMIL: AUROC 0.871, balanced accuracy 0.517 (Prism2 row: 0.862,
  0.495). Part of v7's input change is that Virchow2 now gets attention pooling; the Virchow2 table row stays
  Prism2 as fixed in v3, and this ablation is reported next to it.
- Next by this plan: the 3 Claude runs (about $2) after the user approves the cost; the official fused model is
  the main Claude run's pick.

## Official fused pick (recorded 2026-11-19, before any test-set access)

- The 3 Claude runs (main + 2 repeats; seed 30; all 25 proposals by Claude, none by the fallback; cost $1.73 in
  total) each stopped after 25 trials (15 in a row without an accepted gain).
- **Official fused model = main_llm trial 8**: early fusion, L2-normalised, no PCA, metadata weight 3, C = 1e-4,
  block balancing on. CV AUROC 0.880 +/- 0.025, balanced accuracy 0.518.
- Repeats: repeat_llm_2 trial 14 and repeat_llm_3 trial 20 both pick the same config (late fusion, stacking):
  AUROC 0.880, balanced accuracy 0.544. The three Claude runs agree on AUROC within 0.001.
- CV, paired over the 15 folds (artifacts/results/v7/comparison_cv.csv, scripts/compare_cv.py): the fused pick is
  higher than every single-model row on both metrics. AUROC gain +0.010 (UNI2-h + meta, 1.0 corrected SE) to
  +0.018; balanced accuracy gain +0.022 to +0.034 (each < 1 corrected SE).
- vs the ablation (Virchow2 tiles + per-model ABMIL, not a table row): AUROC +0.010 (SE 0.008, 12/15 folds),
  balanced accuracy +0.001 (a tie).
- The nested-CV numbers of the fused model stay those of the bandit procedure (0.867 / 0.479); Claude is not run
  inside nested CV.

## Last development round (recorded 2026-11-19, the user's decision, before any test-set access)

- v7 is the last development round, whatever the test result. No further change to the features, the search, the
  rules or the picks. Locked: single-model rows = v7/baselines/summary.csv (the v6 picks); fused model = v7
  main_llm trial 8 on the joint ABMIL streams.
- Still allowed on dev data: failure analysis and figures (they do not change any model).
- Then: one final test script, reviewed by the user and run once with the user's explicit approval.

## Final test protocol (written 2026-11-19, before the test set was opened; the test access log is empty)

- Script: scripts/final_test.py; settings: conf/experiments.yaml `final`. One run with --test, only after the
  user approves; the script refuses a second run (test access log entry or output directory present) and refuses
  uncommitted tracked changes (unless --allow-dirty, which is recorded).
- Rows (README table): metadata only; UNI2-h (ABMIL), Virchow2 (Prism2), H-optimus-1 (ABMIL); best single + meta
  = the '+meta' row with the highest CV AUROC = UNI2-h + meta (0.870); fused + meta = v7 main_llm trial 8. The
  other two '+meta' rows are scored too and reported in metrics.csv.
- Refit: each row's locked config on all 167 train+val patients; logistic regression with the CV settings.
  Supervised vectors cross-fitted as in CV (5 parts; seed = abmil.seed + 1000 x 15): training patients get
  out-of-part vectors, test patients the mean of the 5 networks. Virchow2 rows: the fixed Prism2 vectors.
- Metrics on the 42 test patients: macro AUROC and balanced accuracy (primary), accuracy, macro F1, per-class
  AUROC, confusion matrices. Uncertainty: 2000 bootstrap resamples stratified by class (seed 30), percentile 95%
  CIs; paired fused - row differences on the same resamples.
- Reading: the README claim (fused beats every single-model row) is read from the point estimates of both
  primary metrics; the paired CIs say whether a gain is clear or within noise. With 42 patients, most differences
  are expected to be within noise. Everything is reported, whatever the result. No change after the test is seen.
- Before the real run: a dry run (dev patients only: CV fold 0 as a stand-in test set, cross-fit seed index 0)
  must reproduce the logged fold-0 CV scores of every row and the cached fold-0 vectors.
