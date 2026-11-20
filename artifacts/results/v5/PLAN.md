# Search round v5: attention pooling (ABMIL) for UNI2-h and H-optimus-1

Written 2026-11-19, before any ABMIL code was run on the data.

## Why

Across v3, v4 and v4b every good configuration (fused or single) lands at 0.854-0.864 CV macro AUROC. The fused
model ties Prism2 (+ metadata) and clearly beats UNI2-h and H-optimus-1. Prism2 is best or tied on every class.
The one change with a mechanism behind it is the tile -> slide aggregation of the two tile encoders: on the same
Virchow2 tiles, Prism2's attention-based slide encoder beat mean pooling by +0.048, mostly on Cribriform (+0.17)
and Lepidic (+0.08), patterns that cover only part of a slide and are diluted by an average over all tissue. The
README lists attention-based (ABMIL-style) pooling as an aggregation option.

## What changes

UNI2-h and H-optimus-1: mean pooling -> gated-attention MIL pooling (Ilse et al., 2018). Virchow2 keeps Prism2.
Everything downstream is unchanged: the same logistic-regression rows, grids, search loop and selection rule
(v4b: `incumbent: best_explored`), the same agent space and Claude settings.

## ABMIL settings (fixed in advance, not tuned on this data)

- Input: the saved tile embeddings (1536-d), standardised per feature with the mean / SD of the training tiles.
- Network: dropout 0.25 on the input; gated attention a_i = softmax_i(w^T [tanh(V x_i) * sigmoid(U x_i)]) with
  V, U: 1536 -> 128; z = sum_i a_i x_i; linear head 1536 -> 7 on z (dropout 0.25). About 0.4 M parameters.
- Training: AdamW, lr 2e-4, weight decay 1e-4, 30 epochs, one slide per step in random order, at most 4096
  random tiles per slide per step (all tiles at inference), class-balanced cross-entropy, fp32, deterministic GPU
  algorithms, fixed seeds.
- Exported slide vector: the attention-weighted mean of the RAW tile embeddings, sum_i a_i h_i (1536-d), so the
  only learned part is where the model looks; the vector stays in the foundation model's own space, like the
  mean-pooled vector, and goes into the same logistic-regression pipeline.

Values are common ABMIL/CLAM defaults. The only check before the real runs is a smoke test that the training loss
goes down; if a technical problem forces a change, it is recorded here first.

## No label leakage: cross-fitting

ABMIL learns from labels, so a slide vector must never come from a network trained on that slide's own label.
For a set of training patients T and a set of patients to score A:
- split T into 5 stratified parts; for each part, train ABMIL on the other 4 and pool the held-out part
  (out-of-fold vectors for every training patient) and pool A
- training-patient vectors = the out-of-fold vectors; A vectors = the mean over the 5 networks
This is done separately for each of the 15 CV folds (T = fold training patients, A = fold validation patients),
for each inner fold inside nested CV (T, A from the inner patients only), and, at the end, for the test set
(T = all 167 train+val patients, A = the 42 test patients).

## Decision rule (chosen by the user before any v5 result)

The final round is v4b (mean pooling) or v5 (ABMIL), whichever fused model has the higher nested-CV outer mean
macro AUROC. Both rounds are reported. In the chosen round: fused = the main Claude run's selection, baselines =
that round's selections. Then the configs are locked and the test set is used once.

## Rules

1. Same folds, budget (60), patience (15), exploration (10), acceptance (1 x corrected SE), seeds, caps.
2. v5 reruns: all baseline rows, 3 Claude runs, the bandit run, nested CV. Results in artifacts/results/v5.
3. No other change in this round.

## Outcome of the decision rule (recorded 2026-11-19, after the v5 runs, before any test-set access)

Fused nested-CV outer mean macro AUROC: v4b 0.846, v5 0.867. By the rule above, **v5 is the final round**.
Final fused model = the v5 main Claude run's selection (trial 8: early fusion, L2, no PCA, meta weight 3,
C 1e-4, balanced blocks; CV 0.885). Baselines = the v5 selections in artifacts/results/v5/baselines/summary.csv.
