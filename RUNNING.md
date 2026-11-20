# How to run the pipeline

One documented entry point per stage. Every stage has a script in `scripts/` (its docstring gives the details), and
`scripts/run_pipeline.sh` runs them in order:

```bash
bash scripts/run_pipeline.sh <stage> [<stage> ...]     # e.g. bash scripts/run_pipeline.sh tiles embeddings
bash scripts/run_pipeline.sh all                        # split -> dryrun (everything except the one-time test)
```

All settings live in `conf/*.yaml` (data, tiling, embeddings, experiments, agent). Seed: 30 everywhere.

## 1. Setup

| Need | How |
|---|---|
| Python 3.11-3.12, pinned dependencies | `uv sync --group dev --group notebooks` (versions locked in `uv.lock`; PyTorch from the CUDA 12.8 index) |
| Gated Hugging Face models (UNI2-h, Virchow2, Prism2, H-optimus-1) | Request access on each model page, then `uv run huggingface-cli login`. Model revisions are pinned in `conf/embeddings.yaml`. |
| Claude proposer for the agent (optional) | `ANTHROPIC_API_KEY=...` in `.env` (gitignored). Without a key the agent falls back to the deterministic bandit. |
| GPU | One CUDA GPU with 32 GB for the joint ABMIL stages. GPU scripts refuse to start above the power limit in `conf/embeddings.yaml` (`run.max_power_w`, 450 W). |
| Disk | About 41 GB (4x slide levels) + 26 GB (tile embeddings), both gitignored |
| Tests | `uv run pytest` |

## 2. Stages

| # | Stage (`run_pipeline.sh`) | Entry point | Reads | Writes | Time (RTX 5090) |
|---|---|---|---|---|---|
| 1 | `split` | `scripts/make_split.py` | `CLWD.csv` | `artifacts/splits/`: cohort, patient-level split, 5x3 CV folds | seconds |
| 2 | `download` | `scripts/download_slides.py` | authors' server (`conf/data.yaml`) | `artifacts/slides/<WSI>_4x.tif` (4x pyramid level only, 0.415 um/px) | ~1.5 h |
| 3 | `tiles` | `scripts/make_tiles.py` | 4x slides | `artifacts/tiles/`, `artifacts/masks/`, `artifacts/metadata/tiling_summary.csv` | ~2 h CPU |
| 4 | `embeddings` | `scripts/extract_embeddings.py` | 4x slides + tile grid | `artifacts/embeddings/<model>/<WSI>.h5` (224 px at 0.5 um/px, all 3 models) | ~6.9 h GPU |
| 5 | `slide_vectors` | `pool_slides.py`, `extract_prism2.py`, `abmil_features.py`, `abmil_features.py --joint` | tile embeddings | `artifacts/features/`: mean, Prism2, cross-fitted ABMIL and joint ABMIL vectors per CV fold | ~15 min |
| 6 | `baselines` | `scripts/run_baselines.py` | slide vectors | `<results_dir>/baselines/` (single-model and metadata rows) | ~1 min |
| 7 | `agent` | `scripts/run_agent.py` (3 Claude runs + 1 bandit run), `scripts/compare_cv.py` | slide vectors, baseline picks | `<results_dir>/agent/<run>/trials.jsonl` (decision log), `summary.json`, `comparison_cv.csv` | ~15 min per run, ~$0.6 per Claude run |
| 8 | `nested` | `scripts/nested_cv.py` | slide vectors | `<results_dir>/nested_cv/` (selection-noise estimate) | ~1.3 h (v7 run; longer if the inner baseline searches are rerun) |
| 9 | `dryrun` | `scripts/final_test.py` | dev data only (CV fold 0 as stand-in test set) | `artifacts/results/final_dryrun/` (must reproduce the logged fold-0 scores exactly) | ~2 min |
| 10 | `test` | `scripts/final_test.py --test` | test patients, **once** | `artifacts/results/final/` (README table, CIs, predictions) | ~2 min |
| 11 | `analysis` | `test_analysis.py`, `report_figures.py`, both notebooks | saved results only | `artifacts/results/final/test_*.csv`, `artifacts/figures/` | minutes |
| - | `attention` (optional) | `scripts/test_attention.py` | test tiles (logged, visualization only) | `artifacts/figures/failure_test/` | ~2 min |

`<results_dir>` = `results_dir` in `conf/experiments.yaml` and `conf/agent.yaml` (final round: `artifacts/results/v7`).

## 3. Rules the scripts enforce

- **The test set is used once.** `final_test.py --test` refuses if `artifacts/logs/test_access.log` already has an
  entry, if its output folder exists, or if tracked files have uncommitted changes. Every test access is logged
  with its reason. The test run happened on 2026-11-19.
- **Committed results are not overwritten.** `run_pipeline.sh` skips a stage whose outputs exist (split, slide
  vectors, baselines, agent runs, nested CV). To rerun the selection from scratch, set `results_dir` in
  `conf/experiments.yaml` and `conf/agent.yaml` to a new folder (or use `FORCE=1` to overwrite).
- **Slides, tiles and embeddings resume:** finished slides are skipped, so these stages are safe to stop and restart.

## 4. Where the results are

| What | Where |
|---|---|
| README comparison table (test, with 95% CIs) | `artifacts/results/final/table.md` |
| All test metrics, paired differences, per-class AUROC, confusion matrices, predictions | `artifacts/results/final/` |
| Agent decision logs (every trial, Claude's reasoning, scores, decision) | `artifacts/results/v7/agent/<run>/trials.jsonl` |
| Round history and the reason for each change, written before each round ran | `artifacts/results/<round>/PLAN.md` (v1-v7) |
| Dev failure analysis (CV predictions, attention maps) | `notebooks/failure_analysis.ipynb` |
| Test results and failure analysis (read-only view of the saved test run) | `notebooks/test_failure_analysis.ipynb` |
| Figures | `artifacts/figures/report/`, `artifacts/figures/failure_v7/`, `artifacts/figures/failure_test/` |
| Logs (download, tiling, embeddings timing, GPU power, test access) | `artifacts/logs/` |

## 5. Code layout

| Folder | Contents |
|---|---|
| `conf/` | All settings (YAML) |
| `src/luad/data/` | Cohort, split, slide reading, tissue mask, tiling |
| `src/luad/embeddings/` | Foundation-model loading and preprocessing, ABMIL and joint ABMIL |
| `src/luad/fusion/` | Fusion methods registered by name: early, late, gated, late_tuned, early_tuned |
| `src/luad/agent/` | Search loop, Claude proposer, bandit fallback, search space, final-pick rule |
| `src/luad/eval/` | CV, metrics, bootstrap CIs, nested-CV decision, failure-analysis helpers |
| `scripts/` | Stage entry points (above); `scripts/explore/` and `scripts/checks/` hold the one-off preprocessing checks |
| `tests/` | Unit tests (`uv run pytest`) |
