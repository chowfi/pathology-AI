#!/usr/bin/env bash
# One entry point per pipeline stage, in order. See RUNNING.md for inputs, outputs and run times.
#
#   bash scripts/run_pipeline.sh <stage> [<stage> ...]
#   bash scripts/run_pipeline.sh all            # stages split .. dryrun (everything except the one-time test)
#   FORCE=1 bash scripts/run_pipeline.sh agent  # rerun a stage whose committed outputs already exist
#
# Stages: split download tiles embeddings slide_vectors baselines agent nested dryrun | test | analysis attention
# Stages whose outputs are committed are SKIPPED when those outputs exist (they would be overwritten). To rerun
# the selection from scratch, point results_dir in conf/experiments.yaml and conf/agent.yaml to a new folder
# first (RUNNING.md). The test stage is not part of "all": the test set is used once, and final_test.py refuses
# a second run.
set -euo pipefail
cd "$(dirname "$0")/.."

run() { echo "+ $*"; "$@"; }
results_dir() { uv run python -c "from luad.config import load_config; print(load_config('$1')['results_dir'])"; }

# skip the stage if its output exists, unless FORCE=1
fresh() {
  if [ -e "$1" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "skip: $1 exists (committed result; FORCE=1 to overwrite, or set a new results_dir)"; return 1
  fi
}

stage() {
  case "$1" in
    split)          if fresh artifacts/splits/split_seed30.csv; then run uv run python scripts/make_split.py; fi ;;
    download)       run uv run python scripts/download_slides.py ;;          # resumes; finished slides skipped
    tiles)          run uv run python scripts/make_tiles.py ;;               # finished slides skipped
    embeddings)     run uv run python scripts/extract_embeddings.py ;;       # finished slides skipped
    slide_vectors)  if fresh artifacts/features/mean_uni2h.npz; then
                      run uv run python scripts/pool_slides.py
                      run uv run python scripts/extract_prism2.py
                    fi
                    run uv run python scripts/abmil_features.py              # gitignored cache, rebuilt
                    run uv run python scripts/abmil_features.py --joint ;;
    baselines)      if fresh "$(results_dir experiments)/baselines/summary.csv"; then run uv run python scripts/run_baselines.py; fi ;;
    agent)          d="$(results_dir agent)"
                    if fresh "$d/main_llm/summary.json"; then run uv run python scripts/run_agent.py --run main_llm; fi
                    if fresh "$d/repeat_llm_2/summary.json"; then run uv run python scripts/run_agent.py --run repeat_llm_2; fi
                    if fresh "$d/repeat_llm_3/summary.json"; then run uv run python scripts/run_agent.py --run repeat_llm_3; fi
                    if fresh "$d/main_bandit/summary.json"; then run uv run python scripts/run_agent.py --run main_bandit --no-llm; fi
                    run uv run python scripts/compare_cv.py ;;
    nested)         if fresh "$(results_dir experiments)/nested_cv/summary.csv"; then run uv run python scripts/nested_cv.py; fi ;;
    dryrun)         run uv run python scripts/final_test.py ;;                # writes artifacts/results/final_dryrun
    test)           run uv run python scripts/final_test.py --test ;;         # refuses after the first run
    analysis)       run uv run --group notebooks python scripts/test_analysis.py           # saved predictions only
                    run uv run --group notebooks python scripts/report_figures.py
                    run uv run --group notebooks jupyter nbconvert --to notebook --execute --inplace notebooks/failure_analysis.ipynb
                    run uv run --group notebooks jupyter nbconvert --to notebook --execute --inplace notebooks/test_failure_analysis.ipynb ;;
    attention)      run uv run --group notebooks python scripts/test_attention.py ;;       # reads test tiles; logged
    *) echo "unknown stage: $1 (see RUNNING.md)" >&2; exit 2 ;;
  esac
}

[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 2; }
for s in "$@"; do
  if [ "$s" = all ]; then
    for t in split download tiles embeddings slide_vectors baselines agent nested dryrun; do stage "$t"; done
  else
    stage "$s"
  fi
done
