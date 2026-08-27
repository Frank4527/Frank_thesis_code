#!/usr/bin/env bash
# The ONLY file you edit to point this experiment at different data.
# Everything else (run_all.sh, the pipeline drivers) is dataset-agnostic.

EXP_NAME=Bankstatement2Quaterly_rep_rerun

OLD_DS=bank_full          # stage-1 task, the one that gets forgotten   (registry name)
NEW_DS=quarterly          # stage-2 task, the one newly learned         (registry name)

BASE_MODEL=/home/frank/models/Qwen3-VL-8B-Instruct
SEED=0                    # TRAINING seed (init/shuffle)
DATA_SEED=1               # DATA SPLIT seed: which random re-split to use (1|2|3)
export DATA_SEED          # every child process resolves manifests through this

# identical for EVERY training run in this experiment, so cost comparisons are fair
RANK=16
LORA_ALPHA=32
LR=2e-4
EPOCHS=10
BATCH=1
GRAD_ACCUM=4
MAX_PIXELS=589824
WORKERS=4
NGPU=4                    # all four, one job at a time

REHEARSAL_RATIOS="0.01 0.05"   # the two new replay baselines (1% and 5% of the old task)

# fixed paths derived from the above — do not edit
CODE=/home/frank/code
PY=/home/frank/python-env/.venv/bin/python
RUNS=/home/frank/runs/thesis_experiment_runs
EXP=$RUNS/$EXP_NAME

# every manifest this experiment opens is appended here, so seed isolation can be
# checked after the fact instead of assumed
export MANIFEST_AUDIT=$EXP/logs/manifest_access.tsv
