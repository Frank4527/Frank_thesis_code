#!/usr/bin/env bash
# Pretrained-ability recovery sweep: six dials over four GPUs.
#
# The base dial is NOT re-run: Step 0 already measured the pretrained model on
# OCRBench (725/1000), MMBench (88.80%) and bank validation (field F1 0.0025).
#
# Dials are paired so the GPUs finish together. Bank evaluation is slow at LOW beta
# (the model is near-pretrained, does not know the schema, and generates to the token
# cap without emitting EOS) and fast at high beta, so the low dials run alone.
set -u
C=/home/frank/code
PY=/home/frank/python-env/.venv/bin/python
S=/home/frank/runs/thesis_experiment_runs/_pretrained_ability_recovery
mkdir -p "$S/logs"

launch () {   # $1=gpu  $2=comma-separated dials
  local gpu=$1 dials=$2
  echo "[$(date -Is)] GPU$gpu -> $dials"
  nohup sg ucl -c "cd $C && PYTHONPATH=$C HF_HUB_OFFLINE=1 DATA_SEED=1 \
CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -u $C/Pretrained_Ability/sweep.py --dials $dials --device cuda:0 --batch 4" \
    > "$S/logs/sweep_gpu${gpu}.log" 2>&1 &
}

launch 0 "b0.00"
launch 1 "b0.25"
launch 2 "b0.50,b1.00"
launch 3 "b0.75,a0.00"

echo "launched; 6 dials over 4 GPUs"
