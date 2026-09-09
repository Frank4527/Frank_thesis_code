#!/usr/bin/env bash
# Pretrained-ability recovery for the bank stage-1 adapters of splits 2 and 3.
# Same code path as split 1; OCRBench and CORD skipped, so each dial is scored on
# MMBench-dev-EN, DocVQA-val and bank validation only.
set -u
C=/home/frank/code
PY=/home/frank/python-env/.venv/bin/python
R=/home/frank/runs/thesis_experiment_runs

run_phase () {            # $1=seed $2=script $3=extra-flag
  local seed=$1 script=$2 flag=$3
  local adp=$R/Bankstatement2Quaterly_rep_rerun_seed${seed}/adapters/stage1/bank_full_8b_seed0/adapter_last
  local out=$R/_pretrained_ability_recovery_seed${seed}
  mkdir -p "$out/results" "$out/logs"
  local i=0
  for dials in "b0.00" "b0.25" "b0.50,b1.00" "b0.75,a0.00"; do
    nohup sg ucl -c "cd $C && PYTHONPATH=$C HF_HUB_OFFLINE=1 \
PA_SEED=$seed DATA_SEED=$seed PA_ADAPTER=$adp PA_RES=$out/results \
CUDA_VISIBLE_DEVICES=$i PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -u $C/Pretrained_Ability/$script --dials $dials --device cuda:0 --batch 4 $flag" \
      > "$out/logs/$(basename $script .py)_gpu${i}.log" 2>&1 &
    i=$((i+1))
  done
  wait
  echo "[$(date -Is)] seed $seed $script done"
}

for seed in 2 3; do
  echo "[$(date -Is)] === seed $seed: MMBench + bank ==="
  run_phase $seed sweep_seed.py --skip-ocr
  echo "[$(date -Is)] === seed $seed: DocVQA ==="
  run_phase $seed sweep_docvqa_seed.py --skip-cord
done
echo "PA SEEDS 2,3 DONE $(date -Is)" | tee $R/_pretrained_ability_recovery_seed3/DONE
