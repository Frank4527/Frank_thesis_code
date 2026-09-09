#!/usr/bin/env bash
# Pretrained backbone on bank validation, splits 1-3. Bank only, no other benchmark.
# Split 1 is the CONTROL: it must reproduce field_f1 0.0025 from
# _pretrain_reversion/step0_base.json, whose script no longer exists on the box.
set -u
C=/home/frank/code
PY=/home/frank/python-env/.venv/bin/python
R=/home/frank/runs/thesis_experiment_runs
OUT=$R/_pa_base_bank
declare -A ADP=(
  [1]=$R/Bankstatement2Quaterly_rep_rerun/adapters/stage1/bank_full_8b_seed0/adapter_last
  [2]=$R/Bankstatement2Quaterly_rep_rerun_seed2/adapters/stage1/bank_full_8b_seed0/adapter_last
  [3]=$R/Bankstatement2Quaterly_rep_rerun_seed3/adapters/stage1/bank_full_8b_seed0/adapter_last
)
i=0
for seed in 1 2 3; do
  mkdir -p "$OUT/seed$seed"
  nohup sg ucl -c "cd $C && PYTHONPATH=$C HF_HUB_OFFLINE=1 \
PA_SEED=$seed DATA_SEED=$seed PA_ADAPTER=${ADP[$seed]} PA_RES=$OUT/seed$seed \
CUDA_VISIBLE_DEVICES=$i PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEV=cuda:0 \
$PY -u $C/Pretrained_Ability/base_bank_only.py" > "$OUT/seed$seed/run.log" 2>&1 &
  i=$((i+1))
done
wait
echo "PA BASE BANK DONE $(date -Is)" | tee $OUT/DONE
