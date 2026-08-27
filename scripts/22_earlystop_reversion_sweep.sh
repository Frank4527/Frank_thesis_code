#!/usr/bin/env bash
# Reversion applied to the EARLY-STOPPED adapter of each run.
#
# The adapter is not hard-coded: it is re-derived at launch by es_rule.py from that
# run's own pass-1 scores, so it is by construction the checkpoint an in-loop run with
# patience 3 would have returned -- the best seen BEFORE the trigger, not the best in
# hindsight.
#
# Everything is written under <run>/results/es_reversion/ inside the _es_ directories.
# No fixed-budget experiment directory is read from except its merged stage-1 anchor,
# which is read-only, and none is written to.
#
#   direction sweep, 5 dials : the frontier, and its beta=0 endpoint is mag_only
#   magnitude at alpha=0     : dir_only, completing the four-version diagnostic
set -u
R=/home/frank/runs/thesis_experiment_runs
CODE=/home/frank/code
PY=/home/frank/python-env/.venv/bin/python
RULE=$CODE/Thesis_Experiment/pipeline/es_rule.py
source $CODE/Thesis_Experiment/pipeline/gpu_pool.sh

# run : data_seed : OLD_DS : NEW_DS : new_key : adapter_subdir : merged_base
JOBS=(
  "Bankstatement2Quaterly_es_seed3:3:bank_full:quarterly:quarterly_f1:stage2_quarterly_8b_seed0:Bankstatement2Quaterly_rep_rerun_seed3/merged/base_bank_full_8b_seed0"
  "Quaterly_rep2Bankstatement_es_seed2:2:quarterly:bank_full:bank_f1:stage2_bank_full_8b_seed0:Quaterly_rep2Bankstatement_rerun_seed2/merged/base_quarterly_8b_seed0"
  "Quaterly_rep2Bankstatement_es_seed3:3:quarterly:bank_full:bank_f1:stage2_bank_full_8b_seed0:Quaterly_rep2Bankstatement_rerun_seed3/merged/base_quarterly_8b_seed0"
  "Bankstatement2Quaterly_es_seed1:1:bank_full:quarterly:quarterly_f1:stage2_quarterly_8b_seed0:Bankstatement2Quaterly_rep_rerun/merged/base_bank_full_8b_seed0"
  "Bankstatement2Quaterly_es_seed2:2:bank_full:quarterly:quarterly_f1:stage2_quarterly_8b_seed0:Bankstatement2Quaterly_rep_rerun_seed2/merged/base_bank_full_8b_seed0"
  "Quaterly_rep2Bankstatement_es_seed1:1:quarterly:bank_full:bank_f1:stage2_bank_full_8b_seed0:Quaterly_rep2Bankstatement_rerun/merged/base_quarterly_8b_seed0"
)

for J in "${JOBS[@]}"; do
  IFS=: read -r RUN DSEED OLDDS NEWDS NEWKEY ADP BASEREL <<< "$J"
  BASE=$R/$BASEREL
  OUT=$R/$RUN/results/es_reversion
  export TIMINGS=$R/$RUN/timings_es_reversion.jsonl
  export JOBLOG=$R/$RUN/logs/jobs
  export MANIFEST_AUDIT=$R/$RUN/logs/manifest_access_es.tsv
  mkdir -p "$OUT" "$JOBLOG"

  ES=$($PY $RULE $R/$RUN/results/es_traj --new-key $NEWKEY --patience 3 \
        --pattern "step*__new__val.json" --print-best-step --quiet)
  CK=$R/$RUN/adapters/stage2/$ADP/hf/checkpoint-$ES
  if [ ! -d "$CK" ]; then echo "[$(date -Is)] $RUN: MISSING $CK -- skipping"; continue; fi
  echo "=== [$(date -Is)] $RUN  ES adapter = checkpoint-$ES  (old=$OLDDS new=$NEWDS seed=$DSEED)"
  echo "$ES" > $OUT/SELECTED_STEP.txt

  # direction sweep: 5 dials; beta=0 is mag_only (m_ft, D0)
  pool_submit "es_rev_dir_${RUN}" \
    "cd $CODE && sg ucl -c \"cd $CODE && PYTHONPATH=$CODE DATA_SEED=$DSEED \
MANIFEST_AUDIT=$MANIFEST_AUDIT HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
$PY -u Thesis_Experiment/pipeline/sweep.py --base $BASE --adapter $CK \
  --old $OLDDS --new $NEWDS --component direction --alphas 0.0,0.25,0.5,0.75,1.0 \
  --split validation --out $OUT/es-dirsweep__both__val.json \
  --progress $JOBLOG/es_rev_dir_${RUN}.progress\""

  # magnitude at alpha=0 only: dir_only (m0, D_ft), completing the four versions
  pool_submit "es_rev_mag0_${RUN}" \
    "cd $CODE && sg ucl -c \"cd $CODE && PYTHONPATH=$CODE DATA_SEED=$DSEED \
MANIFEST_AUDIT=$MANIFEST_AUDIT HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
$PY -u Thesis_Experiment/pipeline/eval_pair.py --base $BASE --adapter $CK \
  --old $OLDDS --new $NEWDS --split validation --alpha-mag 0.0 --beta-dir 1.0 \
  --config es_dir_only_step${ES} --out $OUT/es-dir_only__both__val.json \
  --progress $JOBLOG/es_rev_mag0_${RUN}.progress\""
done
pool_wait
echo "ES REVERSION SWEEPS DONE $(date -Is)" | tee $R/_es_train/SWEEP_DONE
