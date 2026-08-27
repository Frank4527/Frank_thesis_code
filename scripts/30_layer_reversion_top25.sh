#!/usr/bin/env bash
# Layer-targeted direction reversion.
#
# Nothing is retrained. The stage-1 anchor and the stage-2 adapter come from the
# PRIMARY experiment, exactly as they are; the per-layer drift was measured from
# those same weights by Identification_and_Reversion/layer_drift.py.
#
#   1. layer_select.py  picks the layers at or above the 75th percentile of
#      rho_median = median_row ||sBA||/||W0||, per run, from that run's own drift
#   2. layer_sweep.py   applies beta = 0, 0.25, 0.5, 0.75, 1.0 to ONLY those layers
#      and scores both tasks; every other layer keeps the full trained weights
#
# Results go to _layer_reversion/<primary run>/ so no primary directory is written
# to: the primary experiment is read-only here.
set -u
R=/home/frank/runs/thesis_experiment_runs
CODE=/home/frank/code
PY=/home/frank/python-env/.venv/bin/python
OUTROOT=$R/_layer_reversion
QUANTILE=0.75
MODE=top
source $CODE/Thesis_Experiment/pipeline/gpu_pool.sh

# primary run : drift key : data_seed : OLD_DS : NEW_DS : merged anchor : stage-2 adapter
JOBS=(
  "Bankstatement2Quaterly_rep_rerun:b2q_seed1:1:bank_full:quarterly:base_bank_full_8b_seed0:stage2_quarterly_8b_seed0"
  "Bankstatement2Quaterly_rep_rerun_seed2:b2q_seed2:2:bank_full:quarterly:base_bank_full_8b_seed0:stage2_quarterly_8b_seed0"
  "Bankstatement2Quaterly_rep_rerun_seed3:b2q_seed3:3:bank_full:quarterly:base_bank_full_8b_seed0:stage2_quarterly_8b_seed0"
  "Quaterly_rep2Bankstatement_rerun:q2b_seed1:1:quarterly:bank_full:base_quarterly_8b_seed0:stage2_bank_full_8b_seed0"
  "Quaterly_rep2Bankstatement_rerun_seed2:q2b_seed2:2:quarterly:bank_full:base_quarterly_8b_seed0:stage2_bank_full_8b_seed0"
  "Quaterly_rep2Bankstatement_rerun_seed3:q2b_seed3:3:quarterly:bank_full:base_quarterly_8b_seed0:stage2_bank_full_8b_seed0"
)

mkdir -p $OUTROOT
for J in "${JOBS[@]}"; do
  IFS=: read -r RUN KEY DSEED OLDDS NEWDS MERGED ADP <<< "$J"
  OUT=$OUTROOT/$RUN
  export TIMINGS=$OUT/timings.jsonl
  export JOBLOG=$OUT/logs
  export MANIFEST_AUDIT=$OUT/manifest_access.tsv
  mkdir -p "$OUT" "$JOBLOG"
  BASE=$R/$RUN/merged/$MERGED
  ADAPTER=$R/$RUN/adapters/stage2/$ADP/adapter_last
  SEL=$OUT/selected_layers_${MODE}_q${QUANTILE}.json

  echo "=== [$(date -Is)] $RUN   drift=$KEY  seed=$DSEED  old=$OLDDS new=$NEWDS"
  for p in "$BASE/config.json" "$ADAPTER/adapter_model.safetensors" "$R/_layerdrift/layer_drift_${KEY}.json"; do
    [ -e "$p" ] || { echo "  MISSING $p -- skipping"; continue 2; }
  done

  if [ -f "$OUT/layer-${MODE}-dirsweep__both__val.json" ]; then
    echo "  already swept -- skipping"; continue
  fi
  $PY $CODE/Identification_and_Reversion/layer_select.py \
      --run $KEY --quantile $QUANTILE --mode $MODE --stat rho_median --out $SEL

  pool_submit "lr_${MODE}_${RUN}" \
    "cd $CODE && sg ucl -c \"cd $CODE && PYTHONPATH=$CODE DATA_SEED=$DSEED \
MANIFEST_AUDIT=$MANIFEST_AUDIT HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
$PY -u Thesis_Experiment/pipeline/layer_sweep.py \
  --base $BASE --adapter $ADAPTER --layers $SEL \
  --old $OLDDS --new $NEWDS --betas 0.0,0.25,0.5,0.75,1.0 --split validation \
  --out $OUT/layer-${MODE}-dirsweep__both__val.json \
  --progress $JOBLOG/lr_${MODE}_${RUN}.progress\""
done
pool_wait
echo "LAYER REVERSION DONE $(date -Is)" | tee $OUTROOT/DONE
