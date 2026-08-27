#!/usr/bin/env bash
# BOTTOM CONTROL for the layer-targeted study.
#
# Identical to _layer_reversion_chain.sh and _layer_reversion_random_chain.sh except
# that the 63 layers are the LEAST-drifted quartile instead of the most-drifted or a
# random draw. Layer COUNT, dials, adapters and evaluation are all held constant, so
# any difference between the three arms is attributable to which layers were chosen.
#
# What this arm is for. The matched-beta comparison of top-25% against random-25%
# showed the drift ranking IS informative but TASK-AGNOSTIC: selecting by drift
# recovers more of the old task (+0.12 at beta=0) AND destroys more of the new task
# (-0.17), because the same layers carry both. Bottom-25% tests that account with a
# signed prediction made in advance:
#
#   if the ranking is task-agnostic  -> bottom-25% should be the MIRROR of top-25%:
#                                       LESS old-task recovery and LESS new-task
#                                       damage than random, at every beta
#   if the ranking is uninformative  -> bottom-25% should look like random-25%,
#                                       and the flat-drift explanation wins instead
#
# No experiment run so far discriminates between those two accounts.
#
# Nothing is retrained. The stage-1 anchor and the stage-2 adapter come from the
# PRIMARY experiment, read-only; per-layer drift was measured from those same
# weights by Identification_and_Reversion/layer_drift.py.
set -u
R=/home/frank/runs/thesis_experiment_runs
CODE=/home/frank/code
PY=/home/frank/python-env/.venv/bin/python
OUTROOT=$R/_layer_reversion
QUANTILE=0.75
MODE=bottom
LABEL=bottom          # used in filenames; bottom mode ignores --seed
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
  SEL=$OUT/selected_layers_${LABEL}_q${QUANTILE}.json

  echo "=== [$(date -Is)] $RUN   drift=$KEY  seed=$DSEED  old=$OLDDS new=$NEWDS"
  for p in "$BASE/config.json" "$ADAPTER/adapter_model.safetensors" "$R/_layerdrift/layer_drift_${KEY}.json"; do
    [ -e "$p" ] || { echo "  MISSING $p -- skipping"; continue 2; }
  done

  if [ -f "$OUT/layer-${LABEL}-dirsweep__both__val.json" ]; then
    echo "  already swept -- skipping"; continue
  fi
  $PY $CODE/Identification_and_Reversion/layer_select.py \
      --run $KEY --quantile $QUANTILE --mode $MODE --stat rho_median --out $SEL

  pool_submit "lr_${LABEL}_${RUN}" \
    "cd $CODE && sg ucl -c \"cd $CODE && PYTHONPATH=$CODE DATA_SEED=$DSEED \
MANIFEST_AUDIT=$MANIFEST_AUDIT HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
$PY -u Thesis_Experiment/pipeline/layer_sweep.py \
  --base $BASE --adapter $ADAPTER --layers $SEL \
  --old $OLDDS --new $NEWDS --betas 0.0,0.25,0.5,0.75,1.0 --split validation \
  --out $OUT/layer-${LABEL}-dirsweep__both__val.json \
  --progress $JOBLOG/lr_${LABEL}_${RUN}.progress\""
done
pool_wait
echo "LAYER REVERSION (BOTTOM CONTROL) DONE $(date -Is)" | tee $OUTROOT/DONE_bottom
