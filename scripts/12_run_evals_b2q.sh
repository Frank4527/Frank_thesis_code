#!/usr/bin/env bash
# Phase 2: every evaluation for this experiment, one at a time, each timed.
# Nothing here is dataset-specific — the names come from config.sh.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

export TIMINGS=$EXP/timings.jsonl
export JOBLOG=$EXP/logs/jobs
source "$CODE/Thesis_Experiment/pipeline/jobrunner.sh"
source "$CODE/Thesis_Experiment/pipeline/gpu_pool.sh"
POOL_GPUS=${POOL_GPUS:-"0 1 2 3"}
export PYTHONPATH=$CODE
export PY CODE OLD_DS NEW_DS EXP SEED DATA_SEED MANIFEST_AUDIT BASE_MODEL   # visible to pool subshells

# PHASE=val  : anchors + the three dial sweeps + baselines on validation
# PHASE=test : the test set, using dials SELECTED from this run's own sweeps
# Splitting them is deliberate: the test evals below carry placeholder dials until
# the sweeps have been read, and running them early would write thesis numbers at
# an alpha chosen on a different experiment's data.
PHASE=${1:-val}

# ---- SELECTED DIALS: set these from THIS seed's own validation sweeps -------
# Placeholders until the sweeps land. run_evals.sh test refuses to run while they
# are still unset, so no thesis number is ever written at another seed's alpha.
DIR_B=${DIR_B:-unset}      # beta for direction-only reversion
MAG_A=${MAG_A:-unset}      # alpha for magnitude-only reversion
WISEFT_C=${WISEFT_C:-unset}  # WiSE-FT interpolation coefficient

WBANK=$EXP/merged/base_${OLD_DS}_8b_seed${SEED}
S2=$EXP/adapters/stage2/stage2_${NEW_DS}_8b_seed${SEED}/adapter_last
R1=$EXP/results/stage1
R2=$EXP/results/stage2
RB=$EXP/results/baselines
mkdir -p "$R1" "$R2" "$RB" "$JOBLOG"

N_OLD_VAL=$($PY -c "import sys;sys.path.insert(0,'$CODE');from Thesis_Experiment.datasets import n;print(n('$OLD_DS','validation'))")
N_OLD_TEST=$($PY -c "import sys;sys.path.insert(0,'$CODE');from Thesis_Experiment.datasets import n;print(n('$OLD_DS','test'))")

pair () {   # $1=name $2=base $3=adapter|none $4=alpha_mag $5=beta_dir $6=split $7=outfile $8=n_items
  local ADP=""; [ "$3" != "none" ] && ADP="--adapter $3"
  pool_submit "$1" "cd $CODE && sg ucl -c \"cd $CODE && PYTHONPATH=$CODE DATA_SEED=$DATA_SEED MANIFEST_AUDIT=$MANIFEST_AUDIT HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
    $PY -u Thesis_Experiment/pipeline/eval_pair.py --base $2 $ADP \
      --old $OLD_DS --new $NEW_DS --split $6 --alpha-mag $4 --beta-dir $5 \
      --config $1 --out $7 --progress $JOBLOG/$1.progress\""
}

sweep () {  # $1=name $2=component(direction|magnitude|wiseft) $3=dial values $4=split $5=outfile
  pool_submit "$1" "cd $CODE && sg ucl -c \"cd $CODE && PYTHONPATH=$CODE DATA_SEED=$DATA_SEED MANIFEST_AUDIT=$MANIFEST_AUDIT HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
    $PY -u Thesis_Experiment/pipeline/sweep.py --base $WBANK --adapter $S2 \
      --old $OLD_DS --new $NEW_DS --component $2 --alphas $3 --split $4 \
      --out $5 --progress $JOBLOG/$1.progress\""
}

wiseft_eval () {  # $1=name $2=coeff $3=split $4=outfile
  pool_submit "$1" "cd $CODE && sg ucl -c \"cd $CODE && PYTHONPATH=$CODE DATA_SEED=$DATA_SEED MANIFEST_AUDIT=$MANIFEST_AUDIT HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
    $PY -u Thesis_Experiment/pipeline/eval_pair.py --base $WBANK --adapter $S2 \
      --old $OLD_DS --new $NEW_DS --split $3 --wiseft $2 \
      --config $1 --out $4 --progress $JOBLOG/$1.progress\""
}

if [ "$PHASE" = "val" ] || [ "$PHASE" = "all" ]; then
# ---- validation: the anchors, then the sweeps that select each operating point ----
pair  wbank__val_rerun        "$WBANK" none "1.0" "1.0" validation "$R1/wbank__both__val_rerun.json"        "$N_OLD_VAL"
pair  stage2__val_rerun       "$WBANK" "$S2" "1.0" "1.0" validation "$R2/stage2__both__val_rerun.json"      "$N_OLD_VAL"
sweep sweep_direction_rerun   direction "0.0,0.25,0.5,0.75,1.0" validation "$R2/stage2-dirsweep__both__val_rerun.json"
sweep sweep_magnitude_rerun   magnitude "0.0,0.25,0.5,0.75,1.0" validation "$R2/stage2-magsweep__both__val_rerun.json"
# WiSE-FT baseline: whole-weight interpolation, its own dial selected on validation
sweep sweep_wiseft_rerun      wiseft    "0.0,0.25,0.5,0.75,1.0" validation "$R2/wiseft-sweep__both__val_rerun.json"

# ---- the two NEW baselines on validation ----
for TAG in joint rehearsal001 rehearsal005; do
  ADP=$EXP/adapters/baselines/${TAG}_8b_seed${SEED}/adapter_last
  BASE=$WBANK; [ "$TAG" = "joint" ] && BASE=$BASE_MODEL     # joint starts from the pretrained model
  pair "baseline_${TAG}__val_rerun"  "$BASE" "$ADP" "1.0" "1.0" validation "$RB/${TAG}__both__val_rerun.json"  "$N_OLD_VAL"
done
pool_wait
echo "[$(date -Is)] VALIDATION PHASE COMPLETE - read the sweeps, set the dials, then: run_evals.sh test"
fi

if [ "$PHASE" = "test" ] || [ "$PHASE" = "all" ]; then
for v in DIR_B MAG_A WISEFT_C; do
  if [ "${!v}" = "unset" ]; then
    echo "REFUSING to run test evals: $v is still a placeholder."
    echo "Read this seed's validation sweeps, then rerun with e.g.:"
    echo "  DIR_B=0.5 MAG_A=0.0 WISEFT_C=0.75 bash run_evals.sh test"
    exit 1
  fi
done
for TAG in joint rehearsal001 rehearsal005; do
  ADP=$EXP/adapters/baselines/${TAG}_8b_seed${SEED}/adapter_last
  BASE=$WBANK; [ "$TAG" = "joint" ] && BASE=$BASE_MODEL
  pair "baseline_${TAG}__test_rerun" "$BASE" "$ADP" "1.0" "1.0" test       "$RB/${TAG}__both__test_rerun.json" "$N_OLD_TEST"
done

# ---- test set: ceiling, no-reversion, and the selected reversion points ----
pair  wbank__test_rerun       "$WBANK" none  "1.0" "1.0" test "$R1/wbank__both__test_rerun.json"          "$N_OLD_TEST"
pair  stage2__test_rerun      "$WBANK" "$S2" "1.0" "1.0" test "$R2/stage2__both__test_rerun.json"         "$N_OLD_TEST"
# NOTE: the dials below are placeholders. After the sweeps land, set them to this rerun's own
# argmax (max average on validation) before running these three.
pair  dir_selected__test_rerun "$WBANK" "$S2" "1.0" "$DIR_B" test "$R2/stage2-dir__both__test_rerun.json"     "$N_OLD_TEST"
pair  mag_selected__test_rerun "$WBANK" "$S2" "$MAG_A" "1.0" test "$R2/stage2-mag__both__test_rerun.json"     "$N_OLD_TEST"
pair  combined__test_rerun     "$WBANK" "$S2" "$MAG_A" "$DIR_B" test "$R2/stage2-dirmag__both__test_rerun.json" "$N_OLD_TEST"
# WiSE-FT at its validation-selected coefficient (PLACEHOLDER - set from this seed's own sweep)
wiseft_eval "wiseft_selected__test_rerun" "$WISEFT_C" test "$R2/wiseft__both__test_rerun.json"

pool_wait
fi
echo "[$(date -Is)] EVAL PHASE ($PHASE) COMPLETE"
