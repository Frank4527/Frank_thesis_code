#!/usr/bin/env bash
# Early-stopping analysis, two passes, both through pipeline/eval_pair.py.
#
#  PASS 1  --skip-old : score the NEW task on checkpoints in blocks, stopping the
#                       moment the patience-3 rule fires. This is the only thing the
#                       stopping rule may see, so the old-task number does not yet
#                       exist when the checkpoint is chosen.
#  PASS 2  --skip-new : score the OLD task on checkpoints up to the selected point,
#                       completing the trade-off trajectory without re-running the
#                       new task, and without touching checkpoints past the ES point.
#
# Both passes use the SAME script, the same reconstruction path (--alpha-mag 1.0
# --beta-dir 1.0 -> VERSION_FNS["full"]) and the same cap/batch (evalcore defaults
# 4096/16) as every other evaluation in this project, including the section-6
# checkpoint evals. The two scoring CALLS inside eval_pair.py are untouched, so a
# number produced here is identical to one from a full both-task run.
set -u
R=/home/frank/runs/thesis_experiment_runs
CODE=/home/frank/code
PY=/home/frank/python-env/.venv/bin/python
BLOCK=8
source $CODE/Thesis_Experiment/pipeline/gpu_pool.sh

JOBS=(
  "Bankstatement2Quaterly_es_seed3:3:bank_full:quarterly:quarterly_f1:stage2_quarterly_8b_seed0:Bankstatement2Quaterly_rep_rerun_seed3/merged/base_bank_full_8b_seed0"
  "Quaterly_rep2Bankstatement_es_seed2:2:quarterly:bank_full:bank_f1:stage2_bank_full_8b_seed0:Quaterly_rep2Bankstatement_rerun_seed2/merged/base_quarterly_8b_seed0"
  "Quaterly_rep2Bankstatement_es_seed3:3:quarterly:bank_full:bank_f1:stage2_bank_full_8b_seed0:Quaterly_rep2Bankstatement_rerun_seed3/merged/base_quarterly_8b_seed0"
  "Bankstatement2Quaterly_es_seed1:1:bank_full:quarterly:quarterly_f1:stage2_quarterly_8b_seed0:Bankstatement2Quaterly_rep_rerun/merged/base_bank_full_8b_seed0"
  "Bankstatement2Quaterly_es_seed2:2:bank_full:quarterly:quarterly_f1:stage2_quarterly_8b_seed0:Bankstatement2Quaterly_rep_rerun_seed2/merged/base_bank_full_8b_seed0"
  "Quaterly_rep2Bankstatement_es_seed1:1:quarterly:bank_full:bank_f1:stage2_bank_full_8b_seed0:Quaterly_rep2Bankstatement_rerun/merged/base_quarterly_8b_seed0"
)

submit () {   # $1=name $2=ckpt_dir $3=extra_flag $4=outfile $5..=context
  pool_submit "$1" \
    "cd $CODE && sg ucl -c \"cd $CODE && PYTHONPATH=$CODE DATA_SEED=$DSEED \
MANIFEST_AUDIT=$MANIFEST_AUDIT HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES \
$PY -u Thesis_Experiment/pipeline/eval_pair.py \
  --base $BASE --adapter $2 \
  --old $OLDDS --new $NEWDS --split validation --alpha-mag 1.0 --beta-dir 1.0 $3 \
  --config $1 --out $4 --progress $JOBLOG/$1.progress\""
}

for J in "${JOBS[@]}"; do
  IFS=: read -r RUN DSEED OLDDS NEWDS NEWKEY ADP BASEREL <<< "$J"
  BASE=$R/$BASEREL
  CK=$R/$RUN/adapters/stage2/$ADP/hf
  OUT=$R/$RUN/results/es_traj
  export TIMINGS=$R/$RUN/timings_es_traj.jsonl
  export JOBLOG=$R/$RUN/logs/jobs
  export MANIFEST_AUDIT=$R/$RUN/logs/manifest_access_es.tsv
  mkdir -p "$OUT" "$JOBLOG"
  echo "=== [$(date -Is)] $RUN  old=$OLDDS new=$NEWDS  DATA_SEED=$DSEED"

  ALL=($(ls -d $CK/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n))

  # ---------- PASS 1: new task only, block-wise, stop when the rule fires ----------
  echo "--- pass 1: NEW task ($NEWDS), selection"
  i=0
  while [ $i -lt ${#ALL[@]} ]; do
    for j in $(seq $i $((i + BLOCK - 1))); do
      [ $j -ge ${#ALL[@]} ] && break
      c=${ALL[$j]}
      [ -f "$OUT/step${c}__new__val.json" ] && continue
      submit "es_new_${RUN}_${c}" "$CK/checkpoint-$c" "--skip-old" "$OUT/step${c}__new__val.json"
    done
    pool_wait
    if $PY $CODE/Thesis_Experiment/pipeline/es_rule.py "$OUT" --new-key $NEWKEY \
         --patience 3 --pattern "step*__new__val.json"; then
      echo "  rule fired"; break
    fi
    i=$((i + BLOCK))
  done

  # ---------- PASS 2: old task only, on checkpoints up to the selected point -------
  ES=$($PY $CODE/Thesis_Experiment/pipeline/es_rule.py "$OUT" --new-key $NEWKEY \
        --patience 3 --pattern "step*__new__val.json" --print-best-step --quiet)
  echo "--- pass 2: OLD task ($OLDDS), checkpoints up to the selected step $ES"
  for c in "${ALL[@]}"; do
    [ "$c" -gt "$ES" ] && break
    [ -f "$OUT/step${c}__old__val.json" ] && continue
    submit "es_old_${RUN}_${c}" "$CK/checkpoint-$c" "--skip-new" "$OUT/step${c}__old__val.json"
  done
  pool_wait
  echo "=== [$(date -Is)] $RUN done: new=$(ls $OUT/step*__new__val.json 2>/dev/null | wc -l) old=$(ls $OUT/step*__old__val.json 2>/dev/null | wc -l)  ES step=$ES"
done
echo "ES TRAJECTORY DONE $(date -Is)" | tee $R/_es_train/TRAJ_DONE
