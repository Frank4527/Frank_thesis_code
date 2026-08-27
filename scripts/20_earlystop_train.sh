#!/usr/bin/env bash
# Retrain stage-2 for the three runs with a retained training history, saving a
# checkpoint every 10 optimizer steps.
#
# The FULL epoch budget is trained deliberately. The LR schedule is cosine over
# num_train_epochs, so truncating training would compress the schedule and put the
# checkpoints on a DIFFERENT trajectory than the fixed-budget run they are meant to
# be compared against. Training the full budget also means the final checkpoint must
# reproduce that run's adapter_last -- a fidelity check that costs nothing.
#
# In-loop dev evaluation is OFF (sentinel eval-every): at batch_size=2 a single dev
# eval on 26 pages took >25 min, because an under-trained model never emits EOS and
# generates to the token cap. Checkpoints are scored offline instead, at batch 16,
# four at a time on the pool.
set -u
R=/home/frank/runs/thesis_experiment_runs
CODE=/home/frank/code
PY=/home/frank/python-env/.venv/bin/python
SENTINEL=999999999

mkdir -p $R/_es_train
echo $$ > $R/_es_train/chain.pid

# name : source_run : data_seed : merged_base : new_ds_folder : out_run
JOBS=(
  "b2q_seed3:Bankstatement2Quaterly_rep_rerun_seed3:3:base_bank_full_8b_seed0:Quarterly_rep_En_seed3:Bankstatement2Quaterly_es_seed3:stage2_quarterly_8b_seed0"
  "q2b_seed2:Quaterly_rep2Bankstatement_rerun_seed2:2:base_quarterly_8b_seed0:Bank_Full_seed2:Quaterly_rep2Bankstatement_es_seed2:stage2_bank_full_8b_seed0"
  "q2b_seed3:Quaterly_rep2Bankstatement_rerun_seed3:3:base_quarterly_8b_seed0:Bank_Full_seed3:Quaterly_rep2Bankstatement_es_seed3:stage2_bank_full_8b_seed0"
  "b2q_seed1:Bankstatement2Quaterly_rep_rerun:1:base_bank_full_8b_seed0:Quarterly_rep_En_seed1:Bankstatement2Quaterly_es_seed1:stage2_quarterly_8b_seed0"
  "b2q_seed2:Bankstatement2Quaterly_rep_rerun_seed2:2:base_bank_full_8b_seed0:Quarterly_rep_En_seed2:Bankstatement2Quaterly_es_seed2:stage2_quarterly_8b_seed0"
  "q2b_seed1:Quaterly_rep2Bankstatement_rerun:1:base_quarterly_8b_seed0:Bank_Full_seed1:Quaterly_rep2Bankstatement_es_seed1:stage2_bank_full_8b_seed0"
)

for J in "${JOBS[@]}"; do
  IFS=: read -r NAME SRC DSEED MERGED NEWDS OUTRUN OUTADP <<< "$J"
  BASE=$R/$SRC/merged/$MERGED
  TRAIN=/home/frank/data/Thesis_datasets/$NEWDS/training/manifest.jsonl
  DEV=/home/frank/data/Thesis_datasets/$NEWDS/validation/manifest.jsonl
  OUT=$R/$OUTRUN/adapters/stage2/$OUTADP

  if [ -d "$OUT/adapter_last" ]; then
    echo "[$(date -Is)] SKIP $NAME (already trained)"; continue
  fi
  for p in "$BASE/config.json" "$TRAIN" "$DEV"; do
    [ -e "$p" ] || { echo "[$(date -Is)] $NAME MISSING $p -- skipping"; continue 2; }
  done

  for i in $(seq 1 240); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break
    sleep 30
  done
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && \
    { echo "[$(date -Is)] GPUs busy 2h -- aborting"; exit 1; }

  mkdir -p "$OUT" "$R/$OUTRUN/logs"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "  [DRY] would train $NAME  DATA_SEED=$DSEED -> $OUT"
    continue
  fi
  echo "[$(date -Is)] START $NAME  DATA_SEED=$DSEED  -> $OUT"
  t0=$(date +%s)
  sg ucl -c "cd $CODE && PYTHONPATH=$CODE TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 DATA_SEED=$DSEED \
$PY -m torch.distributed.run --nproc_per_node=4 --master_port=29555 \
  Training_Dora/train_dora_ddp.py \
  --model $BASE --task extraction --train $TRAIN --dev $DEV --out $OUT \
  --rank 16 --lora-alpha 32 --lr 2e-4 --epochs 10 \
  --batch-size 1 --grad-accum 4 --workers 4 --max-pixels 589824 --seed 0 \
  --eval-every $SENTINEL --save-steps 10 --save-only-model" \
  > "$R/$OUTRUN/logs/es_train_${NAME}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[$(date -Is)] END $NAME rc=$rc  wall=$(( (t1-t0)/60 )) min  ckpts=$(ls -d $OUT/hf/checkpoint-* 2>/dev/null | wc -l)"
done

echo "ES STAGE-2 RETRAINING DONE $(date -Is)" | tee $R/_es_train/DONE
