#!/usr/bin/env bash
# Rerun of the bank -> quarterly continual-learning experiment: EVERY train and eval
# on the "document" prompt, plus two new baselines (joint training, rehearsal 1%/5%).
#
# Runs ONE job at a time on ALL 4 GPUs and records what each cost into timings.jsonl.
# Idempotent: a job already recorded "ok" is skipped, so this can be relaunched after
# a reboot or a kill and it resumes where it stopped.
#
#   bash run_all.sh train     # phase 1: mixtures + 4 trainings + 1 merge (the long pole)
#   bash run_all.sh eval      # phase 2: all evaluations
#   bash run_all.sh all
#
# Deliberate deviation from the first b2q run: in-loop dev evaluation is DISABLED
# (--eval-every huge). Every run therefore trains the full $EPOCHS with no early stop,
# so (a) training cost is pure training and comparable across methods, and (b) no
# method is cut short by a noisy 36-document dev metric. We use adapter_last, as before.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

export TIMINGS=$EXP/timings.jsonl
export JOBLOG=$EXP/logs/jobs
source "$CODE/Thesis_Experiment/pipeline/jobrunner.sh"

PHASE=${1:-all}
export PYTHONPATH=$CODE
NO_INLOOP_EVAL=999999999

# fail fast rather than silently training on the wrong split
$PY -c "
import os,sys; sys.path.insert(0,'$CODE')
os.environ['DATA_SEED']='$DATA_SEED'
from Thesis_Experiment.datasets import folder
print('  SEED SANITY: DATA_SEED=$DATA_SEED MANIFEST_AUDIT=$MANIFEST_AUDIT ->', folder('$OLD_DS'), '/', folder('$NEW_DS'))
" || { echo "DATA_SEED=$DATA_SEED MANIFEST_AUDIT=$MANIFEST_AUDIT is not usable"; exit 1; }

mkdir -p "$EXP"/{adapters/{stage1,stage2,baselines},merged,logs/{stage1,stage2,baselines,jobs},results/{stage1,stage2,baselines},data}

WBANK=$EXP/merged/base_${OLD_DS}_8b_seed${SEED}
S1_OUT=$EXP/adapters/stage1/${OLD_DS}_8b_seed${SEED}
S2_OUT=$EXP/adapters/stage2/stage2_${NEW_DS}_8b_seed${SEED}

dspath () { $PY -c "
import sys; sys.path.insert(0,'$CODE')
from Thesis_Experiment.datasets import manifest; print(manifest('$1','$2'))"; }
n_docs () { $PY -c "
import sys; sys.path.insert(0,'$CODE')
from Thesis_Experiment.datasets import n; print(n('$1','training'))"; }

OLD_TRAIN=$(dspath $OLD_DS training); OLD_VAL=$(dspath $OLD_DS validation)
NEW_TRAIN=$(dspath $NEW_DS training); NEW_VAL=$(dspath $NEW_DS validation)

# one training invocation, parameterised by (base model, train manifest, out dir, dev manifest)
train_cmd () {
  echo "cd $CODE && sg ucl -c \"cd $CODE && \
PYTHONPATH=$CODE TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
$PY -m torch.distributed.run --nproc_per_node=$NGPU --master_port=29551 \
  Training_Dora/train_dora_ddp.py \
  --model $1 --task extraction --train $2 --dev $4 --out $3 \
  --rank $RANK --lora-alpha $LORA_ALPHA --lr $LR --epochs $EPOCHS \
  --batch-size $BATCH --grad-accum $GRAD_ACCUM --workers $WORKERS \
  --max-pixels $MAX_PIXELS --seed $SEED --eval-every $NO_INLOOP_EVAL\""
}

if [ "$PHASE" = "train" ] || [ "$PHASE" = "all" ]; then

# 0. mixture manifests for the two new baselines (CPU, seconds)
run_job build_mixtures prep null 0 <<CMD
cd $CODE && PYTHONPATH=$CODE DATA_SEED=$DATA_SEED MANIFEST_AUDIT=$MANIFEST_AUDIT $PY -c "
from Thesis_Experiment.datasets import make_mixture
import json
out = {}
for ratio, tag in [(1.0, 'joint'), (0.01, 'rehearsal001'), (0.05, 'rehearsal005')]:
    n_new, n_old, p = make_mixture('$NEW_DS', '$OLD_DS', ratio, '$EXP/data/train_%s.jsonl' % tag, seed=$SEED)
    out[tag] = {'n_new': n_new, 'n_old_replayed': n_old, 'total': n_new + n_old, 'path': p}
    print(tag, out[tag])
json.dump(out, open('$EXP/data/mixtures.json','w'), indent=2)
"
CMD

# 1. stage-1: the OLD task, from the pretrained base
run_job train_stage1_${OLD_DS} train "$(n_docs $OLD_DS)" $NGPU <<CMD
$(train_cmd "$BASE_MODEL" "$OLD_TRAIN" "$S1_OUT" "$OLD_VAL")
CMD

# 2. merge stage-1 into full weights -> W_bank, the anchor everything reverts toward
run_job merge_wbank merge null 1 <<CMD
cd $CODE && PYTHONPATH=$CODE HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  $PY Thesis_Experiment/sequential/merge_adapter.py \
    --model $BASE_MODEL --adapter $S1_OUT/adapter_last --out $WBANK --device cuda:0
CMD

# 3. stage-2: the NEW task on top of W_bank — this is what causes the forgetting
run_job train_stage2_${NEW_DS} train "$(n_docs $NEW_DS)" $NGPU <<CMD
$(train_cmd "$WBANK" "$NEW_TRAIN" "$S2_OUT" "$NEW_VAL")
CMD

# 4. joint baseline: NOT retrained here. It is order-free -- the same 1043 documents
#    (598 bank + 445 quarterly) trained simultaneously from the pretrained base -- so the
#    adapter and its scores are identical to Bankstatement2Quaterly_rep_rerun, whose result
#    files are copied into results/baselines/. Re-enable below for an independent 2nd sample.
# run_job train_baseline_joint ... (disabled)
# 5. NEW BASELINE — rehearsal: stage-2 with r of the old task replayed alongside
for R in $REHEARSAL_RATIOS; do
  PCT=$($PY -c "print(int(round(float('$R')*100)))")
  TAG=$(printf 'rehearsal%03d' "$PCT")
  MIX=$EXP/data/train_${TAG}.jsonl
  run_job train_baseline_${TAG} train "$(wc -l < $MIX)" $NGPU <<CMD
$(train_cmd "$WBANK" "$MIX" "$EXP/adapters/baselines/${TAG}_8b_seed${SEED}" "$NEW_VAL")
CMD
done

echo "[$(date -Is)] TRAINING PHASE COMPLETE"
fi

if [ "$PHASE" = "eval" ] || [ "$PHASE" = "all" ]; then
  bash "$HERE/run_evals.sh"
fi

touch "$EXP/PHASE_${PHASE}_DONE"
echo "[$(date -Is)] run_all.sh ($PHASE) finished"
