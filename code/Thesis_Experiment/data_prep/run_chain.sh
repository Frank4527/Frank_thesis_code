#!/usr/bin/env bash
# Run the remaining seed-2 and seed-3 pipeline end to end, unattended.
#
# Strictly sequential: never two GPU jobs at once, because training cost is a
# reported result and a contended wall-clock is meaningless.
#
# Order -- q2b of a seed depends on b2q of the SAME seed for the joint baseline:
#   b2q_seed2  train -> val -> select -> test
#   q2b_seed2  train -> val -> select -> test -> copy_joint
#   b2q_seed3  train -> val -> select -> test
#   q2b_seed3  train -> val -> select -> test -> copy_joint
#
# RETRIES TRAINING. The box is shared: on 2026-08-05 another user's job took
# 13.9 GB on GPU 1 mid-run and all three baseline trainings died with CUDA OOM.
# The previous version of this script then waited forever for adapters that were
# never going to appear, and the GPUs sat idle for nine hours. Training is now
# retried, and every wait is bounded.
#
# Memory footprint is deliberately NOT reduced to avoid OOM: batch size, grad
# accumulation and max_pixels must stay identical to seed 1 or the runs stop
# being comparable. Retrying is the only fix that preserves the experiment.
set -u
R=/home/frank/runs/thesis_experiment_runs
C=/home/frank/code/Thesis_Experiment
SEL=$C/data_prep/select_dials.py
CJ=$C/data_prep/copy_joint.sh
CHAINLOG=$R/CHAIN.log
MAX_TRAIN_ATTEMPTS=3
COOLDOWN=600          # give a foreign job time to finish before retrying

say () { echo "[$(date -Is)] $*" | tee -a "$CHAINLOG"; }

echo $$ > "$R/chain.pid"

PAUSE=$R/PAUSE
check_pause () {
  if [ -e "$PAUSE" ]; then
    say "PAUSED -- stopping at a clean boundary. Resume: bash $C/data_prep/resume_chain.sh"
    rm -f "$R/chain.pid"
    exit 0
  fi
}

# ---------------------------------------------------------------- guards
gpus_busy () { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

foreign_gpu_user () {   # prints other users holding GPU memory, if any
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r p; do
    p=$(echo "$p" | tr -d ' '); [ -n "$p" ] || continue
    o=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$o" ] && [ "$o" != "$(whoami)" ] && echo "$o"
  done | sort -u | paste -sd, -
}

wait_for_gpus () {   # until nothing at all is computing
  local waited=0
  while gpus_busy; do
    check_pause
    sleep 60; waited=$((waited+60))
    if [ $((waited % 1800)) -eq 0 ]; then
      say "  still waiting for GPUs (${waited}s); foreign users: [$(foreign_gpu_user)]"
    fi
  done
  [ $waited -gt 0 ] && say "  GPUs clear after ${waited}s"
  return 0
}

wait_for_own_training () {   # never launch a second run_all.sh on top of a live one
  local waited=0
  while pgrep -f "run_all[.]sh" >/dev/null 2>&1; do
    check_pause
    sleep 60; waited=$((waited+60))
    if [ $((waited % 1800)) -eq 0 ]; then say "  a training phase is already running (${waited}s)"; fi
  done
  [ $waited -gt 0 ] && say "  in-flight training phase finished after ${waited}s"
  return 0
}

training_complete () {   # $1=exp dir $2=old $3=new $4=wants_joint  ($5=quiet)
  local e=$1 old=$2 new=$3 joint=$4 quiet=${5:-} miss=0
  local need=("$e/adapters/stage1/${old}_8b_seed0/adapter_last"
              "$e/merged/base_${old}_8b_seed0"
              "$e/adapters/stage2/stage2_${new}_8b_seed0/adapter_last"
              "$e/adapters/baselines/rehearsal001_8b_seed0/adapter_last"
              "$e/adapters/baselines/rehearsal005_8b_seed0/adapter_last")
  [ "$joint" = "yes" ] && need+=("$e/adapters/baselines/joint_8b_seed0/adapter_last")
  for p in "${need[@]}"; do
    if [ ! -e "$p" ]; then
      [ -z "$quiet" ] && say "  missing artifact: ${p#$R/}"
      miss=1
    fi
  done
  return $miss
}

# ---------------------------------------------------------------- phases
do_train () {
  check_pause
  local e=$R/$1
  mkdir -p "$e/logs/jobs"   # the redirect below is opened before run_all.sh can create it
  say "TRAIN  $1"
  wait_for_gpus
  ( cd "$e/scripts" && bash run_all.sh train ) >> "$e/logs/train_phase.out" 2>&1
  say "  train phase returned $?"
}

do_val () {
  check_pause
  local e=$R/$1
  mkdir -p "$e/logs/jobs"   # the redirect below is opened before run_all.sh can create it
  say "EVAL-VAL  $1"
  wait_for_gpus
  ( cd "$e/scripts" && bash run_evals.sh val ) >> "$e/logs/eval_val.out" 2>&1
  say "  val phase returned $?"
}

do_test () {
  check_pause
  local e=$R/$1 env
  mkdir -p "$e/logs/jobs"   # the redirect below is opened before run_all.sh can create it
  say "SELECT DIALS  $1"
  if ! env=$(python3 "$SEL" "$e" --emit-env 2>&1); then
    say "  DIAL SELECTION FAILED: $env"
    say "  STOPPING CHAIN -- $1 needs a human look"
    return 1
  fi
  say "  $env"
  say "EVAL-TEST  $1"
  wait_for_gpus
  ( cd "$e/scripts" && eval "export $env" && bash run_evals.sh test ) >> "$e/logs/eval_test.out" 2>&1
  say "  test phase returned $?"
}

run_experiment () {   # $1=name $2=old $3=new $4=wants_joint
  local name=$1 old=$2 new=$3 joint=$4 attempt=1
  say "===== $name ====="

  while :; do
    wait_for_own_training
    if training_complete "$R/$name" "$old" "$new" "$joint" quiet; then break; fi
    if [ $attempt -gt "$MAX_TRAIN_ATTEMPTS" ]; then
      training_complete "$R/$name" "$old" "$new" "$joint"
      say "  STOPPING CHAIN -- $name training still incomplete after $MAX_TRAIN_ATTEMPTS attempts"
      return 1
    fi
    [ $attempt -gt 1 ] && {
      local fu; fu=$(foreign_gpu_user)
      say "  retry $attempt/$MAX_TRAIN_ATTEMPTS after incomplete training (foreign GPU users: [${fu:-none}])"
      echo "$(date -Is)  $name training retried (attempt $attempt): a job that spans a" \
           "retry has its wall-clock split across a FAILED and an ok record in" \
           "timings.jsonl -- sum them before quoting its training cost." >> "$R/INTERRUPTIONS.log"
      sleep "$COOLDOWN"
    }
    do_train "$name"
    attempt=$((attempt+1))
  done

  do_val "$name" || return 1
  do_test "$name" || return 1
  if [ "$joint" = "no" ]; then
    say "COPY JOINT  $name"
    bash "$CJ" "$R/$name" 2>&1 | tee -a "$CHAINLOG"
  fi
  say "===== $name COMPLETE ====="
  check_pause
}

# ---------------------------------------------------------------- sequence
say "chain starting"
run_experiment Bankstatement2Quaterly_rep_rerun_seed2 bank_full quarterly yes || exit 1
run_experiment Quaterly_rep2Bankstatement_rerun_seed2 quarterly bank_full no  || exit 1
run_experiment Bankstatement2Quaterly_rep_rerun_seed3 bank_full quarterly yes || exit 1
run_experiment Quaterly_rep2Bankstatement_rerun_seed3 quarterly bank_full no  || exit 1

say "ALL FOUR RUNS COMPLETE -- running the seed-isolation audit"
python3 "$C/data_prep/audit_seed_isolation.py" 2>&1 | tail -25 | tee -a "$CHAINLOG"
rm -f "$R/chain.pid"
say "chain finished"
