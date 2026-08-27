#!/usr/bin/env bash
# A 4-slot GPU pool for the eval phase.
#
# Training was run strictly one-job-at-a-time because training cost is a RESULT we
# compare between methods. Evaluation cost is not: every config is the same 8B model
# doing one greedy pass over the same documents, so there is nothing to compare and
# nothing to protect. We therefore run four eval jobs at once, one per GPU.
#
# The clean (uncontended) eval-cost figure comes from the solo reproduction run,
# not from these; jobs launched here are tagged parallel=true in timings.jsonl.

POOL_GPUS=${POOL_GPUS:-"0 1 2 3"}
declare -A SLOT_PID

pool_submit () {   # $1=name $2=command-string
  local name="$1" cmd="$2" gpu=""

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "  [DRY] $name"
    echo "$cmd" | sed 's/^/        /'
    return 0
  fi
  # wait for a free GPU slot
  while [ -z "$gpu" ]; do
    for g in $POOL_GPUS; do
      local pid="${SLOT_PID[$g]:-}"
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && sleep 20
  done

  if grep -q "\"name\": \"$name\".*\"status\": \"ok\"" "$TIMINGS" 2>/dev/null; then
    echo "[$(date -Is)] SKIP $name (already completed)"; return 0
  fi

  echo "[$(date -Is)] LAUNCH $name on GPU$gpu"
  (
    t0=$(date +%s)
    # the GPU must be pinned INSIDE the sg shell where python runs; a VAR=x prefix
    # on the outer command would only apply to the leading `cd`
    bash -c "export CUDA_VISIBLE_DEVICES=$gpu; $cmd" > "$JOBLOG/$name.out" 2>&1
    rc=$?
    t1=$(date +%s)
    python3 - "$name" "$t0" "$t1" "$rc" "$TIMINGS" "$gpu" <<'PY'
import json, sys, datetime
name, t0, t1, rc, out, gpu = sys.argv[1:7]
t0, t1 = int(t0), int(t1)
rec = {"name": name, "kind": "eval",
       "start": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
       "end": datetime.datetime.fromtimestamp(t1).isoformat(timespec="seconds"),
       "wall_seconds": t1 - t0,
       "wall_hms": str(datetime.timedelta(seconds=t1 - t0)),
       "gpus": 1, "gpu_index": int(gpu),
       "status": "ok" if rc == "0" else f"FAILED(rc={rc})",
       "parallel": True,
       "note": "ran alongside up to 3 other evals; wall time is contended, "
               "use the solo reproduction run for eval cost"}
with open(out, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(f"  {name}: {rec['wall_hms']} ({rec['status']})")
PY
  ) &
  SLOT_PID[$gpu]=$!
  sleep 3            # stagger model loading so four 17 GB reads do not collide on NFS
}

pool_wait () {
  [ "${DRY_RUN:-0}" = "1" ] && { echo "  [DRY] pool_wait"; return 0; }
  echo "[$(date -Is)] waiting for all eval slots to drain..."
  for g in $POOL_GPUS; do
    local pid="${SLOT_PID[$g]:-}"
    [ -n "$pid" ] && wait "$pid" 2>/dev/null
  done
  echo "[$(date -Is)] pool empty"
}
