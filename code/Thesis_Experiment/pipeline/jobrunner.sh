#!/usr/bin/env bash
# jobrunner.sh — run ONE job at a time, on ALL GPUs, and record what it cost.
#
# Why one at a time: the point of the rerun is to compare methods, including their
# training cost. Two jobs sharing the box give meaningless wall-clock numbers.
#
# Records per job, into $TIMINGS (JSONL):
#   name, kind (train|eval|merge), start/end ISO, wall_seconds, gpus,
#   n_train / n_eval where known, and FOREIGN GPU PROCESSES seen at start and end
#   (this box is shared — a job that overlapped someone else's work is flagged, not
#   silently averaged in).
#
# Usage:  run_job <name> <kind> <n_items> <gpus> <<'CMD' ... CMD   (command on stdin)

set -u
TIMINGS=${TIMINGS:?set TIMINGS to the timings.jsonl path}
JOBLOG=${JOBLOG:?set JOBLOG to the log directory}
mkdir -p "$(dirname "$TIMINGS")" "$JOBLOG"

_foreign_procs () {   # GPU compute processes that are not ours
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
    while IFS=, read -r pid mem; do
      pid=$(echo "$pid" | tr -d ' ')
      owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      [ -n "$owner" ] && [ "$owner" != "$(whoami)" ] && echo "${owner}:${mem// /}"
    done | paste -sd';' -
}

run_job () {
  local name="$1" kind="$2" nitems="${3:-null}" gpus="${4:-4}"
  local cmd; cmd=$(cat)                      # the command comes in on stdin

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "  [DRY] $name  kind=$kind n_items=$nitems gpus=$gpus"
    echo "$cmd" | sed 's/^/        /'
    return 0
  fi

  # name and status sit on the same line but with other keys between them
  if grep -q "\"name\": \"$name\".*\"status\": \"ok\"" "$TIMINGS" 2>/dev/null; then
    echo "[$(date -Is)] SKIP $name (already completed)"; return 0
  fi

  local f0 t0 t1 rc f1
  f0=$(_foreign_procs); t0=$(date +%s)
  echo "[$(date -Is)] START $name ($kind)  foreign_gpu_procs=[${f0:-none}]"
  bash -c "$cmd" > "$JOBLOG/$name.out" 2>&1
  rc=$?
  t1=$(date +%s); f1=$(_foreign_procs)

  python3 - "$name" "$kind" "$nitems" "$t0" "$t1" "$rc" "${f0:-}" "${f1:-}" "$TIMINGS" "$gpus" <<'PY'
import json, sys, datetime
name, kind, nitems, t0, t1, rc, f0, f1, out, gpus = sys.argv[1:11]
t0, t1 = int(t0), int(t1)
rec = {
    "name": name, "kind": kind,
    "n_items": None if nitems in ("null", "") else int(nitems),
    "start": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
    "end":   datetime.datetime.fromtimestamp(t1).isoformat(timespec="seconds"),
    "wall_seconds": t1 - t0,
    "wall_hms": str(datetime.timedelta(seconds=t1 - t0)),
    "gpus": int(gpus),
    "status": "ok" if rc == "0" else f"FAILED(rc={rc})",
    "foreign_gpu_procs_start": f0 or None,
    "foreign_gpu_procs_end": f1 or None,
    "clean_measurement": (not f0) and (not f1),
}
if rec["n_items"]:
    rec["seconds_per_item"] = round((t1 - t0) / rec["n_items"], 2)
with open(out, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(f"  -> {rec['wall_hms']}  ({rec['status']})"
      + ("" if rec["clean_measurement"] else "  [!] shared GPU during this job"))
PY
  echo "[$(date -Is)] END   $name  rc=$rc"
  return $rc
}
