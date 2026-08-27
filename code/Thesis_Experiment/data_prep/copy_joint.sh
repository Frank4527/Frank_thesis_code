#!/usr/bin/env bash
# Copy the joint-baseline results from the matching-seed b2q run into a q2b run.
#
# Joint training is order-free: it starts from the pretrained base and sees the
# union of both tasks at once, so there is ONE joint model per seed, not one per
# direction. Rather than spend 4.7 h retraining a second draw of the same
# quantity, q2b reuses b2q's -- but ONLY from the same data seed, which is what
# this script enforces.
#
#   bash copy_joint.sh <q2b_experiment_dir>
set -euo pipefail
Q="${1:?usage: copy_joint.sh <q2b_experiment_dir>}"
R=/home/frank/runs/thesis_experiment_runs

qseed=$(grep -oP '^DATA_SEED=\K\d+' "$Q/scripts/config.sh")
[ -n "$qseed" ] || { echo "no DATA_SEED in $Q/scripts/config.sh"; exit 1; }

# find the b2q experiment carrying the SAME data seed
src=""
for e in "$R"/Bankstatement2Quaterly_rep_rerun*; do
  [ -f "$e/scripts/config.sh" ] || continue
  s=$(grep -oP '^DATA_SEED=\K\d+' "$e/scripts/config.sh" || true)
  [ "$s" = "$qseed" ] && src="$e" && break
done
[ -n "$src" ] || { echo "no b2q experiment with DATA_SEED=$qseed"; exit 1; }

echo "  target : $(basename "$Q")  (seed $qseed)"
echo "  source : $(basename "$src")  (seed $qseed)"

mkdir -p "$Q/results/baselines"
n=0
for split in val test; do
  f="$src/results/baselines/joint__both__${split}_rerun.json"
  if [ ! -f "$f" ]; then
    echo "  MISSING $(basename "$f") -- run the b2q seed-$qseed evals first"; exit 1
  fi
  # the source must itself be seed-consistent before it is propagated
  python3 - "$f" "$qseed" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); want = int(sys.argv[2])
got = d.get("data_seed")
if got is not None and got != want:
    sys.exit(f"  REFUSING: {sys.argv[1]} carries data_seed={got}, expected {want}")
PY
  # never overwrite an existing result without keeping the old bytes
  dst="$Q/results/baselines/$(basename "$f")"
  if [ -f "$dst" ] && ! cmp -s "$f" "$dst"; then
    bak="$dst.replaced-$(date +%Y%m%dT%H%M%S)"
    cp -p "$dst" "$bak"
    echo "  differed from existing file; previous bytes kept at $(basename "$bak")"
  fi
  cp -p "$f" "$dst"
  n=$((n+1))
done
echo "  copied $n joint result file(s); adapter path inside them still points at"
echo "  $(basename "$src"), which is the provenance the audit checks."
