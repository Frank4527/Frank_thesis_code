"""Pick each dial from an experiment's OWN validation sweeps.

The rule, fixed in EXPERIMENT_DESIGN.md: choose the dial maximising the mean of the
two tasks' validation field-F1. This script only applies that rule -- it makes no
judgement -- so the chain can run unattended without a human gate that would
otherwise idle the GPUs for hours.

    python select_dials.py <experiment_dir> [--emit-env]

--emit-env prints "MAG_A=.. DIR_B=.. WISEFT_C=.." for the caller to eval.
Exits non-zero if a sweep is missing or malformed, which stops the chain rather
than writing thesis numbers at a dial chosen from bad data.
"""
import json
import os
import re
import sys

EXP = sys.argv[1].rstrip("/")
EMIT = "--emit-env" in sys.argv
EXPECTED_DIALS = 5          # the {0, .25, .5, .75, 1} grid
NEAR_TIE = 0.005            # log, but still proceed
LOG = []


def out(msg):
    LOG.append(msg)
    if not EMIT:
        print(msg)


def load(path):
    if not os.path.exists(path):
        sys.exit(f"MISSING SWEEP: {path}")
    d = json.load(open(path))
    dials = {k: v for k, v in d.items() if re.fullmatch(r"[0-9.]+", k)}
    if len(dials) < EXPECTED_DIALS:
        sys.exit(f"INCOMPLETE SWEEP ({len(dials)}/{EXPECTED_DIALS} dials): {path}")
    return dials


def choose(path, name):
    dials = load(path)
    # the two task keys are "<label>_f1"; there are exactly two per entry
    first = next(iter(dials.values()))
    f1keys = sorted(k for k in first if k.endswith("_f1") and not k.endswith("_vf1"))
    if len(f1keys) != 2:
        sys.exit(f"expected 2 task f1 keys in {path}, found {f1keys}")

    scored = []
    for k, v in dials.items():
        vals = [v.get(x) for x in f1keys]
        if any(x is None for x in vals):
            sys.exit(f"missing f1 in {path} at dial {k}")
        scored.append((float(k), sum(vals) / 2.0, vals))

    best = max(s[1] for s in scored)
    # exact ties -> prefer the LARGER dial (less intervention). A conservative
    # tie-break biases AGAINST the reversion method rather than for it.
    winners = [s for s in scored if s[1] == best]
    pick = max(winners, key=lambda s: s[0])

    out(f"  {name}:")
    for dial, mean, vals in sorted(scored):
        mark = " <-- selected" if dial == pick[0] else ""
        near = ""
        if dial != pick[0] and abs(mean - best) <= NEAR_TIE:
            near = "  (near-tie)"
        out(f"    {dial:<5} {f1keys[0]}={vals[0]:.4f} {f1keys[1]}={vals[1]:.4f} "
            f"mean={mean:.4f}{mark}{near}")
    if len(winners) > 1:
        out(f"    NOTE: {len(winners)} dials tied at {best:.4f}; took the largest "
            f"({pick[0]}) as the conservative choice")
    if pick[0] == 1.0:
        out("    NOTE: argmax is 1.0 -- no reversion beats every reverted point here")
    if pick[0] == 0.0:
        out("    NOTE: argmax is 0.0 -- full reversion to the stage-1 anchor is best")
    return pick[0]


R2 = f"{EXP}/results/stage2"
out(f"selecting dials for {os.path.basename(EXP)}")
mag = choose(f"{R2}/stage2-magsweep__both__val_rerun.json", "magnitude (MAG_A)")
dirb = choose(f"{R2}/stage2-dirsweep__both__val_rerun.json", "direction (DIR_B)")
wc = choose(f"{R2}/wiseft-sweep__both__val_rerun.json", "wiseft (WISEFT_C)")

fmt = lambda x: repr(x) if x != int(x) else str(int(x)) + ".0"
if EMIT:
    print(f"MAG_A={fmt(mag)} DIR_B={fmt(dirb)} WISEFT_C={fmt(wc)}")
else:
    out(f"\n  => MAG_A={fmt(mag)} DIR_B={fmt(dirb)} WISEFT_C={fmt(wc)}")

# leave an auditable record beside the results
os.makedirs(f"{EXP}/results", exist_ok=True)
with open(f"{EXP}/results/DIAL_SELECTION.txt", "w") as fh:
    fh.write("\n".join(LOG) + "\n")
    fh.write(f"\nMAG_A={fmt(mag)} DIR_B={fmt(dirb)} WISEFT_C={fmt(wc)}\n")
