"""Three-arm layer-reversion table: top 25%, random 25%, bottom 25%.

Extends the exchange-rate analysis of thesis Table 5.7 with the bottom-quartile
control. Recovery and cost are measured at full reversion of the selected
layers (beta = 0) against that arm's own beta = 1 row, which is the unmodified
stage-2 model, so all three arms share a baseline.

Runs with no bottom sweep yet are reported as pending rather than skipped.
"""
import json
import os

B = "/home/frank/runs/thesis_experiment_runs/_layer_reversion"
RUNS = [
    ("b2q split 1", "Bankstatement2Quaterly_rep_rerun",       "bank",      "quarterly"),
    ("b2q split 2", "Bankstatement2Quaterly_rep_rerun_seed2", "bank",      "quarterly"),
    ("b2q split 3", "Bankstatement2Quaterly_rep_rerun_seed3", "bank",      "quarterly"),
    ("q2b split 1", "Quaterly_rep2Bankstatement_rerun",       "quarterly", "bank"),
    ("q2b split 2", "Quaterly_rep2Bankstatement_rerun_seed2", "quarterly", "bank"),
    ("q2b split 3", "Quaterly_rep2Bankstatement_rerun_seed3", "quarterly", "bank"),
]
ARMS = [("top", "layer-top"), ("random", "layer-random0"), ("bottom", "layer-bottom")]


def arm(exp, fstem, old, new):
    p = f"{B}/{exp}/{fstem}-dirsweep__both__val.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if "0.0" not in d or "1.0" not in d:
        return None
    rec = d["0.0"][f"{old}_f1"] - d["1.0"][f"{old}_f1"]
    giv = d["1.0"][f"{new}_f1"] - d["0.0"][f"{new}_f1"]
    return rec, giv


print(f"{'run':<13}{'arm':<9}{'recovered':>11}{'given up':>11}{'rate':>8}")
print("-" * 52)
pending = []
for lab, exp, old, new in RUNS:
    for name, fstem in ARMS:
        r = arm(exp, fstem, old, new)
        if r is None:
            print(f"{lab:<13}{name:<9}{'pending':>11}")
            pending.append(f"{lab} {name}")
            continue
        rec, giv = r
        rate = f"{rec / giv:.2f}" if giv > 0.005 else "---"
        print(f"{lab:<13}{name:<9}{rec:>11.3f}{giv:>11.3f}{rate:>8}")
    print()

if pending:
    print(f"{len(pending)} arm(s) not yet available: " + ", ".join(pending))
else:
    print("all 18 arms present")
    for name, fstem in ARMS:
        v = [arm(e, fstem, o, n) for _, e, o, n in RUNS]
        mr = sum(x[0] for x in v) / len(v)
        mg = sum(x[1] for x in v) / len(v)
        print(f"  mean {name:<7} recovered {mr:+.3f}   given up {mg:+.3f}")
