"""Figure 5.5 -- early stopping against reversion, both from the same run.

Two lines per panel: the stage-2 training path, one point per ten-step
checkpoint from the stage-1 anchor onward, and the direction-reversion sweep of
that run's own early-stopped adapter. They meet at the early-stopped model,
which is both the checkpoint the patience-3 rule returned and the sweep's
beta = 1 end, so the comparison is paired (Section 5.3).

    python figures/fig5_5_es_paired.py
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RESULTS, DIALS, OUT, RC, BLUE, VERM, load, tidy   # noqa: E402

ES = os.path.join(RESULTS, "early_stopping")
PRIM = os.path.join(RESULTS, "primary")
RUNS = [
    ("b2q, split 1", "Bankstatement2Quaterly_es_seed1", "Bankstatement2Quaterly_rep_rerun",       "bank",      "quarterly"),
    ("b2q, split 2", "Bankstatement2Quaterly_es_seed2", "Bankstatement2Quaterly_rep_rerun_seed2", "bank",      "quarterly"),
    ("b2q, split 3", "Bankstatement2Quaterly_es_seed3", "Bankstatement2Quaterly_rep_rerun_seed3", "bank",      "quarterly"),
    ("q2b, split 1", "Quaterly_rep2Bankstatement_es_seed1", "Quaterly_rep2Bankstatement_rerun",       "quarterly", "bank"),
    ("q2b, split 2", "Quaterly_rep2Bankstatement_es_seed2", "Quaterly_rep2Bankstatement_rerun_seed2", "quarterly", "bank"),
    ("q2b, split 3", "Quaterly_rep2Bankstatement_es_seed3", "Quaterly_rep2Bankstatement_rerun_seed3", "quarterly", "bank"),
]


def anchor(prim, old, new):
    f = glob.glob(os.path.join(PRIM, prim, "stage1", "*__both__val_rerun.json"))[0]
    d = load(f)
    return (0, d[old + "_f1"], d[new + "_f1"])


def traj(exp, old, new):
    """(step, old, new) for every checkpoint scored on BOTH tasks."""
    out = []
    for f in glob.glob(os.path.join(ES, exp, "es_traj", "*__old__val.json")):
        st = int(re.search(r"step(\d+)_", os.path.basename(f)).group(1))
        g = f.replace("__old__", "__new__")
        if not os.path.exists(g):
            continue
        a, b = load(f), load(g)
        if a.get(old + "_f1") is None or b.get(new + "_f1") is None:
            continue
        out.append((st, a[old + "_f1"], b[new + "_f1"]))
    return sorted(out)


def es_sweep(exp, old, new):
    d = load(os.path.join(ES, exp, "es_reversion", "es-dirsweep__both__val.json"))
    return [(k, d[k][old + "_f1"], d[k][new + "_f1"]) for k in DIALS]


def main():
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.2))
    for ax, (lab, exp, prim, old, new) in zip(axes.ravel(), RUNS):
        t = [anchor(prim, old, new)] + traj(exp, old, new)
        s = es_sweep(exp, old, new)
        step = open(os.path.join(ES, exp, "es_reversion",
                                 "SELECTED_STEP.txt")).read().split()[0]
        print(f"--- {lab}   stopping rule chose step {step}")
        print("    training path : " + "  ".join(f"s{a}:{b:.4f}/{c:.4f}" for a, b, c in t))
        print("    ES dir sweep  : " + "  ".join(f"b{a}:{b:.4f}/{c:.4f}" for a, b, c in s))
        ax.plot([x[1] for x in t], [x[2] for x in t], "--o", color=VERM, lw=1.5,
                ms=3.6, mfc="white", mew=1.2, zorder=3,
                label="stage-2 training path (every 10 steps)")
        ax.plot([x[1] for x in s], [x[2] for x in s], "-s", color=BLUE, lw=1.8,
                ms=4.4, mfc="white", mew=1.4, zorder=4,
                label="direction reversion of the early-stopped adapter")
        ax.set_title(lab, fontsize=8.5)
        ax.set_xlabel(f"old-task F1  ({old})")
        ax.set_ylabel(f"new-task F1  ({new})")
        tidy(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(OUT, "es_paired.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    main()
