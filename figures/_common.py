"""Shared paths, run table and styling for the thesis figures.

Every figure script in this directory reads only from ``results/`` in this
repository, so the figures in the thesis can be regenerated without access to
the GPU box. Each script prints the coordinates it plots, so a reader can check
a figure against the corresponding table in the thesis.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures", "out")
os.makedirs(OUT, exist_ok=True)

# Reversion coefficient, ordered as the sweeps are read: 1 keeps the fine-tuned
# value, 0 reverts fully to the anchor (Section 3.2).
DIALS = ["1.0", "0.75", "0.5", "0.25", "0.0"]
GRID = [1.0, 0.75, 0.5, 0.25, 0.0]

# (label, results directory, old task, new task). Old is the task learned in
# stage 1, new the task learned in stage 2 (Section 5.1).
RUNS = [
    ("b2q, split 1", "Bankstatement2Quaterly_rep_rerun",       "bank",      "quarterly"),
    ("b2q, split 2", "Bankstatement2Quaterly_rep_rerun_seed2", "bank",      "quarterly"),
    ("b2q, split 3", "Bankstatement2Quaterly_rep_rerun_seed3", "bank",      "quarterly"),
    ("q2b, split 1", "Quaterly_rep2Bankstatement_rerun",       "quarterly", "bank"),
    ("q2b, split 2", "Quaterly_rep2Bankstatement_rerun_seed2", "quarterly", "bank"),
    ("q2b, split 3", "Quaterly_rep2Bankstatement_rerun_seed3", "quarterly", "bank"),
]

# Okabe-Ito, colour-blind safe.
BLUE, VERM, YELLOW, GREEN, PINK, BLACK = (
    "#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#000000")

RC = {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
      "ytick.labelsize": 7.5, "legend.fontsize": 8, "figure.dpi": 110}


def load(path):
    with open(path) as fh:
        return json.load(fh)


def sweep(exp, kind):
    """One primary validation sweep. kind is 'dir' or 'mag' (Table 5.2)."""
    return load(os.path.join(
        RESULTS, "primary", exp, "stage2",
        f"stage2-{kind}sweep__both__val_rerun.json"))


def sweep_xy(exp, kind, old, new):
    d = sweep(exp, kind)
    return ([d[k][old + "_f1"] for k in DIALS],
            [d[k][new + "_f1"] for k in DIALS])


def selected(exp, kind, old, new):
    """The coefficient Equation 3.6 returns: the grid point maximising the
    unweighted mean of the two validation scores, ties broken toward the larger
    coefficient, i.e. toward less intervention (Section 4.7.6, Table 5.3)."""
    d = sweep(exp, kind)
    best, best_mean = None, None
    for k in DIALS:                       # DIALS runs 1.0 -> 0.0, so an exact
        m = (d[k][old + "_f1"] + d[k][new + "_f1"]) / 2   # tie keeps the larger
        if best_mean is None or m > best_mean:
            best, best_mean = k, m
    return best, best_mean


def tidy(ax):
    ax.grid(alpha=.18, lw=.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
