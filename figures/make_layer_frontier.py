"""Pareto frontier of the four layer-selection arms, one panel per run.

x = old-task validation field-level F1, y = new-task. Every arm is DIRECTION
reversion only, swept over beta in {1, .75, .5, .25, 0}. All four arms share the
beta=1 endpoint (the unmodified stage-2 model). Only the uniform arm's beta=0
endpoint is the stage-1 model; the selective arms revert 63 of 252 layers, so
their beta=0 is something else, which is the point of the figure.

Prints every plotted coordinate so the figure can be checked against the tables.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/frank/runs/thesis_experiment_runs"
LR = f"{R}/_layer_reversion"
DIALS = ["1.0", "0.75", "0.5", "0.25", "0.0"]

RUNS = [
    ("bank$\\rightarrow$quarterly, split 1", "Bankstatement2Quaterly_rep_rerun",       "bank", "quarterly"),
    ("bank$\\rightarrow$quarterly, split 2", "Bankstatement2Quaterly_rep_rerun_seed2", "bank", "quarterly"),
    ("bank$\\rightarrow$quarterly, split 3", "Bankstatement2Quaterly_rep_rerun_seed3", "bank", "quarterly"),
    ("quarterly$\\rightarrow$bank, split 1", "Quaterly_rep2Bankstatement_rerun",       "quarterly", "bank"),
    ("quarterly$\\rightarrow$bank, split 2", "Quaterly_rep2Bankstatement_rerun_seed2", "quarterly", "bank"),
    ("quarterly$\\rightarrow$bank, split 3", "Quaterly_rep2Bankstatement_rerun_seed3", "quarterly", "bank"),
]
# Okabe-Ito
ARMS = [("uniform (all 252 layers)", None,            "#000000", "o", "-"),
        ("top 25%",                  "layer-top",     "#D55E00", "s", "--"),
        ("random 25%",               "layer-random0", "#0072B2", "^", "-."),
        ("bottom 25%",               "layer-bottom",  "#CC79A7", "D", ":")]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8})


def curve(exp, stem, old, new):
    p = (f"{R}/{exp}/results/stage2/stage2-dirsweep__both__val_rerun.json" if stem is None
         else f"{LR}/{exp}/{stem}-dirsweep__both__val.json")
    d = json.load(open(p))
    return ([d[k][old + "_f1"] for k in DIALS], [d[k][new + "_f1"] for k in DIALS])


fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
print(f"{'run':<34}{'arm':<26}" + "".join(f"{'b='+d:>16}" for d in DIALS))
for ax, (lab, exp, old, new) in zip(axes.ravel(), RUNS):
    for aname, stem, col, mk, ls in ARMS:
        x, y = curve(exp, stem, old, new)
        print(f"{lab.replace(chr(92)+'rightarrow','>').replace('$',''):<34}{aname:<26}"
              + "".join(f"{a:>7.4f}/{b:<8.4f}" for a, b in zip(x, y)))
        ax.plot(x, y, ls, color=col, marker=mk, ms=4.6, lw=1.7, mfc="white",
                mew=1.4, label=aname, zorder=3)
    ax.set_title(lab, fontsize=8)
    ax.locator_params(axis="x", nbins=5)
    ax.set_xlabel(f"old-task F1  ({old})")
    ax.set_ylabel(f"new-task F1  ({new})")
    ax.grid(alpha=.18, lw=.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    print()

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.005))
fig.suptitle("Restricting reversion to 63 of 252 layers truncates the trade-off curve\n"
             "direction reversion swept over $\\beta \\in \\{1, .75, .5, .25, 0\\}$; "
             "all four arms share the $\\beta=1$ endpoint",
             y=1.075, fontsize=9, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig("/tmp/layer_frontier.png", dpi=300, bbox_inches="tight")
print("saved /tmp/layer_frontier.png")
