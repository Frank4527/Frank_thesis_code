"""Rebuild figures/sweeps.png (the central-result six-panel sweep figure)
straight from the validation sweep JSONs, so the thesis figure is reproducible
on the box rather than from a lost local script.

Reads, per run:
    results/stage2/stage2-dirsweep__both__val_rerun.json
    results/stage2/stage2-magsweep__both__val_rerun.json
and plots the OLD task's field-level F1 against the reversion coefficient.
Old task is bank on b2q and quarterly on q2b.

The earlier version of this figure carried a per-row y-label AND a figure-level
one, which overprinted; only the figure-level label is drawn here.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/frank/runs/thesis_experiment_runs"
RUNS = [
    ("Bankstatement2Quaterly_rep_rerun",       "b2q, split 1", "bank"),
    ("Bankstatement2Quaterly_rep_rerun_seed2", "b2q, split 2", "bank"),
    ("Bankstatement2Quaterly_rep_rerun_seed3", "b2q, split 3", "bank"),
    ("Quaterly_rep2Bankstatement_rerun",       "q2b, split 1", "quarterly"),
    ("Quaterly_rep2Bankstatement_rerun_seed2", "q2b, split 2", "quarterly"),
    ("Quaterly_rep2Bankstatement_rerun_seed3", "q2b, split 3", "quarterly"),
]
DIALS = ["1.0", "0.75", "0.5", "0.25", "0.0"]
BLUE, VERM = "#0072B2", "#D55E00"
OUT = "/tmp/sweeps.png"

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "legend.fontsize": 8})


def sweep(exp, kind, task):
    p = f"{R}/{exp}/results/stage2/stage2-{kind}sweep__both__val_rerun.json"
    d = json.load(open(p))
    return [d[k][f"{task}_f1"] for k in DIALS]


fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.5), sharex=True, sharey=True)
x = np.arange(5)
for ax, (exp, lab, task) in zip(axes.ravel(), RUNS):
    ax.plot(x, sweep(exp, "dir", task), "-s", color=BLUE, lw=1.6, ms=4,
            label="direction reversion")
    ax.plot(x, sweep(exp, "mag", task), "--o", color=VERM, lw=1.6, ms=4,
            label="magnitude reversion")
    ax.set_title(f"{lab}   (old: {task})", fontsize=9)
    ax.grid(alpha=.18, lw=.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0, 0].legend(loc="upper left", frameon=True, framealpha=.95,
                  edgecolor="none", borderpad=.3, handlelength=1.8)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(["1", ".75", ".5", ".25", "0"])
fig.supylabel("old-task validation field-level F1", fontsize=9)
fig.supxlabel(r"reversion coefficient   (1 = unmodified stage-2 model  $\longrightarrow$  "
              r"0 = full reversion to the stage-1 anchor)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(f"saved {OUT}")
for exp, lab, task in RUNS:
    dv, mv = sweep(exp, "dir", task), sweep(exp, "mag", task)
    print(f"  {lab}  dir range {max(dv)-min(dv):.3f}   mag range {max(mv)-min(mv):.3f}")
