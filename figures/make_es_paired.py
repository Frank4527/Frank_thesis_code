"""Early stopping against reversion, both curves from the SAME training run.

Two lines per panel, nothing else:
  * the stage-2 training path, one point per 10-step checkpoint
  * the direction-reversion sweep of the early-stopped adapter, beta 1 -> 0

They meet at the early-stopped model, which is both the checkpoint the stopping
rule selected and the beta = 1 end of the sweep. Because both curves come from
the early-stopping retrain, this comparison is paired -- unlike the version that
plotted the primary adapter's sweep against this trajectory.

Prints every plotted coordinate.
"""
import glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/frank/runs/thesis_experiment_runs"
DIALS = ["1.0", "0.75", "0.5", "0.25", "0.0"]
RUNS = [
    ("bank$\\rightarrow$quarterly, split 1", "Bankstatement2Quaterly_es_seed1", "bank", "quarterly"),
    ("bank$\\rightarrow$quarterly, split 2", "Bankstatement2Quaterly_es_seed2", "bank", "quarterly"),
    ("bank$\\rightarrow$quarterly, split 3", "Bankstatement2Quaterly_es_seed3", "bank", "quarterly"),
    ("quarterly$\\rightarrow$bank, split 1", "Quaterly_rep2Bankstatement_es_seed1", "quarterly", "bank"),
    ("quarterly$\\rightarrow$bank, split 2", "Quaterly_rep2Bankstatement_es_seed2", "quarterly", "bank"),
    ("quarterly$\\rightarrow$bank, split 3", "Quaterly_rep2Bankstatement_es_seed3", "quarterly", "bank"),
]
BLUE, VERM = "#0072B2", "#D55E00"

plt.rcParams.update({"font.size": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8})


def traj(exp, old, new):
    """(step, old, new) per checkpoint, only steps scored on both tasks."""
    out = []
    for f in glob.glob(f"{R}/{exp}/results/es_traj/*__old__val.json"):
        st = int(re.search(r"step(\d+)_", os.path.basename(f)).group(1))
        g = f.replace("__old__", "__new__")
        if not os.path.exists(g):
            continue
        a, b = json.load(open(f)), json.load(open(g))
        if old in a and new in b:
            out.append((st, a[old]["field_f1"], b[new]["field_f1"]))
    return sorted(out)


def sweep(exp, old, new):
    d = json.load(open(f"{R}/{exp}/results/es_reversion/es-dirsweep__both__val.json"))
    return [(k, d[k][old + "_f1"], d[k][new + "_f1"]) for k in DIALS]


fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
for ax, (lab, exp, old, new) in zip(axes.ravel(), RUNS):
    t = traj(exp, old, new)
    s = sweep(exp, old, new)
    step = open(f"{R}/{exp}/results/es_reversion/SELECTED_STEP.txt").read().split()[0]
    print(f"--- {lab.replace(chr(92)+'rightarrow','>').replace('$','')}   stopping rule chose step {step}")
    print("    training path : " + "  ".join(f"s{a}:{b:.4f}/{c:.4f}" for a, b, c in t))
    print("    ES dir sweep  : " + "  ".join(f"b{a}:{b:.4f}/{c:.4f}" for a, b, c in s))

    ax.plot([x[1] for x in t], [x[2] for x in t], "--o", color=VERM, lw=1.5, ms=3.6,
            mfc="white", mew=1.2, label="stage-2 training path (every 10 steps)", zorder=3)
    ax.plot([x[1] for x in s], [x[2] for x in s], "-s", color=BLUE, lw=1.8, ms=4.4,
            mfc="white", mew=1.4, label=r"direction reversion of the early-stopped adapter",
            zorder=4)
    ax.set_title(lab, fontsize=8)
    ax.locator_params(axis="x", nbins=5)
    ax.set_xlabel(f"old-task F1  ({old})")
    ax.set_ylabel(f"new-task F1  ({new})")
    ax.grid(alpha=.18, lw=.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.005))
fig.suptitle("Two routes from the same training run: stop early, or train on and revert\n"
             "both curves meet at the early-stopped adapter --- the checkpoint the rule "
             "selected, and the $\\beta=1$ end of its sweep",
             y=1.075, fontsize=9, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig("/tmp/es_paired.png", dpi=300, bbox_inches="tight")
print("saved /tmp/es_paired.png")
