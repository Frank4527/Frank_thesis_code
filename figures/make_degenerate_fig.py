"""Why bank->quarterly split 2 selects a degenerate operating point.

Two panels, one per method whose dial is chosen on this run. Each plots both tasks
and the unweighted mean the selection rule maximises, against the dial. The point
is the mismatch between the two: the mean is nearly flat across the region where
the new task falls off a cliff, so a difference far below the measurement
resolution decides between operating points that are far apart in behaviour.

Prints every plotted coordinate.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/frank/runs/thesis_experiment_runs/Bankstatement2Quaterly_rep_rerun_seed2/results"
DIALS = ["1.0", "0.75", "0.5", "0.25", "0.0"]
BLUE, VERM = "#0072B2", "#D55E00"
PANELS = [("direction reversion", "stage2/stage2-dirsweep__both__val_rerun.json", r"$\beta$", 0.25),
          ("WiSE-FT",             "stage2/wiseft-sweep__both__val_rerun.json",    r"$c$",     0.0)]

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
x = range(5)
for ax, (name, f, sym, sel) in zip(axes, PANELS):
    d = json.load(open(f"{R}/{f}"))
    old = [d[k]["bank_f1"] for k in DIALS]
    new = [d[k]["quarterly_f1"] for k in DIALS]
    mean = [(a + b) / 2 for a, b in zip(old, new)]
    print(f"--- {name}")
    for k, a, b, m in zip(DIALS, old, new, mean):
        mark = "  <-- selected" if abs(float(k) - sel) < 1e-9 else ""
        print(f"    {sym.strip('$')}={k:<5} bank {a:.4f}  quarterly {b:.4f}  mean {m:.4f}{mark}")

    ax.plot(x, old, "-s", color=BLUE, lw=1.7, ms=4.4, mfc="white", mew=1.4,
            label="old task (bank)")
    ax.plot(x, new, "--o", color=VERM, lw=1.7, ms=4.4, mfc="white", mew=1.4,
            label="new task (quarterly)")
    ax.plot(x, mean, "-^", color="black", lw=2.0, ms=4.6, mfc="white", mew=1.4,
            label="mean  (what the rule maximises)")
    i = DIALS.index(f"{sel:g}" if sel else "0.0")
    ax.axvline(i, color="0.55", lw=1.0, ls=":", zorder=1)
    ax.annotate("selected", (i, 0.76), ha="center", fontsize=7.5, color="0.35")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["1", ".75", ".5", ".25", "0"])
    ax.set_xlabel(f"{name} dial  {sym}")
    ax.set_ylim(-0.03, 0.82)
    ax.set_title(name, fontsize=9)
    ax.grid(alpha=.18, lw=.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("validation field-level F1")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("/tmp/degenerate_b2q_s2.png", dpi=300, bbox_inches="tight")
print("saved /tmp/degenerate_b2q_s2.png")
