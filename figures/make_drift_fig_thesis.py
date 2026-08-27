"""Thesis-figure render of Figure 1 of make_drift_plots.py.

Same data, same two quantities, same definitions. This file exists only to
render at the width of a thesis text block: no in-figure title (the caption
does that job), the Okabe-Ito palette used by the sweep figure, and font
sizes that survive being scaled to \\textwidth.

make_drift_plots.py is unchanged and remains the analysis script.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/frank/runs/thesis_experiment_runs/_layerdrift"
RUNS = ["b2q_seed1", "b2q_seed2", "b2q_seed3",
        "q2b_seed1", "q2b_seed2", "q2b_seed3"]
BLUE, VERM = "#0072B2", "#D55E00"          # as figures/sweeps.png
OUT = "/tmp/magdir_hist.png"

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "legend.fontsize": 8})

bins = np.linspace(0, 2.5, 76)
fig, ax = plt.subplots(figsize=(6.5, 3.3))
stats, nunits = [], 0
for key in RUNS:
    p = f"{D}/layer_drift_{key}.npz"
    if not os.path.exists(p):
        print(f"  missing {key}")
        continue
    d = np.load(p, allow_pickle=True)
    mag = np.abs(d["dm"]) / d["m0"] * 100.0     # |magnitude change|, % of the unit
    dirn = d["ddir"] / d["m0"] * 100.0          # direction update, % of the unit
    nunits += mag.size
    ax.hist(mag, bins=bins, density=True, histtype="step", lw=1.0, color=BLUE)
    ax.hist(dirn, bins=bins, density=True, histtype="step", lw=1.0, color=VERM)
    stats.append((key, float(np.median(mag)), float(np.median(dirn))))

mm = np.median([s[1] for s in stats])
md = np.median([s[2] for s in stats])
mlo, mhi = min(s[1] for s in stats), max(s[1] for s in stats)
dlo, dhi = min(s[2] for s in stats), max(s[2] for s in stats)

ax.axvline(mm, color=BLUE, lw=0.8, alpha=.55)
ax.axvline(md, color=VERM, lw=0.8, alpha=.55)

h = [plt.Line2D([], [], color=BLUE, lw=1.6),
     plt.Line2D([], [], color=VERM, lw=1.6)]
ax.legend(h, [r"magnitude  $|m_{ft}-\Vert W_0\Vert|\,/\,\Vert W_0\Vert$"
              + f"     median {mlo:.3f}–{mhi:.3f}%",
              r"direction  $\Vert sBA\Vert\,/\,\Vert W_0\Vert$"
              + f"     median {dlo:.3f}–{dhi:.3f}%"],
          loc="upper right", frameon=True, framealpha=.95, borderpad=.6)

ax.set_xlabel("percentage change per output unit")
ax.set_ylabel("density")
ax.set_xlim(0, 2.5)
ax.grid(alpha=.18, lw=.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(f"saved {OUT}")
print(f"{nunits:,} output units over {len(stats)} runs")
for k, a, b in stats:
    print(f"  {k:<10} mag {a:.3f}%   dir {b:.3f}%   ratio {b/a:.2f}")
print(f"pooled medians: mag {mm:.3f}%  dir {md:.3f}%  ratio {md/mm:.1f}x")
