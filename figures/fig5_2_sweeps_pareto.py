"""Figure 5.2 -- the same two sweeps with both tasks on the axes.

x = old-task validation field-level F1, y = new-task. One line per component,
swept over the grid of Section 3.4. Both lines share the coefficient = 1
endpoint, the unmodified stage-2 model; direction reversion's coefficient = 0
endpoint is the stage-1 anchor. Same numbers as Table 5.2.

    python figures/fig5_2_sweeps_pareto.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RUNS, DIALS, OUT, RC, BLUE, VERM, sweep_xy, tidy   # noqa: E402

ARMS = [("direction reversion", "dir", BLUE, "s", "-"),
        ("magnitude reversion", "mag", VERM, "o", "-")]
PRETTY = {"b2q": "bank$\\rightarrow$quarterly", "q2b": "quarterly$\\rightarrow$bank"}


def main():
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.2))
    for ax, (lab, exp, old, new) in zip(axes.ravel(), RUNS):
        order, split = lab.split(", ")
        print(f"{lab}")
        for aname, kind, colr, mk, ls in ARMS:
            x, y = sweep_xy(exp, kind, old, new)
            print(f"    {aname:<20} " +
                  "  ".join(f"c={c}:{a:.4f}/{b:.4f}" for c, a, b in zip(DIALS, x, y)))
            ax.plot(x, y, ls, color=colr, marker=mk, ms=5.0, lw=1.8, mfc="white",
                    mew=1.6, label=aname, zorder=3)
        ax.set_title(f"{PRETTY[order]}, {split}", fontsize=8.5)
        ax.set_xlabel(f"old-task F1  ({old})")
        ax.set_ylabel(f"new-task F1  ({new})")
        tidy(ax)
        print()
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(OUT, "sweeps_pareto.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    main()
