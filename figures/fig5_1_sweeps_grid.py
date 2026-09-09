"""Figure 5.1 -- the two reversion sweeps, one row per run.

Left column scores the old task, right column the new one, both against the
reversion coefficient, on the validation split. Same numbers as Table 5.2.

    python figures/fig5_1_sweeps_grid.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RUNS, GRID, OUT, RC, BLUE, VERM, sweep_xy, tidy   # noqa: E402

ARMS = [("direction reversion", "dir", BLUE, "s"),
        ("magnitude reversion", "mag", VERM, "o")]


def main(name="sweeps_grid.png"):
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(RUNS), 2, figsize=(7.0, 8.6), sharex=True)
    for row, (lab, exp, old, new) in enumerate(RUNS):
        print(f"{lab}   old={old} new={new}")
        for col, task in enumerate((old, new)):
            ax = axes[row, col]
            for aname, kind, colr, mk in ARMS:
                x, y = sweep_xy(exp, kind, old, new)
                series = x if col == 0 else y
                ax.plot(GRID, series, "-", color=colr, marker=mk, ms=4.4, lw=1.7,
                        label=aname if (row == 0 and col == 0) else None, zorder=3)
                print(f"    {aname:<20} {task:<10} " +
                      "  ".join(f"{c:>4}:{v:.4f}" for c, v in zip(GRID, series)))
            ax.set_title(f"{lab}   ({task})", fontsize=8)
            ax.set_ylim(-0.05, 0.8)
            ax.set_yticks([0.0, 0.5])
            tidy(ax)
        print()
    for ax in axes[-1]:
        ax.set_xlabel("reversion coefficient")
        ax.set_xticks(GRID)
        ax.set_xticklabels(["1", ".75", ".5", ".25", "0"])
        ax.invert_xaxis()
    axes[len(RUNS) // 2, 0].set_ylabel("Field-F1 on old task")
    axes[len(RUNS) // 2, 1].set_ylabel("Field-F1 on new task")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.002))
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)
    return fig, axes


if __name__ == "__main__":
    main()
