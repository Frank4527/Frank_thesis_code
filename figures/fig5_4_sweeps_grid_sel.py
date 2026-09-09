"""Figure 5.4 -- Figure 5.1 with the selected coefficients ringed.

The rings mark the operating point Equation 3.6 returns for each component,
recomputed here from the same sweeps rather than read from a stored file, so
the figure and Table 5.3 cannot drift apart.

    python figures/fig5_4_sweeps_grid_sel.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig5_1_sweeps_grid as base                                     # noqa: E402
from _common import RUNS, DIALS, OUT, BLUE, VERM, sweep_xy, selected  # noqa: E402


def main():
    fig, axes = base.main(name="_sweeps_grid_tmp.png")
    print("selected coefficients (Table 5.3)")
    for row, (lab, exp, old, new) in enumerate(RUNS):
        for kind, colr in (("dir", BLUE), ("mag", VERM)):
            k, mean = selected(exp, kind, old, new)
            i = DIALS.index(k)
            x, y = sweep_xy(exp, kind, old, new)
            sym = "beta*" if kind == "dir" else "alpha*"
            print(f"  {lab}  {sym} = {k}   mean {mean:.4f}"
                  f"   old {x[i]:.4f}  new {y[i]:.4f}")
            for col, series in ((0, x), (1, y)):
                axes[row, col].plot([float(k)], [series[i]], "o", ms=11,
                                    mfc="none", mec=colr, mew=1.6, zorder=5)
    p = os.path.join(OUT, "sweeps_grid_sel.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    os.remove(os.path.join(OUT, "_sweeps_grid_tmp.png"))
    print("saved", p)


if __name__ == "__main__":
    main()
