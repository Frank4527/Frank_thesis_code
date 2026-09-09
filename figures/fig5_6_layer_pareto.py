"""Figures 5.6 and 5.7 -- layer-selective reversion in the two-task plane.

Four direction-reversion sweeps per run: all 252 adapted layers, and the
most-drifted, an arbitrary, and the least-drifted quarter, 63 layers each. All
four share the beta = 1 endpoint, the stage-2 model, marked with a star. Only
the all-layer arm's beta = 0 endpoint is the stage-1 model, which is the point
of the figure (Section 3.6).

Figure 5.7 is the same with the all-layer arm removed and each panel scaled to
its own three curves, so the 63-layer arms can be told apart.

    python figures/fig5_6_layer_pareto.py          # writes both
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (RUNS, DIALS, RESULTS, OUT, RC, BLUE, VERM, YELLOW,   # noqa: E402
                     GREEN, BLACK, load, sweep, tidy)

LR = os.path.join(RESULTS, "layer_reversion")
ARMS = [("all layers",   None,             BLUE,   "o", "-"),
        ("top-25%",      "layer-top",      VERM,   "s", "-"),
        ("random-25%",   "layer-random0",  YELLOW, "^", "--"),
        ("bottom-25%",   "layer-bottom",   GREEN,  "D", ":")]


def curve(exp, stem, old, new):
    d = (sweep(exp, "dir") if stem is None else
         load(os.path.join(LR, exp, f"{stem}-dirsweep__both__val.json")))
    return ([d[k][old + "_f1"] for k in DIALS],
            [d[k][new + "_f1"] for k in DIALS])


def draw(arms, name, share):
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4),
                             sharex=share, sharey=share)
    for ax, (lab, exp, old, new) in zip(axes.ravel(), RUNS):
        for aname, stem, colr, mk, ls in arms:
            x, y = curve(exp, stem, old, new)
            ax.plot(x, y, ls, color=colr, marker=mk, ms=4.6, lw=1.8,
                    label=aname, zorder=3)
        x0, y0 = curve(exp, None, old, new)
        ax.plot([x0[0]], [y0[0]], "*", color=BLACK, ms=10, zorder=6,
                label="stage-2 model")
        ax.set_title(lab.replace(",", ""), fontsize=9)
        tidy(ax)
    fig.supxlabel("old-task field-level F1  (validation)", fontsize=9)
    fig.supylabel("new-task field-level F1  (validation)", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(arms) + 1, frameon=False,
               bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)


def main():
    print("beta = 1 -> 0, old/new validation F1 (Table 5.9 is the beta = 0 column)")
    for lab, exp, old, new in RUNS:
        print(f"--- {lab}")
        for aname, stem, *_ in ARMS:
            x, y = curve(exp, stem, old, new)
            print(f"    {aname:<12} " +
                  "  ".join(f"b{d}:{a:.4f}/{b:.4f}" for d, a, b in zip(DIALS, x, y)))
    draw(ARMS, "layer_pareto.png", share=True)
    draw(ARMS[1:], "layer_zoom.png", share=False)


if __name__ == "__main__":
    main()
