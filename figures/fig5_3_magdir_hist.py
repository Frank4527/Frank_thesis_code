"""Figure 5.3 -- how far each DoRA component moves during stage 2.

One value per adapted output unit, as a percentage of that unit's stage-1 norm:

    magnitude   |m_ft - ||W0|| | / ||W0||
    direction   ||s BA||       / ||W0||

The per-unit arrays are 1,400,832 values per run and about 34 MB each, too
large to ship, so this reads pre-binned counts in
``results/layer_drift/drift_hist.npz`` and renders exactly the same histogram.
That file is produced by ``figures/derive_drift_hist.py`` from the arrays
``code/Identification_and_Reversion/layer_drift.py`` writes.

    python figures/fig5_3_magdir_hist.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import RESULTS, OUT, RC, BLUE, VERM, tidy                # noqa: E402

KEYS = ["b2q_seed1", "b2q_seed2", "b2q_seed3",
        "q2b_seed1", "q2b_seed2", "q2b_seed3"]


def main():
    plt.rcParams.update(RC)
    z = np.load(os.path.join(RESULTS, "layer_drift", "drift_hist.npz"))
    meta = json.load(open(os.path.join(RESULTS, "layer_drift",
                                       "drift_hist_meta.json")))
    bins = z["bins"]
    width = bins[1] - bins[0]
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    n_units = 0
    for k in KEYS:
        n_units += meta[k]["n_units"]
        for series, colr in (("mag", BLUE), ("dir", VERM)):
            c = z[f"{k}__{series}_counts"].astype(float)
            ax.stairs(c / (c.sum() * width), bins, color=colr, lw=1.0)
    mag_meds = [meta[k]["mag_median_pct"] for k in KEYS]
    dir_meds = [meta[k]["dir_median_pct"] for k in KEYS]
    ax.axvline(np.median(mag_meds), color=BLUE, lw=0.8, alpha=.55)
    ax.axvline(np.median(dir_meds), color=VERM, lw=0.8, alpha=.55)
    h = [plt.Line2D([], [], color=BLUE, lw=1.6),
         plt.Line2D([], [], color=VERM, lw=1.6)]
    ax.legend(h, [r"magnitude  $|m_{ft}-\Vert W_0\Vert|\,/\,\Vert W_0\Vert$"
                  + f"     median {min(mag_meds):.3f}–{max(mag_meds):.3f}%",
                  r"direction  $\Vert sBA\Vert\,/\,\Vert W_0\Vert$"
                  + f"     median {min(dir_meds):.3f}–{max(dir_meds):.3f}%"],
              loc="upper right", frameon=True, framealpha=.95, borderpad=.6)
    ax.set_xlabel("percentage change per output unit")
    ax.set_ylabel("density")
    ax.set_xlim(0, 2.5)
    tidy(ax)
    fig.tight_layout()
    p = os.path.join(OUT, "magdir_hist.png")
    fig.savefig(p, dpi=200)
    for k in KEYS:
        m, d = meta[k]["mag_median_pct"], meta[k]["dir_median_pct"]
        print(f"  {k:<10} mag {m:.4f}%   dir {d:.4f}%   ratio {d/m:.2f}")
    print(f"pooled medians: mag {np.median(mag_meds):.4f}%  "
          f"dir {np.median(dir_meds):.4f}%  "
          f"ratio {np.median(dir_meds)/np.median(mag_meds):.2f}x")
    print(f"{n_units:,} output units over {len(KEYS)} runs "
          f"({n_units//len(KEYS):,} each)")
    print("saved", p)


if __name__ == "__main__":
    main()
