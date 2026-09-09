"""Derive the compact histogram Figure 5.3 is drawn from.

``code/Identification_and_Reversion/layer_drift.py`` writes, per run, one value
per adapted output unit into ``_layerdrift/layer_drift_<run>.npz`` (m0, dm,
ddir; 1,400,832 units, about 34 MB per run). That is too large to ship, so this
script reduces it to bin counts on the fixed grid Figure 5.3 uses, which
reproduces the figure exactly, and records each run's median.

Run on the machine holding the arrays:

    python figures/derive_drift_hist.py --src /path/to/_layerdrift \
                                        --out results/layer_drift
"""
import argparse
import json
import os

import numpy as np

RUNS = ["b2q_seed1", "b2q_seed2", "b2q_seed3",
        "q2b_seed1", "q2b_seed2", "q2b_seed3"]
BINS = np.linspace(0, 2.5, 76)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of layer_drift_*.npz")
    ap.add_argument("--out", required=True, help="where to write the compact files")
    a = ap.parse_args()
    packed, meta = {"bins": BINS}, {}
    for k in RUNS:
        d = np.load(os.path.join(a.src, f"layer_drift_{k}.npz"))
        mag = np.abs(d["dm"]) / d["m0"] * 100.0
        dirn = d["ddir"] / d["m0"] * 100.0
        packed[f"{k}__mag_counts"] = np.histogram(mag, bins=BINS)[0]
        packed[f"{k}__dir_counts"] = np.histogram(dirn, bins=BINS)[0]
        meta[k] = {"n_units": int(mag.size),
                   "mag_median_pct": float(np.median(mag)),
                   "dir_median_pct": float(np.median(dirn))}
        print(f"  {k}: {mag.size:,} units  mag {meta[k]['mag_median_pct']:.4f}%"
              f"  dir {meta[k]['dir_median_pct']:.4f}%")
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "drift_hist.npz"), **packed)
    with open(os.path.join(a.out, "drift_hist_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("wrote drift_hist.npz and drift_hist_meta.json to", a.out)


if __name__ == "__main__":
    main()
