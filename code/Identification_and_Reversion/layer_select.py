"""Choose which adapted layers to revert, from the per-layer direction drift.

Pure selection: reads the drift statistics produced by layer_drift.py and writes a
layer list. No model is loaded and no GPU is used, so the choice can be inspected,
diffed and re-derived without repeating any measurement.

The statistic is
    rho_i = median over output rows of  ||s B A||_row / ||W0||_row
i.e. how large the direction update is in that layer, relative to the weight it
modifies. Because sBA is ~99.99% orthogonal to W0 (see the report), rho is also the
per-row rotation angle in radians to three decimals, so "largest rho" and "turned
furthest" are the same ranking.

Selection is per RUN, using that run's own drift. Pooling runs would mean choosing
layers with information from experiments the choice is then tested on.

    python layer_select.py --run b2q_seed3 --quantile 0.75 --out sel.json
    python layer_select.py --run b2q_seed3 --quantile 0.75 --mode random --seed 0
"""
import argparse, json, os
import numpy as np

DRIFT = "/home/frank/runs/thesis_experiment_runs/_layerdrift"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="e.g. b2q_seed3 (a key of _layerdrift/)")
ap.add_argument("--quantile", type=float, default=0.75,
                help="keep layers at or above this quantile of the statistic")
ap.add_argument("--stat", default="rho_median", choices=["rho_median", "rho_mean", "rho_p90"])
ap.add_argument("--mode", default="top", choices=["top", "bottom", "random"],
                help="top = the drifted layers (the hypothesis); bottom and random are controls "
                     "that hold the LAYER COUNT fixed, so a difference cannot be explained by "
                     "how many layers were touched")
ap.add_argument("--seed", type=int, default=0, help="only used by --mode random")
ap.add_argument("--out", required=True)
a = ap.parse_args()

src = f"{DRIFT}/layer_drift_{a.run}.json"
if not os.path.exists(src):
    raise SystemExit(f"no drift statistics for {a.run}: {src}")
layers = json.load(open(src))["layers"]
vals = np.array([l[a.stat] for l in layers])
thr = float(np.quantile(vals, a.quantile))
k = int(round((1.0 - a.quantile) * len(layers)))

if a.mode == "top":
    chosen = [l for l in layers if l[a.stat] >= thr]
elif a.mode == "bottom":
    chosen = sorted(layers, key=lambda l: l[a.stat])[:k]
else:
    chosen = [layers[i] for i in np.random.default_rng(a.seed).choice(len(layers), k, replace=False)]

out = {
    "run": a.run, "statistic": a.stat, "quantile": a.quantile, "mode": a.mode,
    "seed": a.seed if a.mode == "random" else None,
    "threshold": thr, "n_selected": len(chosen), "n_total": len(layers),
    "source": src,
    "layers": sorted(l["layer"] for l in chosen),
    "detail": sorted(({"layer": l["layer"], a.stat: l[a.stat], "block": l["block"],
                       "proj": l["proj"]} for l in chosen), key=lambda d: -d[a.stat]),
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2)
print(f"{a.run}: {a.mode} {len(chosen)}/{len(layers)} layers at {a.stat} "
      f"quantile {a.quantile} (threshold {thr:.6f}) -> {a.out}")
