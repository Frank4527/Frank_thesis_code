"""Reversion sweep: dial ONE component of the stage-2 adapter from trained to base.

  --component direction : trained magnitude + direction dialled  (alpha=0 -> mag_only)
  --component magnitude : trained direction + magnitude dialled  (alpha=0 -> dir_only)

alpha=1 is the untouched stage-2 model; alpha=0 is that component fully reverted to the
merged stage-1 base. Scores both tasks at every alpha on one split.
"""
import argparse
import json
import os
import time

from Thesis_Experiment.pipeline import evalcore
from Thesis_Experiment.datasets import data_seed, label

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="merged stage-1 model (the reversion anchor)")
ap.add_argument("--adapter", required=True, help="stage-2 adapter")
ap.add_argument("--old", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--component", required=True, choices=["direction", "magnitude", "wiseft"])
ap.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
ap.add_argument("--split", default="validation", choices=["training", "validation", "test"])
ap.add_argument("--cap", type=int, default=evalcore.DEFAULT_CAP)
ap.add_argument("--batch", type=int, default=evalcore.DEFAULT_BATCH)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", required=True)
ap.add_argument("--progress", default=None)
args = ap.parse_args()

alphas = [float(a) for a in args.alphas.split(",")]
model, proc = evalcore.load(args.base, args.device)
ad = evalcore.attach(model, args.adapter, args.device)
print(f"{args.component} sweep: {args.old} vs {args.new}, split={args.split}, "
      f"alphas={alphas}, cap={args.cap}", flush=True)

res, t0 = {"data_seed": data_seed()}, time.time()
for i, a in enumerate(alphas):
    # dial the chosen component, leave the other fully trained
    # (wiseft has no component split: it interpolates the whole weight)
    if args.component == "wiseft":
        am, bd = None, None
        evalcore.apply_wiseft(ad, model, a)
    else:
        am, bd = (1.0, a) if args.component == "direction" else (a, 1.0)
        evalcore.apply_reversion(ad, model, am, bd)
    old = evalcore.score(model, proc, args.old, args.split, args.device, args.cap, args.batch)
    new = evalcore.score(model, proc, args.new, args.split, args.device, args.cap, args.batch)
    res[str(a)] = {
        f"{label(args.old)}_f1": old["field_f1"], f"{label(args.old)}_vf1": old["value_f1"],
        f"{label(args.new)}_f1": new["field_f1"], f"{label(args.new)}_vf1": new["value_f1"],
        "alpha_beta": [am, bd] if am is not None else None,
        "coeff": a if args.component == "wiseft" else None,
        "eval_seconds": round(old["eval_seconds"] + new["eval_seconds"], 1),
    }
    if args.progress:
        open(args.progress, "w").write(
            f"{i + 1}/{len(alphas)} alphas done ({time.time() - t0:.0f}s)\n")
    print(f"  a={a}: {label(args.old)}={old['field_f1']}  {label(args.new)}={new['field_f1']}",
          flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2)
print(f"SWEEP DONE -> {args.out}  ({time.time() - t0:.0f}s)", flush=True)
