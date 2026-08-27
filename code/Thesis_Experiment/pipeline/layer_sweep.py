"""Direction reversion applied to a SUBSET of adapted layers.

Same dial and same algebra as pipeline/sweep.py --component direction; the only
difference is that the coefficient is applied to the layers named in a selection
file, while every other adapted layer keeps the full trained weights. At beta=1 the
model is therefore the untouched stage-2 model, exactly as in the uniform sweep, so
the two curves share that endpoint and can be plotted together.

Layers outside the selection go through VERSION_FNS["full"], not revert_direction(1),
for the bf16 reason documented in DoRAAdapter.apply_selective.

    python layer_sweep.py --base <merged stage-1> --adapter <stage-2 adapter>
        --layers sel.json --old bank_full --new quarterly
        --betas 0.0,0.25,0.5,0.75,1.0 --split validation --out out.json
"""
import argparse, json, os, time
import torch

from Thesis_Experiment.pipeline import evalcore
from Thesis_Experiment.datasets import data_seed, label
from Identification_and_Reversion.dora_merge import VERSION_FNS
from Identification_and_Reversion.dora_reversion import revert_direction

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--layers", required=True, help="JSON from layer_select.py")
ap.add_argument("--old", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--betas", default="0.0,0.25,0.5,0.75,1.0")
ap.add_argument("--split", default="validation", choices=["training", "validation", "test"])
ap.add_argument("--cap", type=int, default=evalcore.DEFAULT_CAP)
ap.add_argument("--batch", type=int, default=evalcore.DEFAULT_BATCH)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", required=True)
ap.add_argument("--progress", default=None)
args = ap.parse_args()

sel = json.load(open(args.layers))
paths = sel["layers"]
betas = [float(b) for b in args.betas.split(",")]


def note(m):
    if args.progress:
        open(args.progress, "w").write(m + "\n")
    print(m, flush=True)


t0 = time.time()
note(f"layer sweep: {sel['mode']} {sel['n_selected']}/{sel['n_total']} layers "
     f"({sel['statistic']} q={sel['quantile']}), betas={betas}, cap={args.cap}")
model, proc = evalcore.load(args.base, args.device)
ad = evalcore.attach(model, args.adapter, args.device)

res = {"data_seed": data_seed(), "base": args.base, "adapter": args.adapter,
       "selection": {k: sel[k] for k in ("run", "mode", "statistic", "quantile",
                                         "threshold", "n_selected", "n_total", "seed")},
       "split": args.split, "cap": args.cap, "batch": args.batch}
for b in betas:
    if b == 1.0:
        # nothing is reverted anywhere: take the canonical full path everywhere so
        # this endpoint is bit-identical to the stage-2 model of every other sweep
        ad.apply(model, VERSION_FNS["full"])
        n = 0
    else:
        n = ad.apply_selective(model, lambda l, _b=b: revert_direction(l, _b), paths)
    note(f"  beta={b}: reverted {n} layers, scoring {args.old}")
    old = evalcore.score(model, proc, args.old, args.split, args.device, args.cap, args.batch)
    note(f"  beta={b}: scoring {args.new}")
    new = evalcore.score(model, proc, args.new, args.split, args.device, args.cap, args.batch)
    res[str(b)] = {"beta": b, "n_layers_reverted": n,
                   f"{label(args.old)}_f1": old["field_f1"],
                   f"{label(args.new)}_f1": new["field_f1"],
                   f"{label(args.old)}_vf1": old["value_f1"],
                   f"{label(args.new)}_vf1": new["value_f1"],
                   "eval_seconds": round(old["eval_seconds"] + new["eval_seconds"], 1)}
    note(f"  beta={b}: {label(args.old)}={old['field_f1']:.4f} {label(args.new)}={new['field_f1']:.4f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2)
note(f"layer sweep done in {time.time()-t0:.0f}s -> {args.out}")
