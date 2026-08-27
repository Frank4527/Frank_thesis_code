"""Score ONE model configuration on the old task and the new task.

Covers every row of the results table with one script:
  W_bank          --base W_bank                       (no adapter)
  full DoRA       --base W_bank --adapter stage2
  direction rev   --base W_bank --adapter stage2 --beta-dir 0.5
  magnitude rev   --base W_bank --adapter stage2 --alpha-mag 0.0
  joint baseline  --base <pretrained> --adapter joint
  rehearsal       --base W_bank --adapter rehearsal001
"""
import argparse
import json
import os
import time

from Thesis_Experiment.pipeline import evalcore
from Thesis_Experiment.datasets import data_seed, label

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--old", required=True, help="dataset name of the first task")
ap.add_argument("--new", required=True, help="dataset name of the second task")
ap.add_argument("--split", default="test", choices=["training", "validation", "test"])
ap.add_argument("--alpha-mag", type=float, default=1.0, help="1=trained magnitude, 0=base")
ap.add_argument("--beta-dir", type=float, default=1.0, help="1=trained direction, 0=base")
ap.add_argument("--wiseft", type=float, default=None,
                help="if set, apply WiSE-FT at this coefficient instead of component reversion")
ap.add_argument("--config", default="model", help="name recorded in the result file")
ap.add_argument("--cap", type=int, default=evalcore.DEFAULT_CAP)
ap.add_argument("--batch", type=int, default=evalcore.DEFAULT_BATCH)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", required=True)
ap.add_argument("--progress", default=None)
ap.add_argument("--skip-new", action="store_true",
                help="score ONLY the old task. The mirror of --skip-old: used for the "
                     "second pass of early-stopping analysis, where the new task was "
                     "already scored during checkpoint selection and only the old-task "
                     "column is still missing. The old-task scoring call is unchanged.")
ap.add_argument("--skip-old", action="store_true",
                help="score ONLY the new task. For early-stopping checkpoint selection, "
                     "where the rule reads the new task alone and scoring the old one "
                     "costs 2.5x for a number the rule must ignore. The new-task scoring "
                     "call is unchanged, so numbers are identical to a full run.")
args = ap.parse_args()
if args.skip_old and args.skip_new:
    raise SystemExit("--skip-old and --skip-new together would score nothing")


def note(msg):
    if args.progress:
        with open(args.progress, "w") as f:
            f.write(msg + "\n")
    print(msg, flush=True)


t0 = time.time()
note(f"{args.config}: loading {os.path.basename(args.base)}")
model, proc = evalcore.load(args.base, args.device)
ad = evalcore.attach(model, args.adapter, args.device)
if args.wiseft is not None:
    evalcore.apply_wiseft(ad, model, args.wiseft)
else:
    evalcore.apply_reversion(ad, model, args.alpha_mag, args.beta_dir)

if args.skip_old:
    note(f"{args.config}: SKIPPING {args.old} (--skip-old)")
    old = None
else:
    note(f"{args.config}: scoring {args.old} ({args.split})")
    old = evalcore.score(model, proc, args.old, args.split, args.device, args.cap, args.batch)
if args.skip_new:
    note(f"{args.config}: SKIPPING {args.new} (--skip-new)")
    new = None
else:
    note(f"{args.config}: scoring {args.new} ({args.split})")
    new = evalcore.score(model, proc, args.new, args.split, args.device, args.cap, args.batch)

res = {
    "data_seed": data_seed(),
    "config": args.config,
    "base": args.base,
    "adapter": args.adapter,
    "method": "wiseft" if args.wiseft is not None else "reversion",
    "wiseft_coeff": args.wiseft,
    "alpha_beta": None if args.wiseft is not None else [args.alpha_mag, args.beta_dir],
    "split": args.split,
    label(args.old): old,
    label(args.new): new,
    f"{label(args.old)}_f1": None if old is None else old["field_f1"],
    f"{label(args.new)}_f1": None if new is None else new["field_f1"],
    "avg_f1": None if (old is None or new is None)
              else round((old["field_f1"] + new["field_f1"]) / 2, 4),
    "total_eval_seconds": round(time.time() - t0, 1),
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2)
note(f"{args.config}: DONE  "
     f"{label(args.old)}={'skipped' if old is None else old['field_f1']}  "
     f"{label(args.new)}={'skipped' if new is None else new['field_f1']}  "
     f"avg={res['avg_f1']}  ({res['total_eval_seconds']}s)")
