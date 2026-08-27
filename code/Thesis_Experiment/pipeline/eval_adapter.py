"""Score a trained adapter the ORDINARY way: load it with PEFT, generate, score.

No reversion machinery is involved at all -- no DoRAAdapter, no W0 snapshot, no
weight rewriting. This is the natural path for every configuration that reverts
nothing:  stage-2 full DoRA, joint, rehearsal 1%, rehearsal 5%.

It also serves as an independent check on the reversion code: this script and
`eval_pair.py --alpha-mag 1 --beta-dir 1` should agree, because
VERSION_FNS["full"] (m_ft * D_ft) is by definition what PEFT computes. If they
disagree, the reconstruction is wrong and every reverted number is suspect.

Scoring goes through the same evalcore.score used everywhere else, so prompt,
batch size, token cap and metric are identical to the rest of the pipeline.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from Thesis_Experiment.pipeline import evalcore
from Thesis_Experiment.datasets import data_seed, label

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="base model dir (merged W_bank, or the pretrained model for joint)")
ap.add_argument("--adapter", default=None, help="PEFT adapter dir; omit to score the base model alone")
ap.add_argument("--old", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--split", default="test", choices=["training", "validation", "test"])
ap.add_argument("--config", default="adapter")
ap.add_argument("--cap", type=int, default=evalcore.DEFAULT_CAP)
ap.add_argument("--batch", type=int, default=evalcore.DEFAULT_BATCH)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", required=True)
ap.add_argument("--progress", default=None)
args = ap.parse_args()


def note(msg):
    if args.progress:
        with open(args.progress, "w") as f:
            f.write(msg + "\n")
    print(msg, flush=True)


t0 = time.time()
note(f"{args.config}: loading base {os.path.basename(args.base)}")
proc = AutoProcessor.from_pretrained(args.base)
model = Qwen3VLForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16).to(args.device)

if args.adapter:
    from peft import PeftModel
    note(f"{args.config}: attaching adapter with PEFT (no reversion machinery)")
    model = PeftModel.from_pretrained(model, args.adapter).to(args.device)
model.eval()

note(f"{args.config}: scoring {args.old} ({args.split})")
old = evalcore.score(model, proc, args.old, args.split, args.device, args.cap, args.batch)
note(f"{args.config}: scoring {args.new} ({args.split})")
new = evalcore.score(model, proc, args.new, args.split, args.device, args.cap, args.batch)

res = {
    "data_seed": data_seed(),
    "config": args.config,
    "loader": "peft",                     # <- distinguishes this from the reconstruction path
    "base": args.base,
    "adapter": args.adapter,
    "split": args.split,
    label(args.old): old,
    label(args.new): new,
    f"{label(args.old)}_f1": old["field_f1"],
    f"{label(args.new)}_f1": new["field_f1"],
    "avg_f1": round((old["field_f1"] + new["field_f1"]) / 2, 4),
    "total_eval_seconds": round(time.time() - t0, 1),
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2)
note(f"{args.config}: DONE  {label(args.old)}={old['field_f1']}  {label(args.new)}={new['field_f1']}  "
     f"avg={res['avg_f1']}  ({res['total_eval_seconds']}s)")
