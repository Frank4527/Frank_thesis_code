"""Pretrained-ability recovery sweep.

Anchor    = pretrained Qwen3-VL-8B-Instruct
Fine-tuned = the stage-1 BANK adapter (split 1), which sits directly on that base
Dials     = revert_direction(beta) over all adapted layers, plus revert_magnitude(0)

Seven evaluation points:
    base        (m0, D0)     the pretrained model
    b0.00       (m_ft, D0)   = mag_only
    b0.25 / b0.50 / b0.75    the sweep
    b1.00       (m_ft, D_ft) = full = the stage-1 bank model, unmodified
    a0.00       (m0, D_ft)   = dir_only

Each point is scored on OCRBench, MMBench-dev-EN and bank validation (split 1).
Results are written after EACH benchmark, before any printing.

Run one or more dials per process with --dials, so dials can be spread over GPUs.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/frank/code")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DATA_SEED", os.environ.get("PA_SEED", "1"))

import torch
from datasets import load_from_disk
from Thesis_Experiment.pipeline import evalcore
from Identification_and_Reversion.dora_merge import VERSION_FNS
from Pretrained_Ability.bench_lib import run_ocrbench, run_mmbench

BASE = "/home/frank/models/Qwen3-VL-8B-Instruct"
ADAPTER = os.environ["PA_ADAPTER"]
STUDY = "/home/frank/runs/thesis_experiment_runs/_pretrained_ability_recovery"
RES = os.environ["PA_RES"]
DATA = f"{STUDY}/data"

# name -> (alpha_mag, beta_dir); None means "restore the pretrained weight"
DIALS = {
    "base":  None,
    "b0.00": (1.0, 0.00),
    "b0.25": (1.0, 0.25),
    "b0.50": (1.0, 0.50),
    "b0.75": (1.0, 0.75),
    "b1.00": (1.0, 1.00),
    "a0.00": (0.00, 1.0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dials", default="all")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--limit", type=int, default=0, help="smoke-test row cap; 0 = full")
ap.add_argument("--tag", default="")
ap.add_argument("--skip-bank", action="store_true")
ap.add_argument("--skip-ocr", action="store_true")
a = ap.parse_args()
limit = a.limit or None
want = list(DIALS) if a.dials == "all" else a.dials.split(",")
for d in want:
    if d not in DIALS:
        raise SystemExit(f"unknown dial {d!r}; known: {list(DIALS)}")
os.makedirs(RES, exist_ok=True)

print(f"device={a.device} dials={want} limit={limit} batch={a.batch}", flush=True)
model, proc = evalcore.load(BASE, a.device)
ad = evalcore.attach(model, ADAPTER, a.device)
print(f"adapter attached: {len(ad.layers)} DoRA layers", flush=True)

ocr = None if a.skip_ocr else load_from_disk(f"{DATA}/ocrbench")
mmb = load_from_disk(f"{DATA}/mmbench_dev_en")


def apply_dial(name):
    """Set the model's weights for this dial. Returns a description."""
    spec = DIALS[name]
    if spec is None:
        ad.apply(model, VERSION_FNS["base"])       # exactly W0 everywhere
        return "pretrained base (m0, D0)"
    alpha, beta = spec
    # apply_reversion routes (1,1) to the canonical full path -- the bf16 dial-1 guard
    evalcore.apply_reversion(ad, model, alpha_mag=alpha, beta_dir=beta)
    return f"alpha_mag={alpha} beta_dir={beta}"


def save(name, obj):
    p = f"{RES}/{name}"
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  [saved] {os.path.basename(p)}", flush=True)


suffix = f"_{a.tag}" if a.tag else ""
for dial in want:
    t0 = time.time()
    desc = apply_dial(dial)
    print(f"\n=== dial {dial}: {desc}", flush=True)

    if not a.skip_ocr:
        om, opreds = run_ocrbench(model, proc, ocr, batch_size=a.batch,
                                  device=a.device, limit=limit)
        save(f"dial_{dial}_ocrbench{suffix}.json",
             {"dial": dial, "spec": DIALS[dial], "benchmark": "OCRBench",
              "adapter": ADAPTER, "data_seed": int(os.environ["DATA_SEED"]),
              "max_pixels": 1048576, **om})
        print(f"  OCRBench {om['score']}/{om['n']} = {om['accuracy']*100:.1f}%", flush=True)

    mm, mpreds = run_mmbench(model, proc, mmb, batch_size=a.batch,
                             device=a.device, limit=limit)
    save(f"dial_{dial}_mmbench{suffix}.json",
         {"dial": dial, "spec": DIALS[dial], "benchmark": "MMBench-dev-EN",
          "adapter": ADAPTER, "max_pixels": 1048576, **mm})
    print(f"  MMBench {mm['accuracy']*100:.2f}%  parse-fail {mm['parse_fail_rate']*100:.2f}%  "
          f"routes {mm['extraction_route']}", flush=True)

    if not a.skip_bank:
        bank = evalcore.score(model, proc, "bank_full", "validation", a.device,
                              limit=limit)
        save(f"dial_{dial}_bank{suffix}.json",
             {"dial": dial, "spec": DIALS[dial], "dataset": "bank_full",
              "split": "validation", "data_seed": int(os.environ["DATA_SEED"]),
              **{k: bank[k] for k in ("field_f1", "value_f1", "n")}})
        print(f"  bank val field_f1 {bank['field_f1']:.4f} value_f1 {bank['value_f1']:.4f}",
              flush=True)

    print(f"  dial {dial} done in {time.time()-t0:.0f}s", flush=True)

print("\nSWEEP SEGMENT DONE", flush=True)
