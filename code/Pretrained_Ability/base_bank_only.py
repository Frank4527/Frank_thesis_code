"""Pretrained backbone on bank validation only.

Same evaluation path as sweep_seed.py --dials base: VERSION_FNS["base"] restores
W0 in every adapted layer, so the model is exactly the pretrained backbone. The
MMBench and OCRBench legs are removed. DATA_SEED selects the re-split.
"""
import json, os, sys, time
sys.path.insert(0, "/home/frank/code")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DATA_SEED", os.environ.get("PA_SEED", "1"))

from Thesis_Experiment.pipeline import evalcore
from Identification_and_Reversion.dora_merge import VERSION_FNS

BASE = "/home/frank/models/Qwen3-VL-8B-Instruct"
ADAPTER = os.environ["PA_ADAPTER"]
RES = os.environ["PA_RES"]
DEV = os.environ.get("DEV", "cuda:0")
os.makedirs(RES, exist_ok=True)

model, proc = evalcore.load(BASE, DEV)
ad = evalcore.attach(model, ADAPTER, DEV)
print(f"adapter attached: {len(ad.layers)} DoRA layers", flush=True)
ad.apply(model, VERSION_FNS["base"])          # exactly W0 everywhere
print("restored pretrained base (m0, D0)", flush=True)

t0 = time.time()
bank = evalcore.score(model, proc, "bank_full", "validation", DEV, limit=None)
out = {"dial": "base", "spec": None, "dataset": "bank_full", "split": "validation",
       "data_seed": int(os.environ["DATA_SEED"]), "adapter": ADAPTER,
       **{k: bank[k] for k in ("field_f1", "value_f1", "n")},
       "seconds": round(time.time() - t0, 1)}
with open(f"{RES}/dial_base_bank.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out), flush=True)
