"""STEP 0 GATE: the UNMODIFIED pretrained base on OCRBench and MMBench-dev-EN.

Nothing is trained; nothing existing is modified.

Results are written to disk IMMEDIATELY after each benchmark, before any formatting
or printing. An earlier version printed first and a KeyError in the print discarded
4,329 completed MMBench generations.
"""
import json, os, sys, time
sys.path.insert(0, "/home/frank/code")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import load_from_disk
from Thesis_Experiment.pipeline import evalcore
from Pretrained_Ability.bench_lib import run_ocrbench, run_mmbench

BASE = "/home/frank/models/Qwen3-VL-8B-Instruct"
STUDY = "/home/frank/runs/thesis_experiment_runs/_pretrained_ability_recovery"
RES = f"{STUDY}/results"
DEV = os.environ.get("DEV", "cuda:0")
LIMIT = int(os.environ.get("LIMIT", "0")) or None
BATCH = int(os.environ.get("BATCH", "4"))
os.makedirs(RES, exist_ok=True)


def save(name, obj):
    p = f"{RES}/{name}"
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  [saved] {p}", flush=True)


print("loading pretrained base (no adapter)...", flush=True)
model, proc = evalcore.load(BASE, DEV)
ocr = load_from_disk(f"{STUDY}/data/ocrbench")
mmb = load_from_disk(f"{STUDY}/data/mmbench_dev_en")
print(f"OCRBench {len(ocr)} rows | MMBench {len(mmb)} rows"
      + (f"  (LIMIT={LIMIT})" if LIMIT else ""), flush=True)
t_all = time.time()

# ---------------------------------------------------------------- OCRBench
om, opreds = run_ocrbench(model, proc, ocr, batch_size=BATCH, device=DEV, limit=LIMIT)
save("step0_base_ocrbench.json", {"stage": "step0_base", "benchmark": "OCRBench",
                                  "base": BASE, "max_pixels": 1048576, **om})
with open(f"{RES}/step0_base_ocrbench_preds.jsonl", "w") as f:
    for r, p in zip(list(ocr)[:LIMIT] if LIMIT else list(ocr), opreds):
        f.write(json.dumps({"dataset": r["dataset"], "type": r["question_type"],
                            "gold": str(r["answer"]), "pred": p}, ensure_ascii=False) + "\n")
print(f"\nOCRBench (base): {om['score']}/{om['n']} = {om['accuracy']*100:.1f}%   "
      f"empty {om['empty_output_rate']*100:.1f}%   {om['seconds']:.0f}s", flush=True)
for k, v in om["components"].items():
    if v["n_seen"]:
        print(f"    {k:48s} {v['score']:4d}/{v['max']}", flush=True)

# ---------------------------------------------------------------- MMBench
mm, mpreds = run_mmbench(model, proc, mmb, batch_size=BATCH, device=DEV, limit=LIMIT)
save("step0_base_mmbench.json", {"stage": "step0_base", "benchmark": "MMBench-dev-EN",
                                 "base": BASE, "max_pixels": 1048576, **mm})
with open(f"{RES}/step0_base_mmbench_preds.jsonl", "w") as f:
    for r, p in zip(list(mmb)[:LIMIT] if LIMIT else list(mmb), mpreds):
        f.write(json.dumps({"index": r["index"], "gold": r["answer"], "pred": p},
                           ensure_ascii=False) + "\n")
print(f"\nMMBench-dev-EN (base): {mm['accuracy']*100:.2f}%  ({mm['correct']}/{mm['n']})   "
      f"parse-fail {mm['parse_fail_rate']*100:.2f}%   {mm['seconds']:.0f}s", flush=True)
print(f"    extraction routes: {mm['extraction_route']}")
print(f"    eval mode: {mm['eval_mode']}")
print("    per-ability (first 12):")
for k, v in list(mm["per_ability"].items())[:12]:
    print(f"      {k:30s} {v['acc']*100:5.1f}%  n={v['n']:<5d} pf={v['parse_fail']*100:4.1f}%")

save("step0_base_summary.json",
     {"stage": "step0_base", "base": BASE, "device": DEV, "limit": LIMIT,
      "batch": BATCH, "max_pixels": 1048576,
      "ocrbench": {"score": om["score"], "n": om["n"], "accuracy": om["accuracy"]},
      "mmbench": {"accuracy": mm["accuracy"], "n": mm["n"],
                  "parse_fail_rate": mm["parse_fail_rate"]},
      "total_seconds": round(time.time() - t_all, 1)})
print("\nGATE  OCRBench: a healthy Qwen3-VL-8B should land in the 70s-80s per cent")
print("      MMBench : expect roughly 75-85%; base parse-fail should be near zero,")
print("                since that is the floor the fine-tuned model is compared against.")
