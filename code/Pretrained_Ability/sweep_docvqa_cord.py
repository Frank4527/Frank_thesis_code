"""Second sweep for the pretrained-ability study: DocVQA and CORD.

Same anchor, adapter and dials as sweep.py. Different probes:

  DocVQA (val, 500 q)  -- official metric ANLS: 1 - normalised Levenshtein, max over
                          the acceptable answers, zeroed below tau=0.5, averaged.
                          Scored by Metrics/text_metrics.corpus_anls via
                          Identification_and_Reversion/generate_and_score.evaluate --
                          the same path the DocVQA training used.
  CORD   (test, 100)   -- official metric is the Donut field-level F1 over
                          (field-path, value) pairs, NOT an ANLS conversion. Scored by
                          evalcore.score, which routes through Metrics/field_f1.

Both are read-only. Results are saved after each benchmark, before any printing.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/frank/code")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DATA_SEED", "1")

from Thesis_Experiment.pipeline import evalcore
from Identification_and_Reversion.dora_merge import VERSION_FNS
from Identification_and_Reversion.generate_and_score import evaluate as docvqa_eval
from Training_Dora.qa_data import load_manifest

BASE = "/home/frank/models/Qwen3-VL-8B-Instruct"
ADAPTER = ("/home/frank/runs/thesis_experiment_runs/Bankstatement2Quaterly_rep_rerun"
           "/adapters/stage1/bank_full_8b_seed0/adapter_last")
STUDY = "/home/frank/runs/thesis_experiment_runs/_pretrained_ability_recovery"
RES = f"{STUDY}/results"
DOCVQA = "/home/frank/data/docvqa/val_set/manifest.jsonl"

DIALS = {"base": None, "b0.00": (1.0, 0.00), "b0.25": (1.0, 0.25), "b0.50": (1.0, 0.50),
         "b0.75": (1.0, 0.75), "b1.00": (1.0, 1.00), "a0.00": (0.00, 1.0)}

ap = argparse.ArgumentParser()
ap.add_argument("--dials", default="all")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default="")
ap.add_argument("--skip-cord", action="store_true",
                help="CORD field-level F1 floors at 0 for both the pretrained and the "
                     "bank-tuned model (neither emits CORD's schema), so the official "
                     "metric is uninformative here. Off by default.")
a = ap.parse_args()
limit = a.limit or None
want = list(DIALS) if a.dials == "all" else a.dials.split(",")
for d in want:
    if d not in DIALS:
        raise SystemExit(f"unknown dial {d!r}")
os.makedirs(RES, exist_ok=True)
suffix = f"_{a.tag}" if a.tag else ""

print(f"device={a.device} dials={want} limit={limit}", flush=True)
model, proc = evalcore.load(BASE, a.device)
ad = evalcore.attach(model, ADAPTER, a.device)
print(f"adapter attached: {len(ad.layers)} DoRA layers", flush=True)

rows = load_manifest(DOCVQA)
root = os.path.dirname(DOCVQA)
for r in rows:
    if not os.path.isabs(r["image"]):
        r["image"] = os.path.join(root, r["image"])
if limit:
    rows = rows[:limit]
print(f"DocVQA rows {len(rows)}", flush=True)


def save(name, obj):
    with open(f"{RES}/{name}", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  [saved] {name}", flush=True)


for dial in want:
    t0 = time.time()
    spec = DIALS[dial]
    if spec is None:
        ad.apply(model, VERSION_FNS["base"])
        desc = "pretrained base (m0, D0)"
    else:
        evalcore.apply_reversion(ad, model, alpha_mag=spec[0], beta_dir=spec[1])
        desc = f"alpha_mag={spec[0]} beta_dir={spec[1]}"
    print(f"\n=== dial {dial}: {desc}", flush=True)

    # ---- DocVQA, official ANLS ----
    m, preds = docvqa_eval(model, proc, rows, batch_size=a.batch, max_new_tokens=32,
                           device=a.device, log_every=20)
    save(f"dial_{dial}_docvqa{suffix}.json",
         {"dial": dial, "spec": spec, "benchmark": "DocVQA-val",
          "metric": "ANLS (official)", "adapter": ADAPTER, **m,
          "seconds": round(time.time() - t0, 1)})
    print(f"  DocVQA ANLS {m['anls']:.4f}  exact {m['exact_match']:.4f}  n={m['n']}",
          flush=True)

    # ---- CORD, official Donut field-level F1 ----
    if a.skip_cord:
        print(f"  dial {dial} done in {time.time()-t0:.0f}s (CORD skipped)", flush=True)
        continue
    t1 = time.time()
    c = evalcore.score(model, proc, "cord", "test", a.device, limit=limit)
    save(f"dial_{dial}_cord{suffix}.json",
         {"dial": dial, "spec": spec, "benchmark": "CORD-test",
          "metric": "field-level F1 (Donut), not ANLS", "adapter": ADAPTER,
          **{k: c[k] for k in ("field_f1", "value_f1", "n")},
          "seconds": round(time.time() - t1, 1)})
    print(f"  CORD field_f1 {c['field_f1']:.4f}  value_f1 {c['value_f1']:.4f}  n={c['n']}",
          flush=True)
    print(f"  dial {dial} done in {time.time()-t0:.0f}s", flush=True)

print("\nSEGMENT DONE", flush=True)
