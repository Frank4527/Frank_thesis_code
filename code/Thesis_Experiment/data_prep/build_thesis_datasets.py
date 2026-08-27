"""Assemble Thesis_datasets/ : one canonical home for the official experiments.
DocVQA, Cord, Bank_Po, Bank_De, Bank_En -- each with training/validation/test manifests
(absolute image paths, images referenced in place). Non-destructive."""
import json, os, random

ROOT = "/home/frank/data/Thesis_datasets"
SPLITMAP = {"training_set": "training", "val_set": "validation", "test_set": "test"}

def write(ds, split, rows):
    d = f"{ROOT}/{ds}/{split}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)

def abspath(base, img):
    if os.path.isabs(img): return img
    cand = os.path.join(base, img)
    return cand if os.path.exists(cand) else os.path.join(base, "images", os.path.basename(img))

# ---- CORD (gt_parse) & DocVQA (QA) : copy existing splits, make image paths absolute ----
for ds, src, fname in [("Cord", "/home/frank/data/cord", "gt_parse.jsonl"),
                       ("DocVQA", "/home/frank/data/docvqa", "manifest.jsonl")]:
    for srcsplit, split in SPLITMAP.items():
        base = f"{src}/{srcsplit}"
        rows = []
        for line in open(f"{base}/{fname}"):
            if not line.strip(): continue
            r = json.loads(line)
            r["image"] = abspath(base, r["image"])
            rows.append(r)
        n = write(ds, split, rows)
        print(f"{ds}/{split}: {n}")

# ---- Bank_Po / Bank_De / Bank_En : split each language 80/10/10 by document ----
rng = random.Random(1234)
LANG2DS = {"polish": "Bank_Po", "german": "Bank_De", "english": "Bank_En"}
for lang, ds in LANG2DS.items():
    rows = [json.loads(l) for l in open(f"/home/frank/data/bank_by_language/{lang}.jsonl")]
    rows = [{"image": r["image"], "gt_parse": r["gt_parse"], "bank": r.get("bank"),
             "language": lang} for r in rows]
    rng.shuffle(rows)
    n = len(rows); n_tr = int(round(n * 0.8)); n_va = int(round(n * 0.1))
    parts = {"training": rows[:n_tr], "validation": rows[n_tr:n_tr + n_va], "test": rows[n_tr + n_va:]}
    for split, rr in parts.items():
        print(f"{ds}/{split}: {write(ds, split, rr)}")

print("\n=== Thesis_datasets tree ===")
for ds in ["DocVQA", "Cord", "Bank_Po", "Bank_De", "Bank_En"]:
    counts = {s: sum(1 for _ in open(f"{ROOT}/{ds}/{s}/manifest.jsonl")) for s in ("training","validation","test")}
    print(f"  {ds:8s} training/validation/test = {counts['training']}/{counts['validation']}/{counts['test']}")
