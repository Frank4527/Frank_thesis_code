"""Rebuild the quarterly split for a given seed, from the ORIGINAL source.

Row construction is copied verbatim from build_quarterly_dataset.py so rows are
byte-identical to seed 1; only the shuffle seed and the output directory change.

Split is BY DOCUMENT (pages of one report never straddle splits) and uses the same
document counts as seed 1, so every seed has the same number of documents per split.

    python build_quarterly_seed.py <seed> <out_dir> [--verify-against <dir>]
"""
import json
import os
import random
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
from evolution_ai_datasets.serialization import load_dataset

NEW_ROOT = "/data/ucl_students/dataset-01"
SEED = int(sys.argv[1])
OUT = sys.argv[2]
VERIFY = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--verify-against" else None


def prune(obj):
    if isinstance(obj, dict):
        out = {k: prune(v) for k, v in obj.items()}
        out = {k: v for k, v in out.items() if v is not None}
        return out or None
    if isinstance(obj, list):
        out = [prune(v) for v in obj]
        out = [v for v in out if v is not None]
        return out or None
    if isinstance(obj, str) and obj.strip() in ("", "null", "None"):
        return None
    return obj if obj not in (None, "", [], {}) else None


def val(f):
    return getattr(f, "value", f)


def groups_to_lists(grouped):
    out = {}
    for g, insts in (grouped or {}).items():
        items = []
        for idx in sorted(insts.keys(), key=lambda x: (isinstance(x, str), x)):
            inst = prune({n: val(f) for n, f in insts[idx].items()})
            if inst:
                items.append(inst)
        if items:
            out[g] = items
    return out


ds, _ = load_dataset(Path(NEW_ROOT), load_ocr=False)
docs = {}
for doc in ds.documents:
    rows = []
    for page in doc.pages:
        target = {}
        f = prune({n: val(v) for n, v in (page.fields or {}).items()})
        if f:
            target.update(f)
        target.update(groups_to_lists(page.grouped_fields))
        for k, v in groups_to_lists(page.tables if isinstance(page.tables, dict) else {}).items():
            target.setdefault(k, v)
        target = prune(target)
        if not target or page.image is None or not page.image.file_path:
            continue
        rows.append({"image": os.path.join(NEW_ROOT, page.image.file_path),
                     "gt_parse": target, "doc_type": "quarterly_report"})
    if rows:
        docs[doc.id] = rows

ids = sorted(docs)
random.Random(SEED).shuffle(ids)
n = len(ids)
n_tr, n_va = int(round(n * 0.8)), int(round(n * 0.1))
split_ids = {"training": ids[:n_tr], "validation": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}

print(f"  seed={SEED}  documents={n}  -> train/val/test docs = "
      f"{len(split_ids['training'])}/{len(split_ids['validation'])}/{len(split_ids['test'])}")

for split, dids in split_ids.items():
    d = f"{OUT}/{split}"
    os.makedirs(d, exist_ok=True)
    rows = [r for did in dids for r in docs[did]]
    with open(f"{d}/manifest.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"    {split:11s} {len(dids):3d} docs / {len(rows):4d} pages")

# no document may appear in two splits
allids = [i for v in split_ids.values() for i in v]
assert len(allids) == len(set(allids)) == n, "document leaked across splits"
print("    leakage check: every document appears in exactly one split  OK")

if VERIFY:
    same = True
    for split in split_ids:
        a = open(f"{OUT}/{split}/manifest.jsonl").read()
        b = open(f"{VERIFY}/{split}/manifest.jsonl").read()
        if a != b:
            same = False
            print(f"    VERIFY {split}: DIFFERS")
    print(f"    reproduces {VERIFY}: {same}")
