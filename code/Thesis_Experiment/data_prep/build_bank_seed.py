"""Re-split Bank_Full for a given seed.

Each bank row is its own document (747 rows, 747 unique images), so a row-level
split IS a document-level split -- no leakage risk.

The split is STRATIFIED BY LANGUAGE using seed 1's exact per-language counts, so
every seed has the same split sizes AND the same language mix:
    german 226/28/28,  polish 201/25/25,  english 171/21/22

    python build_bank_seed.py <seed> <out_dir>
"""
import json
import os
import random
import sys
from collections import defaultdict

TD = "/home/frank/data/Thesis_datasets"
SEED = int(sys.argv[1])
OUT = sys.argv[2]
SPLITS = ["training", "validation", "test"]

# the target counts, read from seed 1 rather than hardcoded
target = defaultdict(dict)
pool = defaultdict(list)
for sp in SPLITS:
    for line in open(f"{TD}/Bank_Full_seed1/{sp}/manifest.jsonl"):
        r = json.loads(line)
        pool[r["language"]].append(r)
        target[r["language"]][sp] = target[r["language"]].get(sp, 0) + 1

print(f"  seed={SEED}  target counts per language (from seed 1):")
for lang in sorted(target):
    print(f"    {lang:8s} " + "  ".join(f"{sp}={target[lang][sp]}" for sp in SPLITS)
          + f"   pool={len(pool[lang])}")

out_rows = {sp: [] for sp in SPLITS}
for lang in sorted(pool):
    rows = sorted(pool[lang], key=lambda r: r["image"])   # deterministic starting order
    random.Random(SEED + hash(lang) % 1000).shuffle(rows)
    i = 0
    for sp in SPLITS:
        k = target[lang][sp]
        out_rows[sp].extend(rows[i:i + k])
        i += k
    assert i == len(rows), f"{lang}: {i} assigned of {len(rows)}"

seen = set()
for sp in SPLITS:
    d = f"{OUT}/{sp}"
    os.makedirs(d, exist_ok=True)
    random.Random(SEED).shuffle(out_rows[sp])          # mix languages within the split
    with open(f"{d}/manifest.jsonl", "w") as fh:
        for r in out_rows[sp]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    imgs = [r["image"] for r in out_rows[sp]]
    assert not (seen & set(imgs)), "document leaked across splits"
    seen |= set(imgs)
    from collections import Counter
    by = Counter(r["language"] for r in out_rows[sp])
    print(f"    {sp:11s} {len(out_rows[sp]):4d} docs   " + "  ".join(f"{k}={v}" for k, v in sorted(by.items())))

print(f"    total documents placed: {len(seen)}  (expect 747)")
print("    leakage check: no document in two splits  OK")
