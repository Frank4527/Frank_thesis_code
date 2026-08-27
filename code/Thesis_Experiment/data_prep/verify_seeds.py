"""Full verification of the three-seed dataset split, plus a provenance record.

Checks, for both datasets:
  1. seed1 is still byte-identical to the live original  (nothing was clobbered)
  2. every seed has identical split sizes                (bank: rows; quarterly: documents)
  3. the union of rows across splits is the same set in every seed (same corpus, re-dealt)
  4. no document appears in two splits within a seed     (no leakage)
  5. seeds are genuinely different draws                 (low test-set overlap)
  6. bank language mix is identical across seeds
Then writes SEEDS.md + checksums.md5 into the dataset root.
"""
import hashlib
import json
from collections import Counter

TD = "/home/frank/data/Thesis_datasets"
SPLITS = ["training", "validation", "test"]
SEEDS = [1, 2, 3]
FAIL = []


def check(cond, msg):
    print(f"    [{'OK ' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAIL.append(msg)


def path(ds, seed, sp):
    d = ds if seed is None else f"{ds}_seed{seed}"
    return f"{TD}/{d}/{sp}/manifest.jsonl"


def rows(ds, seed, sp):
    return [json.loads(l) for l in open(path(ds, seed, sp))]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def docid(ds, r):
    # bank: one row per document.
    # quarterly: .../files/<DOC_ID>/pages/page-NNNN/image.png -- the report is <DOC_ID>.
    # (the parent dir is page-NNNN, which is unique per page, so it is NOT the doc key)
    if ds == "Bank_Full":
        return r["image"]
    return r["image"].split("/files/")[1].split("/")[0]


for ds, unit in [("Bank_Full", "documents"), ("Quarterly_rep_En", "pages")]:
    print(f"\n=== {ds} ===")

    print("  1. seed1 preserves the original split")
    for sp in SPLITS:
        check(md5(path(ds, None, sp)) == md5(path(ds, 1, sp)),
              f"{sp}: seed1 byte-identical to live {ds}/")

    print(f"  2. split sizes identical across seeds")
    doc_sizes = {}
    for seed in SEEDS:
        doc_sizes[seed] = {sp: len({docid(ds, r) for r in rows(ds, seed, sp)}) for sp in SPLITS}
    ref = doc_sizes[1]
    for seed in SEEDS[1:]:
        check(doc_sizes[seed] == ref,
              f"seed{seed} document counts {doc_sizes[seed]} == seed1 {ref}")
    if ds == "Quarterly_rep_En":
        for seed in SEEDS:
            pg = {sp: len(rows(ds, seed, sp)) for sp in SPLITS}
            print(f"         seed{seed} pages: {pg}  (accepted: varies by report length)")

    print("  3. same corpus in every seed, only re-dealt")
    universe = {}
    for seed in SEEDS:
        universe[seed] = {docid(ds, r) for sp in SPLITS for r in rows(ds, seed, sp)}
    for seed in SEEDS[1:]:
        check(universe[seed] == universe[1],
              f"seed{seed} covers the same {len(universe[1])} documents as seed1")

    print("  4. no document straddles two splits")
    for seed in SEEDS:
        sets = [{docid(ds, r) for r in rows(ds, seed, sp)} for sp in SPLITS]
        overlap = (sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])
        check(not overlap, f"seed{seed}: no leakage across train/val/test")

    print("  5. seeds are genuinely different draws")
    t = {s: {docid(ds, r) for r in rows(ds, s, "test")} for s in SEEDS}
    for a, b in [(1, 2), (1, 3), (2, 3)]:
        ov = len(t[a] & t[b])
        check(ov < len(t[a]) * 0.6, f"test overlap seed{a}&seed{b} = {ov}/{len(t[a])}")
    print(f"         union of all three test sets = {len(t[1] | t[2] | t[3])} distinct documents")

    if ds == "Bank_Full":
        print("  6. language mix identical across seeds")
        for sp in SPLITS:
            mixes = [tuple(sorted(Counter(r["language"] for r in rows(ds, s, sp)).items()))
                     for s in SEEDS]
            check(len(set(mixes)) == 1, f"{sp}: {dict(mixes[0])} in all three seeds")

print("\n=== writing provenance ===")
lines = ["# Dataset seeds\n",
         "Three independent random re-splits, built for the 3-seed thesis runs.\n",
         "Splitting is BY DOCUMENT in both datasets, so no report's pages straddle a split.\n",
         "\n## Sizes\n"]
for ds in ["Bank_Full", "Quarterly_rep_En"]:
    lines.append(f"\n### {ds}\n\n| seed | training | validation | test |\n|---|---|---|---|\n")
    for seed in SEEDS:
        cells = []
        for sp in SPLITS:
            rs = rows(ds, seed, sp)
            nd = len({docid(ds, r) for r in rs})
            cells.append(f"{nd} docs" if ds == "Bank_Full" else f"{nd} docs / {len(rs)} pages")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |\n")
lines += [
    "\n## Notes\n\n",
    "- `Bank_Full_seed1` / `Quarterly_rep_En_seed1` are byte-identical copies of the\n",
    "  original split; all results produced before 2026-08 correspond to seed 1.\n",
    "- Bank splits are stratified by language, so every seed has the identical\n",
    "  language mix (de 226/28/28, po 201/25/25, en 171/21/22).\n",
    "- Quarterly document counts match exactly (27/3/4) but PAGE counts vary across\n",
    "  seeds because reports run from 2 to 50 pages. Matching pages exactly would\n",
    "  mean choosing which reports go where, so document counts were matched instead.\n",
    "- Built by `code/Thesis_Experiment/tools/build_bank_seed.py` and\n",
    "  `build_quarterly_seed.py`; both are deterministic given the seed.\n",
]
open(f"{TD}/SEEDS.md", "w").writelines(lines)
print(f"  wrote {TD}/SEEDS.md")

with open(f"{TD}/checksums.md5", "w") as fh:
    for ds in ["Bank_Full", "Quarterly_rep_En"]:
        for seed in SEEDS:
            for sp in SPLITS:
                p = path(ds, seed, sp)
                fh.write(f"{md5(p)}  {p[len(TD) + 1:]}\n")
print(f"  wrote {TD}/checksums.md5")

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + "; ".join(FAIL)))
