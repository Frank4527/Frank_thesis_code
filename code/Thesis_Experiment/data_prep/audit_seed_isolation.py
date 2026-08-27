"""Prove, from evidence on disk, that each run used only its own seed's data.

Three independent checks per experiment:
  A. config DATA_SEED vs every result JSON's recorded data_seed
  B. the manifest access log -- which split files the run actually opened
  C. seed-1 manifests still match the checksums taken when they were frozen

Run after any phase completes:  python audit_seed_isolation.py
"""
import glob
import hashlib
import json
import os
import re

R = "/home/frank/runs/thesis_experiment_runs"
TD = "/home/frank/data/Thesis_datasets"
FAIL = []


def bad(msg):
    FAIL.append(msg)
    print(f"    [FAIL] {msg}")


def ok(msg):
    print(f"    [OK ] {msg}")


exps = sorted(d for d in os.listdir(R) if "rerun" in d)

for e in exps:
    print(f"\n=== {e} ===")
    cfg = f"{R}/{e}/scripts/config.sh"
    m = re.search(r"^DATA_SEED=(\d+)", open(cfg).read(), re.M)
    if not m:
        bad(f"{e}: no DATA_SEED in config.sh")
        continue
    seed = int(m.group(1))
    print(f"    config DATA_SEED = {seed}")

    # -- A. result files ----------------------------------------------------
    res = glob.glob(f"{R}/{e}/results/**/*.json", recursive=True)
    stamped = unstamped = 0
    for f in res:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        got = d.get("data_seed")
        if got is None:
            unstamped += 1
        else:
            stamped += 1
            if got != seed:
                bad(f"{os.path.relpath(f, R)}: data_seed={got} but experiment is seed {seed}")
        # a result must not reference another experiment's adapter unless same seed.
        # checked for EVERY result, including ones written before the stamp existed --
        # those are exactly the cross-experiment copies worth catching.
        adp = str(d.get("adapter") or "")
        mm = re.search(r"thesis_experiment_runs/([^/]+)/", adp)
        if mm and mm.group(1) != e:
            other = f"{R}/{mm.group(1)}/scripts/config.sh"
            o = re.search(r"^DATA_SEED=(\d+)", open(other).read(), re.M) if os.path.exists(other) else None
            if not o or int(o.group(1)) != seed:
                bad(f"{os.path.relpath(f, R)}: borrows adapter from {mm.group(1)} "
                    f"(seed {o.group(1) if o else '?'}) but this is seed {seed}")
            else:
                ok(f"{os.path.basename(f)}: borrowed from {mm.group(1)}, same seed {seed}")
    if res:
        ok(f"{stamped} result files carry data_seed={seed}"
           + (f"; {unstamped} predate the stamp" if unstamped else ""))
    else:
        print("    (no results yet)")

    # -- B. manifest access log --------------------------------------------
    log = f"{R}/{e}/logs/manifest_access.tsv"
    if os.path.exists(log):
        paths = {l.split("\t")[1].strip() for l in open(log) if "\t" in l}
        wrong = sorted(p for p in paths
                       if re.search(r"_seed(\d+)/", p) and int(re.search(r"_seed(\d+)/", p).group(1)) != seed)
        if wrong:
            for p in wrong[:5]:
                bad(f"opened foreign-seed manifest: {p}")
        else:
            ok(f"all {len(paths)} manifests opened belong to seed {seed}")
            for p in sorted(paths):
                print(f"          {p.replace(TD + '/', '')}")
    else:
        print("    (no access log yet - written once the run starts)")

# -- C. seed-1 immutability ------------------------------------------------
print("\n=== seed-1 data untouched ===")
sums = f"{TD}/checksums.md5"
if os.path.exists(sums):
    n = drift = 0
    for line in open(sums):
        h, rel = line.strip().split("  ", 1)
        p = f"{TD}/{rel}"
        if not os.path.exists(p):
            bad(f"missing: {rel}")
            continue
        n += 1
        if hashlib.md5(open(p, "rb").read()).hexdigest() != h:
            drift += 1
            bad(f"CHANGED since freeze: {rel}")
    if not drift:
        ok(f"all {n} manifests across all three seeds match their frozen checksums")
    # writability is the other half: they should be read-only
    w = [p for p in glob.glob(f"{TD}/*_seed*/*/manifest.jsonl") if os.access(p, os.W_OK)]
    if w:
        bad(f"{len(w)} seed manifests are still writable")
    else:
        ok("every seed manifest is read-only on disk")
else:
    bad("checksums.md5 missing")

print("\n" + ("AUDIT PASSED" if not FAIL else f"AUDIT FAILED - {len(FAIL)} problem(s)"))
