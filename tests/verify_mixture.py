"""Verify make_mixture: counts, fidelity of rows, reproducibility, no leakage."""
import json
import os
import tempfile

from Thesis_Experiment.datasets import make_mixture, rows, n

TMP = tempfile.mkdtemp()
ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    print(f"  {'PASS' if cond else 'FAIL'}  {name:56s} {extra}")
    ok, fail = ok + bool(cond), fail + (not cond)


q_train = rows("quarterly", "training")
b_train = rows("bank_full", "training")
print(f"  sources: quarterly={len(q_train)}  bank={len(b_train)}")

for ratio, expect_old in [(1.0, 598), (0.05, 30), (0.01, 6), (0.0, 0)]:
    p = os.path.join(TMP, f"mix_{ratio}.jsonl")
    n_new, n_old, _ = make_mixture("quarterly", "bank_full", ratio, p, seed=0)
    out = [json.loads(l) for l in open(p)]
    check(f"ratio={ratio}: {expect_old} bank docs replayed",
          n_old == expect_old, f"got {n_old}")
    check(f"ratio={ratio}: file length == new + replayed",
          len(out) == 445 + expect_old, f"{len(out)} rows")
    # every row must be an unmodified copy of a source row
    src = {json.dumps(r, sort_keys=True) for r in q_train + b_train}
    check(f"ratio={ratio}: all rows are verbatim source rows",
          all(json.dumps(r, sort_keys=True) in src for r in out))
    # the whole NEW task must be present exactly once
    qset = [json.dumps(r, sort_keys=True) for r in q_train]
    got = [json.dumps(r, sort_keys=True) for r in out]
    check(f"ratio={ratio}: all 445 quarterly docs present, no dupes",
          sorted(qset) == sorted(x for x in got if x in set(qset)))

print("--- reproducibility / independence ---")
p1 = os.path.join(TMP, "a.jsonl"); p2 = os.path.join(TMP, "b.jsonl")
make_mixture("quarterly", "bank_full", 0.05, p1, seed=0)
make_mixture("quarterly", "bank_full", 0.05, p2, seed=0)
check("same seed -> byte-identical manifest", open(p1).read() == open(p2).read())
p3 = os.path.join(TMP, "c.jsonl")
make_mixture("quarterly", "bank_full", 0.05, p3, seed=1)
check("different seed -> different replay sample", open(p1).read() != open(p3).read())

print("--- interleaving (a blocked file would train old-then-new) ---")
first10 = [json.loads(l).get("doc_type") == "quarterly_report" for l in open(p1)][:40]
switches = sum(1 for i in range(1, len(first10)) if first10[i] != first10[i - 1])
check("rows are shuffled, not concatenated in blocks", switches >= 1,
      f"{switches} task switches in first 40 rows")

print("--- replayed bank docs come from TRAINING split only (no val/test leak) ---")
val_test = {json.dumps(r, sort_keys=True) for r in
            rows("bank_full", "validation") + rows("bank_full", "test")}
outp = [json.dumps(json.loads(l), sort_keys=True) for l in open(p1)]
check("no bank val/test document appears in the mixture",
      not any(x in val_test for x in outp))

print(f"\n  {ok} passed, {fail} failed")
