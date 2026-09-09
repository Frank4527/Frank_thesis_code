"""Correctness tests for the metric, the stopping rule, layer selection and the
dataset registry -- run against the real modules on the server."""
import os, sys, os, json, subprocess, tempfile
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "code"))

P = F = 0
def chk(name, cond, detail=""):
    global P, F
    if cond: P += 1; print(f"  PASS  {name}")
    else:    F += 1; print(f"  FAIL  {name}   {detail}")

print("=" * 74); print("1. FIELD-LEVEL F1  (Metrics/field_f1.py)"); print("=" * 74)
from Metrics.field_f1 import corpus_field_f1, flatten_fields, normalize

chk("identical prediction and gold gives F1 = 1",
    corpus_field_f1([{"a": "1", "b": "2"}], [{"a": "1", "b": "2"}]) == 1.0)
chk("completely wrong prediction gives F1 = 0",
    corpus_field_f1([{"a": "1"}], [{"b": "2"}]) == 0.0)
chk("empty prediction against empty gold gives 0 (no fields, defined as 0)",
    corpus_field_f1([{}], [{}]) == 0.0)
# one right, one wrong, gold has two: tp=1, fp=1, fn=1 -> 1/(1+1) = 0.5
v = corpus_field_f1([{"a": "1", "b": "X"}], [{"a": "1", "b": "2"}])
chk("tp=1 fp=1 fn=1 gives 0.5", abs(v - 0.5) < 1e-12, f"got {v}")
# a missing field is a false negative: tp=1, fn=1 -> 1/(1+0.5) = 0.667
v = corpus_field_f1([{"a": "1"}], [{"a": "1", "b": "2"}])
chk("one of two fields recovered gives 2/3", abs(v - 2/3) < 1e-9, f"got {v}")
# scoring is over the corpus, not averaged per document
v_corpus = corpus_field_f1([{"a": "1"}, {"b": "X"}], [{"a": "1"}, {"b": "2"}])
chk("corpus-level pooling (not a per-document mean)", abs(v_corpus - 0.5) < 1e-12, f"got {v_corpus}")
chk("value matters, not just the key",
    corpus_field_f1([{"a": "1"}], [{"a": "2"}]) == 0.0)
chk("nested structures are flattened to (path, value) pairs",
    len(flatten_fields(normalize({"x": {"y": "1", "z": "2"}}))) == 2)
# duplicates: gold has two identical values, prediction one -> tp=1, fn=1
g = {"items": [{"v": "1"}, {"v": "1"}]}
p = {"items": [{"v": "1"}]}
vd = corpus_field_f1([p], [g])
chk("repeated values are matched with multiplicity, not set-deduped", 0.0 < vd < 1.0, f"got {vd}")

print()
print("=" * 74); print("2. EARLY-STOPPING RULE  (pipeline/es_rule.py)"); print("=" * 74)
RULE = os.path.join(_REPO, "code/Thesis_Experiment/pipeline/es_rule.py")
PY = sys.executable

def run_rule(scores, patience=3):
    """scores: {step: value} -> (selected_step, triggered)"""
    d = tempfile.mkdtemp()
    for st, v in scores.items():
        json.dump({"quarterly_f1": v, "quarterly": {"field_f1": v, "n": 1}},
                  open(f"{d}/step{st}__new__val.json", "w"))
    sel = subprocess.run([PY, RULE, d, "--new-key", "quarterly_f1", "--patience", str(patience),
                          "--pattern", "step*__new__val.json", "--print-best-step", "--quiet"],
                         capture_output=True, text=True).stdout.strip()
    trig = subprocess.run([PY, RULE, d, "--new-key", "quarterly_f1", "--patience", str(patience),
                           "--pattern", "step*__new__val.json", "--quiet"],
                          capture_output=True, text=True).returncode == 0
    return int(sel), trig

s, t = run_rule({10: .1, 20: .2, 30: .3, 40: .4})
chk("monotone improvement never triggers", (s, t) == (40, False), f"{s},{t}")
s, t = run_rule({10: .1, 20: .5, 30: .4, 40: .3, 50: .2})
chk("three consecutive non-improvements trigger, best-before returned",
    (s, t) == (20, True), f"{s},{t}")
s, t = run_rule({10: .1, 20: .5, 30: .4, 40: .3, 50: .2, 60: .9})
chk("a later, higher peak AFTER the trigger is NOT selected (no hindsight)",
    (s, t) == (20, True), f"{s},{t}")
s, t = run_rule({10: .5, 20: .5, 30: .5, 40: .5})
chk("ties count as non-improvement (strictly-greater rule)", (s, t) == (10, True), f"{s},{t}")
s, t = run_rule({10: .1, 20: .5, 30: .4, 40: .6, 50: .3, 60: .2, 70: .1})
chk("the counter resets on a new best", (s, t) == (40, True), f"{s},{t}")
s, t = run_rule({10: .1, 20: .5, 30: .4, 40: .3}, patience=2)
chk("patience is honoured (2 fires earlier than 3)", (s, t) == (20, True), f"{s},{t}")

print()
print("=" * 74); print("3. LAYER SELECTION  (layer_select.py)"); print("=" * 74)
SEL = os.path.join(_REPO, "code/Identification_and_Reversion/layer_select.py")
out = {}
for mode in ("top", "bottom", "random"):
    o = f"/tmp/_sel_{mode}.json"
    subprocess.run([PY, SEL, "--run", "b2q_seed1", "--quantile", "0.75",
                    "--mode", mode, "--stat", "rho_median", "--out", o],
                   capture_output=True, text=True)
    out[mode] = json.load(open(o))
drift = json.load(open(os.path.join(_REPO, "results/layer_drift/layer_drift_b2q_seed1.json")))
rho = {l["layer"]: l["rho_median"] for l in drift["layers"]}

chk("every arm selects the same count (63 of 252)",
    len({len(out[m]["layers"]) for m in out}) == 1 and len(out["top"]["layers"]) == 63,
    str({m: len(out[m]["layers"]) for m in out}))
chk("top and bottom are disjoint", not (set(out["top"]["layers"]) & set(out["bottom"]["layers"])))
chk("every top layer has drift >= every bottom layer",
    min(rho[l] for l in out["top"]["layers"]) > max(rho[l] for l in out["bottom"]["layers"]))
thr = out["top"]["threshold"]
chk("top set is exactly those at or above the reported threshold",
    all(rho[l] >= thr for l in out["top"]["layers"]))
chk("selected layers are real adapted layers", all(l in rho for m in out for l in out[m]["layers"]))
o2 = "/tmp/_sel_rand2.json"
subprocess.run([PY, SEL, "--run", "b2q_seed1", "--quantile", "0.75", "--mode", "random",
                "--seed", "0", "--stat", "rho_median", "--out", o2], capture_output=True)
chk("random selection is reproducible for a fixed seed",
    json.load(open(o2))["layers"] == out["random"]["layers"])

print()
print("=" * 74); print("4. DATASET REGISTRY GUARD  (Thesis_Experiment/datasets.py)"); print("=" * 74)
code = ("import sys; sys.path.insert(0, %r); "
        "from Thesis_Experiment.datasets import data_seed; print(data_seed())"
        % os.path.join(_REPO, "code"))
env = dict(os.environ); env.pop("DATA_SEED", None)
r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env)
chk("an unset DATA_SEED raises rather than defaulting to split 1",
    r.returncode != 0, f"rc={r.returncode} out={r.stdout.strip()[:40]}")
env["DATA_SEED"] = "2"
r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env)
chk("DATA_SEED is read from the environment", r.stdout.strip() == "2", r.stdout.strip())

print()
print("=" * 74); print(f"{P} passed, {F} failed"); print("=" * 74)
sys.exit(1 if F else 0)
