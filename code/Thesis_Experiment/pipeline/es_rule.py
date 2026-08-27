"""Apply the early-stopping rule to whatever checkpoints have been scored so far.

Mirrors DevEval in train_dora_ddp.py exactly:
    strictly greater sets a new best and resets the counter;
    a tie counts as a failure; stop after `patience` consecutive failures.

Exit 0 = triggered (stop scoring), exit 1 = need more checkpoints.
"""
import glob, json, os, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("results_dir")
ap.add_argument("--new-key", required=True, help="e.g. quarterly_f1 or bank_f1")
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--pattern", default="step*__both__val.json")
ap.add_argument("--print-best-step", action="store_true",
                help="print the SELECTED step (the best checkpoint) and exit 0")
a = ap.parse_args()

pts = []
for f in glob.glob(f"{a.results_dir}/{a.pattern}"):
    d = json.load(open(f))
    step = int(os.path.basename(f).split("step")[1].split("__")[0])
    pts.append((step, d[a.new_key]))
pts.sort()
if not pts:
    if a.print_best_step:
        print(0); sys.exit(0)
    sys.exit(1)

# The selected model is the best checkpoint seen BEFORE the rule fires. An in-loop
# run halts at the trigger and never evaluates anything after it, so taking the best
# over all scored checkpoints would be oracle selection, not early stopping.
best, best_step, bad, trigger = -1.0, None, 0, None
oracle_step, oracle = None, -1.0
for step, v in pts:
    if v > oracle:
        oracle, oracle_step = v, step          # best over everything scored
    if trigger is not None:
        continue                                # past the stopping point
    if v > best:
        best, best_step, bad = v, step, 0
    else:
        bad += 1
        if bad >= a.patience:
            trigger = step
if a.print_best_step:
    print(best_step)
    sys.exit(0)
if not a.quiet:
    print(f"  scored {len(pts)} checkpoints, steps {pts[0][0]}-{pts[-1][0]}")
    print(f"  best so far: step {best_step} = {best:.4f}   consecutive non-improvements: {bad}")
    print(f"  -> {'TRIGGERED at step ' + str(trigger) if trigger else 'not triggered, continue'}")
    if trigger is not None and oracle_step != best_step:
        print(f"  note: best over ALL scored checkpoints is step {oracle_step} ({oracle:.4f}), "
              f"but an in-loop run stops at {trigger} and would return step {best_step}")
sys.exit(0 if trigger else 1)
