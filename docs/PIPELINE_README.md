# Experiment pipeline

Everything that computes a number lives in `code/`. An experiment folder under
`runs/thesis_experiment_runs/<NAME>/scripts/` contains **only** configuration and
two run scripts — no logic, no dataset paths, no prompts.

## Layout

| file | role |
|---|---|
| `Identification_and_Reversion/dora_reversion.py` | the method: decompose W into magnitude x direction, revert either component, build the four diagnostic versions |
| `Identification_and_Reversion/dora_merge.py` | `DoRAAdapter` (binds a trained adapter to a live model, keeps a pristine W0 snapshot), `VERSION_FNS`, `wiseft` |
| `Metrics/field_f1.py` | field-F1 + nTED, matching Donut's `JSONParseEvaluator` |
| `Metrics/document_extraction.py` | prompt, JSON target, generation, evaluation harness |
| `Thesis_Experiment/datasets.py` | **the registry**: dataset name -> manifests + prompt + label, and `make_mixture` for replay/joint baselines |
| `Thesis_Experiment/pipeline/evalcore.py` | load model / attach adapter / apply reversion / score one dataset |
| `Thesis_Experiment/pipeline/eval_pair.py` | one configuration -> one result JSON (old task + new task) |
| `Thesis_Experiment/pipeline/eval_adapter.py` | same, but loads the adapter with **PEFT** and no reversion machinery |
| `Thesis_Experiment/pipeline/sweep.py` | dial one component (direction / magnitude / wiseft) across alphas |
| `Thesis_Experiment/pipeline/jobrunner.sh` | run one job at a time, record wall time + GPU contention |
| `Thesis_Experiment/pipeline/gpu_pool.sh` | run N jobs at a time, one per GPU (evaluation only) |

## Adding an experiment

Copy an experiment folder, edit `config.sh` (`OLD_DS`, `NEW_DS`, hyper-parameters),
run. Adding a dataset is one line in `datasets.py` — and the prompt travels with the
dataset, which is what prevents the train/eval prompt drift that affected the first run.

## Which command produced which result

With `$E` = the experiment folder and `$W` = `$E/merged/base_<OLD_DS>_8b_seed0`:

| result file | produced by |
|---|---|
| `results/stage1/wbank__both__{val,test}.json` | `eval_pair.py --base $W` (no adapter) |
| `results/stage2/stage2__both__{val,test}.json` | `eval_pair.py --base $W --adapter $S2 --alpha-mag 1 --beta-dir 1` |
| `results/stage2/stage2-dirsweep__both__val.json` | `sweep.py --component direction --alphas 0,0.25,0.5,0.75,1` |
| `results/stage2/stage2-magsweep__both__val.json` | `sweep.py --component magnitude --alphas ...` |
| `results/stage2/wiseft-sweep__both__val.json` | `sweep.py --component wiseft --alphas ...` |
| `results/stage2/stage2-dir__both__test.json` | `eval_pair.py --alpha-mag 1 --beta-dir <selected>` |
| `results/stage2/stage2-mag__both__test.json` | `eval_pair.py --alpha-mag <selected> --beta-dir 1` |
| `results/stage2/stage2-dirmag__both__test.json` | `eval_pair.py --alpha-mag <mag*> --beta-dir <dir*>` (ablation) |
| `results/stage2/wiseft__both__test.json` | `eval_pair.py --wiseft <selected>` |
| `results/baselines/{joint,rehearsal001,rehearsal005}__both__{val,test}.json` | `eval_pair.py --base <W or pretrained> --adapter <baseline adapter>` |
| `timings.jsonl` | every job, via `jobrunner.sh` / `gpu_pool.sh` |

Operating points are selected on **validation** by max mean of the two tasks, then
reported once on **test**.

## Numerical precision (important)

Adapted weights are stored in **bf16** (~8 mantissa bits). Two routes that are
mathematically identical in fp32 can therefore write *different* bf16 weights, and
greedy decoding turns that into a different token and a different exact-match field.

Consequences, both handled in `evalcore.apply_reversion`:

* reverting only one component dispatches to `revert_direction` / `revert_magnitude`,
  never to `revert_both` (which renormalises an already-unit vector);
* dial = 1 always uses `VERSION_FNS["full"]`, so "no reversion" is one number
  regardless of which sweep it appears in.

Measured effect when this was **not** done: the same model scored quarterly 0.4518
via one route and 0.4307 via another. Treat differences below ~0.02 on these split
sizes as numerical noise.

## Verification

`tests/` reproduces the checks (CPU only, no GPU needed):

```
python tests/verify_metric.py    # field-F1 / nTED vs hand-computable cases (14 checks)
python tests/verify_math.py      # reversion algebra + revert_both equivalences (23 checks)
python tests/verify_mixture.py   # replay manifests: counts, fidelity, no val/test leakage (20)
```

`tests/ab_paths.py` needs a GPU: it loads one model and runs two code paths
back-to-back, comparing weights -> raw generations -> field-F1. It is what caught the
bf16 issue above, and is the right harness before trusting any refactor of the eval path.
