# Where Does Forgetting Live? — code and results

Submission package for the MSc thesis *Where Does Forgetting Live? Localising and
Reverting Catastrophic Forgetting in DoRA* (UCL, in partnership with Evolution AI).

Everything here is a **byte-identical copy** of the code that produced the results and
of the result files themselves; `CHECKSUMS.txt` records both. Nothing was renamed
inside `code/`, because the modules import each other by package path and renaming
them would break every import — the tidying is in the top-level layout and in
`scripts/`, which nothing imports.

```
thesis_submission/
├── README.md            you are here
├── CHECKSUMS.txt        md5 of every file in this package
├── docs/                design, data pipeline, split definitions, naming, provenance
├── code/                importable modules  (run with PYTHONPATH=code)
├── scripts/             the drivers, numbered in the order they were run
├── tests/               correctness checks (see docs/TESTS.md) -- 114 passing
├── figures/             the scripts behind the thesis's data figures
└── results/             every result file the thesis reports
```

## The 60-second version

The thesis asks where catastrophic forgetting is stored inside a DoRA-adapted
vision-language document extractor. DoRA writes each weight as a per-output-row
**magnitude** times a unit **direction**, `W = m · D`. After training task A then task
B, we can revert either component toward its task-A value independently:

| operation | file |
|---|---|
| `revert_magnitude(α)` → `((1−α)m₀ + αm_ft)·D_ft` | `code/Identification_and_Reversion/dora_reversion.py` |
| `revert_direction(β)` → `m_ft · normalise((1−β)D₀ + βD_ft)` | same file |

Reverting the magnitude does essentially nothing; reverting the direction recovers
most of the forgotten task. That is the central result.

## Start here

**`docs/WALKTHROUGH.md`** takes the study from the data through to the figures in
nine stops, in the order the work happened. If you are reviewing this repository for
the first time, read that instead of this list.

## Reading order

1. `docs/EXPERIMENT_DESIGN.md` — the design, the baselines, the limitations
2. `code/Identification_and_Reversion/dora_reversion.py` — the method, ~80 lines
3. `code/Identification_and_Reversion/dora_merge.py` — how it is applied to a live model
4. `docs/DATA_PIPELINE.md` + `docs/SEEDS.md` — the corpora and the three re-splits
5. `scripts/` in numerical order — what was actually run
6. `results/` — the JSON the tables are built from

## What is in `code/`

| path | role |
|---|---|
| `Identification_and_Reversion/dora_reversion.py` | **the method**: `decompose`, `revert_magnitude`, `revert_direction` |
| `Identification_and_Reversion/dora_merge.py` | `DoRAAdapter`, the four diagnostic versions, `apply_selective` |
| `Identification_and_Reversion/merge_adapter.py` | merges the stage-1 adapter into a base model |
| `Identification_and_Reversion/layer_drift.py` | per-layer, per-row direction drift ρ |
| `Identification_and_Reversion/layer_select.py` | picks layers to revert (top / bottom / random) |
| `Training_Dora/train_dora_ddp.py` | DoRA fine-tuning, 4×GPU DDP; used for every trained model |
| `Metrics/field_f1.py` | field-level F1 (Donut-style), value-F1, nTED |
| `Metrics/document_extraction.py` | generation + scoring loop |
| `Thesis_Experiment/datasets.py` | dataset registry; resolves `DATA_SEED` to manifests |
| `Thesis_Experiment/pipeline/evalcore.py` | model loading, reversion dispatch, scoring |
| `Thesis_Experiment/pipeline/eval_pair.py` | one configuration → one result JSON |
| `Thesis_Experiment/pipeline/sweep.py` | dial one component across α/β |
| `Thesis_Experiment/pipeline/layer_sweep.py` | the same, restricted to selected layers |
| `Thesis_Experiment/pipeline/es_rule.py` | the patience-3 early-stopping rule |
| `Thesis_Experiment/data_prep/` | split construction, verification, seed-isolation audit, dial selection |

Two implementation details worth knowing before reading the code:

- **`evalcore.apply_reversion` dispatches dial 1 to the canonical full-DoRA path.**
  `revert_direction(1.0)` renormalises an already-unit vector; in fp32 that is a
  no-op, but it survives the cast to bf16 as ~2e-4 of weight noise — enough to change
  a greedily decoded token. The dispatch keeps every sweep's β=1 endpoint identical to
  the untouched stage-2 model. See the comment at the top of that function.
- **`merge_adapter.py` sits under `sequential/` in the working tree.** That directory
  is otherwise superseded, but this one file is live and is called by `run_all.sh`.
  It is filed here under `Identification_and_Reversion/` where it belongs.

## What is in `results/`

| directory | contents |
|---|---|
| `primary/<run>/` | the six main runs: stage-1, stage-2, the three reversions, WiSE-FT, the three sweeps, joint, rehearsal 1%/5%, and that run's `DIAL_SELECTION.txt` |
| `early_stopping/<run>/` | `es_traj/` (a checkpoint every 10 steps) and `es_reversion/` (the sweep applied to the early-stopped model, plus `SELECTED_STEP.txt`) |
| `layer_reversion/<run>/` | the three arms — top-25%, random-25%, bottom-25% — with the selected layer lists |
| `layer_drift/` | per-layer drift summaries, one JSON per run |

The `.npz` per-row drift arrays (163 MB) are **not** included: they are a
re-derivable intermediate, and `layer_drift.py` regenerates them from the adapters.

See `docs/NAMING.md` for how to read a result filename.

## Reproducing

Requires 4×A100, the Qwen3-VL-8B-Instruct weights, and the document corpora — none of
which are redistributable. With those in place, run `scripts/` in numerical order with
`PYTHONPATH=code`. `docs/PROVENANCE.md` records which script produced which result
directory.
