# Walkthrough — the study from start to finish

A narrative path through this repository, in the order the work actually happened.
Nine stops, roughly 40 minutes if you open each file as you go. Everything referred
to is in this repository unless marked otherwise.

---

## The question

A model fine-tuned on task A, then on task B, forgets A. DoRA writes every adapted
weight as a per-output-row **magnitude** times a unit **direction**, `W = m · D`, and
trains the two separately. So they can be reverted separately. If forgetting is
carried disproportionately by one of them, reverting only that one should recover A
while keeping B — at no retraining cost.

**Stop 0 — `docs/EXPERIMENT_DESIGN.md`.** The design, the baselines, and the
limitations, written before the results.

---

## 1. The data — `code/Thesis_Experiment/data_prep/`

Two corpora: **bank statements** (747 single-page documents, three languages) and
**quarterly reports** (34 documents spanning 515 annotated pages). Three
document-level re-splits of each.

- `build_bank_seed.py`, `build_quarterly_seed.py` — splitting is **document-wise, not
  page-wise**. A quarterly report runs to 50 pages; splitting by page would put pages
  of one document in both train and test.
- `verify_seeds.py` — the splits are the right size and do not overlap.
- `audit_seed_isolation.py` — reads the access logs and confirms no run ever opened
  another split's manifests.
- `code/Thesis_Experiment/datasets.py` — the registry. `DATA_SEED` has **no default**:
  an unset seed raises rather than quietly falling back to split 1.

**The asymmetry to keep in mind:** an 80/10/10 document split gives bank a 75-document
test set and quarterly a **four**-document one. Bank carries the study; quarterly
results are sign checks. `docs/SEEDS.md` has the exact counts.

## 2. Training — `code/Training_Dora/train_dora_ddp.py`

One function trains everything: stage 1, stage 2, and all three retraining baselines.
DoRA rank 16, α 32, lr 2e-4, 10 epochs, batch 1 × accum 4 × 4 GPUs, bf16.

Two things worth noting while you are in this file: the budget is **fixed at 10 epochs
with no validation-based model selection for any method**, so nothing in the results
table is selection-biased; and the in-loop dev-eval callback is registered only when
`--eval-every` is a real interval — the primary runs pass a sentinel and get no
callback at all.

## 3. The method — `code/Identification_and_Reversion/dora_reversion.py`

The core of the thesis, about 80 lines of tensor algebra.

```
decompose(W)            ->  m = ‖W‖ per row,  D = W/m  (unit rows)
revert_magnitude(α)     ->  ((1−α)·m₀ + α·m_ft) · D_ft
revert_direction(β)     ->  m_ft · normalise((1−β)·D₀ + β·D_ft)
```

Convention: **1 keeps the trained value, 0 reverts fully.** The renormalise in
`revert_direction` is needed because a blend of two unit vectors is not itself unit.

The property that makes any of this possible is that `m_ft` is a *separately trained
parameter*, not `‖W₀ + sBA‖`. `tests/verify_math.py` asserts exactly that — if the two
were equal there would be nothing to revert independently.

**`dora_merge.py`** applies it to a live model: `VERSION_FNS` builds the four
diagnostic versions (base, mag_only, dir_only, full), and `apply_selective` applies a
reversion to a chosen subset of layers.

## 4. Evaluation — `code/Thesis_Experiment/pipeline/` and `code/Metrics/`

- `field_f1.py` — field-level F1, a Donut-style exact match on (field-path, value)
  pairs, pooled over the corpus rather than averaged per document.
- `evalcore.py` — loading, reversion dispatch, scoring. **Read the comment on
  `apply_reversion`**: at dial 1 it takes the canonical full-DoRA path rather than
  `revert_direction(1.0)`, because that call renormalises an already-unit vector and
  the fp32 no-op survives the bf16 cast as ~2e-4 of weight noise. `tests/test_bf16_dial1.py`
  measures it: 0.0013% of weights land on a different bf16 value.
- `sweep.py` — dials one component across {0, .25, .5, .75, 1} and scores both tasks.

## 5. Running it — `scripts/`, numbered in run order

`10`–`14` the primary experiment, `20`–`22` early stopping, `30`–`32` the
layer-targeted study. `docs/PROVENANCE.md` maps each to the results it produced.

Two design points visible in these files: the joint baseline is **order-free**, so it
is trained once per split and shared between the two task orders (see the comment in
`13_run_all_q2b.sh`); and the operating point is chosen per run by `select_dials.py`
from that run's **own** validation sweeps, never pooled.

## 6. The main result — `results/primary/`

Six runs: two task orders × three re-splits.

Open any `stage2/stage2-dirsweep__both__val_rerun.json` and
`stage2-magsweep__both__val_rerun.json` side by side. The magnitude sweep barely
moves; the direction sweep traces a full trade-off curve. That is the finding.

Checks you can make from the files alone:
- the `"1.0"` entries of the two sweeps are identical — same model, two routes
- `dirsweep` at `"0.0"` is `mag_only`; `magsweep` at `"0.0"` is `dir_only`
- `DIAL_SELECTION.txt` shows the rule's arithmetic, including when the argmax is a
  corner (it flags this rather than hiding it)

## 7. Does training less do the same? — `results/early_stopping/`

Stage 2 retrained with a checkpoint every 10 optimiser steps, stopped by a
**patience-3 rule on the new task only** — the old task is never consulted, since a
rule that watched it would be choosing an operating point rather than training less.

`code/Thesis_Experiment/pipeline/es_rule.py` returns the best checkpoint seen *before*
the trigger, not the best overall. Re-run it over any `es_traj/` directory and it will
reproduce that run's `SELECTED_STEP.txt`; on two runs the oracle best is a later step,
which is exactly the difference between early stopping and hindsight.

The trajectories also show *when* the damage happens. At step 10 — under half an epoch
— the new task is still unlearned everywhere while the old task has already lost up to
76% of its score.

## 8. Is forgetting localised in *space* too? — `results/layer_reversion/`

`layer_drift.py` measures, for every adapted layer, how far the update rotates each
output row: ρ = ‖sBA‖_row / ‖W₀‖_row. `layer_select.py` then picks 63 of 252 layers —
the most-drifted quartile, a random quartile, or the least-drifted quartile — and
`layer_sweep.py` reverts only those.

Three arms, same layer budget, same dials; only the selection rule differs. That is
what makes it a controlled comparison. Checks: the three arms select the same count,
top and bottom are disjoint, and every `layer-*-dirsweep` JSON records
`n_layers_reverted` = 63 at every β<1 and 0 at β=1.

## 9. Figures — `figures/`

`make_sweeps_fig_thesis.py` produces the sweep figure and
`make_drift_fig_thesis.py` the magnitude/direction histogram — the two data figures in
the thesis. The remaining figures in the write-up are schematic diagrams and live in
the thesis repository alongside the LaTeX.

---

## If you only look at three things

1. **`code/Identification_and_Reversion/dora_reversion.py`** — the method, in full.
2. **`results/primary/<any run>/stage2/stage2-{dir,mag}sweep__both__val_rerun.json`** —
   the magnitude sweep going nowhere while the direction sweep moves.
3. **`docs/PROVENANCE.md`** — what was run, what was regenerated and why, and the three
   places where the record is imperfect.

## Verifying rather than trusting

```bash
md5sum -c CHECKSUMS.txt                      # 457 files, all covered
PYTHONPATH=code python tests/test_reversion_math.py
PYTHONPATH=code python tests/test_metric_rule_selection.py
```

`docs/TESTS.md` lists all six test files — 114 checks, all passing as of 2026-08-27 —
and says what each one establishes.
