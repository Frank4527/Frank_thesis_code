# Experiment design

Selective reversion of overfitting components in DoRA — what the runs under
`runs/thesis_experiment_runs/` actually do, and what they can and cannot support.

This is the reference for the thesis methods section. Mechanics of the data splits
are in `data/Thesis_datasets/SEEDS.md`; mechanics of seed isolation are in
`runs/thesis_experiment_runs/README.md`.

---

## 1. The question

A model fine-tuned on task A, then fine-tuned again on task B, forgets task A.
DoRA decomposes each weight into a magnitude and a direction. **If forgetting is
carried disproportionately by one of those two components, reverting only that
component should recover task A while keeping task B** — at zero extra training
cost, unlike replay.

Two directions are run so the finding cannot be an artifact of task order:

| experiment | stage 1 (task A) | stage 2 (task B) |
|---|---|---|
| `Bankstatement2Quaterly_rep_rerun*` | bank statements | quarterly reports |
| `Quaterly_rep2Bankstatement_rerun*` | quarterly reports | bank statements |

---

## 2. The decomposition

DoRA writes a weight as a per-output-row magnitude times a unit direction:

    W = m · V/‖V‖ = m · D        m ∈ R^out (one scalar per output row)
                                 D has unit-norm columns along the layer's dim

After stage 2 there are two versions of each: `(m0, D0)` from the stage-1 model
(the anchor, `W0`) and `(m_ft, D_ft)` from the stage-2 model. The reversion
operations interpolate one component back toward the anchor, in
`Identification_and_Reversion/dora_reversion.py`:

    revert_magnitude(α)   m_rev = (1-α)·m0 + α·m_ft ;  W = m_rev · D_ft
    revert_direction(β)   D_rev = normalise((1-β)·D0 + β·D_ft) ;  W = m_ft · D_rev
    revert_both(α, β)     both of the above

Convention throughout: **1 = keep the fine-tuned value, 0 = fully back to the
stage-1 anchor.** Smaller dial = more reversion. The renormalise in
`revert_direction` is required because a blend of two unit vectors is not itself
unit length.

`revert_direction` and `revert_magnitude` are used rather than `revert_both` with
one dial pinned at 1.0, even though those are mathematically identical. The
`revert_both` path renormalises a blend that is already unit length, and the fp32
rounding difference survives the cast to bf16 (~8 mantissa bits) as ~2e-4 weight
differences — enough to change a greedily decoded digit and move field-F1 by ~0.02
on a small set. See `pipeline/evalcore.py:apply_reversion`.

---

## 3. What is compared

Every row is the same 8B model on the same documents; only the weights differ.

| method | what it is | cost beyond stage 2 |
|---|---|---|
| **stage-1 model** | task A only — ceiling on A, floor on B | — |
| **stage-2 full DoRA** | no reversion — the forgetting case | — |
| **magnitude reversion** | `revert_magnitude(α)`, α from validation | none |
| **direction reversion** | `revert_direction(β)`, β from validation | none |
| **combined** | both dials at their selected values | none |
| **WiSE-FT** | `(1-c)·W0 + c·W_ft` on the whole weight, no component split | none |
| **rehearsal 1% / 5%** | stage 2 retrained with 1%/5% of task A replayed | one retrain |
| **joint** | one adapter trained on A ∪ B from the pretrained base | one retrain |

WiSE-FT is the baseline the method must beat: same "interpolate back toward the
anchor" idea, but without the magnitude/direction split. If component-selective
reversion is not better than WiSE-FT, the decomposition adds nothing.

Rehearsal is the standard continual-learning remedy (Scialom et al., EMNLP 2022);
joint training is the multi-task upper bound. Both cost a full retrain, which is
the practical case for preferring reversion if it comes close.

**Joint is direction-free** — it starts from the pretrained base and sees the union
of both tasks at once, so there is one joint model per seed, not one per direction.
The q2b runs reuse the b2q joint model of the *same seed* rather than retraining an
identical-in-expectation second draw (`data_prep/copy_joint.sh`, which enforces the
seed match). Report it as one measurement shared by both directions, not as two.

---

## 4. Protocol

1. **Stage 1** — DoRA fine-tune the pretrained base on task A.
2. **Merge** — fold the stage-1 adapter into full weights. This is `W0`, the anchor
   every reversion interpolates toward.
3. **Stage 2** — DoRA fine-tune on task B *on top of* `W0`. This adapter is what
   gets decomposed.
4. **Validation sweeps** — score both tasks at dial ∈ {0, 0.25, 0.5, 0.75, 1.0} for
   magnitude, direction and WiSE-FT independently.
5. **Select** the dial maximising the **mean of the two tasks' field-F1 on
   validation**, per seed, from that seed's own sweeps.
6. **Test** — apply the selected dials once to the held-out test split.

Steps 4–6 are separated deliberately. `run_evals.sh test` refuses to run while the
dials are unset, so no test number can be written at a dial chosen on different
data. The selected dials for seed 1 are recorded in each run's
`PROVENANCE/SELECTED_DIALS.md`.

`adapter_last` is used throughout: in-loop evaluation is disabled, so every run
trains the full 10 epochs with no early stop. Training cost is therefore pure
training and comparable across methods, and no method is cut short by a noisy
small-validation metric.

**Fixed for every run:** Qwen3-VL-8B-Instruct, DoRA rank 16, α 32, lr 2e-4,
10 epochs, batch 1 × grad-accum 4 × 4 GPUs (effective 16), bf16, max_pixels 589824,
training seed 0. One job at a time on all four GPUs, so wall-clock cost is
comparable between methods.

**Prompt:** one instruction for every dataset, at train and eval time —
`"Extract all fields from this document as JSON."` In the first (July) run bank
used a "receipt" prompt and quarterly a "document" prompt; the prompt then acted as
a task-retrieval cue and *understated* forgetting. With one shared prompt,
forgetting on quarterly→bank is near-total (0.704 → 0.042). This is why the July
experiment was superseded rather than extended.

**Metric:** field-level F1 from the Donut `JSONParseEvaluator` reimplementation
(exact match per extracted field), with `value_f1` and nTED also recorded.

---

## 5. The three splits

Each experiment is run on three independent document-level re-splits of the same
corpus, with per-split sizes held identical (`SEEDS.md`).

**Call these three document-level re-splits, not "three random seeds."** The
training seed is fixed at 0; what varies is which documents land in train/val/test.
A reader seeing "3 seeds" will assume independent training runs.

What they establish, and what they do not:

- Test sets are nearly disjoint across seeds (15% overlap on bank, **0%** on
  quarterly), so each seed genuinely evaluates on different held-out documents.
- **Training sets overlap ~80%** — the same corpus is re-dealt, not resampled.
  The three runs are positively correlated, closer to folds of a resampling
  procedure than to independent replications. Spread across seeds therefore
  *understates* the variability of genuinely independent datasets.
- **Quarterly's effective n is 4 documents per test set**, not 34–116 pages: pages
  of one report share company, layout and fields, so they are clustered, not
  independent observations. Twelve distinct documents across all three seeds is the
  real sample size.
- **Training volume is not constant.** Quarterly has 445/361/373 training pages at a
  fixed 10 epochs, so seed 2 receives ~19% fewer gradient steps than seed 1. Seed
  differences conflate split effect with training-quantity effect.
- Measurement resolution is ~0.05, from observed val→test shifts of the same models
  plus the ~0.02 bf16 floor. Differences below that are not resolvable here.

### How to analyse them

**Compare methods within a seed, then report the three paired differences.** Every
method within a seed is scored on the identical test set, so the split acts as a
blocking factor and pairing cancels "this split was easy or hard" — the noise that
dominates at n=4 documents.

    good : "beat 1% replay on all three splits, by +0.03 / +0.05 / +0.02"
    weak : "0.61 ± 0.04 vs 0.58 ± 0.05"

Report all three per-seed values explicitly; at n=3 the individual numbers carry
more information than a standard deviation. Do not run significance tests — with
three correlated splits and four test documents no p-value would be meaningful.
"Consistent in sign across all three splits" is the defensible claim.

Bank conclusions are reasonably supported (75 test documents per seed, 195 distinct
across seeds, 26% of the corpus). **Quarterly conclusions stay weak regardless of
seeds** — with 34 documents total, re-splitting re-deals the same small hand. That
is a property of the corpus, not of the method.

---

## 6. Reproducing a run

    cd runs/thesis_experiment_runs/<experiment>/scripts
    bash run_all.sh train                    # mixtures, trainings, merge
    bash run_evals.sh val                    # anchors + the three sweeps + baselines
    # read the sweeps, pick each dial's argmax, then:
    MAG_A=<a> DIR_B=<b> WISEFT_C=<c> bash run_evals.sh test

`config.sh` is the only file to edit between experiments: `EXP_NAME`, `DATA_SEED`,
and the two dataset names. Both run scripts are byte-identical across all seeds of
a direction. Add `DRY_RUN=1` to print every command without executing it.

Verify seed isolation at any point:

    python code/Thesis_Experiment/data_prep/audit_seed_isolation.py

---

## 7. Known limitations

1. Training volume varies across quarterly seeds (§5).
2. Quarterly's effective sample size is 4 test documents (§5).
3. Dials are selected on 3 validation documents for quarterly — seed 3's validation
   is 26 pages, so its operating point is the noisiest of the three.
4. Per-document scores were not retained, so bootstrap confidence intervals cannot
   be computed retrospectively.
5. **Training-cost figures are wall-clock on a shared box, and are the weakest
   numbers in the study.** Three separate effects contaminate them:

   - *Thermal throttling.* Identical data (bank, 598 documents) took 2:14:31 on
     seed 1 and 3:00:05 on seed 2 — **+34% for the same work** — with GPU clocks
     falling from 1410 MHz to ~315 MHz at 85 C. `clean_measurement: true` does NOT
     mean comparable: it only means no other user was present.
     `logs/gpu_thermal.csv` samples clocks and temperature alongside the run.
   - *Contention with other users.* On 2026-08-05 another user's job took 13.9 GB
     on one GPU and all three seed-2 baseline trainings died with CUDA OOM.
     `foreign_gpu_procs` in timings.jsonl records who was present for each job.
   - *Retries.* A job that OOMs and is retried has its wall-clock split across a
     FAILED record and a later ok record, and the retry resumes from the last
     checkpoint rather than from zero. Affected jobs are listed in
     `INTERRUPTIONS.log` and must be summed, not read off the ok record.

   Compare method costs **within one experiment run**, where jobs execute
   back-to-back under similar conditions. Treat cross-seed cost comparisons as
   unreliable unless the thermal log and `clean_measurement` agree for both.

   Memory footprint was deliberately not reduced to avoid OOM: batch size, grad
   accumulation and max_pixels must stay identical to seed 1, or the runs stop
   being comparable.
6. **The dial-selection rule can pick a degenerate operating point.** Dials are
   chosen by maximising the *unweighted mean* of the two tasks' validation
   field-F1 (section 4). That criterion is linear, so it is indifferent between a
   balanced point and one where a task has been abandoned, and maximising a linear
   objective tends to select extreme points. On b2q seed 2:

   | dial | bank (old) | quarterly (new) | mean |
   |---|---|---|---|
   | 0.0 | 0.7158 | 0.0000 | 0.3579 |
   | 0.25 | 0.6886 | 0.0517 | **0.3701 selected** |
   | 0.5 | 0.5210 | 0.1974 | 0.3592 |

   The rule separates "new task dead" from "both tasks working" by only 0.0013,
   and selected beta=0.25, which on test gave quarterly 0.1055. For WiSE-FT it
   selected c=0.0, whose test row is identical to the stage-1 model — i.e. the
   criterion chose a model that had learned nothing of the new task.

   The root cause is a validation set too small to be representative. Seed 2's
   quarterly ceiling reads 0.2850 on validation (3 documents) but 0.5585 on test
   — a gap of 0.27, versus 0.087 on seed 1. Test performance is nearly identical
   across seeds (0.5560 vs 0.5585), so seed 2's task was never harder; its
   validation set simply read low, which made the mean bank-dominated.

   The rule was deliberately NOT changed after observing this: it is
   pre-specified and applied identically to every seed, and switching criteria
   once results are visible is a researcher-degrees-of-freedom problem. The
   consequence is that b2q seed 2's reversion rows understate the method — they
   report a badly-chosen operating point, not a failure of the method. Alternative
   criteria that cannot be maximised by discarding a task (harmonic or geometric
   mean of the two scores, or maximising old-task retention subject to a floor on
   the new task) would select beta=0.5 there; this is noted as future work rather
   than applied retrospectively.

7. The canonical weight reconstruction differs from PEFT's `PeftModel` by ~0.005
   field-F1 on quarterly. Exact equality is not achievable: PEFT applies the
   magnitude as an activation scale and never materialises the merged weight.

---

## 8. How the runs were executed

The scientific content is `datasets.py`, `pipeline/`, and the decomposition in
`Identification_and_Reversion/`. Everything under `data_prep/` named `*_chain.sh`,
`watchdog.sh`, `gpu_sampler.sh` or `protect_models.sh` is **operational scaffolding**:
it sequences long unattended runs on a shared machine and has no bearing on the
method or the results. It is documented in `runs/thesis_experiment_runs/RUNNING.md`
and is not thesis material beyond this note.

The one place it touches the results is dial selection. `select_dials.py` applies
the rule in section 4 mechanically, so no operating point depends on when a human
happened to read a sweep. It was validated by reproducing both seed-1 selections
(b2q 0.5 / 0.75 / 0.75, q2b 0.25 / 0.5 / 0.5) from the sweep files alone.
