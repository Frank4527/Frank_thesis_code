# Provenance

Generated 2026-08-27 from the working tree on the Evolution AI
GPU host. Every file in this package is a copy; nothing in the working tree was moved,
renamed or deleted, so `/home/frank/code` and `/home/frank/runs/thesis_experiment_runs`
remain the record of what actually ran.

Working-tree code revision: `b07fdcd`

## Scripts: original name → package name

| in this package | original path (under `/home/frank/`) | what it does | copy check |
|---|---|---|---|
| `scripts/10_config_b2q_example.sh` | `runs/thesis_experiment_runs/Bankstatement2Quaterly_rep_rerun/scripts/config.sh` | per-run configuration; one copy exists per run, this is split 1's | identical |
| `scripts/11_run_all_b2q.sh` | `runs/thesis_experiment_runs/Bankstatement2Quaterly_rep_rerun/scripts/run_all.sh` | builds the rehearsal/joint mixtures, trains stage 1, merges it, trains stage 2 and the baselines | identical |
| `scripts/12_run_evals_b2q.sh` | `runs/thesis_experiment_runs/Bankstatement2Quaterly_rep_rerun/scripts/run_evals.sh` | every evaluation for a b2q run: anchors, the three sweeps, the baselines, then the test set | identical |
| `scripts/13_run_all_q2b.sh` | `runs/thesis_experiment_runs/Quaterly_rep2Bankstatement_rerun/scripts/run_all.sh` | as above for q2b. Differs in one respect: it does NOT retrain the joint baseline | identical |
| `scripts/14_run_evals_q2b.sh` | `runs/thesis_experiment_runs/Quaterly_rep2Bankstatement_rerun/scripts/run_evals.sh` | as above for q2b | identical |
| `scripts/20_earlystop_train.sh` | `runs/thesis_experiment_runs/_es_train_chain.sh` | retrains stage 2 on every run with a checkpoint every 10 optimiser steps | identical |
| `scripts/21_earlystop_trajectory.sh` | `runs/thesis_experiment_runs/_es_traj_chain.sh` | scores each checkpoint on both tasks and applies the patience-3 rule | identical |
| `scripts/22_earlystop_reversion_sweep.sh` | `runs/thesis_experiment_runs/_es_sweep_chain.sh` | applies the direction sweep to the selected early-stopped checkpoint | identical |
| `scripts/30_layer_reversion_top25.sh` | `runs/thesis_experiment_runs/_layer_reversion_chain.sh` | reverts the 63 highest-drift layers across the five dials | identical |
| `scripts/31_layer_reversion_random25.sh` | `runs/thesis_experiment_runs/_layer_reversion_random_chain.sh` | the random control: 63 layers drawn at random, seed 0 | identical |
| `scripts/32_layer_reversion_bottom25.sh` | `runs/thesis_experiment_runs/_layer_reversion_bottom_chain.sh` | the bottom control: the 63 lowest-drift layers | identical |

## Results: which script produced which directory

| directory | produced by |
|---|---|
| `results/primary/<run>/stage1`, `stage2`, `baselines` | `11`–`14` |
| `results/primary/<run>/DIAL_SELECTION.txt` | `code/Thesis_Experiment/data_prep/select_dials.py`, from that run's own validation sweeps |
| `results/early_stopping/<run>/es_traj` | `20` then `21` |
| `results/early_stopping/<run>/es_reversion` | `22` |
| `results/layer_reversion/<run>/layer-top-*` | `30` |
| `results/layer_reversion/<run>/layer-random0-*` | `31` |
| `results/layer_reversion/<run>/layer-bottom-*` | `32` |
| `results/layer_drift/*.json` | `code/Identification_and_Reversion/layer_drift.py` |

## Known provenance gaps

**Split 1 predates the stamping.** `Bankstatement2Quaterly_rep_rerun` and
`Quaterly_rep2Bankstatement_rerun` were run before `data_seed` was recorded into each
result JSON and before the manifest-access audit existed, so their result files carry
`data_seed: null` and there is no `manifest_access.tsv` for them. The splits were
verified retrospectively by size: quarterly validation 36 / test 34 is unique to split
1, and the bank counts match. Splits 2 and 3 carry the stamp.

**Two split-1 sweep files were regenerated on 2026-08-16.**
`stage2-dirsweep__both__val_rerun.json` and `stage2-magsweep__both__val_rerun.json` for
`Bankstatement2Quaterly_rep_rerun` were originally produced on 2026-08-04, before
`apply_reversion` learned to dispatch dial 1 to the canonical full-DoRA path, so their
β=1 / α=1 rows carried bf16 renormalisation noise. Both were re-run with the current
code. The eight unaffected dials reproduced bit-identically and only the dial-1 row
moved, to the value the canonical stage-2 evaluation already reported. The originals
are preserved on the server under
`Bankstatement2Quaterly_rep_rerun/PROVENANCE/pre_dial1_fix_20260816/`.

**`stage2-peftcheck` exists for split 1 only.** It is a cross-check of the weight
reconstruction against PEFT's own `PeftModel`, not a result, and was run once. On that
run the two differ by 0.0046 field-F1 on the newly learned task and 0.0187 on the
forgotten one, where both models score near zero and exact-match F1 is unstable. Every
comparison in the thesis uses one reconstruction throughout, so the reported
differences are internally consistent; this check bounds how far the absolute numbers
could move under a different but equally valid reconstruction convention.

## Not included

- The `.npz` per-row drift arrays (163 MB). Re-derivable with `layer_drift.py`.
- Model weights, adapters and merged checkpoints (hundreds of GB).
- The document corpora, which are not redistributable.
- Training and evaluation logs; the timing figures quoted in the thesis come from the
  `timings.jsonl` records alongside them, and are shared-machine wall-clock.

## Integrity

`CHECKSUMS.txt` lists md5 for every file in this package. Verify with:

```
cd thesis_submission && md5sum -c CHECKSUMS.txt
```

---

## Added 2026-09-09: Experiment 4, and the figure scripts

The package first assembled on 2026-08-27 predated Experiment 4 and carried figure
scripts that no longer matched the figures in the thesis. Both were brought up to
date from the same GPU host; nothing there was moved, renamed or deleted.

### Experiment 4 (pretrained-ability recovery)

| here | copied from `/home/frank/` |
|---|---|
| `code/Pretrained_Ability/*.py` | `code/Pretrained_Ability/` |
| `results/pretrained_ability/split1/` | `runs/thesis_experiment_runs/_pretrained_ability_recovery/results/` |
| `results/pretrained_ability/split2/`, `split3/` | the matching `_seed2`, `_seed3` directories |
| `results/pretrained_ability/backbone/step0_base*.json` | `_pretrain_reversion/step0_base.json`, and `step0_base_{mmbench,ocrbench,summary}.json` from the split-1 results directory |
| `results/pretrained_ability/backbone/base_bank_split{1,2,3}.json` | `_pa_base_bank/seed{1,2,3}/dial_base_bank.json` |
| `scripts/40–42` | `_pretrained_ability_recovery/scripts/run_sweep.sh`, `_pa_seeds23_chain.sh`, `_pa_base_bank_chain.sh` |

All 45 cells of Table 5.12 were checked against these files after copying, and the
backbone row against `backbone/`.

OCRBench and CORD results are included for split 1 because they came from the same
sweep, but neither is reported in the thesis: the OCRBench axis was cut, and CORD
belongs to a study that was dropped.

### Figures

`figures/` now holds one script per data figure in the thesis and nothing else. The
scripts that were here before produced earlier versions of these figures under
different names (`sweeps.png`, `layer_frontier.png`) and no longer matched what the
thesis prints; the scripts behind the current figures were not on the host. The
present scripts were written against the stored results, and each was checked by
confirming the coordinates it plots equal the corresponding table in the thesis:
Table 5.2 for Figures 5.1, 5.2 and 5.4, Tables 5.8 and 5.9 for Figure 5.5, Table 5.11
for Figures 5.6 and 5.7, and the per-run medians for Figure 5.3.

`results/layer_drift/drift_hist.npz` is derived, not copied: the per-unit arrays in
`_layerdrift/layer_drift_*.npz` are about 34 MB per run and cannot be shipped, so
`figures/derive_drift_hist.py` reduces them to the bin counts Figure 5.3 needs.

### Dial-selection logs

`results/primary/*/DIAL_SELECTION.txt` were regenerated from the stored sweeps with
`code/Thesis_Experiment/data_prep/select_dials.py`. The copy of the split-1
bank-then-quarterly log taken in August still showed the pre-fix dial-1 endpoint
(0.0241/0.4559) alongside post-fix values elsewhere in the same file. The selected
dials were unaffected; regenerating removes the inconsistency. The rule reproduces
Tables 5.3 and 5.4 for all six runs.
