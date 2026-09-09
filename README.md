# Where Does Forgetting Live? — code, results and figures

Companion repository for the MSc thesis *Where Does Forgetting Live? Localising and
Reverting Catastrophic Forgetting in DoRA* (UCL Data Science and Machine Learning).

Everything the thesis reports is here: the code that produced each result, the result
files themselves, and scripts that redraw every data figure from those files. The
tables in the next section are the index — find a chapter, table or figure in the
thesis and they tell you which file backs it. Section 4.11 of the thesis describes
this repository, and its Table 4.6 gives the same mapping the other way round, from
each part of the report to the code and the results directory behind it.

The two document corpora are client data held under a data agreement and are **not**
in this repository, nor are the trained adapters (17.5 GB merged). Everything that can
be released is released.

```
README.md          you are here
CHECKSUMS.txt      md5 of every tracked file
docs/              design notes, data pipeline, split definitions, naming, provenance
code/              importable modules            (run with PYTHONPATH=code)
scripts/           the drivers, numbered in the order they were run
figures/           one script per data figure, reading only from results/
results/           every result file the thesis reports
tests/             correctness checks (docs/TESTS.md)
```

## The four experiments

| thesis | experiment | code | driver | results |
|---|---|---|---|---|
| §4.7, §5.2 | **1. Sequential adaptation** — the six runs, the two sweeps, the selected dials, the four baselines | `Identification_and_Reversion/dora_reversion.py`, `dora_merge.py`, `Thesis_Experiment/pipeline/sweep.py`, `data_prep/select_dials.py` | `scripts/11–14` | `results/primary/<run>/` |
| §4.8, §5.3 | **2. Reversion against early stopping** — retrain with checkpoints, patience-3 rule, sweep the early-stopped adapter | `Thesis_Experiment/pipeline/es_rule.py` | `scripts/20–22` | `results/early_stopping/<run>/` |
| §4.9, §5.4 | **3. Partial-layer reversion** — per-layer drift, three 63-layer arms | `Identification_and_Reversion/layer_drift.py`, `layer_select.py`, `Thesis_Experiment/pipeline/layer_sweep.py` | `scripts/30–32` | `results/layer_reversion/<run>/`, `results/layer_drift/` |
| §4.10, §5.5 | **4. Recovering pretrained ability** — the same operator against the pretrained backbone | `Pretrained_Ability/sweep.py`, `sweep_seed.py`, `step0_bench.py`, `base_bank_only.py` | `scripts/40–42` | `results/pretrained_ability/` |

`<run>` is one of the six: `Bankstatement2Quaterly_rep_rerun{,_seed2,_seed3}` (bank
then quarterly, splits 1–3) and `Quaterly_rep2Bankstatement_rerun{,_seed2,_seed3}`
(quarterly then bank).

## Where each table comes from

| table | what it holds | file |
|---|---|---|
| 5.1 | forgetting on test | `primary/<run>/stage1/*__both__test_rerun.json`, `stage2/stage2__both__test_rerun.json` |
| 5.2 | both sweeps, all six runs | `primary/<run>/stage2/stage2-{dir,mag}sweep__both__val_rerun.json` |
| 5.3 | selected α★, β★ | `primary/<run>/DIAL_SELECTION.txt` |
| 5.4 | selected c★ (WiSE-FT) | `primary/<run>/stage2/wiseft-sweep__both__val_rerun.json` |
| 5.5, 5.6 | the nine configurations on test | `primary/<run>/{stage1,stage2,baselines}/*__test_rerun.json` |
| 5.7 | the degenerate selection on b2q split 2 | `primary/Bankstatement2Quaterly_rep_rerun_seed2/stage2/stage2-dirsweep__both__val_rerun.json` |
| 5.8 | the checkpoint the rule returns | `early_stopping/<run>/es_reversion/SELECTED_STEP.txt`, `es_traj/` |
| 5.9 | the first three checkpoints | `early_stopping/<run>/es_traj/step{10,20,30}__{old,new}__val.json` |
| 5.10 | per-layer drift profile | `layer_drift/layer_drift_<run>.json` |
| 5.11 | the four arms at β = 0 | `layer_reversion/<run>/layer-{top,random0,bottom}-dirsweep__both__val.json`; the *all* column from `primary/<run>/stage2/stage2-dirsweep__both__val_rerun.json` |
| 5.12 | the pretrained-anchor sweep | `pretrained_ability/split{1,2,3}/`, `backbone/` |
| 5.13 | what the dial trades | computed from 5.12 |

## Redrawing the figures

```
PYTHONPATH=code python figures/make_all.py     # writes figures/out/
```

Each script reads only from `results/`, prints every coordinate it plots so a figure
can be checked against its table, and covers one thesis figure:

| figure | script | reads |
|---|---|---|
| 5.1 the two sweeps, per run | `fig5_1_sweeps_grid.py` | `primary/` |
| 5.2 the sweeps in the two-task plane | `fig5_2_sweeps_pareto.py` | `primary/` |
| 5.3 how far each component moves | `fig5_3_magdir_hist.py` | `layer_drift/drift_hist.npz` |
| 5.4 the selected coefficients ringed | `fig5_4_sweeps_grid_sel.py` | `primary/` |
| 5.5 training path against reversion | `fig5_5_es_paired.py` | `early_stopping/`, `primary/` |
| 5.6, 5.7 layer arms, and zoomed | `fig5_6_layer_pareto.py` | `layer_reversion/`, `primary/` |

Figures 2.1–2.3 are schematics with no data behind them and have no script.

Figure 5.3 is drawn from binned counts rather than the raw arrays: `layer_drift.py`
writes one value per adapted output unit (1,400,832 per run, ~34 MB each), which is
too large to ship. `figures/derive_drift_hist.py` reduces those arrays to the bin
counts and medians in `results/layer_drift/drift_hist.npz`, from which the figure is
identical.

## Reproducing the numbers without a GPU

The selection rule can be re-run against the stored sweeps, which reproduces Tables
5.3 and 5.4 exactly:

```
ln -s "$PWD/results/primary/<run>" /tmp/r/results
PYTHONPATH=code python code/Thesis_Experiment/data_prep/select_dials.py /tmp/r
```

Everything else in Chapter 5 is a direct read of the JSON files listed above.

## Section-to-file map

`docs/SECTION_TO_FILE.md` maps each part of the thesis to the file implementing it,
matching Appendix A of the thesis.
