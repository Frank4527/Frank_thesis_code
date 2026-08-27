# Thesis section → file

Section numbers follow the current draft and may shift as it is revised; the
descriptions are stable enough to relocate them if so. Paths are relative to the root
of this package. Run anything under `code/` with `PYTHONPATH=code`.

## Method

| thesis | file | what to look at |
|---|---|---|
| §2.3.3 weight normalisation; the DoRA decomposition | `code/Identification_and_Reversion/dora_reversion.py` | `decompose(W)` → per-output-row magnitude `m = ‖W‖` and unit direction `D` |
| §3.2 reverting the magnitude | same file | `revert_magnitude(layer, α)` |
| §3.2 reverting the direction | same file | `revert_direction(layer, β)` — note the renormalisation: a blend of two unit vectors is not itself unit |
| §3.3 the four diagnostic versions (base, mag\_only, dir\_only, full) | `code/Identification_and_Reversion/dora_merge.py` | `VERSION_FNS` |
| §3.3 applying a reversion to a live model | `code/Thesis_Experiment/pipeline/evalcore.py` | `apply_reversion` — **read the comment**: dial 1 is dispatched to the canonical full path for a bf16 reason |
| §3.4 choosing the operating point | `code/Thesis_Experiment/data_prep/select_dials.py` | the max-mean rule, and the note it emits when the argmax is a corner |
| §3.5 layer-selective extension | `code/Identification_and_Reversion/layer_drift.py` | ρ = ‖sBA‖\_row / ‖W₀‖\_row, per layer; also the orthogonal/radial split |
| §3.5 which layers to revert | `code/Identification_and_Reversion/layer_select.py` | `--mode top` / `bottom` / `random`, per run, never pooled |
| §3.5 applying it to a subset | `code/Thesis_Experiment/pipeline/layer_sweep.py`, `dora_merge.apply_selective` | untouched layers take the canonical path, not a reversion at dial 1 |
| WiSE-FT baseline | `code/Thesis_Experiment/pipeline/evalcore.py` | `apply_wiseft` |

## Experimental setup

| thesis | file |
|---|---|
| §4.1 corpora and the document-level splits | `code/Thesis_Experiment/data_prep/build_bank_seed.py`, `build_quarterly_seed.py`, `docs/SEEDS.md` |
| §4.1 why splitting is document-wise, not page-wise | `docs/DATA_PIPELINE.md` |
| §4.2 that no run saw another split's data | `code/Thesis_Experiment/data_prep/verify_seeds.py`, `audit_seed_isolation.py` |
| §4.3 training configuration (DoRA r=16, α=32, lr 2e-4, 10 epochs, 4×GPU) | `code/Training_Dora/train_dora_ddp.py` |
| §4.3 no validation-based model selection | same file — the dev-eval callback is registered only when `--eval-every` is a real interval; the primary runs pass a sentinel |
| §4.4 field-level F1, value-F1, nTED | `code/Metrics/field_f1.py` |
| §4.4 generation and scoring | `code/Metrics/document_extraction.py` |
| §4.5 rehearsal 1%/5% and joint baselines | `scripts/11_run_all_b2q.sh` (mixtures via `make_mixture` in `code/Thesis_Experiment/datasets.py`) |
| §4.5 the joint baseline is shared between task orders | `scripts/13_run_all_q2b.sh` — see the comment where joint training is disabled |
| §4.9 measurement resolution | `docs/EXPERIMENT_DESIGN.md` |

## Results

| thesis | scripts | results |
|---|---|---|
| §5 primary experiment, all six runs | `scripts/11`–`14` | `results/primary/<run>/` |
| §5 the sweeps | `code/Thesis_Experiment/pipeline/sweep.py` | `results/primary/<run>/stage2/stage2-{dir,mag}sweep__both__val_rerun.json` |
| §5 selected operating points | `select_dials.py` | `results/primary/<run>/DIAL_SELECTION.txt` |
| §5 early stopping | `scripts/20`–`22`, `code/Thesis_Experiment/pipeline/es_rule.py` | `results/early_stopping/<run>/` |
| §5 the training trajectories | `scripts/21` | `results/early_stopping/<run>/es_traj/step<N>__{old,new}__val.json` |
| §5 layer-targeted, top-25% | `scripts/30` | `results/layer_reversion/<run>/layer-top-dirsweep__both__val.json` |
| §5 random-25% control | `scripts/31` | `.../layer-random0-dirsweep__both__val.json` |
| §5 bottom-25% control | `scripts/32` | `.../layer-bottom-dirsweep__both__val.json` |
| §5 which layers each arm chose | `layer_select.py` | `.../selected_layers_{top,random0,bottom}_q0.75.json` |
| §6.2 the drift is almost entirely orthogonal to W₀ | `layer_drift.py` | `results/layer_drift/layer_drift_<order>_seed<N>.json` — `rho_perp_median`, `angle_deg_median` |

## Claims that are checkable from the files alone

| claim | how to check |
|---|---|
| the direction sweep at β=1 equals the magnitude sweep at α=1 | compare the `"1.0"` entries of `stage2-dirsweep` and `stage2-magsweep` in any run |
| the sweep endpoints are the diagnostic versions | `dirsweep` at `"0.0"` is `mag_only`; `magsweep` at `"0.0"` is `dir_only` |
| each layer-targeted arm touched exactly 63 of 252 layers | `n_layers_reverted` in any `layer-*-dirsweep` JSON: 63 at every β<1, 0 at β=1 |
| top-25% and bottom-25% are disjoint | intersect the `layers` lists in `selected_layers_top_*` and `selected_layers_bottom_*` |
| the early-stopping step was not chosen with hindsight | re-run `es_rule.py` over `es_traj/` and compare with `SELECTED_STEP.txt` |
| the joint baseline is one measurement shared by both orders | compare `joint__both__test_rerun.json` across the b2q and q2b run of the same split |
