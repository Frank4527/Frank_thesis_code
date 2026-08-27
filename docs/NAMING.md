# Naming conventions

The conventions below are the ones used by the code that produced the results. They
are documented rather than changed, so that every filename in `results/` still matches
the code path that wrote it.

## Result files

```
<config>__<tasks>__<split>[_rerun].json
```

| part | meaning |
|---|---|
| `<config>` | which model/configuration was scored — see the table below |
| `<tasks>` | `both` throughout: every evaluation scores the old **and** the new task |
| `<split>` | `val` or `test` |
| `_rerun` | present on the primary experiment only. It marks the final series run under the single "document" instruction, distinguishing it from an earlier series that used a different prompt per dataset. Files without it in `early_stopping/` and `layer_reversion/` were only ever produced once. |

### `<config>` values

| config | model scored |
|---|---|
| `wbank`, `w_stage1` | the stage-1 model (the old-task ceiling). `wbank` in b2q, `w_stage1` in q2b |
| `stage2` | stage-2, untouched — the forgetting case |
| `stage2-mag` | magnitude reverted at the selected α |
| `stage2-dir` | direction reverted at the selected β |
| `stage2-dirmag` | both reverted — the combined method |
| `stage2-magsweep` | magnitude dialled across α ∈ {0, .25, .5, .75, 1} |
| `stage2-dirsweep` | direction dialled across β ∈ {0, .25, .5, .75, 1} |
| `wiseft`, `wiseft-sweep` | whole-weight interpolation baseline, selected point / full sweep |
| `stage2-peftcheck` | reconstruction cross-check against PEFT's own `PeftModel` |
| `joint`, `rehearsal001`, `rehearsal005` | the retraining baselines |
| `es-dirsweep`, `es-dir_only` | the sweep applied to an early-stopped model |
| `layer-top-dirsweep`, `layer-random0-dirsweep`, `layer-bottom-dirsweep` | the three layer-targeted arms |
| `step<N>__old__val`, `step<N>__new__val` | a training checkpoint at optimiser step N, scored on one task |

Inside a sweep JSON the top-level keys are the dial values (`"0.0"`, `"0.25"`, …),
each holding both tasks' scores. Inside a single-configuration JSON the task names
(`bank`, `quarterly`) are the top-level keys.

## Run directories

```
<StageOneTask>2<StageTwoTask>_<series>[_seed<N>]
```

- `Bankstatement2Quaterly_rep_rerun` — bank → quarterly, split 1
- `Bankstatement2Quaterly_rep_rerun_seed2` / `_seed3` — splits 2 and 3
- `Quaterly_rep2Bankstatement_rerun[_seed2|_seed3]` — the reverse order
- `<order>_es_seed<N>` — the early-stopping retrain of that split

Two spellings are inherited from the original directory names and left as they are so
the paths keep matching the results: **"Quaterly"** (missing the second *r*) and
**"Bankstatement"** singular. They are typos, not different corpora.

"Split 1" has no `_seed` suffix because it was created before the suffix scheme
existed; it is `DATA_SEED=1`. See `docs/PROVENANCE.md` for the note on split-1
provenance.

## Scripts

`scripts/` is renumbered for this package, since nothing imports these files:

```
NN_<stage>_<what>.sh
```

| range | stage |
|---|---|
| `10–14` | the primary experiment: config, training, evaluation, per task order |
| `20–22` | the early-stopping study: retrain, score the trajectory, sweep the chosen checkpoint |
| `30–32` | the layer-targeted study: top-25%, random-25%, bottom-25% |

The original names are recorded in `docs/PROVENANCE.md`, since the logs on the server
refer to them.

## Seeds — two different things

| name | meaning | value |
|---|---|---|
| `SEED` / `--seed` | the **training** seed: initialisation and shuffling | 0 everywhere |
| `DATA_SEED` | which **document-level re-split** to use | 1, 2 or 3 |

The thesis calls these "three document-level re-splits", not "three random seeds",
because only the split varies. `DATA_SEED` has no default in `datasets.py`: an unset
seed raises rather than silently resolving to split 1.
