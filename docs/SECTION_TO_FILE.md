# Section to file

Appendix A of the thesis maps each part of the report to the file implementing it.
This is that table, with the path each file has in this repository, plus the rows the
appendix does not carry.

## Appendix A of the thesis

| thesis | file in the thesis's table | path here |
|---|---|---|
| §3.2 Decomposition and Reversion | `dora_reversion.py` | `code/Identification_and_Reversion/dora_reversion.py` |
| §3.4 Reversion and Coefficient Selection | `dora_merge.py` | `code/Identification_and_Reversion/dora_merge.py` |
| §3.6 Layer-Selective Direction Reversion | `layer_drift.py`, `layer_select.py`, `layer_sweep.py` | `code/Identification_and_Reversion/layer_drift.py`, `layer_select.py`, `code/Thesis_Experiment/pipeline/layer_sweep.py` |
| §4.7.2 The six sequential runs | `run_all.sh`, `config.sh` | `scripts/11_run_all_b2q.sh`, `scripts/13_run_all_q2b.sh`, `scripts/10_config_b2q_example.sh` |
| §4.4 Datasets | `datasets.py` | `code/Thesis_Experiment/datasets.py` |
| §4.7.6 Dial selection | `select_dials.py` | `code/Thesis_Experiment/data_prep/select_dials.py` |
| §4.5 Evaluation Metrics | `field_f1.py`, `document_extraction.py` | `code/Metrics/field_f1.py`, `code/Metrics/document_extraction.py` |
| §5.3 Experiment 2 | `es_rule.py` | `code/Thesis_Experiment/pipeline/es_rule.py` |
| §5.6 Numerical sensitivity in bf16 | `evalcore.py` | `code/Thesis_Experiment/pipeline/evalcore.py` |

The drivers in `scripts/` are the per-experiment `run_all.sh` and `config.sh` of the
thesis's table, renamed and numbered in the order they were run; nothing inside
`code/` was renamed, because the modules import each other by package path.

## Not in Appendix A

Experiment 4 has no row in the thesis's table. Its code is here:

| thesis | what it does | path here |
|---|---|---|
| §4.10, §5.5 | the pretrained-anchor sweep, per split | `code/Pretrained_Ability/sweep.py`, `sweep_seed.py` |
| §4.10, §5.5 | DocVQA and MMBench scoring | `code/Pretrained_Ability/bench_lib.py`, `scoring_official.py`, `sweep_docvqa_cord.py`, `sweep_docvqa_seed.py` |
| §5.5, Table 5.12 | the pretrained backbone's own scores | `code/Pretrained_Ability/step0_bench.py` |
| §5.5, Table 5.12 backbone row | the backbone on bank validation, per split | `code/Pretrained_Ability/base_bank_only.py` |

`base_bank_only.py` was written to fill the two empty cells in Table 5.12's backbone
row. Split 1 was re-run through it as a control and returned 0.0025, reproducing the
published cell, which is what licensed splits 2 and 3 to go in beside it. See
`docs/PROVENANCE.md`.

## Figures

`figures/` holds one script per data figure in the thesis; see the table in
`README.md`. Every script reads only from `results/`.
