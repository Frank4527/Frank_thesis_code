# Dataset seeds
Three independent random re-splits, built for the 3-seed thesis runs.
Splitting is BY DOCUMENT in both datasets, so no report's pages straddle a split.

## Sizes

### Bank_Full

| seed | training | validation | test |
|---|---|---|---|
| 1 | 598 docs | 74 docs | 75 docs |
| 2 | 598 docs | 74 docs | 75 docs |
| 3 | 598 docs | 74 docs | 75 docs |

### Quarterly_rep_En

| seed | training | validation | test |
|---|---|---|---|
| 1 | 27 docs / 445 pages | 3 docs / 36 pages | 4 docs / 34 pages |
| 2 | 27 docs / 361 pages | 3 docs / 71 pages | 4 docs / 83 pages |
| 3 | 27 docs / 373 pages | 3 docs / 26 pages | 4 docs / 116 pages |

## Notes

- `Bank_Full_seed1` / `Quarterly_rep_En_seed1` are byte-identical copies of the
  original split; all results produced before 2026-08 correspond to seed 1.
- Bank splits are stratified by language, so every seed has the identical
  language mix (de 226/28/28, po 201/25/25, en 171/21/22).
- Quarterly document counts match exactly (27/3/4) but PAGE counts vary across
  seeds because reports run from 2 to 50 pages. Matching pages exactly would
  mean choosing which reports go where, so document counts were matched instead.
- Built by `code/Thesis_Experiment/data_prep/build_bank_seed.py` and
  `build_quarterly_seed.py`; both are deterministic given the seed.
