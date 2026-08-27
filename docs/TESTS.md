# Tests

Correctness checks on the method, the metric and the decision logic. They test
mathematical invariants, not just that files are unchanged, so a failure here would
mean a defect in a reported number.

Run from the package root with the working tree available:

```bash
PYTHONPATH=code python tests/test_reversion_math.py
PYTHONPATH=code python tests/test_bf16_dial1.py
PYTHONPATH=code python tests/test_metric_rule_selection.py
PYTHONPATH=code DATA_SEED=1 python tests/verify_math.py
PYTHONPATH=code DATA_SEED=1 python tests/verify_metric.py
PYTHONPATH=code DATA_SEED=1 python tests/verify_mixture.py
```

Last run 2026-08-27 against the code in this package: **114 passed, 0 failed.**

| file | what it establishes | count |
|---|---|---|
| `test_reversion_math.py` | decomposition round-trips; direction rows are unit norm; `base == W0`; `full ==` the model's effective weight; every reversion endpoint equals the diagnostic version it should (`revert_direction(0) == mag_only`, `revert_magnitude(0) == dir_only`, and the four corners of `revert_both`); a magnitude reversion leaves the direction untouched and vice versa; distance from the full model grows monotonically as the dial falls; the slerp variant matches at both endpoints and stays finite for near-parallel rows; the identification decision logic picks the right component in each regime | 34 |
| `test_bf16_dial1.py` | characterises why `apply_reversion` dispatches dial 1 to the canonical path: the two routes agree to ~1e-8 in fp32, but after the bf16 cast **0.0013% of weights land on a different value, up to 2.4e-4 apart** at 4096×4096 — matching the figure quoted in the code comment and the thesis. Also asserts the guard is present in `evalcore.apply_reversion` | 3 |
| `test_metric_rule_selection.py` | field-F1 against hand-computable cases (tp/fp/fn arithmetic, corpus-level pooling rather than a per-document mean, values matter not just keys, repeats matched with multiplicity); the early-stopping rule (ties count as non-improvement, the counter resets on a new best, **a higher peak after the trigger is not selected**, patience is honoured); layer selection (all three arms take 63 of 252, top and bottom are disjoint, top is exactly the layers at or above the reported threshold, random is reproducible for a fixed seed); the registry raises on an unset `DATA_SEED` instead of silently using split 1 | 23 |
| `verify_math.py` | pre-existing: the DoRA algebra and the WiSE-FT endpoints, including that `m_ft ≠ ‖W₀ + sBA‖` — the decoupling the whole method depends on | 23 |
| `verify_metric.py` | pre-existing: field-F1 and nTED edge cases, plus a sanity check of the vendored tree-edit-distance | 14 |
| `verify_mixture.py` | pre-existing: rehearsal mixture construction — the replay fraction is right, the file is shuffled rather than blocked, the same seed gives a byte-identical manifest, and **no bank validation or test document ever appears in a mixture** | 20 |

## Known non-passing file

`code/tests/ab_paths.py` in the working tree does not run: it points at
`Bankstatements_Bank2Quarterly`, an experiment directory from an earlier naming scheme
that no longer exists. It is stale scaffolding, not a check of anything the thesis
reports, and is not included in this package.

## A note on `dora_reversion.py`

Sections 1–4 of that file (decomposition, the four versions, the reversion operations)
are the method the thesis uses. Section 5 — `identify_overfitting_component`,
`revert_identified` — implements an earlier framing in which the method first *decided*
which component was to blame and then reverted it. The thesis instead reports the full
sweep of both components, so this logic is not called anywhere in the pipeline that
produced the results; its only callers are `reversion_eval.py` and `reversion_cord.py`,
from the CORD study that was cut. It is kept because removing it would make the file
differ from the one that ran, and it is covered by the tests above.
