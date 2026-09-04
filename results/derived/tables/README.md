# Generated C-fairness tables

Every file names its attribute: `<family>_attacks_<attribute>_b<budget>`. Age was
only run for the C-fair attacks, so there is a `cfair/cfair_attacks_age_b10.*` set (234 model-wise tests). The partial-knowledge experiments are gender-only.

- `vanilla/`: PowerUser, Bandwagon, ReverseBandwagon (195 model-wise tests).
- `cfair/`: the six C-fairness-targeted attacks (390).
- `partial_knowledge/`: Full, Sampling, and Top-k C-fairness attacks (325
  model-wise tests; row-level table restricted by raw significance).

Each folder holds the LaTeX table, its values as CSV, the raw and
Benjamini--Hochberg-adjusted p-values, and a `*_summary.txt` with the
significance counts and the group-assignment audit.

Regenerate from the repository root:

```bash
.venv/bin/python experiments/tables/attack_tables.py --attribute all
.venv/bin/python experiments/tables/partial_knowledge_table.py
```

Options: `--family vanilla|cfair|both`, `--budget` (default `0.1`) and
`--output-dir` for `attack_tables.py`; the partial-knowledge generator accepts
`--input`, `--budget`, and `--output-dir`. The code is in `experiments/tables/`.

## Table conventions

- Four columns per model family: `delta gamma`, the largest `delta m` (in
  percentage points), the count
  below the four-fifths threshold, and the count of significant reductions. The
  8 conventional and the 5 fairness-aware models are reported separately.
  Interval endpoints may come from different models.
- The threshold count is the post-attack one, so it includes models already
  below the threshold before any injection. The parenthesised figure beside it
  is how many of them the attack put there, and is omitted where nothing is
  below the threshold. The CSV holds the two as separate columns and
  `*_summary.txt` breaks them down per row.
- The significance count follows the same convention: the models reaching
  `p < 0.05` uncorrected, with those surviving Benjamini--Hochberg in
  parentheses. Reporting only the corrected count cannot separate a test that
  never fires from one whose hits the correction removes.

## Pre-attack `gamma > 1`

Fairness-aware models take the privileged/unprivileged split of their
conventional counterpart (MF-\* from MF, BNSLIM-u from SLIM-u), so that an attack
targets the same users with and without the fairness intervention. A model whose
intervention has reversed the disparity then reports `gamma > 1`.

Three combinations are affected: ML100K MF-value (1.07) and MF-absolute (1.14),
and FNYC MF-absolute (1.02). The list is recomputed on every run and written to
`*_summary.txt`.
