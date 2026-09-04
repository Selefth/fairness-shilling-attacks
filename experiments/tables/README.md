# experiments/tables

Builds the paper's LaTeX tables from attack results. Nothing here trains a model
or runs an attack; these scripts read result CSVs and format them.

## Inputs

| Produced by | Output |
|---|---|
| model fitting | `results/runs/{dataset}/clean/metrics_top10.csv` |
| `experiments/vanilla_shilling_baselines.py` | `results/runs/{dataset}/{power_user,bandwagon,reverse_bandwagon}/metrics_top10.csv` |
| `03_train_aug.ipynb` | `results/runs/{dataset}/cfair_*/metrics_top10.csv` |
| `experiments/partial_knowledge_attacks.py` | `results/other_experiments/partial_knowledge/table_values.csv` |

One row per model per attack configuration, with `ndcg_f`, `ndcg_m`,
`ndcg_gender_ratio`, `unprivileged_group`, `budget`, `target_items` and the
Mann–Whitney `p_value`.

## Files

**`attack_tables.py`** — builds both tables. Sections: constants; the two attack
families; shared construction (row selection and validation, `delta gamma`,
`delta m`, significance, the split by model family); each family's row loader;
the C-fair layout; `main`.

```bash
.venv/bin/python experiments/tables/attack_tables.py                     # gender, both families
.venv/bin/python experiments/tables/attack_tables.py --attribute age     # age, C-fair only
.venv/bin/python experiments/tables/attack_tables.py --attribute all     # everything available
```

Options: `--family vanilla|cfair|both` (default `both`), `--attribute
gender|age|all` (default `gender`), `--budget` (default `0.1`), `--output-dir`.

## Sensitive attributes

The attribute is an argument threaded through the functions, not a module mode.
`src/helper_functions/experiment_registry.py` names, per attribute, its two
groups, the datasets it was run on and the attack families it has results for;
every function that needs one takes it and asks the registry.

| attribute | groups | datasets | families |
|---|---|---|---|
| `gender` | `f`, `m` | all five | vanilla, C-fair |
| `age` | `y`, `o` | ML100K, ML1M, LFM1K | C-fair only |

Every artifact names its attribute:
`<family>_attacks_<attribute>_b<budget>`.

The vanilla table covers three attacks and 195 model-wise tests, writing to
`results/tables/vanilla/`. The C-fair table covers six attacks and 390 tests,
writing to `results/tables/cfair/`; its loader joins through
each run's `ndcg_index.csv` to select canonical rows.

The Benjamini–Hochberg correction is `benjamini_hochberg` in
`src.helper_functions.result_utils`, applied across the model-wise tests of one
table. `significance_counts` keeps the uncorrected and corrected counts side by
side and `format_sig` renders them as `n (m)`, on the same convention as the
threshold column.

The below-threshold count separates models the injection pushed under the
four-fifths threshold from models already under it beforehand. The LaTeX cell
carries the post-attack count with the induced one in parentheses, dropped where
the count is zero (`format_gamma_below`); the CSV keeps them as separate columns, and
`crossing_summary` writes the per-row breakdown to `*_summary.txt`.

**`partial_knowledge_table.py`** — builds the row-level table comparing the
clean model, the full-knowledge Influencer-PushR attack, Sampling, and Top-1,
Top-5, and Top-10 partial-knowledge variants. It selects model--dataset pairs
with at least one uncorrected `p < 0.05`, adds significance stars, and bolds the
lowest post-attack benefit ratio.

```bash
.venv/bin/python experiments/tables/partial_knowledge_table.py
```

It writes the wide values, LaTeX, all 325 raw and globally BH-adjusted p-values,
and a short audit summary to `results/tables/partial_knowledge/`. The paper table
uses the raw p-values; the adjustment is included as an audit and does not alter
the selected rows or stars. Options: `--input`, `--budget`, and `--output-dir`.

**`rq1_contrast_check.py`** — prints two measurements, writes no files:

- *Multiple-testing family size.* Corrects all 585 model-wise tests as one
  family and compares against the per-table correction. Vanilla is 0 either way;
  C-fair goes from 22 to 20.
- *Composition of the copied profiles.* The unprivileged share among the highly
  active users PowerUser copies on ML1M — 19.3%, against a 28.3% population
  share. Reads `data/ml-1m/{ratings,users}.dat`, and reports that it skipped the
  measurement if they are absent.

## Model naming

A fairness-aware model is stored under its own name with the protected attribute
appended, and borrows the group split of a conventional counterpart. Both halves
of that convention live in `src.helper_functions.model_naming`.

## Attack results

The tables read one CSV per dataset per attack. Regenerating them requires the
runners listed under Inputs.