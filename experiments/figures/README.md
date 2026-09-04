# experiments/figures

Builds the paper's budget-curve figures from attack results. Nothing here trains
a model or runs an attack; these read result CSVs and emit TikZ, in the same
spirit as `experiments/tables`.

## Inputs

| Produced by | Output |
|---|---|
| model fitting | `results/runs/{dataset}/clean/metrics_top10.csv` (budget 0) |
| `03_train_aug.ipynb` | `results/runs/{dataset}/cfair_*/metrics_top10.csv` |
| `03_train_aug.ipynb` | `results/runs/{dataset}/*/ndcg_index.csv` |

Each figure is a 3-by-N groupplot: one column per attack, one row per dataset,
plotting the benefit ratio gamma against attack budget at 0, 10, 25, 50, 75 and
100%.

## Files

**`attack_figures.py`** — builds every figure family, for either attack set and
either attribute.

```bash
.venv/bin/python experiments/figures/attack_figures.py                       # influencer, gender
.venv/bin/python experiments/figures/attack_figures.py --attack-set favorite
.venv/bin/python experiments/figures/attack_figures.py --attribute age       # three datasets
.venv/bin/python experiments/figures/attack_figures.py --figure mf_fair --attack-set all --attribute all
```

Options: `--figure neighborhood_conventional|mf_conventional|neighborhood_fair|mf_fair|all`
(default `all`), `--attack-set influencer|favorite|all` (default `influencer`),
`--attribute gender|age|all` (default `gender`), `--output-dir` (default
`results/derived/figures/`).

The figures vary along four binary axes -- architecture and model variant
(together the `--figure` name), attack set, and attribute -- so
`--figure all --attack-set all --attribute all` writes all 16. Two axes are
directories and two stay in the filename:

```text
results/derived/figures/<attribute>/<attack set>/<figure>.tex

  gender/influencer/mf_fair.tex
  age/favorite/neighborhood_conventional.tex
```

The `\label` inside each file still names every axis, so a figure read outside
its directory is still identifiable.

Three registries vary independently, and a new combination is a dictionary entry
rather than a new script.

**`ATTACK_SETS`** — which three attacks are the columns. Both sets exist for both
attributes on every dataset the attribute covers.

| set | column 1 | column 2 | column 3 | `\label` suffix |
|---|---|---|---|---|
| `influencer` | Influencer-PushR | Influencer-PushLF | Influencer-NukeF | *(none)* |
| `favorite` | Favorite-PushR | Favorite-PushLF | ReverseFavorite-NukeF | `_favorite` |

**`FIGURES`** — which models are the curves.

| family | models | legend cols | y clipped |
|---|---|---|---|
| `neighborhood_conventional` | SLIM-u, EASE-u, SLIM-i, EASE-i | 4 | no |
| `mf_conventional` | SVD, MF, NMF, NeuMF | 4 | no |
| `neighborhood_fair` | BNSLIM-u vs SLIM-u | 2 | yes |
| `mf_fair` | MF-value/absolute/under/over vs MF | 5 | yes |

The two fairness-aware families set `restrict y to domain=0:1.3`, because an
intervention that overshoots can push gamma well above 1 when both group benefits
are near zero. The run prints how many points were clipped, and in which datasets,
so a truncated curve is never silent.

**`explore.py`** — interactive exploration, backing `notebooks/04_plots.ipynb`.
Where `attack_figures.py` builds the fixed paper figures, this draws any
dataset / attribute / attack / model / metric on screen.

```python
from experiments.figures import explore
explore.options()                    # what exists, read live from results/
explore.plot_attack("ml-1m")         # every model, default attack
explore.plot_attack("ml-1m", attack_id="power_user", models_to_plot=["svd", "mf"])
```

Four panels against budget: the benefit ratio, the absolute benefit gap, and each
group's score. Only `dataset` is required; the rest default to what the results
contain, and an unavailable combination raises with the available options rather
than plotting an empty axis. `options()` reads the files.

**`check_tikz_against_csv.py`** — audits a written `.tex` against the CSVs. Parses
the `\addplot[style] table {...}` blocks and compares every point. It re-reads the
CSVs itself rather than calling `attack_figures`, so a figure pasted in by hand, or
generated before a result was recomputed, is caught rather than trusted.

```bash
.venv/bin/python experiments/figures/check_tikz_against_csv.py \
    results/derived/figures/gender/favorite/mf_conventional.tex \
    --attack-set favorite \
    --datasets ml-100k ml-1m lastfm-1k ftky fnyc \
    --models svd mf nmf neumf
```

`--attack-set` has to match the figure's columns. Pass `--datasets` and `--models`
explicitly: the defaults cover only ML100K, ML1M and the conventional MF family,
so a figure audited on the defaults alone leaves rows and curves unchecked.

## Sensitive attributes

Same registry shape as the tables. Both attack sets exist for both attributes, so
every family can be built for either.

| attribute | groups | datasets |
|---|---|---|
| `gender` | `f`, `m` | all five |
| `age` | `y`, `o` | ML100K, ML1M, LFM1K |

A fairness-aware model takes the attribute as a suffix (`mf-value-gender`,
`mf-value-age`), so the model names are resolved per attribute rather than
hardcoded.

