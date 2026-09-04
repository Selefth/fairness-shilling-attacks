# Fairness Shilling Attacks in Recommender Systems

Code for the EDBT 2027 paper *"Modeling and Studying Fairness in Recommender
Systems as a Data Manipulation Problem"*: recommendation models, C-Fair attack
strategies, vanilla shilling baselines, and the pipeline that evaluates their
effect on top-N utility and group fairness.

## Setup

```bash
poetry install
poetry shell
```

## Data

The raw datasets are not in the repository. Download them and unpack each one
into the matching folder under `data/`:

| Folder | Source |
|---|---|
| `foursquare/` | https://sites.google.com/site/yangdingqi/home/foursquare-dataset |
| `lastfm-1k/` | http://ocelma.net/MusicRecommendationDataset/ |
| `ml-100k/`, `ml-1m/` | https://grouplens.org/datasets/movielens/ |

The notebooks build the experimental datasets from these files. The Foursquare
check-ins give two datasets, `fnyc` and `ftky`.

## Layout

```text
configs/      attack definitions (attack_configs.yaml)
data/         raw datasets
notebooks/    the C-Fair pipeline
experiments/  scripted experiments, plus tables/ and figures/
src/          attacks/, models/, helper_functions/
results/      runs/, other_experiments/, derived/, logs/, recbole/
```

Every path under `results/` is defined in `src/helper_functions/paths.py`.

## Running the pipeline

Run the notebooks in order:

1. `00_data_stats.ipynb` – dataset statistics.
2. `01_tuning.ipynb` – hyperparameter tuning on real data. The tuned values are
   already in the repository, so you can skip this.
3. `02_train_real.ipynb` – train on real data, no attack.
4. `03_train_aug.ipynb` – train on real + fake profiles (C-Fair attacks).
5. `04_plots.ipynb` – explore stored results (see `experiments/figures/explore.py`).

`03_train_aug.ipynb` reads its attacks from `configs/attack_configs.yaml` and
runs everything in `ATTACK_IDS_TO_RUN`. Edit that list to run a subset:

```python
ATTACK_IDS_TO_RUN = ["cfair_influencer_push_random"]
```

Each attack writes one directory:

```text
results/runs/<dataset>/<attack_id>/
  metrics_top10.csv    aggregate metrics, one row per model
  ndcg.npz             per-user NDCG scores
  ndcg_index.csv       which rows of the CSV belong to the current run
```

The unattacked models live in `results/runs/<dataset>/clean/`, with the same
three files.

## Scripted experiments

```bash
poetry run python experiments/vanilla_shilling_baselines.py      # vanilla shilling baselines
poetry run python experiments/partial_knowledge_attacks.py       # C-Fair under partial attacker knowledge
poetry run python experiments/ml1m_sparsity_cfair_influencer.py  # ML-1M sparsity, C-Fair Influencer
```

## Tables and figures

The paper's tables and figures come from `experiments/tables/` and
`experiments/figures/`. To rebuild all of them from the
stored runs:

```bash
poetry run python experiments/build_all.py
```

They land in `results/derived/`, as `tables/<attack family>/` and
`figures/<attribute>/<attack set>/`.

## Attack configuration

`configs/attack_configs.yaml` has two sections: `cfair_attacks`, which target the
group fairness, and `vanilla_attacks`, the standard shilling heuristics.
Attack classes are resolved through `src/attacks/registry.py`, and
`perform_attack` in `src/helper_functions/attack_utils.py` runs both kinds.

For attacks that use selected items, `num_selected_values: ["auto"]` sets the
fake profile length from the real profiles:

```text
selected_items = round(average profile length) - target_items
```

The average is taken over the targeted unprivileged group for C-Fair attacks,
and over all training users for vanilla attacks.

## What is tracked

Tracked: the code, `configs/`, `pyproject.toml` and `poetry.lock`, the tuned
hyperparameters (`results/runs/<dataset>/opt_params.json`) for all five
datasets, the ml-100k `metrics_top10.csv` and `ndcg_index.csv` files as a
reference to check against, and `results/derived/`.

Not tracked, regenerate it: the raw datasets, all `ndcg.npz` arrays, the other
four datasets' metrics, `user_labels.npz`, `results/other_experiments/` and
`results/logs/`.

`metrics_top10.csv` is appended to, so it also holds rows from earlier runs.
The `ndcg_index.csv` records which rows belong to the current run,
and `result_utils.canonical_run_keys()` uses it to select them, which is why
it is tracked alongside the CSVs.
