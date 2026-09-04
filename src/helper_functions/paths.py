"""Every location under results/, in one place.

Everything here returns an absolute path, so a script works regardless of the
directory it is launched from. Nothing is created or written; these are names,
not side effects.

Layout::

    results/
      runs/                     expensive; hours of compute, never regenerated casually
        <dataset>/
          opt_params.json
          user_labels.npz
          clean/                metrics_top<N>.csv, ndcg.npz, ndcg_index.csv
          <attack_id>/          metrics_top<N>.csv, ndcg.npz, ndcg_index.csv
      other_experiments/        sweeps that aggregate across many runs
        partial_knowledge/      clean/, attacked/, plus its own summary CSVs
        sparsity_cfair_influencer/
      derived/                  rebuilt from runs/ in seconds by experiments/build_all.py
        tables/<attack_family>/
        figures/<attribute>/<attack_set>/<figure_name>.tex
      logs/
      recbole/                  RecBole-only working artifacts
        datasets/
        checkpoints/<experiment_name>/
        log/<model>/

The layout has one rule: a run is a directory, and everything that run produced
lives inside it. The attack id is a directory name and appears nowhere in a
filename, so re-running one attack is deleting one folder, and a run can never
be half-migrated with its metrics in one place and its per-user scores in
another. Every directory under ``runs/<dataset>/`` holds the same three files,
with no exceptions.

``other_experiments/`` is for work that sweeps configurations and reports one
aggregated table rather than one result per run. Each keeps the ``clean/`` and
``attacked/`` pair, so the per-user scores are found the same way everywhere.

To move any of it, change this file and migrate the data; call sites do not
need to know. The two helpers at the bottom that take an explicit ``folder``
exist for callers that already hold one.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

RUNS = RESULTS / "runs"
OTHER_EXPERIMENTS = RESULTS / "other_experiments"
DERIVED = RESULTS / "derived"
LOGS = RESULTS / "logs"

# RecBole otherwise scatters these working files across root-level dataset/,
# saved/, and log/ directories. Keep all framework-owned artifacts together.
RECBOLE = RESULTS / "recbole"
RECBOLE_DATASETS = RECBOLE / "datasets"
RECBOLE_CHECKPOINTS = RECBOLE / "checkpoints"

DEFAULT_LIST_SIZE = 10

# Filename conventions. Kept as functions/constants so that a caller holding a
# folder path builds the same names as the helpers below.
NDCG_NPZ = "ndcg.npz"
NDCG_INDEX = "ndcg_index.csv"
USER_LABELS = "user_labels.npz"
OPT_PARAMS = "opt_params.json"

# The run id of the unattacked model, alongside the attack ids in the same
# dataset folder. Attack ids are lowercase identifiers, so this cannot collide.
CLEAN_RUN = "clean"

# In a sweep there is no single attack id to name the directory after, so the
# attacked half is simply "attacked".
ATTACKED_RUN = "attacked"


def metrics_name(list_size: int = DEFAULT_LIST_SIZE) -> str:
    """Aggregate metrics filename, identical in every run directory."""
    return f"metrics_top{list_size}.csv"


# ---------------------------------------------------------------------------
# per dataset
# ---------------------------------------------------------------------------


def dataset_dir(dataset: str) -> Path:
    return RUNS / dataset


def run_dir(dataset: str, run_id: str) -> Path:
    """One run's directory: the clean model, or one attack.

    ``run_id`` is ``CLEAN_RUN`` or an attack id. Everything that run produced
    lives here.
    """
    return dataset_dir(dataset) / run_id


def run_metrics(
    dataset: str, run_id: str, list_size: int = DEFAULT_LIST_SIZE
) -> Path:
    return run_dir(dataset, run_id) / metrics_name(list_size)


def run_ndcg(dataset: str, run_id: str) -> Path:
    return run_dir(dataset, run_id) / NDCG_NPZ


def run_index(dataset: str, run_id: str) -> Path:
    return run_dir(dataset, run_id) / NDCG_INDEX


def opt_params(dataset: str) -> Path:
    return dataset_dir(dataset) / OPT_PARAMS


def user_labels(dataset: str) -> Path:
    """Per-user group labels, one array per sensitive attribute."""
    return dataset_dir(dataset) / USER_LABELS


# Named views onto the same run directories, for callers that think in terms of
# "before the attack" and "after the attack" rather than run ids.


def clean_results(dataset: str, list_size: int = DEFAULT_LIST_SIZE) -> Path:
    """Pre-attack aggregate metrics, one row per model."""
    return run_metrics(dataset, CLEAN_RUN, list_size)


def attack_results(
    dataset: str, attack_id: str, list_size: int = DEFAULT_LIST_SIZE
) -> Path:
    """Post-attack aggregate metrics for one attack, across budgets and attributes."""
    return run_metrics(dataset, attack_id, list_size)


def pre_attack_dir(dataset: str) -> Path:
    return run_dir(dataset, CLEAN_RUN)


def pre_attack_ndcg(dataset: str) -> Path:
    return run_ndcg(dataset, CLEAN_RUN)


def pre_attack_index(dataset: str) -> Path:
    return run_index(dataset, CLEAN_RUN)


def post_attack_dir(dataset: str, attack_id: str) -> Path:
    return run_dir(dataset, attack_id)


def post_attack_ndcg(dataset: str, attack_id: str) -> Path:
    return run_ndcg(dataset, attack_id)


def post_attack_index(dataset: str, attack_id: str) -> Path:
    return run_index(dataset, attack_id)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def available_datasets() -> list[str]:
    """Datasets with a run directory, read from disk rather than a hardcoded list."""
    if not RUNS.is_dir():
        return []
    return sorted(d.name for d in RUNS.iterdir() if d.is_dir())


def available_attacks(
    dataset: str, list_size: int = DEFAULT_LIST_SIZE
) -> list[str]:
    """Attack ids with stored metrics for this dataset.

    A run counts as present when its metrics file exists, so a directory left
    behind by an interrupted run is not mistaken for a result.
    """
    folder = dataset_dir(dataset)
    if not folder.is_dir():
        return []
    return sorted(
        d.name
        for d in folder.iterdir()
        if d.is_dir()
        and d.name != CLEAN_RUN
        and (d / metrics_name(list_size)).exists()
    )


# ---------------------------------------------------------------------------
# standalone experiments
# ---------------------------------------------------------------------------


def experiment_dir(name: str) -> Path:
    """A sweep that aggregates across runs, rather than one run per directory."""
    return OTHER_EXPERIMENTS / name


def experiment_clean_dir(name: str) -> Path:
    return experiment_dir(name) / CLEAN_RUN


def experiment_attacked_dir(name: str) -> Path:
    return experiment_dir(name) / ATTACKED_RUN


def recbole_checkpoint_dir(experiment_name: str) -> Path:
    """Validation-selected RecBole checkpoints for one experiment profile."""
    return RECBOLE_CHECKPOINTS / experiment_name


def partial_knowledge_dir() -> Path:
    return experiment_dir("partial_knowledge")


def partial_knowledge_runs() -> Path:
    return partial_knowledge_dir() / "runs.csv"


def partial_knowledge_clean_results() -> Path:
    return partial_knowledge_dir() / "clean_results.csv"


def partial_knowledge_table_values() -> Path:
    return partial_knowledge_dir() / "table_values.csv"


def partial_knowledge_clean_dir() -> Path:
    return experiment_clean_dir("partial_knowledge")


def partial_knowledge_post_dir() -> Path:
    return experiment_attacked_dir("partial_knowledge")


def sparsity_dir() -> Path:
    """The sparsity sweep over the C-Fair influencer attack.

    No dataset argument: the sweep records its dataset in a column of every CSV
    it writes, the way ``partial_knowledge`` does, rather than in the path.
    """
    return experiment_dir("sparsity_cfair_influencer")


def vanilla_progress_log(list_size: int = DEFAULT_LIST_SIZE) -> Path:
    return log_file(f"vanilla_shilling_baselines_top_{list_size}.log")


def log_file(name: str) -> Path:
    """Progress logs from the long-running runners and notebooks."""
    return LOGS / name


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def attack_configs() -> Path:
    """The YAML defining every attack configuration."""
    return REPO_ROOT / "configs" / "attack_configs.yaml"


def data_dir(dataset: str) -> Path:
    """Raw source data, as downloaded."""
    return REPO_ROOT / "data" / dataset


# ---------------------------------------------------------------------------
# derived artifacts
# ---------------------------------------------------------------------------


def tables_dir(family: str) -> Path:
    """Generated tables for one attack family: vanilla, cfair, partial_knowledge."""
    return DERIVED / "tables" / family


def figures_dir(attribute: str | None = None, attack_set: str | None = None) -> Path:
    """Generated figures, nested by the axes the paper reads them along.

    The figures vary along four binary axes: architecture and model variant
    (together the figure name), attack set, and sensitive attribute. Two of them
    are directories and the other two stay in the filename, so the tree is the
    index and no name has to spell out all four.
    """
    folder = DERIVED / "figures"
    if attribute is None:
        return folder
    folder = folder / attribute
    return folder if attack_set is None else folder / attack_set


def figure_file(figure_name: str, attack_set: str, attribute: str) -> Path:
    """One figure: ``<attribute>/<attack_set>/<figure_name>.tex``.

    The ``\\label`` inside the file still names every axis, so a figure remains
    identifiable if it is ever read outside its directory.
    """
    return figures_dir(attribute, attack_set) / f"{figure_name}.tex"


# ---------------------------------------------------------------------------
# folder-relative helpers
# ---------------------------------------------------------------------------


def clean_results_in(folder, list_size: int = DEFAULT_LIST_SIZE) -> Path:
    """Clean results inside an already-known dataset folder.

    For callers that receive a dataset folder rather than a dataset id; keeps the
    filename convention here instead of at the call site.
    """
    return Path(folder) / CLEAN_RUN / metrics_name(list_size)


def user_labels_in(folder) -> Path:
    return Path(folder) / USER_LABELS
