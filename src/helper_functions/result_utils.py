from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import pandas as pd
from src.helper_functions import paths


def budget_to_fake_users(budget, unprivileged_group_size):
    """Convert a budget fraction into an absolute fake-user count."""
    return max(1, round(unprivileged_group_size * budget))


def add_group_assignments_from_row(assignments, model_name, row, groups_config):
    """Update group assignments from one clean-result row."""
    for attr in groups_config:
        unpriv, priv = infer_unprivileged_privileged(row, attr, groups_config)
        if unpriv is not None:
            assignments.setdefault(model_name, {})[attr] = (unpriv, priv)

    return assignments


def infer_unprivileged_privileged(row, attr, groups_config, metric="ndcg"):
    """Infer unprivileged/privileged groups from group metric columns."""
    g1, g2 = groups_config[attr]
    score1, score2 = row.get(f"{metric}_{g1}"), row.get(f"{metric}_{g2}")
    if pd.isna(score1) or pd.isna(score2):
        return None, None
    return (g1, g2) if score1 < score2 else (g2, g1)


def load_clean_results(folder, list_size):
    """Read the pre-attack results for one dataset."""
    return pd.read_csv(paths.clean_results_in(folder, list_size))


def canonical_run_keys(dataset, attack_id, attribute, target_items=None, budget=None):
    """The run_keys of the current NDCG checkpoint for one attack.

    A result CSV is appended to, so it keeps rows from earlier runs.

    The per-attack ``ndcg_index.csv`` records which run is current, so restricting
    to these keys selects the live rows without relying on row order.

    ``None`` and the empty set mean different things, and callers rely on the
    difference. ``None`` is "there is no usable index here", so there is no filter
    to apply. An empty set is "the index exists and matches nothing", which for a
    caller that expects a fixed number of rows should empty the frame and fail
    loudly rather than quietly pass the unfiltered rows through.
    """
    index_path = paths.post_attack_index(dataset, attack_id)
    if not index_path.exists():
        return None
    index = pd.read_csv(index_path)
    if "key" not in index.columns or "attribute" not in index.columns:
        return None
    index = index[index["attribute"].eq(attribute)]
    if target_items is not None and "target_items" in index.columns:
        index = index[index["target_items"].eq(target_items)]
    if budget is not None and "budget" in index.columns:
        index = index[np.isclose(index["budget"].astype(float), budget)]
    return set(index["key"].dropna())


def select_canonical_rows(
    attack, dataset, attack_id, attribute, target_items=None, budget=None
):
    """Restrict already-loaded attack rows to the current NDCG checkpoint.

    For callers that require a complete set of rows: an index that matches
    nothing empties the frame, so the caller's own row-count check fires.
    """
    keys = canonical_run_keys(dataset, attack_id, attribute, target_items, budget)
    if keys is None or "run_key" not in attack.columns:
        return attack
    return attack[attack["run_key"].isin(keys)]


def get_group_assignments(folder, list_size, groups_config):
    """Load clean model outcomes and derive privileged/unprivileged groups."""
    assignments = defaultdict(dict)
    clean_results = load_clean_results(folder, list_size)

    for _, row in clean_results.iterrows():
        add_group_assignments_from_row(assignments, row["model"], row, groups_config)

    return assignments


def store_user_labels(folder, R_test, groups_map):
    """Store demographic labels once per dataset for later significance tests."""
    labels_path = paths.user_labels_in(folder)
    if os.path.exists(labels_path):
        return

    labels = {
        attr: np.full(R_test.shape[0], "na", dtype=object)
        for attr in groups_map
    }

    for attr, group_indices in groups_map.items():
        for group, idxs in group_indices.items():
            labels[attr][idxs] = group

    np.savez_compressed(labels_path, **labels)


def add_group_ratios(all_metrics, attr, unpriv, priv, metrics, groups_config):
    """Add unprivileged/privileged ratios for the active sensitive attribute."""
    for metric in metrics:
        for group in groups_config[attr]:
            key = f"{metric}_{group}"
            all_metrics[key] = all_metrics.get(key, None)

        all_metrics[f"{metric}_{attr}_ratio"] = (
            all_metrics.get(f"{metric}_{unpriv}") / all_metrics.get(f"{metric}_{priv}", 1e-9)
        )

    return all_metrics


def group_benefit_gap(row, attr, groups_config, metric="ndcg"):
    """Absolute benefit gap between the two groups of a sensitive attribute.

    Companion to ``add_group_ratios``: that function reports the unprivileged /
    privileged ratio, this one the absolute difference between the same two
    quantities. The gap is symmetric, so it does not depend on which group is
    the unprivileged one.
    """
    group_a, group_b = groups_config[attr]
    return abs(float(row[f"{metric}_{group_a}"]) - float(row[f"{metric}_{group_b}"]))


def log_progress(message, progress_log_path):
    """Print and persist progress messages for long-running experiments."""
    progress_log_path = Path(progress_log_path)
    line = f"{datetime.now().isoformat(timespec='seconds')} | {message}"
    print(line, flush=True)
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_log_path, "a") as f:
        f.write(line + "\n")


@contextmanager
def training_heartbeat(label, progress_log_path, interval_seconds=60):
    """Log periodic progress while a blocking model fit is still running."""
    stop = threading.Event()
    started_at = time.monotonic()

    def beat():
        while not stop.wait(interval_seconds):
            elapsed = int(time.monotonic() - started_at)
            log_progress(
                f"Still training {label}: elapsed={elapsed}s",
                progress_log_path,
            )

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    except Exception as exc:
        elapsed = int(time.monotonic() - started_at)
        log_progress(
            f"Failed training {label}: elapsed={elapsed}s, "
            f"error={type(exc).__name__}: {exc}",
            progress_log_path,
        )
        raise
    finally:
        stop.set()
        thread.join(timeout=1)


def install_graceful_stop_handler():
    """Treat SIGTERM like Ctrl-C so experiment scripts can checkpoint before exit."""
    def handle_stop_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_stop_signal)


def exit_after_interruption(message, progress_log_path):
    """Log an intentional interruption and exit with the conventional Ctrl-C code."""
    log_progress(message, progress_log_path)
    sys.exit(130)


def normalize_key_value(value):
    """Normalize CSV-loaded values so resume keys compare cleanly."""
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, float):
        return round(value, 10)
    return value


def make_run_key(*values):
    """Build a deterministic checkpoint key from the values that define a run."""
    normalized = [normalize_key_value(value) for value in values]
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)
    return f"rk_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:20]}"


def verify_clean_vector_matches_aggregate(
    scores, aggregate_row, attr, group_indices, groups_config,
    label="", metric="ndcg", tolerance=1e-6,
):
    """Fail loudly when a stored per-user vector disagrees with its aggregate row.

    The per-user vectors and the aggregate columns of ``metrics_top{N}.csv`` are
    two views of the same run, so the mean of a vector over one group must equal
    that group's aggregate. They can drift apart when a checkpoint hands back a
    copy taken before a model was regenerated: the aggregate is re-read from the
    CSV on every run and refreshes, while the vector is skipped as "complete" and
    does not. Nothing downstream notices, because gamma comes from the aggregate
    and the significance test comes from the vector -- so the two silently
    describe different models.

    Checkpoint keys identify a run, not the data behind it, so this cannot be
    caught by key comparison. Call it wherever clean vectors are loaded.
    """
    for group in groups_config[attr]:
        indices = group_indices.get(group)
        if indices is None or len(indices) == 0:
            continue
        group_scores = scores[indices]
        group_scores = group_scores[~np.isnan(group_scores)]
        if len(group_scores) == 0:
            continue
        expected = aggregate_row.get(f"{metric}_{group}")
        if expected is None or pd.isna(expected):
            continue
        actual = float(group_scores.mean())
        if abs(actual - float(expected)) > tolerance:
            raise ValueError(
                f"Stored per-user {metric} disagrees with the aggregate row for "
                f"{label or 'this run'}, group {group!r}: vector mean={actual:.6f}, "
                f"{metric}_{group}={float(expected):.6f}. The per-user vector is "
                f"stale relative to the aggregate -- regenerate it rather than "
                f"trusting either."
            )


def mannwhitney_group_drop_test(
    scores_clean, scores_post, group_indices, unprivileged, privileged
):
    """Test whether the unprivileged group lost more benefit than the privileged one.

    ``unprivileged`` and ``privileged`` are REQUIRED and must be the groups the
    rest of the pipeline used: the split derived from the conventional model, so
    that a fairness-aware model is judged on the same users as its counterpart
    (see ``model_naming.get_base_model_for``). 

    Passing the groups explicitly keeps the test aligned with gamma, with
    the attack's target, and with the fairness objective the model was trained on.

    Returns ``(nan, nan, False)`` when either score vector is missing or a group
    has no scored users -- the "could not test" outcome, distinct from "tested
    and found nothing".
    """
    if scores_clean is None or scores_post is None:
        return np.nan, np.nan, False

    from scipy.stats import mannwhitneyu

    if unprivileged == privileged:
        raise ValueError(
            f"unprivileged and privileged must differ; both are {unprivileged!r}"
        )
    missing = [g for g in (unprivileged, privileged) if g not in group_indices]
    if missing:
        raise KeyError(
            f"groups {missing} are not in group_indices "
            f"(available: {sorted(group_indices)})"
        )

    # Mann-Whitney U test:
    #   H0: drop_unprivileged <= drop_privileged
    #   H1: drop_unprivileged >  drop_privileged  (unprivileged harmed more)
    drops = scores_clean - scores_post
    drop_unpriv = drops[group_indices[unprivileged]]
    drop_priv = drops[group_indices[privileged]]
    drop_unpriv = drop_unpriv[~np.isnan(drop_unpriv)]
    drop_priv = drop_priv[~np.isnan(drop_priv)]

    if len(drop_unpriv) == 0 or len(drop_priv) == 0:
        return np.nan, np.nan, False

    u_stat, p_value = mannwhitneyu(drop_unpriv, drop_priv, alternative="greater")
    return u_stat, p_value, bool(p_value < 0.05)


def benjamini_hochberg(p_values):
    """Benjamini--Hochberg adjusted p-values, controlling the false discovery rate.

    Companion to ``mannwhitney_group_drop_test``: one test is run per
    dataset-model-attack combination, so raw p-values cannot be compared against
    a fixed threshold. NaNs are preserved and excluded from the ranking.
    """
    p_values = pd.Series(p_values)
    valid = p_values.notna()
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    p = p_values[valid].to_numpy(dtype=float)
    if len(p) == 0:
        return adjusted

    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty(len(sorted_p), dtype=float)
    running_min = 1.0
    for idx in range(len(sorted_p) - 1, -1, -1):
        running_min = min(running_min, sorted_p[idx] * len(sorted_p) / (idx + 1))
        adjusted_sorted[idx] = running_min

    restored = np.empty(len(sorted_p), dtype=float)
    restored[order] = adjusted_sorted
    adjusted.loc[valid] = np.minimum(restored, 1.0)
    return adjusted


def existing_run_keys(rows, _key_columns=None):
    """Return completed-run keys from existing canonical result rows."""
    return {
        row["run_key"]
        for row in rows
        if "run_key" in row and pd.notna(row["run_key"])
    }


def has_required_result_values(row, required_columns):
    """Return whether a result row has all required non-null values."""
    return (
        "run_key" in row
        and pd.notna(row["run_key"])
        and all(column in row and pd.notna(row[column]) for column in required_columns)
    )


def dedupe_records_by_key(records, key_column):
    """Keep the last record for each checkpoint key."""
    deduped = {}
    missing_key_records = []
    for row in records:
        key = row.get(key_column)
        if pd.isna(key):
            missing_key_records.append(row)
        else:
            deduped[key] = row
    return missing_key_records + list(deduped.values())


def complete_checkpoint_keys(results, ndcg_index, ndcg_store, required_columns):
    """Return run keys whose CSV row, index row, and NDCG vector are all present."""
    indexed_keys = {
        row["key"]
        for row in ndcg_index
        if "key" in row and pd.notna(row["key"])
    }
    stored_keys = set(ndcg_store)
    return {
        row["run_key"]
        for row in results
        if has_required_result_values(row, required_columns)
        and row["run_key"] in indexed_keys
        and row["run_key"] in stored_keys
    }


def filter_checkpoint_by_required_results(results, ndcg_index, ndcg_store, _key_columns, required_columns):
    """Keep canonical rows and NDCG entries that can be joined by run_key."""
    del required_columns

    canonical_results = [
        row for row in results
        if "run_key" in row and pd.notna(row["run_key"])
    ]
    kept_run_keys = existing_run_keys(canonical_results)
    filtered_ndcg_index = [
        row for row in ndcg_index
        if row.get("key") in kept_run_keys
    ]
    kept_ndcg_keys = {
        row["key"] for row in filtered_ndcg_index if row["key"] in ndcg_store
    }
    filtered_ndcg_index = [
        row for row in filtered_ndcg_index if row["key"] in kept_ndcg_keys
    ]

    filtered_ndcg_store = {
        key: value for key, value in ndcg_store.items() if key in kept_ndcg_keys
    }
    return canonical_results, filtered_ndcg_index, filtered_ndcg_store


def read_csv_records(path):
    """Read a CSV into records, treating absent or empty files as no records."""
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def load_checkpoint(
    result_path,
    ndcg_index_path,
    ndcg_path,
    key_columns,
    required_result_columns=None,
):
    """Load partial experiment outputs so a long run can resume."""
    results = read_csv_records(result_path)
    ndcg_index = read_csv_records(ndcg_index_path)
    ndcg_store = {}
    if ndcg_path.exists():
        with np.load(ndcg_path, allow_pickle=True) as runs:
            ndcg_store = {key: runs[key] for key in runs.files}

    run_id = 0
    for row in ndcg_index:
        key = row.get("key", "")
        if isinstance(key, str) and key.startswith("run_"):
            run_id = max(run_id, int(key.split("_", 1)[1]))

    results, ndcg_index, ndcg_store = filter_checkpoint_by_required_results(
        results,
        ndcg_index,
        ndcg_store,
        key_columns,
        required_result_columns,
    )

    completed = complete_checkpoint_keys(
        results,
        ndcg_index,
        ndcg_store,
        required_result_columns or [],
    )

    return results, ndcg_index, ndcg_store, run_id, completed


def write_checkpoint(result_path, ndcg_index_path, ndcg_path, results, ndcg_index, ndcg_store, headers=None):
    """Write partial result CSV, NDCG index, and per-user NDCG arrays."""
    results[:] = dedupe_records_by_key(results, "run_key")
    ndcg_index[:] = dedupe_records_by_key(ndcg_index, "key")
    indexed_keys = {
        row["key"] for row in ndcg_index if "key" in row and pd.notna(row["key"])
    }
    ndcg_store_keys = set(ndcg_store)
    for key in ndcg_store_keys - indexed_keys:
        ndcg_store.pop(key, None)

    df = pd.DataFrame(results)
    if headers is not None:
        ordered_headers = list(headers)
        extra_columns = [col for col in df.columns if col not in ordered_headers]
        df = df.reindex(columns=ordered_headers + extra_columns)

    result_tmp = result_path.with_suffix(result_path.suffix + ".tmp")
    index_tmp = ndcg_index_path.with_suffix(ndcg_index_path.suffix + ".tmp")
    ndcg_tmp = ndcg_path.with_suffix(".tmp.npz")

    df.to_csv(result_tmp, index=False)
    pd.DataFrame(ndcg_index).to_csv(index_tmp, index=False)
    np.savez_compressed(ndcg_tmp, **ndcg_store)

    os.replace(result_tmp, result_path)
    os.replace(index_tmp, ndcg_index_path)
    os.replace(ndcg_tmp, ndcg_path)
