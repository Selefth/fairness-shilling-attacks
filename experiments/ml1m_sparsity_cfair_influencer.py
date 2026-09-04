"""Controlled ML1M sparsity experiment for C-fairness shilling attacks.

The experiment keeps the ML1M user population, demographic labels, validation
set, and test set fixed. Only the chronological training split is subsampled,
with each user keeping at least one training interaction.
"""

import itertools
import json
import os
from pathlib import Path
import sys


import pandas as pd

from src.attacks.registry import CFAIR_ATTACK_REGISTRY
from src.helper_functions import paths
from src.helper_functions.attack_utils import perform_attack
from src.helper_functions.config_loader import load_attack_configs
from src.helper_functions.data_loader import load_dataset_by_name
from src.helper_functions.data_splitter import (
    chronological_split_per_user,
    convert_to_csr,
    create_mappings,
    map_user_indices,
    subsample_train_per_user,
)
from src.helper_functions.metrics_accuracy import compute_metrics, tndcg_at_n
from src.helper_functions.result_utils import (
    add_group_ratios,
    budget_to_fake_users,
    exit_after_interruption,
    infer_unprivileged_privileged,
    install_graceful_stop_handler,
    load_checkpoint,
    log_progress,
    mannwhitney_group_drop_test,
    make_run_key,
    write_checkpoint,
)
from src.helper_functions.train_utils import (
    get_base_model_for,
    initialize_base_models,
    initialize_fair_models,
    model_result_name,
    train_model,
)


DATASET = "ml-1m"
RETAIN_RATIOS = [1.0, 0.7, 0.5, 0.3]
SUBSAMPLE_SEEDS = [42, 43, 44]
BUDGETS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
NUM_TARGET_ITEMS = 10

SEED_FOR_MODELS = 42
RATING_THRES = 4
LIST_SIZE = 10
MIN_RATING, MAX_RATING = 1, 5
FIT_WITH_VALIDATION = True

METRICS = {"ndcg": tndcg_at_n}
GROUPS_CONFIG = {"gender": ["f", "m"]}

# Use None for all models, or specify a subset such as ["neumf"].
# If selecting a fairness-aware model, include its base model too, e.g. ["mf", "mf-over"].
MODEL_IDS_TO_RUN = None
SELECTED_MODEL_IDS = None if MODEL_IDS_TO_RUN is None else set(MODEL_IDS_TO_RUN)

ATTACK_CONFIG_PATH = paths.attack_configs()
ATTACK_CONFIGS = load_attack_configs(
    ATTACK_CONFIG_PATH, "cfair_attacks", CFAIR_ATTACK_REGISTRY
)

# Controlled default: one C-Fair Influencer variant. Add the other two if needed.
ATTACK_IDS_TO_RUN = ["cfair_influencer_push_random"]

OUTPUT_DIR = paths.sparsity_dir()
CLEAN_DIR = paths.experiment_clean_dir("sparsity_cfair_influencer")
ATTACKED_DIR = paths.experiment_attacked_dir("sparsity_cfair_influencer")

PROGRESS_LOG_PATH = paths.log_file("ml1m_sparsity_cfair_influencer.log")

SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
SUMMARY_NDCG_INDEX_PATH = ATTACKED_DIR / paths.NDCG_INDEX
SUMMARY_NDCG_PATH = ATTACKED_DIR / paths.NDCG_NPZ

CLEAN_PATH = OUTPUT_DIR / "clean_results.csv"
CLEAN_NDCG_INDEX_PATH = CLEAN_DIR / paths.NDCG_INDEX
CLEAN_NDCG_PATH = CLEAN_DIR / paths.NDCG_NPZ
AGGREGATE_PATH = OUTPUT_DIR / "summary_by_seed_mean.csv"

CLEAN_KEY_COLUMNS = ["dataset", "retain_ratio", "seed", "model", "attribute"]
SUMMARY_KEY_COLUMNS = [
    "dataset",
    "retain_ratio",
    "seed",
    "model",
    "attack_id",
    "attribute",
    "target_items",
    "budget",
]


def model_is_selected(model_name, result_model_name=None):
    if SELECTED_MODEL_IDS is None:
        return True
    return model_name in SELECTED_MODEL_IDS or result_model_name in SELECTED_MODEL_IDS


def sparsity_stats(R_train_original, R_train_retained, R_fit):
    total_entries = R_train_original.shape[0] * R_train_original.shape[1]
    original_nnz = R_train_original.nnz
    retained_nnz = R_train_retained.nnz
    fit_nnz = R_fit.nnz

    return {
        "train_interactions_original": original_nnz,
        "train_interactions_retained": retained_nnz,
        "fit_interactions": fit_nnz,
        "retained_train_fraction": retained_nnz / original_nnz,
        "train_density_original": original_nnz / total_entries,
        "train_density_retained": retained_nnz / total_entries,
        "fit_density": fit_nnz / total_entries,
        "users_removed": 0,
        "items_removed": 0,
    }


def delta_m(metrics, unpriv, priv):
    return metrics[f"ndcg_{priv}"] - metrics[f"ndcg_{unpriv}"]


def evaluate_group_metrics(R_hat, R_test, attribute, group_indices, unpriv, priv):
    metrics = compute_metrics(
        R_hat, R_test, RATING_THRES, LIST_SIZE, METRICS, groups=group_indices
    )
    return add_group_ratios(metrics, attribute, unpriv, priv, METRICS, GROUPS_CONFIG)


def train_or_reuse_clean(
    dataset,
    retain_ratio,
    seed,
    attribute,
    model_kind,
    model_name,
    opt_params,
    R_train_retained,
    R_fit,
    R_val,
    R_test,
    group_indices,
    base_group_assignments,
    stats,
    clean_results,
    clean_ndcg_index,
    clean_ndcg_store,
    clean_rows_by_key,
    clean_ndcg_by_key,
    clean_run_id,
):
    """Train or reuse the clean model for one sparsity/model/attribute setting.

    This helper first checks whether the clean result already
    exists in the checkpoint files. If so, it returns the saved metrics and
    per-user NDCG scores. Otherwise, it trains the model, evaluates group
    fairness on `R_test`, stores the clean metrics/NDCG checkpoint, and returns
    them for later comparison against post-attack results.

    For base models, the unprivileged and privileged groups are inferred from
    the clean validation result. For fairness-aware models, the group assignment
    is copied from the corresponding base model so clean and attacked runs use
    the same group direction.
    """
    model_tag = model_result_name(model_kind, model_name, attribute)
    current_key = make_run_key(dataset, retain_ratio, seed, model_tag, attribute)

    if current_key in clean_rows_by_key:
        if current_key not in clean_ndcg_by_key:
            raise RuntimeError(f"Missing clean NDCG checkpoint for {current_key}.")
        row = clean_rows_by_key[current_key]
        ndcg = clean_ndcg_by_key[current_key]
        log_progress(
            f"Skipping clean {dataset}/retain={retain_ratio}/seed={seed}/{model_tag}",
            PROGRESS_LOG_PATH,
        )
        return row, ndcg, clean_run_id

    if model_kind == "base":
        model = initialize_base_models(opt_params, SEED_FOR_MODELS)[model_name]
        if model_name == "neumf":
            R_hat = train_model(
                model,
                R_train_retained,
                R_test,
                val_matrix=R_val,
                mask_matrix=R_fit,
            )
        else:
            R_hat = train_model(model, R_fit, R_test)

        provisional = compute_metrics(
            R_hat, R_test, RATING_THRES, LIST_SIZE, METRICS, groups=group_indices
        )
        unpriv, priv = infer_unprivileged_privileged(provisional, attribute, GROUPS_CONFIG)
        metrics = add_group_ratios(provisional, attribute, unpriv, priv, METRICS, GROUPS_CONFIG)
        base_group_assignments[model_name] = (unpriv, priv)
    else:
        base_model = get_base_model_for(model_name)
        if base_model not in base_group_assignments:
            raise RuntimeError(f"Missing base group assignment for {base_model}.")
        unpriv, priv = base_group_assignments[base_model]
        model = initialize_fair_models(opt_params, SEED_FOR_MODELS)[model_name]
        R_hat = train_model(model, R_fit, R_test, unprivileged_users=group_indices[unpriv])
        metrics = evaluate_group_metrics(
            R_hat, R_test, attribute, group_indices, unpriv, priv
        )
    metrics["delta_m"] = delta_m(metrics, unpriv, priv)

    key = current_key
    ndcg = tndcg_at_n(R_hat, R_test, RATING_THRES, LIST_SIZE)
    clean_ndcg_store[key] = ndcg
    clean_ndcg_index.append({
        "key": key,
        "dataset": dataset,
        "retain_ratio": retain_ratio,
        "seed": seed,
        "model": model_tag,
        "attribute": attribute,
    })

    row = {
        "run_key": current_key,
        "dataset": dataset,
        "retain_ratio": retain_ratio,
        "seed": seed,
        "model": model_tag,
        "attribute": attribute,
        "unprivileged_group": unpriv,
        "privileged_group": priv,
        **stats,
        **metrics,
    }
    clean_results.append(row)
    clean_rows_by_key[current_key] = row
    clean_ndcg_by_key[current_key] = ndcg

    write_checkpoint(
        CLEAN_PATH, CLEAN_NDCG_INDEX_PATH, CLEAN_NDCG_PATH,
        clean_results, clean_ndcg_index, clean_ndcg_store,
    )
    log_progress(
        f"Checkpointed clean {dataset}/retain={retain_ratio}/seed={seed}/{model_tag}",
        PROGRESS_LOG_PATH,
    )
    return row, ndcg, clean_run_id


def write_aggregate(summary_rows):
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return

    group_cols = [
        "dataset",
        "retain_ratio",
        "model",
        "attack_id",
        "attribute",
        "target_items",
        "budget",
    ]
    value_cols = [
        "gamma_clean",
        "gamma_post",
        "delta_gamma",
        "delta_m_clean",
        "delta_m_post",
        "delta_delta_m",
        "p_value",
    ]
    existing_values = [col for col in value_cols if col in df.columns]
    aggregate = (
        df.groupby(group_cols, dropna=False)[existing_values]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple) else col
        for col in aggregate.columns
    ]
    aggregate.to_csv(AGGREGATE_PATH, index=False)


def run_experiment():
    for folder in (OUTPUT_DIR, CLEAN_DIR, ATTACKED_DIR, PROGRESS_LOG_PATH.parent):
        folder.mkdir(parents=True, exist_ok=True)

    selected_attacks = [
        attack for attack in ATTACK_CONFIGS
        if attack["attack_id"] in ATTACK_IDS_TO_RUN
    ]
    unknown_attack_ids = set(ATTACK_IDS_TO_RUN) - {
        attack["attack_id"] for attack in ATTACK_CONFIGS
    }
    if unknown_attack_ids:
        raise ValueError(f"Unknown attack ids: {sorted(unknown_attack_ids)}")

    data, groups_gender, groups_age = load_dataset_by_name(DATASET)
    with open(paths.opt_params(DATASET), "r") as f:
        opt_params = json.load(f)

    train_df, val_df, test_df = chronological_split_per_user(
        data, return_dataframes=True
    )
    uid_to_index, iid_to_index = create_mappings(data)

    def to_csr(df):
        return convert_to_csr(df, uid_to_index, iid_to_index)

    R_train_original = to_csr(train_df)
    R_val = to_csr(val_df)
    R_test = to_csr(test_df)

    raw_groups_map = {"gender": groups_gender, "age": groups_age}
    groups_map = {
        attr: map_user_indices(raw_groups_map[attr], uid_to_index)
        for attr in GROUPS_CONFIG
        if raw_groups_map.get(attr)
    }

    base_model_names = list(initialize_base_models(opt_params, SEED_FOR_MODELS).keys())
    fair_model_names = list(initialize_fair_models(opt_params, SEED_FOR_MODELS).keys())
    model_specs = [("base", model_name) for model_name in base_model_names] + [
        ("fair", model_name) for model_name in fair_model_names
    ]

    clean_results, clean_ndcg_index, clean_ndcg_store, clean_run_id, _ = load_checkpoint(
        CLEAN_PATH, CLEAN_NDCG_INDEX_PATH, CLEAN_NDCG_PATH, CLEAN_KEY_COLUMNS
    )
    clean_rows_by_key = {
        row["run_key"]: row
        for row in clean_results
        if pd.notna(row.get("run_key"))
    }
    clean_ndcg_by_key = {
        row["key"]: clean_ndcg_store[row["key"]]
        for row in clean_ndcg_index
        if row.get("key") in clean_ndcg_store
    }
    (
        summary_results,
        summary_ndcg_index,
        summary_ndcg_store,
        summary_run_id,
        summary_completed,
    ) = load_checkpoint(
        SUMMARY_PATH,
        SUMMARY_NDCG_INDEX_PATH,
        SUMMARY_NDCG_PATH,
        SUMMARY_KEY_COLUMNS,
    )

    log_progress(
        f"Starting ML1M sparsity C-Fair Influencer experiment: retain={RETAIN_RATIOS}, "
        f"subsample_seeds={SUBSAMPLE_SEEDS}, "
        f"budgets={BUDGETS}, attacks={ATTACK_IDS_TO_RUN}, "
        f"models={'all' if MODEL_IDS_TO_RUN is None else MODEL_IDS_TO_RUN}",
        PROGRESS_LOG_PATH,
    )

    variants = [(1.0, SEED_FOR_MODELS)] + [
        (retain_ratio, seed)
        for retain_ratio, seed in itertools.product(RETAIN_RATIOS, SUBSAMPLE_SEEDS)
        if retain_ratio != 1.0
    ]

    for retain_ratio, seed in variants:
        if retain_ratio == 1.0:
            # Full-data baseline: keep the chronological training split unchanged.
            R_train_retained = R_train_original
        else:
            train_retained_df = subsample_train_per_user(train_df, retain_ratio, seed)
            R_train_retained = to_csr(train_retained_df)
        R_fit = R_train_retained + R_val if FIT_WITH_VALIDATION else R_train_retained
        stats = sparsity_stats(R_train_original, R_train_retained, R_fit)

        log_progress(
            f"Variant retain={retain_ratio}, seed={seed}: "
            f"retained {stats['train_interactions_retained']}/"
            f"{stats['train_interactions_original']} train interactions, "
            f"train density={stats['train_density_retained']:.6f}, "
            f"fit density={stats['fit_density']:.6f}",
            PROGRESS_LOG_PATH,
        )

        for attr, group_indices in groups_map.items():
            base_group_assignments = {}
            clean_cache = {}
            for model_kind, model_name in model_specs:
                model_tag = model_result_name(model_kind, model_name, attr)
                if not model_is_selected(model_name, model_tag):
                    continue

                clean_row, ndcg_clean, clean_run_id = train_or_reuse_clean(
                    DATASET,
                    retain_ratio,
                    seed,
                    attr,
                    model_kind,
                    model_name,
                    opt_params,
                    R_train_retained,
                    R_fit,
                    R_val,
                    R_test,
                    group_indices,
                    base_group_assignments,
                    stats,
                    clean_results,
                    clean_ndcg_index,
                    clean_ndcg_store,
                    clean_rows_by_key,
                    clean_ndcg_by_key,
                    clean_run_id,
                )
                clean_cache[model_tag] = (clean_row, ndcg_clean)
                if model_kind == "base":
                    base_group_assignments[model_name] = (
                        clean_row["unprivileged_group"],
                        clean_row["privileged_group"],
                    )

            for attack in selected_attacks:
                attack_id = attack["attack_id"]

                for model_kind, model_name in model_specs:
                    model_tag = model_result_name(model_kind, model_name, attr)
                    if not model_is_selected(model_name, model_tag):
                        continue

                    if model_tag not in clean_cache:
                        raise RuntimeError(f"Missing clean cache for {model_tag}.")

                    clean_row, ndcg_clean = clean_cache[model_tag]
                    unpriv = clean_row["unprivileged_group"]
                    priv = clean_row["privileged_group"]
                    fake_user_base = len(group_indices[unpriv])

                    num_targets = NUM_TARGET_ITEMS
                    for budget in BUDGETS:
                        fake_users = (
                            0
                            if budget == 0
                            else budget_to_fake_users(budget, fake_user_base)
                        )
                        current_key = make_run_key(
                            DATASET, retain_ratio, seed, model_tag, attack_id, attr,
                            num_targets, budget,
                        )
                        if current_key in summary_completed:
                            log_progress(
                                f"Skipping completed {DATASET}/retain={retain_ratio}/seed={seed}/"
                                f"{attack_id}/{model_tag}/attr={attr}/budget={budget}",
                                PROGRESS_LOG_PATH,
                            )
                            continue

                        if budget == 0:
                            ndcg_post = ndcg_clean
                            post_metrics = {
                                key: value
                                for key, value in clean_row.items()
                                if key in METRICS
                                or any(key.startswith(f"{metric}_") for metric in METRICS)
                            }
                        else:
                            log_progress(
                                f"Running {DATASET}/retain={retain_ratio}/seed={seed}/"
                                f"{attack_id}/{model_tag}/attr={attr}/budget={budget}",
                                PROGRESS_LOG_PATH,
                            )
                            attack_train = (
                                R_train_retained
                                if model_kind == "base" and model_name == "neumf"
                                else R_fit
                            )
                            R_aug = perform_attack(
                                attack["attack_cls"],
                                attack_train,
                                uids_group=group_indices[unpriv],
                                fake_users=fake_users,
                                num_targets=num_targets,
                                min_rating=MIN_RATING,
                                max_rating=MAX_RATING,
                                attack_config=attack["attack_config"],
                            )

                            if model_kind == "base":
                                model = initialize_base_models(opt_params, SEED_FOR_MODELS)[model_name]
                                if model_name == "neumf":
                                    R_hat = train_model(
                                        model,
                                        R_aug,
                                        R_test,
                                        val_matrix=R_val,
                                        mask_matrix=R_fit,
                                    )
                                else:
                                    R_hat = train_model(model, R_aug, R_test)
                            else:
                                model = initialize_fair_models(opt_params, SEED_FOR_MODELS)[model_name]
                                unprivileged_users_aug = list(group_indices[unpriv]) + list(
                                    range(R_test.shape[0], R_aug.shape[0])
                                )
                                R_hat = train_model(
                                    model, R_aug, R_test,
                                    unprivileged_users=unprivileged_users_aug,
                                )

                            ndcg_post = tndcg_at_n(R_hat, R_test, RATING_THRES, LIST_SIZE)
                            post_metrics = evaluate_group_metrics(
                                R_hat, R_test, attr, group_indices, unpriv, priv
                            )

                        ndcg_key = current_key
                        summary_ndcg_store[ndcg_key] = ndcg_post
                        summary_ndcg_index.append({
                            "key": ndcg_key,
                            "dataset": DATASET,
                            "retain_ratio": retain_ratio,
                            "seed": seed,
                            "model": model_tag,
                            "attack_id": attack_id,
                            "attribute": attr,
                            "target_items": num_targets,
                            "budget": budget,
                        })

                        u_stat, p_value, significant = mannwhitney_group_drop_test(
                            ndcg_clean, ndcg_post, group_indices, unpriv, priv
                        )
                        gamma_clean = clean_row[f"ndcg_{attr}_ratio"]
                        gamma_post = post_metrics[f"ndcg_{attr}_ratio"]
                        delta_m_clean = clean_row["delta_m"]
                        delta_m_post = delta_m(post_metrics, unpriv, priv)

                        row = {
                            "run_key": current_key,
                            "dataset": DATASET,
                            "retain_ratio": retain_ratio,
                            "seed": seed,
                            "model": model_tag,
                            "attack_id": attack_id,
                            "attribute": attr,
                            "target_items": num_targets,
                            "budget": budget,
                            "fake_users": fake_users,
                            "unprivileged_group": unpriv,
                            "privileged_group": priv,
                            **stats,
                            "gamma_clean": gamma_clean,
                            "gamma_post": gamma_post,
                            "delta_gamma": gamma_post - gamma_clean,
                            "delta_m_clean": delta_m_clean,
                            "delta_m_post": delta_m_post,
                            "delta_delta_m": delta_m_post - delta_m_clean,
                            "mannwhitney_u": u_stat,
                            "p_value": p_value,
                            "significant_p_0_05": significant,
                        }
                        row.update(post_metrics)

                        summary_results.append(row)
                        summary_completed.add(current_key)

                        write_checkpoint(
                            SUMMARY_PATH, SUMMARY_NDCG_INDEX_PATH, SUMMARY_NDCG_PATH,
                            summary_results, summary_ndcg_index, summary_ndcg_store,
                        )
                        log_progress(
                            f"Checkpointed {DATASET}/retain={retain_ratio}/seed={seed}/"
                            f"{attack_id}/{model_tag}/attr={attr}/budget={budget}",
                            PROGRESS_LOG_PATH,
                        )

    write_aggregate(summary_results)
    log_progress(f"Wrote summary to {SUMMARY_PATH}", PROGRESS_LOG_PATH)
    log_progress(f"Wrote aggregate to {AGGREGATE_PATH}", PROGRESS_LOG_PATH)


def main():
    install_graceful_stop_handler()
    try:
        run_experiment()
    except KeyboardInterrupt:
        exit_after_interruption(
            "Interrupted by user. Completed rows are already checkpointed.",
            PROGRESS_LOG_PATH,
        )


if __name__ == "__main__":
    main()
