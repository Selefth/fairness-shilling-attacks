"""Run partial-knowledge variants of the PushR C-Fair Influencer attack.

Experiment details:
- models: all base and gender fairness-aware models with clean outputs
- attack strategy: PushR
- budget: 10%
- protected attribute: gender
- Sampling repeated with seeds 42, 43, 44

Outputs:
- results/other_experiments/partial_knowledge/runs.csv
- results/other_experiments/partial_knowledge/clean_results.csv
- results/other_experiments/partial_knowledge/user_labels.npz
- results/other_experiments/partial_knowledge/clean/
- results/other_experiments/partial_knowledge/attacked/
- results/other_experiments/partial_knowledge/table_values.csv
"""

import json
import os
from pathlib import Path
import sys


import numpy as np
import pandas as pd

from src.attacks.registry import CFAIR_ATTACK_REGISTRY
from src.helper_functions import paths
from src.helper_functions.attack_utils import perform_attack, resolve_num_selected
from src.helper_functions.config_loader import load_attack_configs
from src.helper_functions.data_loader import load_dataset_by_name
from src.helper_functions.data_splitter import chronological_split_per_user, map_user_indices
from src.helper_functions.metrics_accuracy import compute_metrics, tndcg_at_n
from src.helper_functions.result_utils import (
    add_group_ratios,
    verify_clean_vector_matches_aggregate,
    budget_to_fake_users,
    exit_after_interruption,
    get_group_assignments,
    install_graceful_stop_handler,
    load_checkpoint,
    log_progress,
    make_run_key,
    mannwhitney_group_drop_test,
    write_checkpoint,
)
from src.helper_functions.train_utils import (
    get_base_model_for,
    initialize_base_models,
    initialize_fair_models,
    model_result_name,
    train_model,
)


DATASETS = ["ml-100k", "ml-1m", "lastfm-1k", "ftky", "fnyc"]
GROUPS_CONFIG = {"gender": ["f", "m"]}
MODEL_IDS_TO_RUN = None  # None means all models; e.g. ["slim-u", "mf-over-gender"] for a subset.

SEED = 42
SAMPLING_SEEDS = [42, 43, 44]
RATING_THRES = 4
LIST_SIZE = 10
TARGET_ITEMS = 10
BUDGET = 0.1
MIN_RATING, MAX_RATING = 1, 5

ATTACK_IDS = [
    "cfair_influencer_push_random",
    "cfair_sampling_push_random",
    "cfair_influencer_top1_push_random",
    "cfair_influencer_top5_push_random",
    "cfair_influencer_top10_push_random",
]
ATTACK_NAMES = {
    "none": "None",
    "cfair_influencer_push_random": "C-Fair Influencer",
    "cfair_sampling_push_random": "Sampling",
    "cfair_influencer_top1_push_random": "Top-1",
    "cfair_influencer_top5_push_random": "Top-5",
    "cfair_influencer_top10_push_random": "Top-10",
}
REPEAT_SEEDS = {
    "cfair_sampling_push_random": SAMPLING_SEEDS,
}

OUTPUT_DIR = paths.partial_knowledge_dir()
TABLE_VALUES_PATH = paths.partial_knowledge_table_values()
PROGRESS_LOG_PATH = paths.log_file("partial_knowledge_attacks.log")
ATTACK_CONFIG_PATH = paths.attack_configs()

RESUME_KEY_COLUMNS = [
    "dataset",
    "model",
    "attack_id",
    "attribute",
    "target_items",
    "selected_items",
    "filler_items",
    "budget",
    "seed",
]


def average_ndcg(run_rows, ndcg_store):
    arrays = [ndcg_store[row["run_key"]] for row in run_rows if row["run_key"] in ndcg_store]
    if len(arrays) == 1:
        return arrays[0]
    return np.nanmean(np.vstack(arrays), axis=0)


def make_table_rows(dataset, clean_results, clean_ndcg, run_rows, ndcg_store, group_indices, specs):
    table_rows = []

    for kind, name in specs:
        model = model_result_name(kind, name, "gender")
        clean_row = clean_results[clean_results["model"] == model].iloc[0]
        table_rows.append({
            "dataset": dataset,
            "model": model,
            "model_kind": kind,
            "attack_id": "none",
            "attack": ATTACK_NAMES["none"],
            "budget": 0.0,
            "runs": 1,
            "seeds": "",
            "c_fairness": clean_row["ndcg_gender_ratio"],
            "c_fairness_rounded": round(clean_row["ndcg_gender_ratio"], 2),
            "p_value": np.nan,
            "significant_p_0_05": False,
        })

    df = pd.DataFrame(run_rows)
    if df.empty:
        return table_rows
    df = df[
        df.apply(
            lambda row: int(row["seed"]) in REPEAT_SEEDS.get(row["attack_id"], [SEED]),
            axis=1,
        )
    ]

    for (model, attack_id), group in df.groupby(["model", "attack_id"], sort=False):
        rows = group.to_dict("records")
        post_ndcg = average_ndcg(rows, ndcg_store)
        # The split recorded by the runner, i.e. the conventional model's.
        _, p_value, significant = mannwhitney_group_drop_test(
            clean_ndcg[model],
            post_ndcg,
            group_indices,
            group["unprivileged_group"].iloc[0],
            group["privileged_group"].iloc[0],
        )
        c_fairness = group["c_fairness"].mean()

        table_rows.append({
            "dataset": dataset,
            "model": model,
            "model_kind": group["model_kind"].iloc[0],
            "attack_id": attack_id,
            "attack": ATTACK_NAMES[attack_id],
            "budget": BUDGET,
            "runs": len(group),
            "seeds": ";".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
            "c_fairness": c_fairness,
            "c_fairness_rounded": round(c_fairness, 2),
            "p_value": p_value,
            "significant_p_0_05": significant,
        })

    return table_rows


def run_experiment():
    install_graceful_stop_handler()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths.partial_knowledge_clean_dir().mkdir(parents=True, exist_ok=True)
    paths.partial_knowledge_post_dir().mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    attack_configs = load_attack_configs(
        ATTACK_CONFIG_PATH, "cfair_attacks", CFAIR_ATTACK_REGISTRY
    )
    attacks_by_id = {attack["attack_id"]: attack for attack in attack_configs}
    attacks = [attacks_by_id[attack_id] for attack_id in ATTACK_IDS]

    run_rows, ndcg_index, ndcg_store, run_id, completed = load_checkpoint(
        paths.partial_knowledge_runs(),
        paths.partial_knowledge_post_dir() / paths.NDCG_INDEX,
        paths.partial_knowledge_post_dir() / paths.NDCG_NPZ,
        RESUME_KEY_COLUMNS,
        required_result_columns=["p_value"],
    )
    clean_rows, clean_ndcg_index_out, clean_ndcg_store_out, clean_run_id, clean_completed = load_checkpoint(
        paths.partial_knowledge_clean_results(),
        paths.partial_knowledge_clean_dir() / paths.NDCG_INDEX,
        paths.partial_knowledge_clean_dir() / paths.NDCG_NPZ,
        ["dataset", "model", "attribute"],
    )
    user_labels = {}
    all_table_rows = []

    log_progress(
        f"Starting partial-knowledge attacks with sampling_seeds={SAMPLING_SEEDS}",
        PROGRESS_LOG_PATH,
    )

    for dataset in DATASETS:
        source_dir = paths.dataset_dir(dataset)

        log_progress(f"Loading {dataset}", PROGRESS_LOG_PATH)
        data, groups_gender, _ = load_dataset_by_name(dataset)
        with open(paths.opt_params(dataset), "r") as f:
            opt_params = json.load(f)

        clean_results = pd.read_csv(paths.clean_results(dataset, LIST_SIZE))

        specs = [("base", name) for name in initialize_base_models(opt_params, SEED)]
        specs += [("fair", name) for name in initialize_fair_models(opt_params, SEED)]
        specs = [
            (kind, name)
            for kind, name in specs
            if model_result_name(kind, name, "gender") in set(clean_results["model"])
        ]
        if MODEL_IDS_TO_RUN is not None:
            selected = set(MODEL_IDS_TO_RUN)
            specs = [
                (kind, name)
                for kind, name in specs
                if name in selected or model_result_name(kind, name, "gender") in selected
            ]

        model_tags = [model_result_name(kind, name, "gender") for kind, name in specs]

        clean_ndcg_index = pd.read_csv(paths.pre_attack_index(dataset))
        with np.load(paths.pre_attack_ndcg(dataset), allow_pickle=True) as clean_ndcg_runs:
            clean_ndcg_store = {key: clean_ndcg_runs[key] for key in clean_ndcg_runs.files}
        clean_ndcg = {}
        for model in model_tags:
            key = clean_ndcg_index[clean_ndcg_index["model"] == model].iloc[0]["key"]
            clean_ndcg[model] = clean_ndcg_store[key]

        log_progress(
            f"{dataset}: running {len(model_tags)} models: {model_tags}",
            PROGRESS_LOG_PATH,
        )

        R_train, R_val, R_test, uid_to_index, _ = chronological_split_per_user(data)
        R_fit = R_train + R_val

        groups_map = {"gender": map_user_indices(groups_gender, uid_to_index)}
        group_indices = groups_map["gender"]
        group_assignments = get_group_assignments(str(source_dir), LIST_SIZE, GROUPS_CONFIG)

        # The per-user vectors and the aggregate row must describe the same run.
        # They drifted apart once, silently, when a checkpoint returned a NeuMF
        # copy taken before the model was regenerated.
        for model_tag in model_tags:
            verify_clean_vector_matches_aggregate(
                clean_ndcg[model_tag],
                clean_results[clean_results["model"] == model_tag].iloc[0],
                "gender",
                group_indices,
                GROUPS_CONFIG,
                label=f"{dataset}/{model_tag} (clean)",
            )

        labels = np.full(R_test.shape[0], "na", dtype=object)
        for group, idxs in group_indices.items():
            labels[idxs] = group
        user_labels[f"{dataset.replace('-', '_')}_gender"] = labels
        np.savez_compressed(paths.user_labels_in(OUTPUT_DIR), **user_labels)

        for kind, name in specs:
            model = model_result_name(kind, name, "gender")
            clean_key = make_run_key(dataset, model, "gender")
            # Deliberately NOT skipped when already checkpointed. This block only
            # copies the pre-attack vector and re-reads the aggregate row, so
            # redoing it is free -- and skipping it is exactly how a stale copy
            # survived here across runs. write_checkpoint dedupes by run_key, so
            # rewriting an existing entry is safe.

            group_ref = name if kind == "base" else get_base_model_for(name)
            unpriv, priv = group_assignments[group_ref]["gender"]
            clean_row = clean_results[clean_results["model"] == model].iloc[0]

            ndcg_key = clean_key
            clean_ndcg_store_out[ndcg_key] = clean_ndcg[model]
            clean_ndcg_index_out.append({
                "key": ndcg_key,
                "dataset": dataset,
                "model": model,
                "model_kind": kind,
                "attribute": "gender",
            })
            clean_rows.append({
                "run_key": ndcg_key,
                "dataset": dataset,
                "model": model,
                "model_kind": kind,
                "attribute": "gender",
                "unprivileged_group": unpriv,
                "privileged_group": priv,
                "ndcg": clean_row["ndcg"],
                "ndcg_unprivileged": clean_row[f"ndcg_{unpriv}"],
                "ndcg_privileged": clean_row[f"ndcg_{priv}"],
                "c_fairness": clean_row["ndcg_gender_ratio"],
            })
            clean_completed.add(clean_key)

        write_checkpoint(
            paths.partial_knowledge_clean_results(),
            paths.partial_knowledge_clean_dir() / paths.NDCG_INDEX,
            paths.partial_knowledge_clean_dir() / paths.NDCG_NPZ,
            clean_rows,
            clean_ndcg_index_out,
            clean_ndcg_store_out,
        )

        for attack in attacks:
            attack_id = attack["attack_id"]
            seeds = REPEAT_SEEDS.get(attack_id, [SEED])

            for kind, name in specs:
                model = model_result_name(kind, name, "gender")
                group_ref = name if kind == "base" else get_base_model_for(name)
                unpriv, priv = group_assignments[group_ref]["gender"]
                fake_users = budget_to_fake_users(BUDGET, len(group_indices[unpriv]))
                R_attack = R_train if kind == "base" and name == "neumf" else R_fit
                num_selected = resolve_num_selected(
                    attack["num_selected_values"][0],
                    R_attack,
                    TARGET_ITEMS,
                    group_indices[unpriv],
                )
                num_fillers = attack["num_filler_values"][0]

                for seed in seeds:
                    run_key = make_run_key(
                        dataset,
                        model,
                        attack_id,
                        "gender",
                        TARGET_ITEMS,
                        num_selected,
                        num_fillers,
                        BUDGET,
                        seed,
                    )
                    if run_key in completed:
                        continue

                    log_progress(
                        f"Running {dataset}/{attack_id}/{model}/seed={seed}",
                        PROGRESS_LOG_PATH,
                    )
                    R_aug = perform_attack(
                        attack["attack_cls"],
                        R_attack,
                        group_indices[unpriv],
                        fake_users,
                        TARGET_ITEMS,
                        num_selected,
                        num_fillers,
                        MIN_RATING,
                        MAX_RATING,
                        seed,
                        attack["attack_config"],
                    )

                    if kind == "base":
                        model_instance = initialize_base_models(opt_params, SEED)[name]
                        if name == "neumf":
                            R_hat = train_model(
                                model_instance,
                                R_aug,
                                R_test,
                                val_matrix=R_val,
                                mask_matrix=R_fit,
                            )
                        else:
                            R_hat = train_model(model_instance, R_aug, R_test)
                    else:
                        model_instance = initialize_fair_models(opt_params, SEED)[name]
                        unprivileged_users_aug = list(group_indices[unpriv]) + list(
                            range(R_test.shape[0], R_aug.shape[0])
                        )
                        R_hat = train_model(
                            model_instance,
                            R_aug,
                            R_test,
                            unprivileged_users=unprivileged_users_aug,
                        )
                    ndcg_post = tndcg_at_n(R_hat, R_test, RATING_THRES, LIST_SIZE)
                    metrics = compute_metrics(
                        R_hat,
                        R_test,
                        RATING_THRES,
                        LIST_SIZE,
                        {"ndcg": tndcg_at_n},
                        groups=group_indices,
                    )
                    metrics = add_group_ratios(
                        metrics, "gender", unpriv, priv, {"ndcg": tndcg_at_n}, GROUPS_CONFIG
                    )

                    _, p_value, significant = mannwhitney_group_drop_test(
                        clean_ndcg[model], ndcg_post, group_indices, unpriv, priv
                    )

                    ndcg_key = run_key
                    ndcg_store[ndcg_key] = ndcg_post
                    ndcg_index.append({
                        "key": ndcg_key,
                        "dataset": dataset,
                        "model": model,
                        "model_kind": kind,
                        "attack_id": attack_id,
                        "attribute": "gender",
                        "seed": seed,
                    })

                    row = {
                        "run_key": run_key,
                        "dataset": dataset,
                        "model": model,
                        "model_kind": kind,
                        "attack_id": attack_id,
                        "attack": ATTACK_NAMES[attack_id],
                        "attribute": "gender",
                        "target_items": TARGET_ITEMS,
                        "selected_items": num_selected,
                        "filler_items": num_fillers,
                        "budget": BUDGET,
                        "seed": seed,
                        "fake_users": fake_users,
                        "unprivileged_group": unpriv,
                        "privileged_group": priv,
                        "c_fairness": metrics["ndcg_gender_ratio"],
                        "p_value": p_value,
                        "significant_p_0_05": significant,
                    }
                    row.update(metrics)
                    run_rows.append(row)
                    completed.add(run_key)

                    write_checkpoint(
                        paths.partial_knowledge_runs(),
                        paths.partial_knowledge_post_dir() / paths.NDCG_INDEX,
                        paths.partial_knowledge_post_dir() / paths.NDCG_NPZ,
                        run_rows,
                        ndcg_index,
                        ndcg_store,
                    )

        dataset_run_rows = [row for row in run_rows if row["dataset"] == dataset]
        all_table_rows.extend(
            make_table_rows(
                dataset,
                clean_results,
                clean_ndcg,
                dataset_run_rows,
                ndcg_store,
                group_indices,
                specs,
            )
        )
        pd.DataFrame(all_table_rows).to_csv(TABLE_VALUES_PATH, index=False)
        log_progress(f"Updated {TABLE_VALUES_PATH}", PROGRESS_LOG_PATH)


def main():
    try:
        run_experiment()
    except KeyboardInterrupt:
        exit_after_interruption(
            "Interrupted by user. Completed rows are already checkpointed.",
            PROGRESS_LOG_PATH,
        )


if __name__ == "__main__":
    main()
