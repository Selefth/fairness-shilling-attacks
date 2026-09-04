"""Run group-agnostic shilling baselines for C-fairness.

This experiment answers a different question from the C-fair attacks in
``03_train_aug.ipynb``: can standard shilling heuristics, without explicitly
targeting the unprivileged group, still damage C-fairness?

The attacks are intentionally constructed from the whole training matrix. For a
comparable budget, however, each budget fraction is converted to an
absolute number of fake users relative to the unprivileged group size.
"""

import os
import sys
from pathlib import Path


import json
import itertools
import numpy as np
import pandas as pd

from src.helper_functions.data_loader import *
from src.helper_functions.data_splitter import *
from src.helper_functions.train_utils import *
from src.helper_functions.metrics_accuracy import *
from src.helper_functions.attack_utils import perform_attack, resolve_num_selected, select_target_iids
from src.helper_functions.config_loader import load_attack_configs
from src.helper_functions import paths
from src.helper_functions.result_utils import (
    add_group_ratios,
    budget_to_fake_users,
    exit_after_interruption,
    get_group_assignments,
    install_graceful_stop_handler,
    load_checkpoint,
    log_progress,
    make_run_key,
    mannwhitney_group_drop_test,
    store_user_labels,
    write_checkpoint,
)
from src.attacks.registry import VANILLA_ATTACK_REGISTRY


DATASETS = ["ml-100k", "ml-1m", "lastfm-1k", "ftky", "fnyc"]
SEED = 42
RATING_THRES = 4
LIST_SIZE = 10

METRICS = {"ndcg": tndcg_at_n}
GROUPS_CONFIG = {"gender": ["f", "m"]}

NUM_T_ITEMS = [10]
BUDGETS = [0.1, 0.25]
GAMMA_THRESHOLD = 0.8

ATTACK_CONFIG_PATH = paths.attack_configs()
ATTACK_CONFIGS = load_attack_configs(
    ATTACK_CONFIG_PATH, "vanilla_attacks", VANILLA_ATTACK_REGISTRY
)
PROGRESS_LOG_PATH = paths.vanilla_progress_log(LIST_SIZE)

# Server-control reminder:
#   nohup poetry run python experiments/vanilla_shilling_baselines.py > vanilla_baselines.out 2>&1 &
#   echo $! > vanilla_baseline.pid
#   tail -f results/logs/vanilla_shilling_baselines_top_10.log
#   ps -p $(cat vanilla_baseline.pid) -o pid,etime,stat,cmd
#   kill -INT $(cat vanilla_baseline.pid)
# Re-run the same nohup command to resume from completed checkpointed rows.

# Edit this list when you want a subset, e.g. ["bandwagon"].
ATTACK_IDS_TO_RUN = [attack["attack_id"] for attack in ATTACK_CONFIGS]

headers = [
    "run_key", "dataset", "model", "attack_id", "attribute", "target_items", "selected_items",
    "filler_items", "budget", "fake_users", "unprivileged_group", "privileged_group",
    *METRICS.keys(),
    *[f"{m}_{g}" for groups in GROUPS_CONFIG.values() for g in groups for m in METRICS],
    *[f"{m}_{attr}_ratio" for attr in GROUPS_CONFIG for m in METRICS],
    "mannwhitney_u", "p_value",
]

RESUME_KEY_COLUMNS = [
    "dataset",
    "model",
    "attack_id",
    "attribute",
    "target_items",
    "selected_items",
    "filler_items",
    "budget",
]


def main() -> None:
    install_graceful_stop_handler()

    selected_attacks = [
        attack for attack in ATTACK_CONFIGS
        if attack["attack_id"] in ATTACK_IDS_TO_RUN
    ]

    unknown_attack_ids = set(ATTACK_IDS_TO_RUN) - {attack["attack_id"] for attack in ATTACK_CONFIGS}
    if unknown_attack_ids:
        raise ValueError(f"Unknown attack ids: {sorted(unknown_attack_ids)}")

    log_progress(
        f"Starting group-agnostic baseline for datasets={DATASETS}, "
        f"attacks={ATTACK_IDS_TO_RUN}, budgets={BUDGETS}, target_items={NUM_T_ITEMS}",
        PROGRESS_LOG_PATH,
    )

    try:
        for dataset in DATASETS:
            log_progress(f"Loading dataset={dataset}", PROGRESS_LOG_PATH)
            data, groups_gender, _ = load_dataset_by_name(dataset)
            folder = paths.dataset_dir(dataset)

            with open(paths.opt_params(dataset), "r") as f:
                opt_params = json.load(f)

            R_train, R_val, R_test, uid_to_index, iid_to_index = chronological_split_per_user(data)
            R_train_full = R_train + R_val

            groups_map = {"gender": map_user_indices(groups_gender, uid_to_index)}

            store_user_labels(folder, R_test, groups_map)

            group_assignments = get_group_assignments(folder, LIST_SIZE, GROUPS_CONFIG)
            clean_ndcg_index = pd.read_csv(paths.pre_attack_index(dataset))
            with np.load(paths.pre_attack_ndcg(dataset), allow_pickle=True) as clean_runs:
                clean_ndcg_by_model = {
                    row["model"]: clean_runs[row["key"]]
                    for _, row in clean_ndcg_index.iterrows()
                    if row["key"] in clean_runs.files
                }

            base_model_names = list(initialize_base_models(opt_params, SEED).keys())
            fair_model_names = list(initialize_fair_models(opt_params, SEED).keys())
            model_specs = [("base", model_name) for model_name in base_model_names] + [
                ("fair", model_name) for model_name in fair_model_names
            ]

            for attack in selected_attacks:
                attack_id = attack["attack_id"]
                post_attack_ndcg_dir = paths.post_attack_dir(dataset, attack_id)
                os.makedirs(post_attack_ndcg_dir, exist_ok=True)
                result_path = paths.attack_results(dataset, attack_id, LIST_SIZE)
                ndcg_index_path = Path(f"{post_attack_ndcg_dir}/ndcg_index.csv")
                ndcg_path = Path(f"{post_attack_ndcg_dir}/ndcg.npz")

                rng = np.random.default_rng(SEED)
                results, ndcg_index, ndcg_store, run_id, completed = load_checkpoint(
                    result_path,
                    ndcg_index_path,
                    ndcg_path,
                    RESUME_KEY_COLUMNS,
                    required_result_columns=["p_value"],
                )
                log_progress(
                    f"Dataset={dataset} attack={attack_id}: loaded {len(completed)} completed rows",
                    PROGRESS_LOG_PATH,
                )

                for attr in GROUPS_CONFIG:
                    group_indices = groups_map[attr]
                    if not group_indices:
                        continue

                    for model_kind, model_name in model_specs:
                        group_reference_model = (
                            model_name if model_kind == "base" else get_base_model_for(model_name)
                        )
                        unpriv, priv = group_assignments[group_reference_model][attr]
                        fake_user_base = len(groups_map[attr][unpriv])

                        combinations = itertools.product(
                            NUM_T_ITEMS,
                            attack["num_selected_values"],
                            attack["num_filler_values"],
                            BUDGETS,
                        )

                        for num_targets, num_selected, num_fillers, budget in combinations:
                            num_selected = resolve_num_selected(
                                num_selected, R_train_full, num_targets
                            )
                            fake_users = budget_to_fake_users(budget, fake_user_base)
                            target_iids = select_target_iids(R_train_full, num_targets, rng)

                            model_tag = model_result_name(model_kind, model_name, attr)
                            current_key = make_run_key(
                                dataset, model_tag, attack_id, attr, num_targets,
                                num_selected, num_fillers, budget, SEED
                            )
                            if current_key in completed:
                                log_progress(
                                    f"Skipping completed {dataset}/{attack_id}/{model_tag}: "
                                    f"selected={num_selected}, targets={num_targets}, budget={budget}",
                                    PROGRESS_LOG_PATH,
                                )
                                continue

                            log_progress(
                                f"Running {dataset}/{attack_id}/{model_tag}: "
                                f"selected={num_selected}, targets={num_targets}, budget={budget}",
                                PROGRESS_LOG_PATH,
                            )
                            R_aug = perform_attack(
                                attack["attack_cls"], R_train_full,
                                target_iids=target_iids,
                                fake_users=fake_users,
                                num_selected=num_selected,
                                seed=SEED,
                            )

                            if model_kind == "base":
                                model = initialize_base_models(opt_params, SEED)[model_name]
                                R_hat = train_model(model, R_aug, R_test)
                            else:
                                unprivileged_users_aug = list(groups_map[attr][unpriv]) + list(
                                    range(R_test.shape[0], R_aug.shape[0])
                                )
                                model = initialize_fair_models(opt_params, SEED)[model_name]
                                R_hat = train_model(
                                    model, R_aug, R_test,
                                    unprivileged_users=unprivileged_users_aug
                                )

                            key = current_key
                            post_ndcg = tndcg_at_n(R_hat, R_test, RATING_THRES, LIST_SIZE)
                            ndcg_store[key] = post_ndcg
                            u_stat, p_value, _ = mannwhitney_group_drop_test(
                                clean_ndcg_by_model.get(model_tag), post_ndcg, group_indices,
                                unpriv, priv,
                            )
                            ndcg_index.append({
                                "key": key,
                                "dataset": dataset,
                                "model": model_tag,
                                "attack_id": attack_id,
                                "attribute": attr,
                                "unprivileged_group": unpriv,
                                "privileged_group": priv,
                                "target_items": num_targets,
                                "selected_items": num_selected,
                                "filler_items": num_fillers,
                                "budget": budget,
                                "fake_users": fake_users,
                            })

                            all_metrics = compute_metrics(
                                R_hat, R_test, RATING_THRES, LIST_SIZE, METRICS,
                                groups=group_indices
                            )
                            all_metrics = add_group_ratios(
                                all_metrics, attr, unpriv, priv, METRICS, GROUPS_CONFIG
                            )

                            row = {
                                "run_key": current_key,
                                "dataset": dataset,
                                "model": model_tag,
                                "attack_id": attack_id,
                                "attribute": attr,
                                "unprivileged_group": unpriv,
                                "privileged_group": priv,
                                "target_items": num_targets,
                                "selected_items": num_selected,
                                "filler_items": num_fillers,
                                "budget": budget,
                                "fake_users": fake_users,
                                "mannwhitney_u": u_stat,
                                "p_value": p_value,
                            }
                            row.update(all_metrics)
                            results.append(row)
                            completed.add(current_key)

                            write_checkpoint(
                                result_path, ndcg_index_path, ndcg_path, results,
                                ndcg_index, ndcg_store, headers
                            )
                            log_progress(
                                f"Checkpointed {dataset}/{attack_id}/{model_tag}",
                                PROGRESS_LOG_PATH,
                            )

    except KeyboardInterrupt:
        exit_after_interruption(
            "Interrupted by user. Completed rows are already checkpointed.",
            PROGRESS_LOG_PATH,
        )


if __name__ == "__main__":
    main()
