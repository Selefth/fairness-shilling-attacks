"""RecBole glue: working paths, atomic files, remapping, relevance masks.

The RecBole counterpart of cornac_utils.py.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from recbole.config import Config
from recbole.trainer import Trainer
from recbole.trainer import trainer as recbole_trainer_module
from recbole.utils import init_logger

from src.helper_functions import paths
from src.helper_functions.data_loader import load_dataset_by_name
from src.helper_functions.data_splitter import chronological_split_per_user


def make_recbole_config(*, model, dataset, config_dict) -> Config:
    """Build Config without letting RecBole re-parse the caller's CLI flags."""
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        return Config(model=model, dataset=dataset, config_dict=config_dict)
    finally:
        sys.argv = original_argv


@contextmanager
def recbole_working_directory():
    """Route RecBole's non-configurable relative paths under results/recbole/."""
    paths.RECBOLE.mkdir(parents=True, exist_ok=True)
    previous_cwd = os.getcwd()
    try:
        os.chdir(paths.RECBOLE)
        yield
    finally:
        os.chdir(previous_cwd)


def init_recbole_logger(config) -> None:
    """Initialize RecBole's fixed relative log path under results/recbole/."""
    # RecBole 1.2.1 hard-codes LOGROOT="./log/" and offers no config key.
    with recbole_working_directory():
        init_logger(config)


def make_recbole_trainer(config, model, experiment_name: str) -> Trainer:
    tensorboard_dir = paths.RECBOLE / "log_tensorboard" / experiment_name
    original_get_tensorboard = recbole_trainer_module.get_tensorboard
    recbole_trainer_module.get_tensorboard = lambda _logger: SummaryWriter(
        str(tensorboard_dir)
    )
    try:
        return Trainer(config, model)
    finally:
        recbole_trainer_module.get_tensorboard = original_get_tensorboard


def build_atomic_files(
    target_dir: Path,
    recbole_dataset: str,
    source_dataset: str,
    sensitive_attribute: str,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, dict]:
    """Write train/valid/test .inter files from the chronological split.
 
    Loaded through benchmark_filename, which also bypasses RecBole's splitter
    and its k-core filtering.
    """
    df, groups_gender, groups_age = load_dataset_by_name(source_dataset)
    groups = {"gender": groups_gender, "age": groups_age}[sensitive_attribute]

    train_df, val_df, test_df = chronological_split_per_user(
        df, test_size=test_size, val_size=val_size, return_dataframes=True
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    for suffix, part in (("train", train_df), ("valid", val_df), ("test", test_df)):
        out = part[["User_id", "Item_id", "Interaction", "Timestamp"]].copy()
        out["Timestamp"] = (
            pd.to_datetime(out["Timestamp"], utc=True).astype("int64") // 10**9
        )
        out.columns = [
            "user_id:token",
            "item_id:token",
            "rating:float",
            "timestamp:float",
        ]
        out.to_csv(
            target_dir / f"{recbole_dataset}.{suffix}.inter", sep="\t", index=False
        )

    return df, groups


def group_row_indices(dataset, groups: dict) -> dict:
    """Raw user ids to RecBole row indices, via the remap table.
 
    Index 0 is the [PAD] user and holds no token, so it maps into no group.
    """
    token2id = dataset.field2token_id[dataset.uid_field]
    out = {}
    for name, users in groups.items():
        ids = [token2id[str(u)] for u in users if str(u) in token2id]
        out[name] = np.asarray(sorted(ids), dtype=np.int64)
    return out


def build_relevance_masks(dataset, train_ds, valid_ds, test_ds, device):
    """Dense [U, I] masks: test positives, and the candidate set.

    The paper evaluates M against the ground truth of the *test* split with the
    model's parameters frozen, so the objective must not be computed on the
    interactions the graph was built from.

    Dense masks are fine for ml-100k (943 x 1682). For ml-1m this is ~22M
    entries per mask and the user axis should be chunked instead.
    """
    n_users, n_items = dataset.user_num, dataset.item_num

    def to_dense(ds):
        m = ds.inter_matrix(form="coo")
        out = torch.zeros(n_users, n_items, dtype=torch.float32)
        out[torch.as_tensor(m.row, dtype=torch.long),
            torch.as_tensor(m.col, dtype=torch.long)] = 1.0
        return out

    pos_mask = to_dense(test_ds)
    seen = to_dense(train_ds) + to_dense(valid_ds)
    cand_mask = (seen == 0).to(torch.float32)
    cand_mask[:, 0] = 0.0        # RecBole's [PAD] item
    cand_mask[0, :] = 0.0        # RecBole's [PAD] user

    return pos_mask.to(device), cand_mask.to(device)


def evaluable_users(pos_mask: torch.Tensor, group_ids: np.ndarray) -> torch.Tensor:
    """Drop users with no test positives; their per-user RMSE is 0/eps."""
    has_test = (pos_mask.sum(dim=1) > 0).cpu().numpy()
    kept = group_ids[has_test[group_ids]]
    return torch.as_tensor(kept, dtype=torch.long, device=pos_mask.device)
