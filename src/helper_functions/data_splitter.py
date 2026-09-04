
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import warnings

def create_mappings(data):
    """Creates mappings for User_id and Item_id to indices."""
    unique_users = data["User_id"].unique()
    unique_items = data["Item_id"].unique()

    uid_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
    iid_to_index = {iid: idx for idx, iid in enumerate(unique_items)}

    return uid_to_index, iid_to_index

def convert_to_csr(data, uid_to_index, iid_to_index):
    """Convert DataFrame to CSR matrix."""
    rows = data["User_id"].map(uid_to_index).values
    cols = data["Item_id"].map(iid_to_index).values
    vals = data["Interaction"].values
    return csr_matrix((vals, (rows, cols)), shape=(len(uid_to_index), len(iid_to_index)))

def subsample_train_per_user(df_train, retain_ratio, seed, user_col="User_id"):
    """
    For each user, keep floor(retain_ratio * n_user) training interactions.
    n_user is that user's number of interactions in the training split.
    Users with very short histories keep at least one training interaction.
    """
    if retain_ratio <= 0 or retain_ratio > 1:
        raise ValueError("retain_ratio must be in the interval (0, 1].")

    rng = np.random.default_rng(seed)
    retained_indices = []

    for _, user_df in df_train.groupby(user_col, sort=False):
        n_interactions = len(user_df)
        n_keep = max(1, int(np.floor(retain_ratio * n_interactions)))
        retained_indices.extend(
            rng.choice(user_df.index.to_numpy(), size=n_keep, replace=False)
        )

    return df_train.loc[retained_indices].sort_index().reset_index(drop=True)


def chronological_split_per_user(data, test_size=0.20, val_size=0.10, return_dataframes=False):
    """
    Split data into train, validation, and test sets on a per-user basis chronologically.
    -------
    :param data: A pandas DataFrame containing 'User_id', 'Item_id', 'Timestamp', and 'Interaction' columns.
    :param test_size: The fraction of interactions to allocate to the test set for each user.
    :param val_size: The fraction of interactions to allocate to the validation set for each user.

    :param return_dataframes: If True, return train/validation/test DataFrames instead of CSR matrices.

    :return:
    If return_dataframes is True:
    - train_data, val_data, test_data

    Otherwise:
    - R_train: CSR sparse matrix representing the training set.
    - R_val: CSR sparse matrix representing the validation set.
    - R_test: CSR sparse matrix representing the test set.
    - uid_to_index: Dictionary mapping user ids to row indices in the sparse matrices.
    - iid_to_index: Dictionary mapping item ids to column indices in the sparse matrices.
    """
    train_data, val_data, test_data = [], [], []

    for _, user_data in data.groupby("User_id"):
        user_data_sorted = user_data.sort_values(by="Timestamp")
        n = len(user_data_sorted)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_test - n_val

        if n_val + n_test == 0:
            warnings.warn(
                f"User {user_data_sorted['User_id'].iloc[0]} has no validation/test items. ",
                UserWarning
            )

        train_data.append(user_data_sorted.iloc[:n_train])
        val_data.append(user_data_sorted.iloc[n_train:n_train + n_val])
        test_data.append(user_data_sorted.iloc[n_train + n_val:])

    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    test_data = pd.concat(test_data)

    if return_dataframes:
        return train_data, val_data, test_data

    # Create user and item mappings
    uid_to_index, iid_to_index = create_mappings(data)

    # Convert the train, val, and test sets to CSR matrices
    R_train = convert_to_csr(train_data, uid_to_index, iid_to_index)
    R_val = convert_to_csr(val_data, uid_to_index, iid_to_index)
    R_test = convert_to_csr(test_data, uid_to_index, iid_to_index)

    return R_train, R_val, R_test, uid_to_index, iid_to_index

def map_user_indices(user_groups, uid_map):
    """
    Map user IDs to their respective matrix row indices for each group.
    -------
    :param user_groups: Dictionary where keys are group names and values are lists of user IDs.
    :param uid_map: Dictionary mapping user IDs to row indices in the user-item matrix.

    :return: 
    - mapped_groups: Dictionary with the same group names as keys and lists of user indices as values.
    """
    mapped_groups = {}
    for group, users in user_groups.items():
        mapped = []; missing = []
        for uid in users:
            if uid in uid_map:
                mapped.append(uid_map[uid])
            else:
                missing.append(uid)
        if missing:
            warnings.warn(
                f"{len(missing)} user(s) in group '{group}' were not found in uid_map and were skipped.",
                UserWarning
            )
        mapped_groups[group] = mapped
    return mapped_groups
