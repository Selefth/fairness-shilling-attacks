from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return np.nan

    eps = 1e-12
    pa = (a + eps) / (a.sum() + eps * a.size)
    pb = (b + eps) / (b.sum() + eps * b.size)
    return float(jensenshannon(pa, pb, base=2) ** 2)


def compute_group_js_divergence(
    df: pd.DataFrame,
    groups: Mapping[str, Sequence],
    user_col: str = "User_id",
    item_col: str = "Item_id",
) -> float:
    if groups is None or len(groups) < 2:
        return np.nan

    group_keys = list(groups.keys())[:2]
    first_users = set(groups[group_keys[0]])
    second_users = set(groups[group_keys[1]])
    if not first_users or not second_users:
        return np.nan

    item_codes, item_uniques = pd.factorize(df[item_col], sort=False)
    valid_items = item_codes >= 0
    user_ids = df[user_col]
    first_mask = user_ids.isin(first_users).to_numpy() & valid_items
    second_mask = user_ids.isin(second_users).to_numpy() & valid_items
    if not first_mask.any() or not second_mask.any():
        return np.nan

    n_items = len(item_uniques)
    first_counts = np.bincount(item_codes[first_mask], minlength=n_items)
    second_counts = np.bincount(item_codes[second_mask], minlength=n_items)
    return js_divergence(first_counts, second_counts)
