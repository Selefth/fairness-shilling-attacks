from collections import OrderedDict

import numpy as np


def full_index_map(size):
    """Return a Cornac raw-id map that preserves the repo's integer row/column ids."""
    return OrderedDict((int(idx), int(idx)) for idx in range(size))


def csr_to_cornac_uir(matrix, implicit=True):
    """Yield Cornac UIR triplets from a CSR-like user-item matrix."""
    coo = matrix.tocoo()
    ratings = np.ones_like(coo.data, dtype=float) if implicit else coo.data.astype(float)

    for user_id, item_id, rating in zip(coo.row, coo.col, ratings):
        yield int(user_id), int(item_id), float(rating)


def dataframe_to_cornac_uir(
    data,
    user_col="User_id",
    item_col="Item_id",
    rating_col="Interaction",
    implicit=False,
):
    """Yield Cornac UIR triplets from the repo's interaction dataframe format."""
    for row in data[[user_col, item_col, rating_col]].itertuples(index=False, name=None):
        user_id, item_id, rating = row
        yield user_id, item_id, 1.0 if implicit else float(rating)


def build_cornac_dataset_from_csr(matrix, seed=None, implicit=True, uid_map=None, iid_map=None):
    """Build a Cornac Dataset while preserving the original matrix id space."""
    from cornac.data import Dataset

    uid_map = full_index_map(matrix.shape[0]) if uid_map is None else uid_map
    iid_map = full_index_map(matrix.shape[1]) if iid_map is None else iid_map

    return Dataset.build(
        csr_to_cornac_uir(matrix, implicit=implicit),
        fmt="UIR",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=seed,
    )


def build_cornac_dataset_from_dataframe(
    data,
    seed=None,
    implicit=False,
    uid_map=None,
    iid_map=None,
    user_col="User_id",
    item_col="Item_id",
    rating_col="Interaction",
):
    """Build a Cornac Dataset from a dataframe of user-item-rating interactions."""
    from cornac.data import Dataset

    return Dataset.build(
        dataframe_to_cornac_uir(
            data,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            implicit=implicit,
        ),
        fmt="UIR",
        global_uid_map=uid_map,
        global_iid_map=iid_map,
        seed=seed,
    )
