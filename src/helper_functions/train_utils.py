import numpy as np
from scipy.sparse import csr_matrix, vstack
from src.models.ease import EASE
from src.models.slim import SLIM
from src.models.bnslim import BNSLIM_ADMM
from src.models.mf_fair import FairMF
from src.models.neumf import CornacNeuMF, NEUMF_DEFAULT_CONFIG
from recpack.algorithms.factorization import SVD, NMF

# The naming conventions live in model_naming, which imports nothing, so a result
# file can be read back without importing the model stack above. Re-exported here
# because the experiment runners import them from this module.
from src.helper_functions.model_naming import (  # noqa: F401
    BASE_MODEL_MAP,
    get_base_model_for,
    model_result_name,
)


def get_model_params(opt_params, model_name, defaults):
    """Merge tuned model params with code defaults for newly added models."""
    params = defaults.copy()
    params.update(opt_params.get(model_name, {}))
    return params


def initialize_base_models(opt_params, seed):
    """
    Initialize standard recommendation models with provided hyperparameters.

    Parameters:
        opt_params (dict): Dictionary containing optimized hyperparameters per model.
        seed (int): Random seed for reproducibility (used in stochastic models).

    Returns:
        dict: A dictionary mapping model names to initialized model instances.
    """
    neumf_params = get_model_params(opt_params, "neumf", NEUMF_DEFAULT_CONFIG)
    # The experiment seed is passed explicitly below; ignore any stale seed in opt_params.
    neumf_params.pop("seed", None)

    return {
        "ease-i": EASE(l2=opt_params["ease-i"]["l2"]),
        "ease-u": EASE(l2=opt_params["ease-u"]["l2"], method="user"),
        "slim-i": SLIM(l1=opt_params["slim-i"]["l1"], l2=opt_params["slim-i"]["l2"]),
        "slim-u": SLIM(l1=opt_params["slim-u"]["l1"], l2=opt_params["slim-u"]["l2"], method="user"),
        "svd": SVD(num_components=opt_params["svd"]["num_factors"], seed=seed),
        "mf": FairMF(
            learning_rate=opt_params["mf"]["learning_rate"],
            l2=opt_params["mf"]["l2"],
            num_factors=opt_params["mf"]["num_factors"],
            seed=seed
        ),
        "nmf": NMF(
            num_components=opt_params["nmf"]["num_factors"],
            alpha=opt_params["nmf"]["alpha"],
            l1_ratio=opt_params["nmf"]["l1_ratio"],
            seed=seed
        ),
        "neumf": CornacNeuMF(**neumf_params, seed=seed),
    }

def initialize_fair_models(opt_params, seed):
    """
    Initialize fairness-aware models with provided hyperparameters.

    Parameters:
        opt_params (dict): Dictionary of optimized hyperparameters.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary mapping fairness-aware model names to model instances.
    """
    models = {
        "bnslim-u": BNSLIM_ADMM(
            l1=opt_params["slim-u"]["l1"],
            l2=opt_params["slim-u"]["l2"],
            l3=1,
            method="user"
        )
    }

    for ftype in ["value", "absolute", "under", "over"]:
        models[f"mf-{ftype}"] = FairMF(
            learning_rate=opt_params["mf"]["learning_rate"],
            l2=opt_params["mf"]["l2"],
            num_factors=opt_params["mf"]["num_factors"],
            fairness_type=ftype,
            seed=seed
        )

    return models


# get_base_model_for and model_result_name now live in model_naming and are
# re-exported at the top of this module.


def _align_validation_matrix(val_matrix, train_shape):
    if val_matrix.shape[1] != train_shape[1]:
        raise ValueError("Validation and training matrices must have the same item dimension.")
    if val_matrix.shape[0] > train_shape[0]:
        raise ValueError("Validation matrix cannot have more users than the training matrix.")
    if val_matrix.shape[0] == train_shape[0]:
        return val_matrix

    fake_rows = train_shape[0] - val_matrix.shape[0]
    empty_fake_val = csr_matrix((fake_rows, train_shape[1]), dtype=val_matrix.dtype)
    return vstack([val_matrix, empty_fake_val], format="csr")


def train_model(model, R_train, R_test, unprivileged_users=None, val_matrix=None, mask_matrix=None):
    """
    Trains a recommendation model and returns its prediction matrix,
    with observed training entries masked out.
    """
    fit_kwargs = {}
    if unprivileged_users is not None:
        fit_kwargs["unprivileged_users"] = unprivileged_users
    if val_matrix is not None and isinstance(model, CornacNeuMF):
        fit_kwargs["val_matrix"] = _align_validation_matrix(val_matrix, R_train.shape)

    model.fit(R_train, **fit_kwargs)

    R_hat = model.predict(R_train).toarray()
    R_hat = R_hat[:R_test.shape[0], :]  # ensure alignment with test matrix
    mask_source = R_train if mask_matrix is None else mask_matrix
    R_hat[mask_source[:R_test.shape[0], :].nonzero()] = -np.inf  # mask seen items

    return R_hat
