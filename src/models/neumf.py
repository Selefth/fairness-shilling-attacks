import importlib.util
import logging

import numpy as np
from scipy.sparse import csr_matrix

from src.helper_functions.cornac_utils import build_cornac_dataset_from_csr


logger = logging.getLogger(__name__)


NEUMF_DEFAULT_CONFIG = {
    "num_factors": 8,
    "layers": [64, 32, 16, 8],
    "act_fn": "relu",
    "reg": 0.0,
    "learner": "adam",
    "lr": 0.001,
    "batch_size": 256,
    "num_neg": 4,
    "num_epochs": 50,
    "early_stopping": {"min_delta": 0.0005, "patience": 10},
    "backend": "pytorch",
    "implicit": True,
    "verbose": False,
}


class CornacNeuMF:
    """Cornac NeuMF wrapper with the repo's fit/predict interface."""

    def __init__(
        self,
        num_factors=NEUMF_DEFAULT_CONFIG["num_factors"],
        layers=NEUMF_DEFAULT_CONFIG["layers"],
        act_fn=NEUMF_DEFAULT_CONFIG["act_fn"],
        reg=NEUMF_DEFAULT_CONFIG["reg"],
        learner=NEUMF_DEFAULT_CONFIG["learner"],
        lr=NEUMF_DEFAULT_CONFIG["lr"],
        batch_size=NEUMF_DEFAULT_CONFIG["batch_size"],
        num_neg=NEUMF_DEFAULT_CONFIG["num_neg"],
        num_epochs=NEUMF_DEFAULT_CONFIG["num_epochs"],
        early_stopping=NEUMF_DEFAULT_CONFIG["early_stopping"],
        backend=NEUMF_DEFAULT_CONFIG["backend"],
        implicit=NEUMF_DEFAULT_CONFIG["implicit"],
        verbose=NEUMF_DEFAULT_CONFIG["verbose"],
        seed=None,
    ):
        self.num_factors = num_factors
        self.layers = tuple(layers)
        self.act_fn = act_fn
        self.reg = reg
        self.learner = learner
        self.lr = lr
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.num_epochs = num_epochs
        self.early_stopping = (
            early_stopping.copy() if isinstance(early_stopping, dict) else early_stopping
        )
        self.backend = backend
        self.implicit = implicit
        self.verbose = verbose
        self.seed = seed

        self.model_ = None
        self.train_set_ = None
        self.val_set_ = None
        self.uid_map_ = None
        self.iid_map_ = None
        self.shape_ = None

    def _resolve_backend(self):
        if self.backend != "auto":
            return self.backend
        if importlib.util.find_spec("torch") is not None:
            return "pytorch"
        return "tensorflow"

    def _init_model(self):
        from cornac.models import NeuMF as CornacNeuMFModel

        class NeuMFWithTorchFactors(CornacNeuMFModel):
            def _build_model_pt(self):
                from cornac.models.ncf.backend_pt import NeuMF

                model = NeuMF(
                    num_users=self.num_users,
                    num_items=self.num_items,
                    num_factors=self.num_factors,
                    layers=self.layers,
                    act_fn=self.act_fn,
                )
                if self.pretrained:
                    model.from_pretrained(
                        self.pretrained_gmf.model, self.pretrained_mlp.model, self.alpha
                    )
                return model

        self.model_ = NeuMFWithTorchFactors(
            name="neumf",
            num_factors=self.num_factors,
            layers=self.layers,
            act_fn=self.act_fn,
            reg=self.reg,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            num_neg=self.num_neg,
            lr=self.lr,
            learner=self.learner,
            backend=self._resolve_backend(),
            early_stopping=self.early_stopping,
            verbose=self.verbose,
            seed=self.seed,
        )

    def fit(self, X, unprivileged_users=None, val_matrix=None):
        """Fit standard NeuMF-End from random initialization."""
        if np.any(X.getnnz(axis=1) >= X.shape[1]):
            raise ValueError(
                "Cornac NeuMF negative sampling requires each user to have at least "
                "one unrated item."
            )

        self.shape_ = X.shape
        self.train_set_ = build_cornac_dataset_from_csr(
            X,
            seed=self.seed,
            implicit=self.implicit,
        )
        self.uid_map_ = self.train_set_.uid_map
        self.iid_map_ = self.train_set_.iid_map

        self.val_set_ = None
        if val_matrix is not None:
            self.val_set_ = build_cornac_dataset_from_csr(
                val_matrix,
                seed=self.seed,
                implicit=self.implicit,
                uid_map=self.uid_map_,
                iid_map=self.iid_map_,
            )

        self._init_model()
        self.model_.fit(self.train_set_, val_set=self.val_set_)
        return self

    def predict(self, X):
        """Return a score matrix aligned to the input matrix row/item order."""
        if self.model_ is None:
            raise RuntimeError("The model has not been fit yet.")

        n_users, n_items = X.shape
        if any(self.uid_map_.get(uid) != uid for uid in range(n_users)):
            raise RuntimeError("Cornac user ids are not aligned with matrix row ids.")
        if any(self.iid_map_.get(iid) != iid for iid in range(n_items)):
            raise RuntimeError("Cornac item ids are not aligned with matrix column ids.")

        scores = np.empty((n_users, n_items), dtype=np.float32)
        item_indices = np.arange(n_items)

        for raw_uid in range(n_users):
            _, user_scores = self.model_.rank(raw_uid, item_indices=item_indices, k=-1)
            scores[raw_uid, :] = np.asarray(user_scores, dtype=np.float32)

        return csr_matrix(scores)
