
import numpy as np
from scipy.sparse import csr_matrix

class EASE:
    """
    Implementation of EASE (Embarrassingly Shallow AutoEncoders) recommender system.
    Allows choosing between item-item and user-user similarities by specifying the `method` parameter.
    Reference: Harald Steck. Embarrassingly shallow autoencoders for sparse data. WWW 2019.
    """

    def __init__(self, l2=1e2, method="item"):
        """
        Initialize the EASE model with regularization parameter and method choice.
        """
        self.l2 = l2
        self.method = method
        self.similarity_matrix_ = None

    def fit(self, X: csr_matrix):
        """
        Compute the coefficient matrix of EASE.
        """
        # Transpose the matrix for user-based approach
        if self.method == "user": X = X.T

        # Compute the P matrix
        P = (X.T @ X).toarray().astype("float32")
        dIndices = np.diag_indices(X.shape[1])
        P[dIndices] += self.l2

        # Compute the coefficient matrix W
        P = np.linalg.inv(P)
        # W = P / (-np.diag(P))
        W = P / (-np.einsum('ii->i', P)) # more efficient
        W[dIndices] = 0
        self.similarity_matrix_ = W

    def predict(self, X: csr_matrix) -> csr_matrix:
        """
        Predict the ranking scores using the fitted model.
        """
        # Compute scores
        scores = self.similarity_matrix_.T @ X if self.method == "user" else X @ self.similarity_matrix_
        # Convert to csr_matrix if not already one
        scores = csr_matrix(scores) if not isinstance(scores, csr_matrix) else scores
        return scores

    def check_fit_complete(self):
        """
        Check if the model has been fit.
        """
        if self.similarity_matrix_ is None:
            raise RuntimeError("The model has not been fit yet.")
