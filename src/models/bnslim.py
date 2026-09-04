
import numpy as np
from scipy.sparse import csr_matrix, identity

class BNSLIM_ADMM:
    """
    Implementation of BNSLIM ADMM recommender system.
    Reference: Eleftherakis S. et al., 'Optimizing Neighborhoods for Fair top-N Recommendation', UMAP24.
    """

    def __init__(self, l1=0.5, l2=5, l3=1e3, rho=1e3, thr=1e-4, maxIter=50, method="item"):
        """
        Initialize parameters for BNSLIM.
        -------
        :param l1: Sparsity-inducing regularizer (float). Controls the number of non-zero entries in the similarity matrix.
        :param l2: Overfitting control regularizer (float). Penalizes large values to prevent overfitting.
        :param l3: Balanced regularizer (float). Adds a balancing constraint to the objective.
        :param rho: ADMM penalty parameter (float). Affects the convergence behavior of the optimization algorithm.
        :param thr: Stopping threshold for convergence (float). Training stops when updates fall below this threshold.
        :param maxIter: Maximum number of iterations (int) for the optimization loop.
        :param method: Neighborhood type (str). Either 'item' or 'user', depending on the similarity to model.
        """
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.rho = rho
        self.thr = thr
        self.maxIter = maxIter
        self.method = method

    def _fit(self, X: csr_matrix, p: np.ndarray):
        G = X.T @ X
        P = G + (self.l2 + self.rho) * identity(X.shape[1], format='csr')
        P = P.toarray() + (self.l3 * np.outer(p, p))
        P = np.linalg.inv(P)
        Z = np.zeros((X.shape[1],X.shape[1]))
        Y = np.zeros((X.shape[1],X.shape[1]))
        error = 10*self.thr; k = 0

        # ADMM iterations
        while (error > self.thr) and (k < self.maxIter):
            # W update
            Q = P @ (G + self.rho * Z - Y)
            # gamma = np.diag(Q) / np.diag(P)
            gamma = np.einsum('ii->i', Q) / np.einsum('ii->i', P) # more efficient
            W = Q - P * gamma
            np.fill_diagonal(W, 0)

            # Z update
            Z = W + (1/self.rho) * Y
            Z = np.multiply(np.sign(Z),np.maximum(np.abs(Z) - (self.l1/self.rho),0))
            np.fill_diagonal(Z, 0)

            Z[Z < 0] = 0

            # Y update
            Y = Y + self.rho * (W - Z)

            error = np.max(np.abs(W - Z))
            k += 1

        self.similarity_matrix_ = W; self.iters = k

    def fit(self, X: csr_matrix, unprivileged_users: list):
        """
        Fit the BNSLIM model to the provided interaction data.
        -------
        :param X: The user-item interaction matrix (scipy.sparse.csr_matrix).
        :param unprivileged_users: List of user indices corresponding to the unprivileged/protected group.

        :return:
        - self: The trained model instance.
        """
        X = X.T if self.method == "user" else X

        if unprivileged_users is None:
            raise ValueError("BNSLIM_ADMM.fit requires unprivileged_users.")

        p = -np.ones(X.shape[1])
        p[unprivileged_users] = 1

        self._fit(X, p)

        return self

    def predict(self, X: csr_matrix) -> csr_matrix:
        # Compute scores
        scores = self.similarity_matrix_.T @ X if self.method == "user" else X @ self.similarity_matrix_
        # Convert to csr_matrix if not already one
        scores = csr_matrix(scores) if not isinstance(scores, csr_matrix) else scores

        return scores
