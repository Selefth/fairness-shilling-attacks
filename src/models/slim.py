
import numpy as np
from scipy.sparse import csr_matrix, identity

class SLIM:
    """
    Implementation of Sparse Linear Method (SLIM) recommender system.
    Allows choosing between item-item and user-user similarities by specifying the `method` parameter.
    Reference: Harald Steck, et al. Admm slim: Sparse recommendations for many users. WSDM 2020.
    """

    def __init__(self, l1=0.5, l2=5, pos=True, rho=1e3, thr=1e-4, maxIter=50, method="item"):
        """
        Initialize parameters for the SLIM recommender.
        -------
        :param l1: Sparsity-inducing regularizer (float). Controls the number of non-zero entries in the similarity matrix.
        :param l2: Overfitting control regularizer (float). Penalizes large coefficient values in the similarity matrix.
        :param pos: Enforces non-negativity constraints (bool). If set to True, only positive relationships are modeled.
        :param rho: ADMM penalty parameter (float). Important for the convergence of the optimization process.
        :param thr: Stopping threshold for convergence (float). Optimization stops when improvements are smaller than this value.
        :param maxIter: Maximum number of iterations (int) for the optimizer.
        :param method: Specifies whether to use item or user neighborhoods (str).
        """
        self.l1 = l1
        self.l2 = l2
        self.pos = pos
        self.rho = rho
        self.thr = thr
        self.maxIter = maxIter
        self.method = method
        self.similarity_matrix_ = None

    def fit(self, X: csr_matrix):
        """
        Compute the coefficient matrix of SLIM.
        """
        # Transpose the matrix for user-based approach
        if self.method == "user": X = X.T
        
        G = X.T @ X
        P = G + (self.l2 + self.rho) * identity(X.shape[1], format='csr')
        P = np.linalg.inv(P.toarray())
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

            if self.pos == True:
                Z[Z < 0] = 0

            # Y update
            Y = Y + self.rho * (W - Z)

            error = np.max(np.abs(W - Z))
            k += 1

        self.similarity_matrix_ = W; self.iters = k

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
