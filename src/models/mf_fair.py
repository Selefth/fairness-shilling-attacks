
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from scipy.sparse import csr_matrix
import random

import warnings

class MFModule(nn.Module):
    """
    Implementation of the MF (Matrix Factorization) model with optional fairness constraints 
    Reference: Yao S. and Huang B., 'Beyond Parity: Fairness Objectives for Collaborative Filtering', NeurIPS 2017.
    """

    def __init__(self, num_users, num_items, num_factors=100):
        super().__init__()

        self.num_factors = num_factors
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding = nn.Embedding(num_users, num_factors)  # User embedding
        self.item_embedding = nn.Embedding(num_items, num_factors)  # Item embedding

        # Initialize weights using Xavier initialization
        init.xavier_normal_(self.user_embedding.weight)
        init.xavier_normal_(self.item_embedding.weight)

    def forward(
        self, user_tensor: torch.Tensor, item_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute predicted scores for all items given a batch of users.
        -------
        :param user_tensor: A tensor (torch.Tensor) containing the indices of the batched users.
        :param item_tensor: A tensor (torch.Tensor) containing the indices of all items.
        
        :return: 
        - torch.Tensor: A tensor of predicted scores with shape (num_users_in_batch, num_items).
        """
        U_batch = self.user_embedding(user_tensor)
        V = self.item_embedding(item_tensor)

        return U_batch.matmul(V.T)

class FairMF:
    def __init__(self, max_epochs=250, min_delta=1e-4, learning_rate=1e-3, patience=5, l2=1e-5, fair_lambda=1, num_factors=64, seed=None, fairness_type=None):
        self.max_epochs = max_epochs # max number of epochs to train
        self.learning_rate = learning_rate # how much to update the weights at each update
        self.patience = patience # number of epochs to wait for an improvement
        self.min_delta = min_delta # a threshold for "significant" change
        self.l2_lambda = l2 # L2 regularization strength
        self.fair_lambda = fair_lambda # fairness regularization strength
        self.num_factors = num_factors # embedding size
        self.seed = seed # for reproducibility across PyTorch and NumPy
        self.fairness_type = fairness_type # type of fairness regularizer to use

        # If a seed is provided, apply it
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def _init_model(self, X:csr_matrix):
        num_users, num_items = X.shape
        self.model_ = MFModule(num_users, num_items, num_factors=self.num_factors).to(self.device)

        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

    def fit(self, X: csr_matrix, unprivileged_users: list = None) -> None:
        """
        Train the model over the input data for a specified number of epochs.
        -------
        :param X: The user-item interaction matrix (scipy.sparse.csr_matrix) with shape (num_users, num_items).
        :param unprivileged_users: Optional (list). User indices belonging to the unprivileged/protected group.
        """ 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        group_indicator = None
        if unprivileged_users is not None:
            group_indicator = torch.zeros(X.shape, dtype=torch.bool)
            group_indicator[unprivileged_users, :] = True

        if group_indicator is None or not self.fairness_type:
            if self.fairness_type and group_indicator is None:
                warnings.warn(
                    f"Fairness type is set to '{self.fairness_type}', but no unprivileged_users were provided. "
                    "The classical MF model will be trained instead.",
                    UserWarning
                )
            elif group_indicator is not None and not self.fairness_type:
                warnings.warn(
                    "unprivileged_users were provided, but no fairness_type was set. "
                    "The classical MF model will be trained instead.",
                    UserWarning
                )
            group_indicator = None
        else:
            group_indicator = group_indicator.to(self.device)

        self._init_model(X)
        self.model_.train()

        self.best_loss = float('inf')
        self.patience_counter = 0

        user_tensor = torch.arange(X.shape[0]).to(self.device)
        item_tensor = torch.arange(X.shape[1]).to(self.device)
        expected_scores = torch.tensor(X.toarray(), dtype=torch.float32, device=self.device) # from sparse to tensor (naive), all unobserved values become 0

        self.epochs = 0  # number of epochs completed
        
        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad() # reset gradients
            scores = self.model_(user_tensor, item_tensor) # compute predictions (forward pass)
            loss = self._compute_loss(expected_scores, scores, group_indicator if group_indicator is not None else None)
            loss.backward() # compute gradients
            self.optimizer.step() # update weights according to the gradients

            # Track loss and early stopping
            current_loss = loss.item()
            self.epochs += 1
            if self.best_loss - current_loss > self.min_delta:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # print(f"Epoch {epoch+1}/{self.max_epochs}, Loss: {current_loss:.4f}")

            # Early stopping check
            if self.patience_counter >= self.patience:
                break
        
    def _compute_loss(self, true_scores: torch.Tensor, pred_scores: torch.Tensor, group_indicator: torch.Tensor = None) -> torch.FloatTensor:
        """
        Compute the combined loss as the sum of MSE and L2 regularization. 
        If fairness is enabled, a fairness loss component is also added.
        -------
        :param true_scores: A 2D tensor (torch.Tensor) containing the ground-truth rating scores.
        :param pred_scores: A 2D tensor (torch.Tensor) containing the predicted rating scores.
        :param group_indicator: Optional (torch.Tensor). A 2D boolean tensor indicating the protected group.
        
        :return: 
        - torch.Tensor: A scalar tensor representing the combined loss value.
        """
        observed_items_mask = true_scores != 0  # boolean 2D tensor where True indicates observed interactions
        mse_loss = F.mse_loss(pred_scores[observed_items_mask], true_scores[observed_items_mask], reduction='mean')
        l2_penalty = self.l2_lambda * (self.model_.user_embedding.weight.norm(2)**2 + 
                                       self.model_.item_embedding.weight.norm(2)**2) / 2
        fairness_loss = self._get_fairness_loss(pred_scores, true_scores, observed_items_mask, group_indicator) if group_indicator is not None else 0

        return mse_loss + l2_penalty + (self.fair_lambda * fairness_loss)
    
    def _get_fairness_loss(self, pred_scores, true_scores, observed_items_mask, group_indicator) -> torch.FloatTensor:
        if self.fairness_type == 'nonparity':
            return self._nonparity_unfairness(pred_scores[observed_items_mask], group_indicator[observed_items_mask])
        elif self.fairness_type == 'value':
            return self._value_unfairness(pred_scores, true_scores, observed_items_mask, group_indicator)
        elif self.fairness_type == 'absolute':
            return self._absolute_unfairness(pred_scores, true_scores, observed_items_mask, group_indicator)
        elif self.fairness_type == 'under':
            return self._under_unfairness(pred_scores, true_scores, observed_items_mask, group_indicator)
        elif self.fairness_type == 'over':
            return self._over_unfairness(pred_scores, true_scores, observed_items_mask, group_indicator)
        else:
            return 0
    
    def _nonparity_unfairness(self, pred_scores, group_indicator) -> torch.FloatTensor:
        avg_score_1 = pred_scores[group_indicator].mean()
        avg_score_2 = pred_scores[~group_indicator].mean()
        return F.smooth_l1_loss(avg_score_1, avg_score_2)
    
    def _value_unfairness(self, pred_scores, true_scores, observed_items_mask, group_indicator) -> torch.FloatTensor:     
        avg_pred, avg_true = self._get_item_ratings(pred_scores, true_scores, observed_items_mask, group_indicator)
        diff = avg_pred - avg_true
        loss_input = torch.abs(diff[:, 0] - diff[:, 1])
        loss_target = torch.zeros_like(loss_input, device=self.device)
        return F.smooth_l1_loss(loss_input, loss_target)

    def _absolute_unfairness(self, pred_scores, true_scores, observed_items_mask, group_indicator) -> torch.FloatTensor:
        avg_pred, avg_true = self._get_item_ratings(pred_scores, true_scores, observed_items_mask, group_indicator)
        diff = torch.abs(avg_pred - avg_true)
        loss_input = torch.abs(diff[:, 0] - diff[:, 1])
        loss_target = torch.zeros_like(loss_input, device=self.device)
        return F.smooth_l1_loss(loss_input, loss_target)

    def _under_unfairness(self, pred_scores, true_scores, observed_items_mask, group_indicator) -> torch.FloatTensor:
        """
        Computes underestimation unfairness: the difference in the magnitude of underestimation 
        (true > predicted) between two groups.
        """
        avg_pred, avg_true = self._get_item_ratings(pred_scores, true_scores, observed_items_mask, group_indicator)
        
        # Compute the difference where true ratings are greater than predicted ratings otherwise 0
        zero_tensor = torch.tensor(0.0, device=self.device)
        under_diff = torch.where(avg_true > avg_pred, avg_true - avg_pred, zero_tensor)
        
        # Compute the absolute difference in underestimations between groups
        loss_input = torch.abs(under_diff[:, 0] - under_diff[:, 1])
        loss_target = torch.zeros_like(loss_input, device=self.device)
        
        return F.smooth_l1_loss(loss_input, loss_target)

    def _over_unfairness(self, pred_scores, true_scores, observed_items_mask, group_indicator) -> torch.FloatTensor:
        """
        Computes overestimation unfairness: the difference in the magnitude of overestimation 
        (predicted > true) between two groups.
        """
        avg_pred, avg_true = self._get_item_ratings(pred_scores, true_scores, observed_items_mask, group_indicator)
        
        # Compute the difference where predicted ratings are greater than true ratings
        zero_tensor = torch.tensor(0.0, device=self.device)
        over_diff = torch.where(avg_pred > avg_true, avg_pred - avg_true, zero_tensor)
        
        # Compute the absolute difference in overestimations between groups
        loss_input = torch.abs(over_diff[:, 0] - over_diff[:, 1])
        loss_target = torch.zeros_like(loss_input, device=self.device)
        
        return F.smooth_l1_loss(loss_input, loss_target)
    
    def _get_item_ratings(self, pred_scores, true_scores, observed_items_mask, group_indicator):
        """
        Compute the average predicted and true ratings per item for each sensitive group.
        -------
        :param pred_scores: Predicted rating scores (torch.Tensor) with shape [num_users, num_items].
        :param true_scores: True rating scores (torch.Tensor) with shape [num_users, num_items].
        :param observed_items_mask: Boolean tensor (torch.Tensor) indicating observed interactions, shape [num_users, num_items].
        :param group_indicator: Boolean tensor (torch.Tensor) indicating protected group membership, shape [num_users, num_items].

        :return: 
        - avg_pred: A tensor (torch.Tensor) of shape [num_items, 2], where:
            - Column 0: average predicted scores for the protected group (Group 1).
            - Column 1: average predicted scores for the non-protected group (Group 2).
        - avg_true: A tensor (torch.Tensor) of shape [num_items, 2], where:
            - Column 0: average true scores for the protected group (Group 1).
            - Column 1: average true scores for the non-protected group (Group 2).
        """
        # Ensure tensors have the same shape
        assert pred_scores.shape == true_scores.shape == group_indicator.shape, \
            "pred_scores, true_scores, and group_indicator must have the same shape."
        
        group_1_mask = group_indicator & observed_items_mask # protected users' observed interactions
        group_2_mask = ~group_indicator & observed_items_mask # non-protected users' observed interactions

        # Group 1 (protected group)
        sum_pred_group1 = pred_scores.masked_fill(~group_1_mask, 0).sum(dim=0)  # Sum observed scores for Group 1
        sum_true_group1 = true_scores.masked_fill(~group_1_mask, 0).sum(dim=0)
        count_group1 = group_1_mask.sum(dim=0)  # Count observed interactions for Group 1

        # Group 2 (non-protected group)
        sum_pred_group2 = pred_scores.masked_fill(~group_2_mask, 0).sum(dim=0)  # Sum observed scores for Group 2
        sum_true_group2 = true_scores.masked_fill(~group_2_mask, 0).sum(dim=0)
        count_group2 = group_2_mask.sum(dim=0)  # Count observed interactions for Group 2

        # Avoid division by zero
        count_group1 = torch.where(count_group1 > 0, count_group1, torch.ones_like(count_group1))
        count_group2 = torch.where(count_group2 > 0, count_group2, torch.ones_like(count_group2))

        # Compute means
        avg_pred = torch.stack([sum_pred_group1 / count_group1, sum_pred_group2 / count_group2], dim=1)
        avg_true = torch.stack([sum_true_group1 / count_group1, sum_true_group2 / count_group2], dim=1)

        return avg_pred, avg_true
    
    def predict(self, X: csr_matrix) -> csr_matrix:
        # TODO: Prediction does not need X, only the model (fix without breaking the pipeline)
        """
        Predict the rating scores for all users in the given user-item interaction matrix.
        -------
        :param X: The user-item interaction matrix (scipy.sparse.csr_matrix) with shape (num_users, num_items).

        :return: 
        - csr_matrix: A sparse matrix containing the predicted rating scores.
        """
        self.model_.eval()
        with torch.no_grad():
            user_tensor = torch.arange(X.shape[0]).to(self.device)
            item_tensor = torch.arange(X.shape[1]).to(self.device)

            # Compute predictions for all users and items
            predictions = self.model_(user_tensor, item_tensor).detach().cpu().numpy()

        # Convert the predictions to a sparse matrix format to match the input
        return csr_matrix(predictions)
