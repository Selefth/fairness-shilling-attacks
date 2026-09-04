from typing import List, Union
import numpy as np
from scipy.sparse import csr_matrix, vstack
from numpy.random import default_rng

from src.helper_functions.attack_utils import *
    
class PowerUserAttack:
    def __init__(self, target_iids: List[int], fake_users: Union[float, int], seed: int = 42):
        """
        Initialize the Power User Attack.
        -------
        :param target_iids: Inner IDs of the target items (column indices of im_real) that every fake profile will rate.
        :param fake_users: Number of fake users to inject.
                           - If an integer (`int`), it specifies the exact number of fake users.
                             Must be an integer >= 1.
                           - If a float (`float`), it specifies the percentage of the number of users
                             in the targeted group. For example, 0.5 means 50%, 1.0 means 100%,
                             and 2.0 means 200% of the targeted group size.
                             Must be a float > 0. 
        :param seed: RNG seed for reproducibility.
        """
   
        self.target_iids = target_iids
        self.fake_users = fake_users
        self.rng = default_rng(seed)

    def _get_num_fake_users(self, num_real_users: int) -> int:
        """Return the absolute number of fake users from fake_users (fraction or exact number)."""
        if isinstance(self.fake_users, float) and self.fake_users > 0:
            return round(num_real_users * self.fake_users)
        elif isinstance(self.fake_users, int) and self.fake_users >= 1:
            return self.fake_users
        else:
            raise ValueError("fake_users must be either a float > 0 or an integer >= 1")

    def attack(self, im_real: csr_matrix) -> csr_matrix:
        """
        Perform the Power User Attack.
        -------
        :param im_real: Real user-item interaction matrix.
        :return: A CSR sparse matrix containing the interactions of real and fake users.
        """

        # Step 1: Identify the most active user(s) within the targeted group
        interaction_counts = im_real.getnnz(axis=1)
        most_active_users = np.argsort(interaction_counts)[::-1]  # Sort users by activity in descending order
        num_fake_users = self._get_num_fake_users(im_real.shape[0])
        users_to_replicate = most_active_users[:num_fake_users]

        # Step 2: Create fake profiles by replicating the profiles of real users
        im_fake = im_real[users_to_replicate,:].tolil(copy=True)
        max_rating = im_real.data.max()
        im_fake[:, self.target_iids] = max_rating

        im_all = vstack([im_real, im_fake.tocsr()])

        return im_all