
from typing import List, Union
from scipy.sparse import csr_matrix, lil_matrix, vstack
from numpy.random import default_rng

from src.helper_functions.attack_utils import *

class ReverseBandwagonAttack:
    def __init__(self, target_iids: List[int], fake_users: Union[float, int], num_selected: int, seed: int = 42):
        """
        Initialize the Reverse Bandwagon Attack.
        -------
        :param target_iids: Inner IDs of the target items (column indices of im_real) that every fake profile will rate.
        :param fake_users: Number of fake users to inject.
                           - If an integer (`int`), it specifies the exact number of fake users.
                             Must be an integer >= 1.
                           - If a float (`float`), it specifies the percentage of the number of users
                             in the targeted group. For example, 0.5 means 50%, 1.0 means 100%,
                             and 2.0 means 200% of the targeted group size.
                             Must be a float > 0. 
        :param num_selected: Number of selected items per attack profile.
        :param seed: RNG seed for reproducibility.
        """

        self.target_iids = target_iids
        self.fake_users = fake_users
        self.num_selected = num_selected
        self.rng = default_rng(seed)

    def _get_num_fake_users(self, num_real_users_total: int) -> int:
        """Return the absolute number of fake users from self.fake_users (float fraction or int)."""
        if isinstance(self.fake_users, float) and self.fake_users > 0:
            return round(num_real_users_total * self.fake_users)
        elif isinstance(self.fake_users, int) and self.fake_users >= 1:
            return self.fake_users

    def attack(self, im_real: csr_matrix) -> csr_matrix:
        """
        Perform the Reverse Bandwagon Attack.
        -------
        :param im_real: The real user-item interaction matrix.
        :return: A CSR sparse matrix containing the interactions of real and fake users.
        """
        num_items = im_real.shape[1]
        num_fake_users = self._get_num_fake_users(im_real.shape[0])
        im_fake = lil_matrix((num_fake_users, num_items))

        selected_iids, _ = find_least_favorite_items(im_real, num_selected=self.num_selected)
        min_rating = im_real.data.min() 
        im_fake[:, selected_iids] = min_rating
        im_fake[:, self.target_iids] = min_rating

        im_all = vstack([im_real, im_fake.tocsr()]) # concatenate im_real and im_fake vertically

        return im_all
    