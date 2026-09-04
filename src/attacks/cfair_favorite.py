
from typing import List, Union
from scipy.sparse import csr_matrix, lil_matrix, vstack
from numpy.random import default_rng

from src.helper_functions.attack_utils import *

class CFairFavoriteAttack:
    """
    Attack aimed at distorting the recommendation accuracy for a specific user group.
    The attack identifies "favorite items" among the group and assigns them the rounded average ratings from the group, simulating believable profiles.  
    Then, it selects a subset of items (either randomly or based on the group's least favorite items) from the catalog as targets, which receive the maximum rating.
    """
    def __init__(self, uids_group: List[int], fake_users: Union[float, int], num_targets: int, num_selected: int, max_rating: int, mode: str = "random", seed: int = 42):
        """
        Initialize the attack model.
        -------
        :param uids_group: The inner IDs of the targeted user group (should match the indices of im_real).
        :param fake_users: Determines the number of fake users to inject.
                           - If an integer (`int`), it specifies the exact number of fake users.
                             Must be an integer >= 1.
                           - If a float (`float`), it specifies the percentage of the number of users
                             in the targeted group. For example, 0.5 means 50%, 1.0 means 100%,
                             and 2.0 means 200% of the targeted group size.
                             Must be a float > 0.
        :param num_targets: The number of target items per attack profile.
        :param num_selected: The number of selected items per attack profile.
        :param max_rating: The maximum rating value.
        :param mode: The attack mode. Can be one of the following:
                     - `"random"`: Push random items (assign max rating to random items).
                     - `"least_favorite"`: Push least favorite items (assign max rating to the group's least favorite items).
        :param seed: Seed for random number generation to ensure reproducibility.
        """
        if mode not in ["random", "least_favorite"]:
            raise ValueError('mode must be "random" or "least_favorite".')

        self.uids_group = uids_group
        self.num_fake_users = self._get_num_fake_users(fake_users)
        self.num_targets = num_targets
        self.num_selected = num_selected
        self.max_rating = max_rating
        self.mode = mode
        self.rng = default_rng(seed)

    def _get_num_fake_users(self, fake_users) -> int:
        """Determine the number of fake users based on fake_users input (fraction or exact number)."""
        if isinstance(fake_users, float) and fake_users > 0:
            return round(len(self.uids_group) * fake_users)
        elif isinstance(fake_users, int) and fake_users >= 1:
            return fake_users
        else:
            raise ValueError("fake_users must be either a float > 0 or an integer >= 1")

    def attack(self, im_real: csr_matrix) -> csr_matrix:
        """
        Perform the favorite item attack on C-fairness.
        -------
        :param im_real: The real user-item interaction matrix.
        :return: A CSR sparse matrix containing the interactions of real and fake users.
        """
        # Ensure num_targets is less than the catalog size
        num_items = im_real.shape[1]
        if self.num_targets >= num_items:
            raise ValueError(f"num_targets must be less than the total number of items in the catalog. Catalog size is {num_items}, but num_targets is {self.num_targets}.")

        im_group = im_real[self.uids_group, :] # extract the group's real user-item interactions

        im_fake = lil_matrix((self.num_fake_users, num_items))

        # Step 1: Identify the favorite items within the targeted group and assign them realistic ratings
        selected_iids, selected_ratings = find_favorite_items(im_group, num_selected=self.num_selected)
        im_fake[:, selected_iids] = selected_ratings

        # Step 2: Select and rate target items
        if self.mode == "least_favorite":
            target_iids = find_least_favorite_items(im_group, self.num_targets, selected_iids)[0]
            im_fake[:, target_iids] = self.max_rating
        else:
            # Random items vary for each fake user
            possible_targets = set(range(num_items)) - set(selected_iids) # exclude selected_iids from possible targets
            if len(possible_targets) < self.num_targets:
                raise ValueError(f"Not enough targets available. Possible targets: {len(possible_targets)}, Requested: {self.num_targets}.")
            for u in range(self.num_fake_users):
                target_iids = self.rng.choice(list(possible_targets), size=self.num_targets, replace=False)
                im_fake[u, target_iids] = self.max_rating

        im_all = vstack([im_real, im_fake.tocsr()]) # concatenate im_real and im_fake vertically

        return im_all
    