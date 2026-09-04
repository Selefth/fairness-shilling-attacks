
from typing import List, Union
import numpy as np
from scipy.sparse import csr_matrix, vstack
from numpy.random import default_rng

from src.helper_functions.attack_utils import *
    
class CFairInfluencerAttack:
    """
    Attack aimed at distorting the recommendation accuracy for a specific user group. 
    The attacker selects users within the target user group who have influence over the group, replicates their profiles, and assigns ratings to target items based on the attack mode.
    To identify influencers, we consider users within the group who have the highest number of interactions. 
    Optionally, a relaxed Top-k variant can be enabled to restrict profile copying to only the top-k most active users in the targeted group.  
    """
    def __init__(self, uids_group: List[int], fake_users: Union[float, int], num_targets: int, min_rating: int, max_rating: int, mode: str = "random", topk: int = 0, seed: int = 42): 
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
        :param min_rating: The minimum rating value.
        :param max_rating: The maximum rating value.
        :param mode: The attack mode. Can be one of the following:
                     - `"random"`: Push random items (assign max rating to random items).
                     - `"least_favorite"`: Push least favorite items (assign max rating to the group's least favorite items).
                     - `"favorite"`: Nuke favorite items (assign min rating to the group's favorite items).
        :param topk: If > 0, restrict profile copying to only the top-k most active users in the targeted group.
        :param seed: Seed for random number generation to ensure reproducibility.
        """
        if mode not in ["random", "least_favorite", "favorite"]:
            raise ValueError('mode must be "random", "least_favorite", or "favorite".')
        
        if not isinstance(topk, int) or topk < 0:
            raise ValueError("topk must be an integer >= 0")
        
        self.uids_group = uids_group
        self.num_fake_users = self._get_num_fake_users(fake_users)
        
        if topk > 0 and topk >= self.num_fake_users:
            import warnings
            warnings.warn(
                "topk >= num_fake_users: larger topk values do not change behavior "
                "(equivalent to copying the top-num_fake_users users).",
                UserWarning
            )
        
        self.num_targets = num_targets
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.mode = mode
        self.topk = topk
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
        Perform the influencer attack on C-fairness.
        -------
        :param im_real: The real user-item interaction matrix.
        :return: A CSR sparse matrix containing the interactions of real and fake users.
        """

        # Step 1: Identify the most active user(s) within the targeted group
        im_group = im_real[self.uids_group]; interaction_counts = im_group.getnnz(axis=1)

        most_active_indices = np.argsort(interaction_counts)[::-1]  # Sort users by activity in descending order
        ranked_users = np.array(self.uids_group)[most_active_indices]
        num_real_users = len(self.uids_group)

        # Optional relaxed variant: restrict profile copying to only the top-k most active users
        if self.topk > 0:
            k = min(self.topk, num_real_users)
            ranked_users = ranked_users[:k]

        # If the number of fake users exceeds the available pool, replicate users from the pool
        pool_size = len(ranked_users)
        if self.num_fake_users > pool_size:
            users_to_replicate = np.tile(ranked_users, (self.num_fake_users // pool_size) + 1)
            users_to_replicate = users_to_replicate[:self.num_fake_users]
        else:
            users_to_replicate = ranked_users[:self.num_fake_users]

        # Step 2: Create fake profiles by replicating the profiles of real users
        im_fake = im_real[users_to_replicate,:].tolil(copy=True)

        # Determine the rating score of target items based on attack mode
        rating_score = self.max_rating if self.mode in ["random", "least_favorite"] else self.min_rating

        # Step 3: Select and rate target items 
        for u in range(self.num_fake_users):
            unrated_items = np.where(im_fake[u].toarray() == 0)[1]
            if len(unrated_items) < self.num_targets:
                raise ValueError(f"User {u} does not have enough unrated items.")
            
            if self.mode == "least_favorite":
                rated_items = im_fake[u].nonzero()[1]
                target_iids = find_least_favorite_items(im_group, self.num_targets, rated_items)[0]
            elif self.mode == "favorite":
                rated_items = im_fake[u].nonzero()[1]
                target_iids = find_favorite_items(im_group, self.num_targets, rated_items)[0]
            else:
                target_iids = self.rng.choice(unrated_items, self.num_targets, replace=False)

            im_fake[u, target_iids] = rating_score
        
        im_all = vstack([im_real, im_fake.tocsr()])

        return im_all
    