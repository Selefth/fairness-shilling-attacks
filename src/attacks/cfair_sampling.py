from typing import List, Union
import numpy as np
from scipy.sparse import csr_matrix, vstack
from numpy.random import default_rng

from src.helper_functions.attack_utils import *


class CFairSamplingAttack:
    """
    Relaxed variant of the C-Fair Influencer Attack ("Random"):
    instead of selecting the most active (influential) users within the
    targeted (unprivileged) group, we sample users uniformly at random
    from the group and replicate their profiles.

    After copying each sampled user's profile, we inject ratings on a
    target set according to the chosen attack mode:
      - "random": push random unrated items (assign max rating)
      - "least_favorite": push group's least favorite items (assign max rating)
      - "favorite": nuke group's favorite items (assign min rating)
    """

    def __init__(
        self,
        uids_group: List[int],
        fake_users: Union[float, int],
        num_targets: int,
        min_rating: int,
        max_rating: int,
        mode: str = "random",
        seed: int = 42,
    ):
        """
        :param uids_group: Inner IDs of the targeted user group (indices in im_real).
        :param fake_users: # fake users injected, either:
                           - int >= 1 (exact count), or
                           - float > 0 (fraction of |uids_group|, e.g., 0.1 = 10%)
        :param num_targets: Number of target items per fake profile.
        :param min_rating: Minimum rating value (used for "favorite" / NUKEF).
        :param max_rating: Maximum rating value (used for "random" / "least_favorite" / push).
        :param mode: One of {"random", "least_favorite", "favorite"}.
        :param seed: RNG seed.
        """
        if mode not in ["random", "least_favorite", "favorite"]:
            raise ValueError('mode must be "random", "least_favorite", or "favorite".')

        if not isinstance(uids_group, list) or len(uids_group) == 0:
            raise ValueError("uids_group must be a non-empty list of user IDs.")

        self.uids_group = uids_group
        self.num_fake_users = self._get_num_fake_users(fake_users)
        self.num_targets = num_targets
        self.min_rating = min_rating
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
        Perform the sampling-based attack on C-fairness.

        :param im_real: Real user-item interaction matrix (CSR), shape (num_users, num_items).
        :return: Augmented CSR matrix containing real + fake users (vstack).
        """
        # Group submatrix (used by favorite/least_favorite helpers)
        im_group = im_real[self.uids_group]

        # Step 1: Sample users uniformly at random from the targeted group
        # If budget exceeds group size, allow repeats (sampling with replacement).
        replace = self.num_fake_users > len(self.uids_group)
        users_to_replicate = self.rng.choice(
            np.array(self.uids_group, dtype=int),
            size=self.num_fake_users,
            replace=replace,
        )

        # Step 2: Copy sampled profiles
        im_fake = im_real[users_to_replicate, :].tolil(copy=True)

        # Step 3: Inject target ratings according to attack mode
        rating_score = self.max_rating if self.mode in ["random", "least_favorite"] else self.min_rating

        for u in range(self.num_fake_users):
            # sanity: ensure enough unrated items exist for *any* mode
            unrated_items = np.where(im_fake[u].toarray() == 0)[1]
            if len(unrated_items) < self.num_targets:
                raise ValueError(f"User {u} does not have enough unrated items.")

            if self.mode == "least_favorite":
                rated_items = im_fake[u].nonzero()[1]
                # helper should return items not in rated_items (same pattern as influencer.py)
                target_iids = find_least_favorite_items(im_group, self.num_targets, rated_items)[0]
            elif self.mode == "favorite":
                rated_items = im_fake[u].nonzero()[1]
                target_iids = find_favorite_items(im_group, self.num_targets, rated_items)[0]
            else:  # "random"
                target_iids = self.rng.choice(unrated_items, self.num_targets, replace=False)

            im_fake[u, target_iids] = rating_score

        # Combine real + fake
        im_all = vstack([im_real, im_fake.tocsr()])
        return im_all