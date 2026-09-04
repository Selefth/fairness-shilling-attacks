
import inspect
import numpy as np
from scipy.sparse import csr_matrix

def select_target_iids(R: csr_matrix, k: int, rng) -> np.ndarray:
    """Pick k random target items from the whole item catalog."""
    return rng.choice(R.shape[1], size=int(k), replace=False)

def resolve_num_selected(num_selected, R_train: csr_matrix, num_targets: int, user_indices=None):
    """
    Resolve selected-item counts.
    Use "auto" to keep fake profile length close to the average profile length:
    selected items = round(avg profile length) - target items.
    If user_indices is provided, the average is computed over those users only.
    """
    if num_selected != "auto":
        return num_selected

    profile_matrix = R_train[user_indices, :] if user_indices is not None else R_train
    profile_lengths = profile_matrix.getnnz(axis=1)
    if len(profile_lengths) == 0:
        raise ValueError("Cannot resolve num_selected='auto' with no users.")

    return max(0, round(float(profile_lengths.mean())) - int(num_targets))

def find_favorite_items(im: csr_matrix, num_selected: int, exclude_items: np.ndarray = None) -> np.ndarray:
    """
    Find items that have total interactions above the average interactions an item has,
    and among them, select the num_selected highest-rated items.
    """
    return find_items_by_mode(im, num_selected, mode='favorite', exclude_items=exclude_items)

def find_least_favorite_items(im: csr_matrix, num_selected: int, exclude_items: np.ndarray = None) -> np.ndarray:
    """
    Find items that have total interactions above the average interactions an item has,
    and among them, select the num_selected lowest-rated items.
    """
    return find_items_by_mode(im, num_selected, mode='least_favorite', exclude_items=exclude_items)

def find_items_by_mode(im: csr_matrix, num_selected: int, mode: str, exclude_items: np.ndarray = None) -> np.ndarray:
    """
    Find items that have total interactions above the average interactions an item has,
    and among them, select either the num_selected highest-rated or lowest-rated items depending on the mode.
    
    :param im: User-item interaction matrix.
    :param num_selected: The number of items to select.
    :param mode: 'favorite' for highest-rated, 'least_favorite' for lowest-rated.
    :param exclude_items: An optional array of item indices to exclude from the selection (unique values).
    :return: A tuple containing an array of indices representing the selected items 
             and an array of their corresponding average ratings rounded.
    """
    # Compute total interactions for each item (popularity)
    item_pop = im.getnnz(axis=0)

    # Compute the average number of interactions across all items
    average_interactions = np.mean(item_pop)

    # Get indices of items with interactions above the average interactions
    popular_items = item_pop > average_interactions
    popular_indices = np.where(popular_items)[0]

    if exclude_items is not None:
        popular_indices = np.setdiff1d(popular_indices, exclude_items, assume_unique=True)

    if len(popular_indices) < num_selected:
        condition = "after applying exclusions." if exclude_items is not None else "with interactions above the average."
        raise ValueError(f"Requested {num_selected} items, but only {len(popular_indices)} items are available {condition}")

    # Compute average rating for each item
    average_ratings = np.array(im.sum(axis=0)).flatten() / np.maximum(im.getnnz(axis=0), 1)

    # Get average ratings for the popular items
    popular_item_ratings = average_ratings[popular_indices]

    # Depending on the mode, sort items either by descending or ascending average rating
    if mode == 'favorite':
        # Sort popular items by their average rating (descending order for highest-rated items)
        sorted_popular_indices = np.argsort(-popular_item_ratings)[:num_selected]
    elif mode == 'least_favorite':
        # Sort popular items by their average rating (ascending order for lowest-rated items)
        sorted_popular_indices = np.argsort(popular_item_ratings)[:num_selected]
    else:
        raise ValueError("Mode must be either 'favorite' or 'least_favorite'.")
    
    # Map back to original item indices
    selected_iids = popular_indices[sorted_popular_indices]

    return selected_iids, np.round(average_ratings[selected_iids])

# Check if a parameter is present in the attack class
def has_parameter(attack_cls, parameter_name):
    constructor = inspect.signature(attack_cls.__init__)
    return parameter_name in constructor.parameters

def perform_attack(
    attack_cls, data_train, uids_group=None, budget=None, num_targets=None, num_selected=None,
    num_fillers=None, min_rating=None, max_rating=None, seed=42, attack_config=None,
    target_iids=None, fake_users=None
):
    """
    Prepares attack parameters and executes the attack.
    Returns the augmented data matrix.
    """
    if fake_users is None:
        fake_users = budget

    attack_kwargs = {
        "seed": seed,
    }
    if has_parameter(attack_cls, 'uids_group'):
        attack_kwargs['uids_group'] = uids_group
    if has_parameter(attack_cls, 'target_iids'):
        attack_kwargs['target_iids'] = list(target_iids)
    if has_parameter(attack_cls, 'fake_users'):
        attack_kwargs['fake_users'] = fake_users
    if has_parameter(attack_cls, 'num_targets'):
        attack_kwargs['num_targets'] = num_targets
    if has_parameter(attack_cls, 'num_selected'):
        attack_kwargs['num_selected'] = num_selected
    if has_parameter(attack_cls, 'num_fillers'):
        attack_kwargs['num_fillers'] = num_fillers
    if has_parameter(attack_cls, 'min_rating'):
        attack_kwargs['min_rating'] = min_rating
    if has_parameter(attack_cls, 'max_rating'):
        attack_kwargs['max_rating'] = max_rating
    if attack_config:
        attack_kwargs.update(attack_config)

    attack_instance = attack_cls(**attack_kwargs)
    return attack_instance.attack(data_train)
