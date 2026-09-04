import numpy as np
import bottleneck as bn

# NOTE:
# Ranking metrics work for both explicit and implicit feedback.
# If the data is binary (i.e., 0/1 interactions), set `threshold=1` to treat '1' as a relevant item.

def precision_at_n(R_hat,R_held,threshold,n):
    """
    Computes the Precision at the given value of n, considering items rated above a specified threshold as relevant.
    This metric does not take the rank of recommended items into account.
    For users with no relevant items (due to either data splitting and/or the relevance threshold), precision is not computed.
    -------
    :param R_hat: The estimated user feedback matrix (numpy.ndarray) where -np.inf marks entries used as model input.
    :param R_held: The held out user feedback matrix (scipy.sparse). This matrix contains the actual user-item interactions that are withheld during model training.
    :param threshold: Relevance threshold to determine whether an item is considered relevant.
    :param n: Number of top items to consider in the evaluation (e.g., top-10).

    :return: A 1D numpy array of per-user Precision@N scores.
    -------
    The function marks items in R_hat with predicted ratings in the top-N as 'recommended'. 
    It then identifies items in R_held with actual ratings above threshold as 'relevant'. 
    True positives are items that are both 'recommended' and 'relevant'.
    """

    users = R_hat.shape[0]
    idx = bn.argpartition(-R_hat, n, axis=1)  # get indices of top-n items

    R_hat_binary = np.zeros_like(R_hat, dtype=bool)
    R_hat_binary[np.arange(users)[:, np.newaxis], idx[:, :n]] = True  # mark recommended items

    R_held_binary = (R_held >= threshold).toarray()  # binarize R_held

    true_positives = np.logical_and(R_held_binary, R_hat_binary).sum(axis=1)  # count of true positives
    relevant_items = R_held_binary.sum(axis=1) # the number of relevant items in R_held for each user

    valid_indices = relevant_items > 0  # users with relevant items
    precision = np.full(users, np.nan, dtype=float)
    precision[valid_indices] = true_positives[valid_indices] / n
    
    return precision

def recall_at_n(R_hat,R_held,threshold,n):
    """
    Computes the Recall at the given value of n, considering items rated above a specified threshold as relevant.
    This metric does not take the rank of recommended items into account.
    For users with no relevant items (due to either data splitting and/or the relevance threshold), recall is not computed.
    -------
    :param R_hat: The estimated user feedback matrix (numpy.ndarray) where -np.inf marks entries used as model input.
    :param R_held: The held out user feedback matrix (scipy.sparse). This matrix contains the actual user-item interactions that are withheld during model training.
    :param threshold: Relevance threshold to determine whether an item is considered relevant.
    :param n: Number of top items to consider in the evaluation (e.g., top-10).

    :return: A 1D numpy array of per-user Recall@N scores.
    -------
    The function marks items in R_hat with predicted ratings in the top-N as 'recommended'. 
    It then identifies items in R_held with actual ratings above threshold as 'relevant'. 
    True positives are items that are both 'recommended' and 'relevant'.
    """

    users = R_hat.shape[0]

    # find the indices that partition the array so that the first n elements are the largest n elements
    idx = bn.argpartition(-R_hat, n, axis=1)

    R_hat_binary = np.zeros_like(R_hat, dtype=bool)
    R_hat_binary[np.arange(users)[:, np.newaxis], idx[:, :n]] = True

    # binarize R_held
    R_held_binary = (R_held >= threshold).toarray()

    true_positives = np.logical_and(R_held_binary, R_hat_binary).sum(axis=1) # the number of relevant items retrieved in the top-n for each user
    relevant_items = R_held_binary.sum(axis=1) # the number of relevant items in R_held for each user

    # recall@N for each user
    valid_indices = relevant_items > 0
    recall = np.full(users, np.nan, dtype=float)
    recall[valid_indices] = true_positives[valid_indices] / np.minimum(n, relevant_items[valid_indices])
    
    return recall

def get_topn_indices(R_hat,n):
    """
    Helper function to get sorted indices of top-n items in each row of R_hat.
    """
    users = R_hat.shape[0]
    
    # find the indices that partition the array so that the first n elements are the largest n elements
    idx_topn_part = bn.argpartition(-R_hat, n, axis=1)

    # keep only the largest n elements of R_hat
    topn_part = R_hat[np.arange(users)[:, np.newaxis], idx_topn_part[:, :n]]

    # find the indices of the sorted top-n predicted relevance scores in R_hat
    idx_part = np.argsort(-topn_part, axis=1)
    idx_topn = idx_topn_part[np.arange(users)[:, np.newaxis], idx_part]
    
    return idx_topn

def tndcg_at_n(R_hat,R_held,threshold,n):
    """
    Computes the truncated Normalized Discounted Cumulative Gain (NDCG) at the given value of n, considering items
    rated above a specified threshold as relevant.
    For users with no relevant items (due to either data splitting and/or the relevance threshold), tndcg is not computed.
    -------
    :param R_hat: The estimated user feedback matrix (numpy.ndarray) where -np.inf marks entries used as model input.
    :param R_held: The held out user feedback matrix (scipy.sparse). This matrix contains the actual user-item interactions that are withheld during model training.
    :param threshold: Relevance threshold to determine whether an item is considered relevant.
    :param n: Number of top items to consider in the evaluation (e.g., top-10).

    :return: A 1D numpy array of per-user truncated NDCG@N scores.
    """

    users = R_hat.shape[0]

    idx_topn = get_topn_indices(R_hat, n)

    # binarize R_held
    R_held_binary = R_held.copy()
    R_held_binary.data = np.where(R_held_binary.data >= threshold, 1, 0)
    R_held_binary.eliminate_zeros()

    tp = 1. / np.log2(np.arange(2, n + 2))
    dcg = (R_held_binary[np.arange(users)[:, np.newaxis], idx_topn].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(i, n)]).sum() for i in R_held_binary.getnnz(axis=1)])
    # to handle cases where IDCG is zero (there are no relevant items as per the threshold)
    valid_indices = idcg > 0
    tndcg = np.full(users, np.nan, dtype=float)
    tndcg[valid_indices] = dcg[valid_indices] / idcg[valid_indices]

    return tndcg

def compute_metrics(predictions, ground_truth, threshold, n, metrics, groups=None):
    """
    Compute top-n metrics over all users and optionally for specified user groups as well.

    -------
    :param predictions: A NumPy array of predicted scores (shape: [num_users, num_items]).
    :param ground_truth: A sparse matrix of true interactions (same shape as predictions).
    :param threshold: Relevance threshold to determine whether an item is considered relevant.
    :param n: Number of top items to consider in the evaluation (e.g., top-10).
    :param metrics: A dictionary mapping metric names (e.g., "precision") to functions 
                         that return arrays of per-user metric scores.
    :param groups: Optional dictionary where keys are group names (e.g., "f", "m") and values are lists 
                   of user indices belonging to each group.

    :return:
    A dictionary with metric names as keys and their mean values (float) as values.
    If groups is provided, also includes keys of the form "metric_group_name" with mean values per group.
    """
    results = {}
    metric_scores = {m: func(predictions, ground_truth, threshold, n)
                for m, func in metrics.items()}

    # overall means
    for m, vals in metric_scores.items():
        results[m] = np.nanmean(vals)

    # group means
    if groups:
        for g_name, ids in groups.items():
            for m, vals in metric_scores.items():
                results[f"{m}_{g_name}"] = np.nanmean(vals[ids])

    return results

def mse(predictions,ground_truth):
    """
    Computes the Mean Squared Error (MSE) between observed true and predicted scores.
    -------
    :param predictions: A NumPy array of predicted scores (shape: [num_users, num_items]).
    :param ground_truth: A sparse matrix of true interactions (same shape as predictions).

    :return: The masked mse score (float).
    """
    true_scores_dense = ground_truth.toarray()
    mask = true_scores_dense != 0 # non-zero positions correspond to observed interactions
    diff = (true_scores_dense - predictions)[mask]
    squared_errors = diff ** 2

    return np.mean(squared_errors)