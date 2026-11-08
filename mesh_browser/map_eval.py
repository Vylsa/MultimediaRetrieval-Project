import numpy as np

def mean_average_precision(D, labels):
    """Compute Mean Average Precision (mAP) given a distance matrix D 
    and integer class labels.

    For each query item i, items are sorted by distance (excluding self).
    Precision is calculated at each rank where a same-class item appears.
    """
    n = len(labels)
    D = D.copy()
    np.fill_diagonal(D, np.inf)  # exclude self-match
    
    # Sort distances and get neighbor labels (drop the self-match at the end)
    sorted_indices = np.argsort(D, axis=1)[:, :-1]
    sorted_labels = labels[sorted_indices]  # shape (n, n-1)
    
    # Match matrix: True if neighbor has the same class as the query
    query_labels = labels[:, np.newaxis]
    matches = (sorted_labels == query_labels)  # shape (n, n-1)
    
    # Total relevant items per query (excluding self)
    total_relevant = matches.sum(axis=1, keepdims=True)
    total_relevant = np.maximum(total_relevant, 1)  # avoid division by zero
    
    # Cumulative matches and ranks (1 to n-1)
    cumulative_matches = np.cumsum(matches, axis=1)
    ranks = np.arange(1, n, dtype=float)[np.newaxis, :]
    
    # Precision at each rank
    precision_at_k = cumulative_matches / ranks
    
    # Average precision per query
    ap_per_query = (precision_at_k * matches).sum(axis=1) / total_relevant.squeeze()
    
    return float(ap_per_query.mean())