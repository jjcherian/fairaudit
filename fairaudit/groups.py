import copy
import numpy as np

from itertools import combinations
from typing import Tuple, Union

def get_intersections(
    X : np.ndarray, 
    discretization : dict = {},
    depth : int = None
) -> np.ndarray:
    """
    Construct groups formed by intersections of other attributes.
    
    Parameters
    ----------
    X : np.ndarray
    discretization : dict = {}
        Keys index columns of X
        Values specify input to the "bins" argument of np.digitize(...)
    depth : int = None
        If None, we consider all intersections, otherwise
        we all consider intersections of up to specified depth.
    Returns
    ---------
    groups : np.ndarray
        Boolean numpy array of size (n_points, n_groups)
    """
    for idx, disc in discretization.items():
        X[:,idx] = np.digitize(
            X[:,idx],
            disc
        )
    
    feature_list = np.arange(X.shape[1])
    if depth == None:
        depth = X.shape[1]
    # all_groups = []
    group_dummies = []
    for n_intersect in range(1, depth + 1):
        for c in combinations(feature_list, n_intersect):
            unique_groups, indices = np.unique(X[:,c], return_inverse=True, axis=0)

            # generate dummies
            dummies = np.zeros((X.shape[0], len(unique_groups)), dtype=int)
            dummies[(range(X.shape[0]), indices)] = int(1)
            group_dummies.append(dummies)
    
    group_dummies = np.concatenate(group_dummies, axis=1, dtype=int)

    return group_dummies.astype(bool)

def get_rectangles(
    X : np.ndarray,
    discretization : dict = {}
) -> np.ndarray:
    """
    Construct rectangles formed by attributes.

    Parameters
    ----------

    discretization : dict 
        Keys index columns of X
        Values specify input to the "bins" argument of np.digitize(...)

    Returns
    ---------
    groups : np.ndarray
        Boolean numpy array of size (n_points, n_groups)
    """
    X = copy.deepcopy(X)
    for idx, disc in discretization.items():
        X[:,idx] = np.digitize(
            X[:,idx],
            disc
        )
    
    n = X.shape[0]
    p = X.shape[1]

    coordinate_dummies = []
    for c in range(p):
        unique_vals, indices = np.unique(X[:,c], return_inverse=True, axis=0)

        num_unique = len(unique_vals)
        # generate unique dummies
        unique_dummies = np.zeros((n, num_unique), dtype=int)
        unique_dummies[(range(n), indices)] = int(1)

        num_intervals = (num_unique * (num_unique + 1)) // 2
        interval_dummies = np.zeros((n, num_intervals), dtype=int)

        idx = len(unique_vals)
        add_dummies = np.cumsum(unique_dummies, axis=1, dtype=int)
        interval_dummies[:,0:idx] = add_dummies

        for c_prime in range(1, len(unique_vals)):
            # update dummies by subtracting out contribution from first unique dummy
            # and removing first column
            num_added = num_unique - c_prime
            add_dummies = add_dummies[:,1:num_unique] - add_dummies[:,0,None]
            interval_dummies[:,idx:(idx + num_added)] = add_dummies
            idx += num_added

        interval_dummies = interval_dummies.clip(max=int(1))

        coordinate_dummies.append(interval_dummies)
    
    chars = [chr(idx + 97) for idx in range(p)]
    einsum_str = ','.join(f'i{c}' for c in chars)
    einsum_str += '->i' + ''.join(chars)
    group_dummies = np.einsum(einsum_str, *coordinate_dummies)
    group_dummies = group_dummies.reshape(n, -1, order='C') # flatten into (n, n_rectangles)

    # filter out duplicate groups
    group_dummies = np.unique(group_dummies, axis=1)
    if np.all(group_dummies[:,0] == 0):
        group_dummies = group_dummies[:,1:]

    return group_dummies.astype(bool)

def score_intervals(
    X : np.ndarray,
    L : np.ndarray,
    threshold : float,
    epsilon : float,
    w : np.ndarray,
    type : str
) -> Union[float, Tuple[float, float]]:
    ind = X.argsort()
    X = X[ind]
    L = L[ind]
    w = w[ind]

    # TODO: wrong...should be using bootstrap threshold
    if type == "lower":
        arr = -1 * (w - 1) * (L - threshold - epsilon)
        score = _max_subarray(arr)
        return -1 * score / len(w)
    elif type == "upper":
        arr = (w - 1) * (L - threshold - epsilon)
        score = _max_subarray(arr)
        return score / len(w)
    else:
        arr = -1 * (w - 1) * (L - threshold - epsilon)
        inf_score = -1 * _max_subarray(arr) / len(w)
        arr = (w - 1) * (L - threshold - epsilon)
        sup_score = _max_subarray(arr) / len(w)
        return inf_score, sup_score


# find maximum-sum subinterval of arr
def _max_subarray(numbers):
    best_sum = 0  # or: float('-inf')
    current_sum = 0
    for i, x in enumerate(numbers):
        current_sum = max(0, current_sum + x)
        best_sum = max(best_sum, current_sum)
    return best_sum