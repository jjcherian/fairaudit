import numpy as np
import scipy.stats

from groups import Groups, score_intervals
from shifts import score_rkhs
from metrics import Metric

from tqdm import tqdm
from typing import Tuple

BOOTSTRAP_DEFAULTS = {
    "B": 500,
    "method": "multinomial",
    "student": None,
    "student_threshold": "adaptive",
    "seed": 0
}

def _compute_bound_statistic(
    Y : np.ndarray, 
    Z : np.ndarray, 
    L : np.ndarray, 
    threshold : float,
    metric : Metric,
    group_dummies : np.ndarray,
    w : np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    Y_b = np.repeat(Y, w, axis=0)
    Z_b = np.repeat(Z, w, axis=0)
    threshold_b = metric.compute_threshold(Z_b, Y_b)

    L = L - threshold
    L_b = w * (L - threshold_b)

    # form group statistics
    # (L, L_b, 1, w)
    mat = np.concatenate(
        (L, L_b, np.ones_like(w), w),
        axis=1
    )

    # returns (sum_{i \in G} L/n, \sum_{i \in G} w_i * L_i / n, |G|/n, \sum_{i \in G} w_i / n)
    # for each group
    # resulting matrix is (n_groups, 4)
    group_mat = (group_dummies.T @ mat) / len(w)

    stats = group_mat[:,1] * group_mat[:,2] - group_mat[:,0] * group_mat[:,3]

    return stats.flatten(), group_mat

def _compute_fixed_statistic(
    Y : np.ndarray,
    Z : np.ndarray,
    L : np.ndarray,
    threshold : float,
    metric : Metric,
    group_dummies : np.ndarray,
    w : np.ndarray,
    epsilon : float
) -> Tuple[np.ndarray, np.ndarray]:
    Y_b = np.repeat(Y, w, axis=0)
    Z_b = np.repeat(Z, w, axis=0)
    threshold_b = metric.compute_threshold(Z_b, Y_b)

    L = L - threshold - epsilon
    L_b = w * (L - threshold_b - epsilon)

    # form group statistics
    # (L, L_b, 1)
    mat = np.concatenate(
        (L, L_b, np.ones_like(w)),
        axis=1
    )

    # returns (sum_{i \in G} L/n, \sum_{i \in G} w_i * L_i / n, |G|/n)
    # for each group
    # resulting matrix is (n_groups, 3)
    group_mat = (group_dummies.T @ mat) / len(w)

    stats = group_mat[:,1] - group_mat[:,0]

    return stats.flatten(), group_mat

def estimate_bootstrap_distribution(
    Y : np.ndarray,
    Z : np.ndarray,
    L : np.ndarray,
    threshold : float,
    groups : Groups,
    metric : Metric,
    epsilon : float = None,
    bootstrap_params : dict = {}
) -> np.ndarray:
    n = Y.shape[0]
    n_groups = groups.dummies.shape[1]

    B = bootstrap_params.get("B", BOOTSTRAP_DEFAULTS["B"])
    method = bootstrap_params.get("method", BOOTSTRAP_DEFAULTS["method"])

    b_statistics = np.empty((B, n_groups))
    group_statistics = np.empty((B, n_groups, 4))
    
    rng = np.random.default_rng(
        seed=bootstrap_params.get("seed", BOOTSTRAP_DEFAULTS["seed"])
    )
    for b in tqdm(range(B)):
        if method == 'multinomial':
            w = rng.multinomial(n, [1/n] * n, size=1).reshape(-1,1)
        elif method == 'gaussian':
            w = rng.standard_normal(size=n)
        else:
            raise ValueError("Invalid multiplier bootstrap method.")
        if epsilon is None:
            b_statistics[b], group_statistics[b] = _compute_bound_statistic(
                Y,
                Z,
                L,
                threshold,
                metric,
                groups.dummies,
                w
            )
        else:
            b_statistics[b], group_statistics[b] = _compute_fixed_statistic(
                Y,
                Z,
                L,
                threshold,
                metric,
                groups.dummies,
                w, 
                epsilon
            )
    
    std_devs = np.ones((n_groups,))
    studentization = bootstrap_params.get("student", BOOTSTRAP_DEFAULTS["student"])
    if studentization:
        student_threshold = bootstrap_params.get("student_threshold", BOOTSTRAP_DEFAULTS["student_threshold"])
        std_devs = studentize(b_statistics, group_statistics, studentization, student_threshold)
        b_statistics /= std_devs

    return b_statistics, std_devs

def studentize(
    statistics : np.ndarray, 
    group_statistics : np.ndarray,
    student : str,
    student_threshold : float
) -> np.ndarray:
    emp_probs = np.mean(group_statistics[:,:,2], axis=0)
    if student == "mad":
        studentization = scipy.stats.median_absolute_deviation(statistics)
        studentization *= 1/scipy.stats.norm.ppf(3/4)
    elif student == "iqr":
        studentization = scipy.stats.iqr(statistics)
    elif student == "prob_bound":
        studentization = emp_probs**(3/2)
    else:
        raise ValueError(f"Unsupported studentization method: {student}.")
    return studentization.clip(student_threshold)

def estimate_critical_value(
    function_class : str,
    alpha : float,
    L : np.ndarray,
    bootstrap_params : dict = {},
    **kwargs
):    
    B = bootstrap_params.get("B", BOOTSTRAP_DEFAULTS["B"])
    method = bootstrap_params.get("method", BOOTSTRAP_DEFAULTS["method"])
    rng = np.random.default_rng(
        seed=bootstrap_params.get("seed", BOOTSTRAP_DEFAULTS["seed"])
    )

    n = len(L)
    scores = []
    for b in tqdm(range(B)):
        if method == 'multinomial':
            w = rng.multinomial(n, [1/n] * n, size=1).reshape(-1,1)
        elif method == 'gaussian':
            w = rng.standard_normal(size=n)
        else:
            raise ValueError("Invalid multiplier bootstrap method.")
        if function_class == "RKHS":
            score = score_rkhs(L, kwargs["K_sqrt"], w, seed=b)
        elif function_class == "interval":
            score = score_intervals(
                kwargs["X"], 
                L, 
                kwargs["threshold"], 
                kwargs["epsilon"], 
                w, 
                kwargs["type"]
            )
        else:
            raise ValueError(f"Invalid function class {function_class}.")
        scores.append(score)

    if function_class == "RKHS":
        qtile = np.quantile(scores, 1 - alpha/2)
    else:
        if kwargs["type"] == "lower":
            qtile = np.quantile(scores, alpha)
        elif kwargs["type"] == "upper":
            qtile = np.quantile(scores, 1 - alpha)
        else:
            qtile_l = np.quantile(np.asarray(scores)[:,0], alpha/2)
            qtile_u = np.quantile(np.asarray(scores)[:,1], 1 - alpha/2)
            return qtile_l, qtile_u
    return qtile

        

    