import numpy as np
import scipy.stats

from fairaudit.groups import score_intervals
from fairaudit.shifts import score_rkhs, score_rkhs_nonneg
from fairaudit.metrics import Metric

from tqdm import tqdm
from typing import Tuple

BOOTSTRAP_DEFAULTS = {
    "B": 500,
    "method": "multinomial",
    "student": None,
    "student_threshold": None,
    "prob_threshold": 1e-10,
    "w_0": 1,
    "seed": 1
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
    all_dummies = np.amax(group_dummies, axis=1)

    Y_b = np.repeat(Y[all_dummies], w.flatten()[all_dummies], axis=0)
    Z_b = np.repeat(Z[all_dummies], w.flatten()[all_dummies], axis=0)
    
    threshold_b = metric.compute_threshold(Z_b, Y_b)

    L_n = (L - threshold).reshape(-1,1)
    L_b = (w.flatten() * (L - threshold_b)).reshape(-1,1)

    n = len(w)

    # form group statistics
    # (L, L_b, 1, w)
    mat = np.concatenate(
        (L_n / n, L_b / n, np.ones_like(w) / n, w / n),
        axis=1
    )

    # returns (sum_{i \in G} L/n, \sum_{i \in G} w_i * L_i / n, |G|/n, \sum_{i \in G} w_i / n)
    # for each group
    # resulting matrix is (n_groups, 4)
    group_mat = (group_dummies.T @ mat)

    stats = group_mat[:,1] * group_mat[:,2] - group_mat[:,0] * group_mat[:,3]

    return stats.flatten(), group_mat[:,2]

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
    all_dummies = np.amax(group_dummies, axis=1)

    Y_b = np.repeat(Y[all_dummies], w.flatten()[all_dummies], axis=0)
    Z_b = np.repeat(Z[all_dummies], w.flatten()[all_dummies], axis=0)
    
    threshold_b = metric.compute_threshold(Z_b, Y_b)

    L_n = (L - threshold - epsilon).reshape(-1,1)
    L_b = (w.flatten() * (L - threshold_b - epsilon).flatten()).reshape(-1,1)

    # form group statistics
    # (L, L_b, 1)
    mat = np.concatenate(
        (L_n, L_b, np.ones_like(w)),
        axis=1
    )

    # returns (sum_{i \in G} L/n, \sum_{i \in G} w_i * L_i / n, |G|/n)
    # for each group
    # resulting matrix is (n_groups, 3)
    group_mat = (group_dummies.T @ mat) / len(w)

    stats = group_mat[:,1] - group_mat[:,0]

    return stats, group_mat[:,2]

def estimate_bootstrap_distribution(
    X : np.ndarray,
    Y : np.ndarray,
    Z : np.ndarray,
    L : np.ndarray,
    threshold : float,
    group_dummies : np.ndarray,
    metric : Metric,
    epsilon : float = None,
    bootstrap_params : dict = {}
) -> np.ndarray:
    n = Y.shape[0]
    n_groups = group_dummies.shape[1]

    B = bootstrap_params.get("B", BOOTSTRAP_DEFAULTS["B"])
    method = bootstrap_params.get("method", BOOTSTRAP_DEFAULTS["method"])

    b_statistics = np.zeros((B, n_groups))
    group_probs = np.zeros((n_groups,))
    
    rng = np.random.default_rng(
        seed=bootstrap_params.get("seed", BOOTSTRAP_DEFAULTS["seed"])
    )
    for b in range(B): #tqdm(range(B)):
        if method == 'multinomial':
            w = rng.multinomial(n, [1/n] * n, size=1).reshape(-1,1)
        elif method == 'gaussian':
            w = rng.standard_normal(size=n).reshape(-1,1)
        else:
            raise ValueError("Invalid multiplier bootstrap method.")
        if epsilon is None:
            b_statistics[b], group_probs = _compute_bound_statistic(
                Y,
                Z,
                L,
                threshold,
                metric,
                group_dummies,
                w
            )
        else:
            b_statistics[b], group_probs = _compute_fixed_statistic(
                Y,
                Z,
                L,
                threshold,
                metric,
                group_dummies,
                w, 
                epsilon
            )
    
    std_devs = np.ones((n_groups,))
    studentization = bootstrap_params.get("student", BOOTSTRAP_DEFAULTS["student"])
    if studentization:
        student_threshold = bootstrap_params.get("student_threshold", BOOTSTRAP_DEFAULTS["student_threshold"])
        prob_threshold = bootstrap_params.get("prob_threshold", BOOTSTRAP_DEFAULTS["prob_threshold"])
        std_devs = studentize(b_statistics, group_probs, studentization, student_threshold, prob_threshold)
        # b_statistics /= std_devs

    return b_statistics, std_devs

def studentize(
    statistics : np.ndarray, 
    group_probs : np.ndarray,
    student : str,
    student_threshold : float,
    prob_threshold : float = 1e-8
) -> np.ndarray:
    if student == "mad":
        studentization = scipy.stats.median_abs_deviation(statistics)
        studentization *= 1/scipy.stats.norm.ppf(3/4)
        studentization[group_probs <= prob_threshold] = np.inf
    elif student == "iqr":
        studentization = scipy.stats.iqr(statistics)
        studentization[group_probs <= prob_threshold] = np.inf
    elif student == "prob_bound":
        studentization = group_probs**(3/2)
    elif student == "prob_bool":
        studentization = group_probs**(1/2)
    else:
        raise ValueError(f"Unsupported studentization method: {student}.")
    return studentization.clip(student_threshold)

def get_rescaling(
    X : np.ndarray,
    Y : np.ndarray,
    Z : np.ndarray,
    metric : Metric,
    g_dummies : np.ndarray,
    bootstrap_params : dict = {}
):
    method = bootstrap_params.get("student", BOOTSTRAP_DEFAULTS["student"])
    prob_threshold = bootstrap_params.get("prob_threshold", BOOTSTRAP_DEFAULTS["prob_threshold"])

    if method is None:
        rescaling = np.ones((g_dummies.shape[1],))
    else:
        exp = 3/2 if method == "prob_bound" else 1/2

        group_probs = np.mean(g_dummies, axis=0)
        rescaling = group_probs.clip(prob_threshold)
        rescaling = rescaling**(exp)

        L_var = metric.compute_metric_variance(Z, Y)
        L_cond_var = metric.compute_metric_variance(Z, Y, g_dummies)
        threshold_var = metric.compute_threshold_variance(Z, Y, X)
        L_threshold_cond_cov = metric.compute_metric_threshold_covariance(Z, Y, X, g_dummies)

        sigma_G = np.sqrt(L_cond_var) + group_probs * (threshold_var - 2 * L_threshold_cond_cov)

        w_0 = bootstrap_params.get("w_0", BOOTSTRAP_DEFAULTS["w_0"])
        weight = group_probs / (group_probs + w_0)
        rescaling *= weight * sigma_G + (1 - weight) * np.sqrt(L_var)
    return rescaling

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
    for _ in range(B): #tqdm(range(B)): # removed tqdm for now
        if method == 'multinomial':
            w = rng.multinomial(n, [1/n] * n, size=1).reshape(-1)
        elif method == 'gaussian':
            w = rng.standard_normal(size=n)
        else:
            raise ValueError("Invalid multiplier bootstrap method.")
        if function_class == "RKHS":
            student_threshold = bootstrap_params.get("student_threshold", BOOTSTRAP_DEFAULTS["student_threshold"])
            score = score_rkhs(
                L, 
                kwargs["K_sqrt"], 
                w,
                kwargs["type"],
                student_threshold,
                kwargs["K_basis"]
            )
        elif function_class == "RKHS_nonneg":
            score = score_rkhs_nonneg(
                L,
                kwargs["K_sqrt"],
                w,
                kwargs["type"]
            )
        elif function_class == "intervals":
            score = score_intervals(
                kwargs["X"].flatten(), 
                L.flatten(), 
                kwargs["threshold"], 
                kwargs["epsilon"], 
                w, 
                kwargs["type"]
            )
        else:
            raise ValueError(f"Invalid function class {function_class}.")
        scores.append(score)

    if kwargs["type"] == "lower":
        qtile = np.quantile(scores, alpha)
    elif kwargs["type"] == "upper":
        qtile = np.quantile(scores, 1 - alpha)
    else:
        qtile_l = np.quantile(np.asarray(scores)[:,0], alpha/2)
        qtile_u = np.quantile(np.asarray(scores)[:,1], 1 - alpha/2)
        return qtile_l, qtile_u
    return qtile

        

    
