import numpy as np

from scipy.stats import norm
from sklearn.metrics.pairwise import pairwise_kernels
from statsmodels.stats import multitest
from typing import List, Tuple, Union

from bootstrap import estimate_bootstrap_distribution, estimate_critical_value
from groups import Groups
from metrics import Metric


class Auditor:
    def __init__(
        self, 
        X : np.ndarray,
        Y : np.ndarray,
        Z : np.ndarray,
        metric : Metric
    ):
        """
        Constructor for auditing performance discrepancies.

        Parameters
        ----------
        X : covariates (n, k)
        Y : outcomes (n, 1)
        Z : predictions (n, 1)
        metric : Metric
        """
        assert X.shape[0] == Y.shape[0] == Z.shape[0]

        self.X = X
        self.Y = Y
        self.Z = Z
        self.metric = metric

        self.L = metric.evaluate(Z, Y)

        # filled in by calibrate_groups(...)
        self.epsilon = None
        self.type = None

        # filled in by calibrate_rkhs(...)
        self.K = None

        # filled in by either calibrate function
        self.critical_values = None
        self.groups_list = None

    def calibrate_groups(
        self,
        alpha : float,
        type : str,
        groups : Union[Groups, str],
        epsilon : float = None,
        bootstrap_params : dict = None
    ) -> None:
        """
        Obtain bootstrap critical values for a specific group collection

        Parameters
        ----------
        alpha : float
        type : str
        groups : Groups
        epsilon : float = None
        bootstrap_params : dict = None
        """
        if isinstance(groups, str):
            groups_name = groups
            groups = Groups([], np.ones((len(self.L), 1)))
        else:
            groups_name = "exhaustive"

        if self.metric.requires_conditioning():
            groups_list = self.metric.get_conditional_groups(
                groups,
                self.Z,
                self.Y
            )
            self.groups_list = groups_list
        else:
            self.groups_list = [groups]

        self.type = type
        self.epsilon = epsilon

        for groups in groups_list:
            all_dummies = np.amax(groups.dummies, axis=1)
            threshold = self.metric.compute_threshold(
                self.Z[all_dummies], 
                self.Y[all_dummies]
            )
            if groups_name != "exhaustive":
                kwargs = {
                    "X": self.X,
                    "threshold": threshold,
                    "epsilon": epsilon,
                    "type": type
                }
                c_value = estimate_critical_value(
                    groups_name,
                    alpha,
                    self.L,
                    bootstrap_params,
                    **kwargs
                )
            else:
                b_statistics, _ = estimate_bootstrap_distribution(
                    self.Y,
                    self.Z,
                    self.L,
                    threshold,
                    groups,
                    self.metric,
                    epsilon,
                    bootstrap_params
                )
                c_value = _compute_critical_value(
                    b_statistics,
                    alpha,
                    type
                )
            self.critical_values.append(c_value)

    def query_group(
        self,
        group : int
    ) -> Tuple[float, bool]:
        if self.critical_values is None:
            raise ValueError("Run calibrate before querying for groups.")

        results = []
        for groups, critical_value in zip(self.groups_list, self.critical_values):
            dummies = groups.dummies[:,group]
            metric_value = np.sum(self.L * dummies) / dummies.sum()
            results.append(self._certify_group(metric_value, dummies, groups, critical_value))

        return results

    def calibrate_rkhs(
        self,
        alpha : float,
        kernel : str,
        kernel_params : dict = {},
        bootstrap_params : dict = {}
    ) -> None:
        vacuous_group = Groups([], np.ones((self.X.shape[0], 1)))
        if self.metric.requires_conditioning():
            self.groups_list = self.metric.get_conditional_groups(
                vacuous_group,
                self.Z,
                self.Y
            )
        else:
            self.groups_list = [vacuous_group]
        
        self.K = []
        self.critical_values = []
        for group in self.groups_list:
            K = pairwise_kernels(
                X=self.X[group.dummies], 
                metric=kernel, 
                **kernel_params
            ) + 1e-6 * np.eye(len(self.X))
            self.K.append(K)

            K_sqrt = np.linalg.cholesky(K)

            self.critical_values.append(
                estimate_critical_value("RKHS", alpha, bootstrap_params, dict(K_sqrt=K_sqrt))
            )

    def query_rkhs(
        self,
        weights : np.ndarray,
    ) -> Tuple[List[float], List[float]]:

        lbs = []
        ubs = []
        for K, crit_value, groups in zip(self.K, self.critical_values, self.groups_list):
            h_plus = (K @ weights).clip(0)
            Lh_plus = np.mean(self.L[groups.dummies] * h_plus)

            # quadratic in epsilon
            a = np.mean(h_plus)**2
            b = -2 * np.mean(h_plus) * Lh_plus
            c = Lh_plus**2 - crit_value / (h_plus**2)

            lb_gp = (-1 * b - np.sqrt(b**2 - 4 * a * c)) / (2*a)
            ub_gp = (-1 * b + np.sqrt(b**2 - 4 * a * c)) / (2*a)
            lbs.append(lb_gp)
            ubs.append(ub_gp)

        return lbs, ubs


    def _certify_group(
        self,
        metric_value : float,
        group_dummies : np.ndarray,
        groups : Groups,
        critical_value : Union[float, Tuple[float, float]]
    ) -> Union[bool, float]:
        all_dummies = np.amax(groups.dummies, axis=1)
        threshold = self.metric.compute_threshold(self.Z[all_dummies], self.Y[all_dummies])
        group_prob = np.sum(group_dummies) / len(group_dummies)

        if self.epsilon:
            if self.type == "lower":
                return metric_value < threshold + self.epsilon + critical_value / group_prob
            elif self.type == "upper":
                return metric_value > threshold + self.epsilon + critical_value / group_prob
            else:
                bool_l = metric_value < threshold + self.epsilon + critical_value[0] / group_prob
                bool_u = metric_value > threshold + self.epsilon + critical_value[1] / group_prob
                return bool_l & bool_u
        else:
            if self.type == "lower":
                return metric_value - threshold - critical_value / group_prob**2
            elif self.type == "upper":
                return metric_value - threshold - critical_value / group_prob**2
            else:
                eps_l = metric_value - threshold - critical_value[0] / group_prob**2
                eps_u = metric_value - threshold - critical_value[1] / group_prob**2
                return max(eps_l, eps_u)

    def flag_groups(
        self,
        groups : np.ndarray,
        type : str,
        alpha : float,
        epsilon : float = 0,
        bootstrap_params : dict = None  
    ) -> np.ndarray:
        if self.metric.requires_conditioning():
            groups_list = self.metric.get_conditional_groups(
                groups,
                self.Z,
                self.Y
            )
        else:
            groups_list = [groups]

        n_groups = groups.dummies.shape[1]
        all_p_values = np.ones((len(groups_list), n_groups))
        for i, grps in enumerate(groups_list):
            all_dummies = np.amax(grps.dummies, axis=1)
            threshold = self.metric.compute_threshold(
                self.Z[all_dummies], 
                self.Y[all_dummies]
            )
            _, s_grps = estimate_bootstrap_distribution(
                self.Y,
                self.Z,
                self.L,
                threshold,
                grps,
                self.metric,
                epsilon,
                bootstrap_params
            )

            metric_values = np.sum((self.L - threshold - epsilon) * grps.dummies) / grps.dummies.sum(axis=0)

            if type == "lower":
                all_p_values[i,:] = 1 - norm.cdf(metric_values / s_grps)
            elif type == "upper":
                all_p_values[i,:] = norm.cdf(metric_values / s_grps)
            else:
                all_p_values[i,:] = 1 - 2 * norm.cdf(np.abs(metric_values) / s_grps)

        bh_rejections = multitest.multipletests(all_p_values.flatten(), alpha, method='fdr_bh')
        flags = np.amax(bh_rejections[0].reshape((-1, n_groups)), axis=0)

        return flags


def _compute_critical_value(
    b_statistics : np.ndarray,
    alpha : float, 
    type : str
) -> float:
    if type == "lower":
        critical_values = np.quantile(b_statistics.min(axis=1), alpha)
    elif type == "upper":
        critical_values = np.quantile(b_statistics.max(axis=1), 1 - alpha)
    elif type == "interval":
        critical_values = (
            np.quantile(b_statistics.min(axis=1), alpha/2),
            np.quantile(b_statistics.max(axis=1), 1 - alpha/2)
        )
    else:
        raise ValueError(f"No critical values for hypothesis type: {type}.")
    return critical_values


