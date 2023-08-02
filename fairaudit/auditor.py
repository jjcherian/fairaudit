import numpy as np

from scipy.stats import norm
from sklearn.metrics.pairwise import pairwise_kernels
from statsmodels.stats import multitest
from typing import List, Tuple, Union

from fairaudit.bootstrap import estimate_bootstrap_distribution, estimate_critical_value, get_rescaling
from fairaudit.metrics import Metric


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
        Z : predictions (n, p)
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

        # filled in by calibrate_rkhs(...)
        self.K = None
        self.student_threshold = None

        # filled in by either calibrate function
        self.critical_values = None
        self.groups_list = None
        self.students = []
        self.type = None

    def calibrate_groups(
        self,
        alpha : float,
        type : str,
        groups : Union[np.ndarray, str],
        epsilon : float = None,
        bootstrap_params : dict = {}
    ) -> None:
        """
        Obtain bootstrap critical values for a specific group collection.

        Parameters
        ----------
        alpha : float
            Type I error threshold
        type : str
            Takes one of three values ('lower', 'upper', 'interval').
            See epsilon documentation for what these correpsond to.
        groups : Union[np.ndarray, str]
            Either a string for a supported collection of groups or a numpy array
            likely obtained by calling `get_intersections` or `get_rectangles` from group.py
            Array dimensions should be (n_points, n_groups)
        epsilon : float = None
            epsilon = None calibrates for issuing confidence bounds. 
                type = "upper" issues lower confidence bounds, 
                type = "lower" issues upper confidence bounds, and 
                "interval" issues confidence intervals.
            If a non-null value is passed in, we issue a Boolean certificate. 
                type = "upper" tests the null that epsilon(G) >= epsilon
                type = "lower" tests the null that epsilon(G) <= epsilon
                type = "interval" tests the null that |epsilon(G)| <= epsilon
        bootstrap_params : dict = {}
            Allows the user to specify a random seed, number of bootstrap resamples,
            and studentization parameters for the bootstrapping process.
        """
        if isinstance(groups, str):
            if epsilon == None:
                raise ValueError(f"Only fixed-epsilon certification supported with {groups}.")
            groups_name = groups
            group_dummies = np.ones((len(self.L), 1), dtype=bool)
        else:
            groups_name = "exhaustive"
            group_dummies = groups

        if self.metric.requires_conditioning():
            groups_list = self.metric.get_conditional_groups(
                group_dummies,
                self.Z,
                self.Y
            )
            self.groups_list = groups_list
        else:
            self.groups_list = [group_dummies]

        self.type = type
        self.epsilon = epsilon
        self.critical_values = []
        self.students = []

        for g_dummies in self.groups_list:
            all_dummies = np.amax(g_dummies, axis=1)
            threshold = self.metric.compute_threshold(
                self.Z[all_dummies], 
                self.Y[all_dummies]
            )
            if groups_name != "exhaustive":
                kwargs = {
                    "X": self.X[all_dummies],
                    "threshold": threshold,
                    "epsilon": epsilon,
                    "type": type
                }
                c_value = estimate_critical_value(
                    groups_name,
                    alpha,
                    self.L[all_dummies],
                    bootstrap_params,
                    **kwargs
                )
                students = np.asarray([1])
            else:
                s_hat = get_rescaling(self.X, self.Y, self.Z, self.metric, g_dummies, bootstrap_params)
                if type == "interval" and epsilon is not None:
                    b_statistics_l, students = estimate_bootstrap_distribution(
                        self.X,
                        self.Y,
                        self.Z,
                        self.L,
                        threshold,
                        g_dummies,
                        self.metric,
                        epsilon,
                        bootstrap_params
                    )

                    b_statistics_l /= s_hat

                    b_statistics_u, students = estimate_bootstrap_distribution(
                        self.X,
                        self.Y,
                        self.Z,
                        self.L,
                        threshold,
                        g_dummies,
                        self.metric,
                        -1 * epsilon,
                        bootstrap_params
                    )

                    b_statistics_u /= s_hat

                    c_value_l = _compute_critical_value(
                        b_statistics_l,
                        alpha,
                        "lower"
                    )
                    c_value_u = _compute_critical_value(
                        b_statistics_u,
                        alpha,
                        "upper"
                    )
                    c_value = (c_value_l, c_value_u)
                else:
                    b_statistics, students = estimate_bootstrap_distribution(
                        self.X,
                        self.Y,
                        self.Z,
                        self.L,
                        threshold,
                        g_dummies,
                        self.metric,
                        epsilon,
                        bootstrap_params
                    )
                    b_statistics /= s_hat
                    c_value = _compute_critical_value(
                        b_statistics,
                        alpha,
                        type
                    )
            self.critical_values.append(c_value)
            self.students.append(s_hat)

    def query_group(
        self,
        group : Union[np.ndarray, int]
    ) -> Tuple[List[Union[float, bool]], List[float], List[float]]:
        """
        Query calibrated auditor for certificate for a particular group
        
        Parameters
        ----------
        group : Union[np.ndarray, int]

        Returns
        -------
        certificate : List[Union[float, bool]]
        value : List[float]
        threshold : List[float]
        """
        if self.critical_values is None:
            raise ValueError("Run calibrate before querying for groups.")

        results = []
        metric_values = []
        thresholds = []
        for g_dummies, critical_value, student in zip(self.groups_list, self.critical_values, self.students):
            if isinstance(group, int):
                dummies = g_dummies[:,group]
                student_value = student[group]
            else: 
                all_dummies = np.amax(g_dummies, axis=1)
                dummies = group.flatten() * all_dummies

                ind = np.all(dummies[:,None] == g_dummies, axis=0)
                if ind.any(): # if there were any matches to existing groups find corresponding student
                    student_value = student[ind]
                    assert len(np.unique(student_value)) == 1
                    student_value = student_value[0]
                else: # don't studentize if no matches found...
                    student_value = student.max()
            if dummies.sum() == 0:
                if self.type in ("lower", "upper"):
                    if self.epsilon is None:
                        results.append(np.nan)
                    else:
                        results.append(False)
                else:
                    if self.epsilon is None:
                        results.append([np.nan, np.nan])
                    else:
                        results.append(False)
                metric_values.append(np.nan)
            else:
                metric_value = np.sum(self.L * dummies) / dummies.sum()
                
                bounds, est, threshold = self._certify_group(metric_value, dummies, g_dummies, critical_value, student_value)
                results.append(bounds)
                metric_values.append(est)
                thresholds.append(threshold)

        return results, metric_values, thresholds
    
    def _certify_group(
        self,
        metric_value : float,
        dummies : np.ndarray,
        group_dummies : np.ndarray,
        critical_value : Union[float, Tuple[float, float]],
        student_value : float
    ) -> Union[bool, float]:
        all_dummies = np.amax(group_dummies, axis=1)
        threshold = self.metric.compute_threshold(self.Z[all_dummies], self.Y[all_dummies])
        group_prob = np.mean(dummies)

        if self.epsilon != None:
            if group_prob == 0:
                return False, metric_value, threshold
            if self.type == "lower":
                cert = metric_value < threshold + self.epsilon + student_value * critical_value / group_prob
                return cert, metric_value, threshold
            elif self.type == "upper":
                cert =  metric_value > threshold + self.epsilon + student_value * critical_value / group_prob
                return cert, metric_value, threshold
            else:
                bool_l = metric_value < threshold + self.epsilon + student_value * critical_value[0] / group_prob
                bool_u = metric_value > threshold - self.epsilon + student_value * critical_value[1] / group_prob
                return bool_l & bool_u, metric_value, threshold
        else:
            if self.type == "lower" or self.type == "upper":
                if group_prob == 0:
                    if critical_value < 0:
                        return np.inf, metric_value, threshold
                    else:
                        return -1 * np.inf, metric_value, threshold
                eps = metric_value - threshold - student_value * critical_value / group_prob**2
                return eps.item(), metric_value, threshold
            else:
                if group_prob == 0:
                    return [-np.inf, np.inf], metric_value - threshold
                
                eps_l = metric_value - threshold - student_value * critical_value[0] / group_prob**2
                eps_u = metric_value - threshold - student_value * critical_value[1] / group_prob**2
                return [eps_u.item(), eps_l.item()], metric_value, threshold

    def calibrate_rkhs(
        self,
        alpha : float,
        type : str,
        kernel : str,
        kernel_params : dict = {},
        bootstrap_params : dict = {}
    ) -> None:
        """
        Obtain bootstrap critical value for a specified RKHS.

        Parameters
        ----------
        alpha : float
            Type I error threshold
        type : str
            Takes one of three values ('lower', 'upper', 'interval').
                type = "upper" issues lower confidence bounds, 
                type = "lower" issues upper confidence bounds, 
                type = "interval" issues confidence intervals.
        kernel : str
            Name of scikit-learn kernel the user would like to use. 
            Suggested kernels: 'rbf' 'laplacian' 'sigmoid'
        kenrnel_params : dict = {}
            Additional parameters required to specify the kernel, 
            e.g. {'gamma': 1} for RBF kernel
        bootstrap_params : dict = {}
            Allows the user to specify a random seed, number of bootstrap resamples,
            and studentization parameters for the bootstrap process.
        """
        vacuous_group = np.ones((len(self.L), 1), dtype=bool)
        if self.metric.requires_conditioning():
            self.groups_list = self.metric.get_conditional_groups(
                vacuous_group,
                self.Z,
                self.Y
            )
        else:
            self.groups_list = [vacuous_group]
        
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.K = []
        self.critical_values = []
        self.type = type
        for group_dummies in self.groups_list:
            K = pairwise_kernels(
                X=self.X[group_dummies].reshape(-1,self.X.shape[1]), 
                metric=kernel, 
                **kernel_params
            ) + 1e-6 * np.eye(len(self.X[group_dummies]))
            self.K.append(K)

            K_sqrt = np.linalg.cholesky(K)
            if bootstrap_params.get("student_threshold", None):
                self.student_threshold = bootstrap_params["student_threshold"]
                K_basis = _approximate_matrix(K_sqrt.T @ K_sqrt, perc=.9)
                print(f"Low-rank approximation size: {K_basis.shape[1]}")
            else:
                self.student_threshold = 1
                K_basis = None

            self.critical_values.append(
                estimate_critical_value(
                    "RKHS", 
                    alpha, 
                    self.L[group_dummies.flatten()], 
                    bootstrap_params, 
                    **dict(K_sqrt=K_sqrt, type=type, K_basis=K_basis)
                )
            )

    def query_rkhs(
        self,
        weights : np.ndarray,
    ) -> Tuple[List[float], List[float]]:
        """
        Query calibrated auditor for certificate for a particular RKHS
        function.
        
        Parameters
        ----------
        weights : np.ndarray

        Returns
        -------
        certificate : List[float]
        value : List[float]
        """
        bounds = []
        vals = []
        for K, crit_value, group_dummies in zip(self.K, self.critical_values, self.groups_list):
            if len(self.groups_list) > 1:
                K_eval = pairwise_kernels(
                    X=self.X[group_dummies],
                    Y=self.X,
                    metric=self.kernel
                    **self.kernel_params
                )
                h_plus = (K_eval @ weights).clip(0) 
            else:
                h_plus = (K @ weights).clip(0)

            Lh_plus = np.mean(self.L[group_dummies.flatten()] * h_plus)
            val = Lh_plus / np.mean(h_plus)

            # self.student_threshold = 1 if there is no studenization
            student_val = max(np.mean(h_plus) * np.sqrt(np.mean(h_plus**2)), self.student_threshold**(3/2)) 
            bound = val - crit_value * student_val / (np.mean(h_plus)**2)

            bounds.append(bound)
            vals.append(val)

        return bounds, vals

    def flag_groups(
        self,
        groups : np.ndarray,
        type : str,
        alpha : float,
        epsilon : float = 0,
        bootstrap_params : dict = {"student" : "mad", "student_threshold" : -np.inf}  
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.metric.requires_conditioning():
            groups_list = self.metric.get_conditional_groups(
                groups,
                self.Z,
                self.Y
            )
        else:
            groups_list = [groups]

        n_groups = groups.shape[1]
        all_p_values = np.ones((len(groups_list), n_groups))
        all_metric_values = np.zeros((len(groups_list), n_groups))
        for i, g_dummies in enumerate(groups_list):
            all_dummies = np.amax(g_dummies, axis=1)
            threshold = self.metric.compute_threshold(
                self.Z[all_dummies], 
                self.Y[all_dummies]
            )
            _, s_grps = estimate_bootstrap_distribution(
                self.X,
                self.Y,
                self.Z,
                self.L,
                threshold,
                g_dummies,
                self.metric,
                epsilon,
                bootstrap_params
            )

            test_statistics = np.sum((self.L - threshold - epsilon)[:,None] * g_dummies, axis=0) 
            test_statistics = test_statistics / np.sum(all_dummies)

            all_metric_values[i,:] = np.sum((self.L - threshold)[:,None] * g_dummies, axis=0) / g_dummies.sum(axis=0)

            if type == "lower":
                all_p_values[i,:] = 1 - norm.cdf(test_statistics / s_grps)
            elif type == "upper":
                all_p_values[i,:] = norm.cdf(test_statistics / s_grps)
            else:
                all_p_values[i,:] = 1 - 2 * norm.cdf(np.abs(test_statistics) / s_grps)
        bh_rejections = multitest.multipletests(all_p_values.flatten(), alpha, method='fdr_bh')
        flags = np.amax(bh_rejections[0].reshape((-1, n_groups)), axis=0)

        return flags, all_metric_values


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


def _approximate_matrix(
    mat : np.ndarray,
    perc : float = 0.9
):
    w, v = np.linalg.eigh(mat)
    v_exp = np.cumsum(w[::-1]) / np.sum(w)
    start_idx = -1 - int(np.argmax(v_exp > perc))

    return v[:,start_idx:]