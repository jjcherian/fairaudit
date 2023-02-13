import numpy as np

from groups import Groups
from typing import List, Union, Callable

def mean_predict(Z, Y):
    return np.mean(Z)

def mean_positive(Z, Y):
    return np.mean(np.isclose(Z[np.isclose(Y, 1)], 0))

def mean_negative(Z, Y):
    return np.mean(np.isclose(Z[np.isclose(Y, 0)], 1))

def error_rate(Z, Y):
    return ~np.isclose(Z, Y)
    
def predictive_equality(Z, Y):
    return np.isclose(Z, 1)

def equal_opportunity(Z, Y):
    return np.isclose(Z, 0)

def statistical_parity(Z, Y):
    return np.isclose(Z, 1)

_METRICS = dict(
    predictive_equality=predictive_equality,
    equal_opportunity=equal_opportunity,
    statistical_parity=statistical_parity
)

_THRESHOLDS = dict(
    predictive_equality=mean_negative,
    equal_opportunity=mean_positive,
    statistical_parity=mean_predict
)

class Metric:
    def __init__(
        self, 
        name : str,
        evaluation_function : Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
        threshold_function : Callable[[np.ndarray, np.ndarray], float] = None,
        metric_params : dict = {}
    ):
        self.metric_name = name
        if name in ('predictive_equality'):
            metric_params['y_values'] = [0]
        if name in ('equal_opportunity'):
            metric_params['y_values'] = [1]

        self.metric_params = metric_params

        if self.metric_name not in _METRICS:
            _METRICS[self.metric_name] = evaluation_function
            _THRESHOLDS[self.metric_name] = threshold_function
    
    def evaluate(
        self,
        Z : np.ndarray,
        Y : np.ndarray
    ) -> np.ndarray:
        return _METRICS[self.metric_name](Z, Y)
    
    def compute_threshold(
        self,
        Z : np.ndarray,
        Y : np.ndarray
    ) -> np.float32:
        return _THRESHOLDS[self.metric_name](Z, Y)

    def requires_conditioning(self) -> bool:
        return 'calibration_bins' in self.metric_params or 'y_values' in self.metric_params

    def get_conditional_groups(
        self,
        group_dummies : np.ndarray,
        Z : np.ndarray = None,
        Y : np.ndarray = None
    ) -> np.ndarray:
        # group_list = groups.definitions
        # group_dummies = groups.dummies

        if 'calibration_bins' in self.metric_params:
            Z_disc = np.digitize(
                Z,
                bins=self.metric_params['calibration_bins']
            )

            z_vals, z_indices = np.unique(Z_disc, return_inverse=True)
            z_dummies = np.zeros((Z.shape[0], len(z_vals)), dtype=int)
            z_dummies[(range(Z.shape[0]), z_indices)] = int(1)

            # group_list = [[dict(grp, Z=z_val) for grp in group_list] for z_val in z_vals]
            group_dummies = np.einsum('ij,ik->kij', group_dummies, z_dummies)

        if 'y_values' in self.metric_params:
            y_vals = np.ones((Y.shape[0], len(self.metric_params['y_values'])))
            y_dummies = np.isclose(Y, y_vals)

            if group_dummies.ndim > 2:
                # group_list = [
                #     [
                #         dict(grp, Y=y_val) for grp in g_list
                #     ]
                #     for g_list in group_list
                #     for y_val in y_vals
                # ]
                group_dummies = np.concatenate(np.einsum('kij,il->klij', group_dummies, y_dummies), axis=0)
            else:
                # group_list = [[dict(grp, Y=y_val) for grp in group_list] for y_val in y_vals]
                group_dummies = np.einsum('ij,ik->kij', group_dummies, y_dummies)

        return group_dummies




