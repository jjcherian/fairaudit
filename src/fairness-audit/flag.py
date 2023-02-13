import numpy as np
import pandas as pd

from typing import Union

def flag_unfairness(
    X : np.ndarray, 
    y : np.ndarray, 
    pred : np.ndarray, 
    subpopulations : np.ndarray,
    metric : Union[function, str],
    normalized : bool = True
):
    """
    flag_unfairness is the wrapper function for flagging unfairness

    Parameters
    ----------

    X : np.ndarray
    y : np.ndarray
    pred : np.ndarray
    subpopulations : np.ndarray
    metric : Union[function, str]
    normalized : bool
    """

    df = pd.DataFrame({"y": y, "pred": pred, "subpop": subpopulations})

        


def _compute_statistic(df, metric):
    subpop_metrics = df.groupby(by=["subpop"]).apply(func=metric)
    return subpop_metrics
