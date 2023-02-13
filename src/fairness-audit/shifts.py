import numpy as np


def score_rkhs(
    L : np.ndarray,
    K_sqrt : np.ndarray,
    w : np.ndarray,
    seed : int,
    num_iters : int = 15
) -> float:
    n = len(w)
    L = L.reshape(-1,1)
    w = w.reshape(-1,1)
    ones = np.ones_like(L).reshape(-1,1)
    A = (1 / n**2) * ( (w * L) @ ones.T - w @ L.T)

    M = K_sqrt @ (A + A.T) @ K_sqrt

    opt = (1/4) * _power_iteration(M, seed, num_iters)

    return opt

def _power_iteration(
    M : np.ndarray,
    seed : int,
    num_iters : int = 15
) -> float:
    rng = np.random.default_rng(seed=seed)

    v = rng.standard_normal(size=(M.shape[0], 1))
    for _ in range(num_iters):
        v = M @ v
        v = v / np.linalg.norm(v)
    return v.T @ M @ v