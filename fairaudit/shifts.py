import numpy as np

def score_rkhs_nonneg(
    L : np.ndarray,
    K_sqrt : np.ndarray,
    w : np.ndarray,
    type : str
):
    opt = None
    return opt

def _nonneg_qr(
    M : np.ndarray,
    K_sqrt : np.ndarray,
    basis : np.ndarray,
    type : str,
    n_samples : int = 1e5
):
    Q, _ = np.linalg.qr(basis)

    B = Q.T @ M @ Q
    P = K_sqrt.T @ Q

    P_tilde, R_tilde = np.linalg.qr(P)
    B_tilde = R_tilde @ B @ R_tilde.T

    V, D = np.linalg.eigh(B_tilde)

    Q_tilde = P_tilde @ V

    proj_matrix = np.eye(K_sqrt.shape[0]) - Q_tilde @ Q_tilde.T

    # sample in the 4d subspace
    rng = np.random.default_rng(seed=0)
    S = rng.standard_normal(size=(int(n_samples), K_sqrt.shape[0]))
    S = S / np.linalg.norm(S, axis=1).reshape(-1,1)
    
    

def score_rkhs(
    L : np.ndarray,
    K_sqrt : np.ndarray,
    w : np.ndarray,
    type : str,
    student_threshold : float,
    K_basis : np.ndarray = None
) -> float:
    n = len(w)
    L = L.reshape(-1,1)
    w = w.reshape(-1,1)
    ones = np.ones_like(L).reshape(-1,1)
    A = (1/n**2) * ( (w * L) @ ones.T - w @ L.T)
    M = (A + A.T) / 2

    basis = np.concatenate((w, L, ones, (w * L)), axis=1)

    if student_threshold is None:
        opt = _qr_no_student(M, K_sqrt, basis, type)
    else:
        opt = _qr_student(M, K_sqrt, np.concatenate((basis, K_basis), axis=1), student_threshold, type)
    return opt

def _qr_no_student(
    M : np.ndarray,
    K_sqrt : np.ndarray,
    basis : np.ndarray,
    type : str
):
    Q, _ = np.linalg.qr(basis)

    B = Q.T @ M @ Q
    P = K_sqrt.T @ Q

    _, R_tilde = np.linalg.qr(P)
    B_tilde = R_tilde @ B @ R_tilde.T
    evals = np.linalg.eigvalsh(B_tilde)

    if type == "lower":
        return np.min(evals)
    elif type == "upper":
        return np.max(evals)
    return np.max(np.abs(evals))

def _qr_student(
    M : np.ndarray,
    K_sqrt : np.ndarray,
    basis : np.ndarray,
    student_threshold : float,
    type : str,
    n_samples : int = 1e6
):
    Q, _ = np.linalg.qr(basis)

    B = Q.T @ M @ Q
    P = K_sqrt.T @ Q

    P_tilde, R_tilde = np.linalg.qr(P)
    B_tilde = R_tilde @ B @ R_tilde.T

    n = K_sqrt.shape[0]
    target = K_sqrt.T @ np.ones((n,1))
    c = np.linalg.norm(target)

    v = P_tilde[:,0].reshape(-1,1) - (target / c)
    v_norm = v / np.linalg.norm(v)
    hh_ref = np.eye(n) - 2 * v_norm @ v_norm.T

    equiv_mat = P_tilde @ B_tilde @ P_tilde.T

    Q_star = hh_ref @ P_tilde
    B_star = Q_star.T @ equiv_mat @ Q_star

    rng = np.random.default_rng(seed=0)
    S = rng.standard_normal(size=(int(n_samples), B_star.shape[0]))
    S = S / np.linalg.norm(S, axis=1).reshape(-1,1)
    
    outputs = (S.dot(B_star)*S).sum(axis=1)

    K_approx = Q_star.T @ K_sqrt.T @ K_sqrt @ Q_star
    denom = np.abs(S[:,0]) * np.sqrt((S.dot(K_approx)*S).sum(axis=1))
    outputs /= (c/n**(3/2)) * denom.clip(student_threshold**(3/2) * (n**(3/2)/ c))

    if type == "lower":
        return np.min(outputs)
    elif type == "upper":
        return np.max(outputs)
    return np.max(np.abs(outputs))