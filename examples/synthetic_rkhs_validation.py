import numpy as np
import pandas as pd
import pickle

from auditor import Auditor
from metrics import Metric

from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_kernels

def eval_mse(Z, Y):
    return (Z - Y)**2

def mean_mse(Z, Y):
    return np.mean((Z - Y)**2)

def run_trial_constant(sample_size, kernel_params, boot_params, alpha):

    metric = Metric("mse", eval_mse, mean_mse)

    rng = np.random.default_rng(seed=boot_params['seed'])
    X = rng.uniform(0, 1, size=(sample_size,1))
    Y = rng.standard_normal(size=sample_size)

    auditor = Auditor(
        X=X,
        Y=Y,
        Z=np.zeros_like(Y),
        metric=metric
    )

    auditor.calibrate_rkhs(
        alpha=alpha*2,
        type='upper',
        kernel="rbf",
        kernel_params=kernel_params,
        bootstrap_params=boot_params
    )

    critical_value = auditor.critical_values[0]

    K = pairwise_kernels(X=X, metric='rbf', **kernel_params) + 1e-6 * np.eye(sample_size)
    K_sqrt = np.linalg.cholesky(K)
    
    L = eval_mse(np.zeros_like(Y), Y).reshape(-1,1)
    ones = np.ones_like(L)

    A = (1/sample_size**2) * ( L @ ones.T - ones @ ones.T)
    M = (A + A.T) / 2

    basis = np.concatenate((L, ones), axis=1)
    Q, _ = np.linalg.qr(basis)

    B = Q.T @ M @ Q
    P = K_sqrt.T @ Q

    _, R_tilde = np.linalg.qr(P)
    B_tilde = R_tilde @ B @ R_tilde.T
    evals = np.linalg.eigvalsh(B_tilde)
    opt = evals.max()
    fwer = opt > critical_value
    return fwer


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def run_trial_ols(sample_size, d, kernel_params, boot_params, alpha):

    metric = Metric("mse", eval_mse, mean_mse)

    rng = np.random.default_rng(seed=boot_params['seed'])

    # for now d = 1
    d = 1
    n_coord = 101
    x_coords = np.linspace(0, 1, n_coord)
    x_grid = cartesian_product(*[x_coords]*d)

    n_train = 1000

    theta_0 = rng.standard_normal((d,1))

    x_train = rng.choice(x_coords, size=(n_train,d))
    y_train = rng.normal(loc=(x_train @ theta_0).flatten(), scale=np.sqrt(x_train.max(axis=1)))

    theta_hat, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)

    X = rng.choice(x_coords, size=(sample_size, d))
    Y = rng.normal(loc= (X @ theta_0).flatten(), scale=np.sqrt(X.max(axis=1)))
    Z = (X @ theta_hat).flatten()

    auditor = Auditor(
        X=X,
        Y=Y,
        Z=Z,
        metric=metric
    )

    auditor.calibrate_rkhs(
        alpha=alpha*2,
        type='upper',
        kernel="rbf",
        kernel_params=kernel_params,
        bootstrap_params=boot_params
    )

    critical_value = auditor.critical_values[0]

    bools = X[...,None] == x_grid
    mse = eval_mse(Z, Y).reshape(-1,1)
    L = (mse[:,None] * bools).sum(axis=0)
    counts = bools.sum(axis=0)

    L_pop = np.max(x_grid, axis=1).reshape(-1,1) + (x_grid @ (theta_0 - theta_hat))**2
    ones = np.ones_like(L_pop)

    K = pairwise_kernels(X=x_grid, metric='rbf', **kernel_params) + 1e-6 * np.eye(len(x_coords)**d)
    K_sqrt = np.linalg.cholesky(K)


    A = (1/(sample_size*len(x_coords)**d)) * ((L @ ones.T) - (L_pop @ counts.T))
    M = (A + A.T) / 2

    basis = np.concatenate((L, ones, counts, L_pop), axis=1)
    Q, _ = np.linalg.qr(basis)

    B = Q.T @ M @ Q
    P = K_sqrt.T @ Q

    _, R_tilde = np.linalg.qr(P)
    B_tilde = R_tilde @ B @ R_tilde.T
    evals = np.linalg.eigvalsh(B_tilde)
    opt = evals.max()
    fwer = opt > critical_value

    return fwer

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='OLS')
    parser.add_argument('-s', '--sample_size', nargs='+', type=int)
    parser.add_argument('-g', '--gamma', default=1, type=float)
    parser.add_argument('-n', '--num_trials', default=1000, type=int)
    parser.add_argument('-a', '--alpha', default=0.1, type=float)
    return parser.parse_args()


args = parse_args()

fwer_vals = {}

for sample_size in args.sample_size:
    fwers = [] 
    for i in tqdm(np.arange(0, args.num_trials)):
        boot_params = {'seed': i}
        if args.dataset == "Constant":
            fwer = run_trial_constant(sample_size, {'gamma': args.gamma}, boot_params, args.alpha)
        elif args.dataset == "OLS":
            fwer = run_trial_ols(sample_size, 1, {'gamma': args.gamma}, boot_params, args.alpha)
        fwers.append(fwer)


    fwer_vals[sample_size] = np.mean(fwers)
    
    print(args, sample_size, np.mean(fwers))
    
with open(f'test_results/results_{args.dataset}_{args.sample_size}_{args.gamma}_{args.alpha}.pkl', 'wb') as fp:
    pickle.dump(fwer_vals, fp)
