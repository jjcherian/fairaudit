import math
import numpy as np
import pandas as pd
import pickle
import scipy.stats

from auditor import Auditor
from groups import get_rectangles
from metrics import Metric

from tqdm import tqdm

with open('ground_truth.pkl', 'rb') as fp:
    coverages_gt = pickle.load(fp)
audit_trail = pd.read_csv('../../oos_data.csv', index_col=0)

def run_trial(size, seed, disc, alpha=0.05):
    rng = np.random.default_rng(seed=seed)
    x = rng.uniform(0, 5, size=size).reshape(-1,1)
    intervals_at = get_rectangles(x, {0:np.arange(0, 5, disc)})
    
    z_star = 1.96

    y = rng.standard_normal(size=len(x))
    z1 = -1 * z_star * np.ones_like(y).reshape(-1,1)
    z2 = z_star * np.ones_like(y).reshape(-1,1)
    z = np.concatenate((z1, z2), axis=1)
    auditor = Auditor(
        X=x,
        Y=y,
        Z=z,
        metric=Metric('equalized_coverage')
    )
    
    auditor.calibrate_groups(
        alpha=alpha,
        type='upper',
        groups=intervals_at,
        bootstrap_params={'seed': seed, 'B': 500, 'method': 'multinomial'}
    )
    coverages = np.ones((len(intervals_at.T),))
    gaps = np.zeros((len(intervals_at.T),))
    for i, interval_dummies in enumerate(intervals_at.T):
        coverage_bound = auditor.query_group(interval_dummies)[0][0]
        min_val = auditor.X[interval_dummies].min()
        max_val = auditor.X[interval_dummies].max()
        min_val = math.floor(10 * min_val) / 10
        max_val = math.ceil(10 * max_val) / 10
        cover = coverage_bound <= 1 - 2 * scipy.stats.norm.cdf(-1 * z_star) # coverages_gt[(min_val, max_val)]
        coverages[i] = cover
        gaps[i] = coverage_bound - (1 - 2 * scipy.stats.norm.cdf(-1 * z_star))
    
    # print('realized coverage gap minimizer', intervals_at[:,gaps.argmax()].mean())

    fwer = np.all(coverages)
    return fwer

def run_cqr_trial(audit_trail, boot_params, disc, alpha=0.05):
    discretization = np.arange(0, 5, disc)
    auditor = Auditor(
        X=audit_trail['x'].to_numpy().reshape(-1,1),
        Y=audit_trail['y'].to_numpy(),
        Z=audit_trail[['z1', 'z2']].to_numpy(),
        metric=Metric('equalized_coverage')
    )

    import IPython
    IPython.embed()

    intervals_at = get_rectangles(audit_trail['x'].to_numpy().reshape(-1,1), {0:discretization})

    
    auditor.calibrate_groups(
        alpha=alpha,
        type='upper',
        groups=intervals_at,
        bootstrap_params=boot_params 
    )

    X = audit_trail['x'].to_numpy().reshape(-1,1)
    fwer = True
    numer_9, denom_9 = (0,0)
    numer_85, denom_85 = (0,0)
    for min_val in discretization:
        val_list = np.arange(min_val + disc, 5 + disc, disc)
        for max_val in val_list:
            if round(max_val, 1) > 5.05:
                continue
            interval_dummies = (X <= max_val) & (X >= min_val)
            if interval_dummies.any() == False:
                if coverages_gt[(round(min_val,1), round(max_val,1))] >= 0.9:
                    denom_9 += 1
                if coverages_gt[(round(min_val,1), round(max_val,1))] >= 0.85:
                    denom_85 += 1
                continue
            coverage_bound = auditor.query_group(interval_dummies)[0][0]
            cover = coverage_bound <= coverages_gt[(round(min_val, 1), round(max_val, 1))]
            if not cover:
                fwer = False
            if coverages_gt[(round(min_val,1), round(max_val,1))] >= 0.9:
                denom_9 += 1
                if coverage_bound >= 0.9:
                    numer_9 += 1
            if coverages_gt[(round(min_val,1), round(max_val,1))] >= 0.85:
                denom_85 += 1
                if coverage_bound >= 0.85:
                    numer_85 += 1
    if denom_9 == 0:
        power_9 = 0
    else:
        power_9 = numer_9 / denom_9
    if denom_85 == 0:
        power_85 = 0
    else:
        power_85 = numer_85 / denom_85
                
    return fwer, (power_9, power_85)

def run_cqr_trial_boolean(audit_trail, boot_params, eps, disc, alpha=0.05):
    auditor = Auditor(
        X=audit_trail['x'].to_numpy().reshape(-1,1),
        Y=audit_trail['y'].to_numpy(),
        Z=audit_trail[['z1', 'z2']].to_numpy(),
        metric=Metric('equalized_coverage')
    )

    discretization = np.arange(0,5,disc)
    intervals_at = get_rectangles(audit_trail['x'].to_numpy().reshape(-1,1), {0:discretization})

    auditor.calibrate_groups(
        alpha=alpha,
        type='upper',
        groups=intervals_at,
        epsilon=eps,
        bootstrap_params=boot_params 
    )

    X=audit_trail['x'].to_numpy().reshape(-1,1)
    fwer = True
    true_rejections = 0
    all_rejections = 0
    for min_val in discretization:
        for max_val in np.arange(min_val + disc, 5 + disc, disc):
            if round(max_val, 1) > 5.05:
                continue
            cover = coverages_gt[(round(min_val,1), round(max_val,1))] >= eps
            if cover:
                all_rejections += 1

            interval_dummies = (X <= max_val) & (X >= min_val)
            coverage_boolean = auditor.query_group(interval_dummies)[0][0]
            if coverage_boolean:
                if cover:
                    true_rejections += 1
                else:
                    fwer = False   


            if interval_dummies.any() == False:
                if coverage_boolean:
                    all_rejections += 1
                continue

    if all_rejections == 0:
        power = 1
    else:
        power = true_rejections / all_rejections

    return fwer, power

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def eval_mse(Z, Y):
    return (Z - Y)**2

def mean_mse(Z, Y):
    return np.mean((Z - Y)**2)


def run_trial_boolean_ols(sample_size, boot_params, eps, alpha):
    rng = np.random.default_rng(seed=boot_params['seed'])

    d = 1
    n_train = 1000

    theta_0 = rng.standard_normal((d,1))
    x_train = rng.uniform(0, 1, size=(n_train,d))
    y_train = rng.normal(loc=x_train @ theta_0, scale=np.sqrt(eps)).flatten()

    theta_hat, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)

    X = rng.uniform(0, 1, size=(sample_size,d))
    Y = rng.normal(loc=(X @ theta_0).flatten(), scale=np.sqrt(eps)).flatten()
    Z = (X @ theta_hat).flatten()

    metric = Metric("mse", eval_mse, lambda x, y: 0)

    auditor = Auditor(
        X=X,
        Y=Y,
        Z=Z,
        metric=metric
    )

    auditor.calibrate_groups(
        alpha=alpha,
        type='lower',
        groups="intervals",
        epsilon=eps,
        bootstrap_params=boot_params 
    )

    def _max_subarray(numbers):
        best_sum = 0  # or: float('-inf')
        current_sum = 0
        for i, x in enumerate(numbers):
            current_sum = max(0, current_sum + x)
            best_sum = max(best_sum, current_sum)
        return best_sum
    
    L = eval_mse(Z, Y).flatten()
    arr = (L - eps) / sample_size
    if -1 * _max_subarray(-1 * arr) < auditor.critical_values[0]:
        return False, 0
    return True, 0

def run_trial_boolean_olsh(sample_size, boot_params, eps, alpha):
    rng = np.random.default_rng(seed=boot_params['seed'])

    d = 1
    n_train = 1000

    theta_0 = rng.standard_normal((d,1))
    x_train = rng.uniform(0, 1, size=(n_train,d))
    y_train = rng.normal(loc=(x_train @ theta_0).flatten(), scale=np.sqrt(x_train.flatten())).flatten()

    theta_hat, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)

    X = rng.uniform(0, 1, size=(sample_size,d))
    Y = rng.normal(loc=(X @ theta_0).flatten(), scale=np.sqrt(X.flatten())).flatten()
    Z = (X @ theta_hat).flatten()

    metric = Metric("mse", eval_mse, lambda x, y: 0)

    auditor = Auditor(
        X=X,
        Y=Y,
        Z=Z,
        metric=metric
    )

    discretization = np.arange(0,1,disc)
    intervals_at = get_rectangles(X, {0:discretization})

    auditor.calibrate_groups(
        alpha=alpha,
        type='lower',
        groups=intervals_at,
        epsilon=eps,
        bootstrap_params=boot_params 
    )


    fwer = True
    true_rejections = 0
    all_rejections = 0
    for min_val in discretization:
        for max_val in np.arange(min_val + disc, 1 + disc, disc):
            if round(max_val, 1) > 1.05:
                continue
                
            true_eps = (max_val + min_val) / 2 + (theta_hat - theta_0)**2  * (max_val**2 + min_val**2 + min_val * max_val)/3
            cover = true_eps <= eps
            if cover:
                all_rejections += 1

            interval_dummies = (X <= max_val) & (X >= min_val)
            coverage_boolean = auditor.query_group(interval_dummies)[0][0]
            if coverage_boolean:
                if cover:
                    true_rejections += 1
                else:
                    fwer = False   

            if interval_dummies.any() == False:
                if coverage_boolean:
                    all_rejections += 1
                continue

    if all_rejections == 0:
        power = 1
    else:
        power = true_rejections / all_rejections
    return fwer, power

def run_trial_olsh(sample_size, boot_params, alpha):
    rng = np.random.default_rng(seed=boot_params['seed'])

    d = 1
    n_train = 1000

    theta_0 = rng.standard_normal((d,1))
    x_train = rng.uniform(0, 1, size=(n_train,d))
    y_train = rng.normal(loc=(x_train @ theta_0).flatten(), scale=np.sqrt(x_train.flatten())).flatten()

    theta_hat, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)

    X = rng.uniform(0, 1, size=(sample_size,d))
    Y = rng.normal(loc=(X @ theta_0).flatten(), scale=np.sqrt(X.flatten())).flatten()
    Z = (X @ theta_hat).flatten()

    metric = Metric("mse", eval_mse, lambda x, y: 0)

    auditor = Auditor(
        X=X,
        Y=Y,
        Z=Z,
        metric=metric
    )

    discretization = np.arange(0,1,disc)
    intervals_at = get_rectangles(X, {0:discretization})

    auditor.calibrate_groups(
        alpha=alpha,
        type='lower',
        groups=intervals_at,
        epsilon=None,
        bootstrap_params=boot_params 
    )

    if boot_params['seed'] == 3:
        import IPython
        IPython.embed()

    fwer = True
    numer_5, denom_5 = (0,0)
    numer_4, denom_4 = (0,0)
    for min_val in discretization:
        for max_val in np.arange(min_val + disc, 1 + disc, disc):
            if round(max_val, 1) > 1.05:
                continue
                
            true_eps = (max_val + min_val) / 2 + (theta_hat - theta_0)**2  * (max_val**2 + min_val**2 + min_val * max_val)/3
            interval_dummies = (X <= max_val) & (X >= min_val)

            if interval_dummies.any() == False:
                if true_eps <= 0.5:
                    denom_5 += 1
                if true_eps <= 0.4:
                    denom_4 += 1
                continue

            eps_hat = auditor.query_group(interval_dummies)[0][0]
            if true_eps <= 0.5:
                denom_5 += 1
                if eps_hat <= 0.5:
                    numer_5 += 1
            if true_eps <= 0.4:
                denom_4 += 1
                if eps_hat <= 0.4:
                    numer_4 += 1
            if eps_hat < true_eps:
                fwer = False

    if denom_5 == 0:
        power_5 = 1
    else:
        power_5 = numer_5 / denom_5
    if denom_4 == 0:
        power_4 = 1
    else:
        power_4 = numer_4 / denom_4
                
    return fwer, (power_5, power_4)

def run_trial_boolean(size, seed, boot_params, eps, alpha=0.05):
    rng = np.random.default_rng(seed=seed)

    x = rng.uniform(-1, 1, size=size).reshape(-1,1)    
    y = rng.standard_normal(size=len(x))
    z1 = -np.inf * np.ones_like(y).reshape(-1,1)
    z2 = scipy.stats.norm.ppf(eps) * np.ones_like(y).reshape(-1,1)
    z = np.concatenate((z1, z2), axis=1)
    auditor = Auditor(
        X=x,
        Y=y,
        Z=z,
        metric=Metric('equalized_coverage')
    )

    # 1 - 2 * scipy.stats.norm.cdf(-1 * z_star)

    auditor.calibrate_groups(
        alpha=alpha,
        type='upper',
        groups="intervals",
        epsilon=eps,
        bootstrap_params=boot_params 
    )

    def _max_subarray(numbers):
        best_sum = 0  # or: float('-inf')
        current_sum = 0
        for i, x in enumerate(numbers):
            current_sum = max(0, current_sum + x)
            best_sum = max(best_sum, current_sum)
        return best_sum
    
    L = ((y <= z2.flatten()) & (y >= z1.flatten())).astype(int)
    arr = (L - eps) / size

    if _max_subarray(arr) > auditor.critical_values[0]:
        return False, 0
    return True, 0

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="CQR")
    parser.add_argument('--studentize', action='store_true')
    parser.add_argument('-s', '--sample_size', nargs='+', type=int)
    parser.add_argument('-e', '--epsilon', default=None, type=float)
    parser.add_argument('-n', '--num_trials', default=1000, type=int)
    parser.add_argument('-d', '--discretization', default=None, type=float)
    parser.add_argument('-a', '--alpha', default=0.1, type=float)
    return parser.parse_args()


args = parse_args()

fwer_vals = {}
power_vals = {}
disc = args.discretization

for sample_size in args.sample_size:
    fwers = [] 
    powers = []
    rng = np.random.default_rng(seed=0)
    for i in tqdm(np.arange(0, args.num_trials)):
        audit_trail_sample = audit_trail.sample(n=sample_size, replace=True, random_state=rng)

        if args.studentize:
            prob_threshold = (25 / sample_size)
            if args.epsilon is None:
                boot_params = {'seed': i, 'B': 500, 'method': 'multinomial',
                               'student': 'prob_bound', 'student_threshold': prob_threshold**(3/2)}
            else:
                boot_params = {'seed': i, 'B': 500, 'method': 'multinomial',
                               'student': 'prob_bool', 'student_threshold': prob_threshold**(1/2)}
        else:     
            boot_params = {'seed': i, 'B': 500, 'method': 'multinomial'}

        if args.dataset == "CQR":
            if args.epsilon is None:
                fwer, power = run_cqr_trial(audit_trail_sample, boot_params, disc, args.alpha)
            else:
                fwer, power = run_cqr_trial_boolean(audit_trail_sample, boot_params, args.epsilon, disc, args.alpha)
        elif args.dataset == "OLS":
            fwer, power = run_trial_boolean_ols(sample_size, boot_params, args.epsilon, args.alpha)
        elif args.dataset == "OLSH":
            if args.epsilon is None:
                fwer, power = run_trial_olsh(sample_size, boot_params, args.alpha)
            else:
                fwer, power = run_trial_boolean_olsh(sample_size, boot_params, args.epsilon, args.alpha)
            
        fwers.append(fwer)
        powers.append(power)

    fwer_vals[sample_size] = np.mean(fwers)
    power_vals[sample_size] = np.mean(np.asarray(powers), axis=0)
    
    print(args, sample_size, np.mean(fwers), np.mean(np.asarray(powers), axis=0))
    
with open(f'test_results/results_{args.dataset}_{args.sample_size}_{args.epsilon}_{args.discretization}_{args.alpha}_{args.studentize}.pkl', 'wb') as fp:
    pickle.dump((fwer_vals, power_vals), fp)
