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

def run_trial(size, seed, disc):
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
        alpha=0.1,
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

def run_cqr_trial(audit_trail, boot_params, disc):
    intervals_at = get_rectangles(audit_trail['x'].to_numpy().reshape(-1,1), {0:np.arange(0, 5, disc)})
    auditor = Auditor(
        X=audit_trail['x'].to_numpy().reshape(-1,1),
        Y=audit_trail['y'].to_numpy(),
        Z=audit_trail[['z1', 'z2']].to_numpy(),
        metric=Metric('equalized_coverage')
    )
    
    auditor.calibrate_groups(
        alpha=0.1,
        type='upper',
        groups=intervals_at,
        bootstrap_params=boot_params 
    )
    fwer = True
    numer_9, denom_9 = (0,0)
    numer_85, denom_85 = (0,0)
    for interval_dummies in intervals_at.T:
        coverage_bound = auditor.query_group(interval_dummies)[0][0]
        min_val = auditor.X[interval_dummies].min()
        max_val = auditor.X[interval_dummies].max()
        min_val = round(math.floor(10 * min_val) / 10, 1)
        max_val = round(math.ceil(10 * max_val) / 10, 1)
        cover = coverage_bound <= coverages_gt[(min_val, max_val)]
        if not cover:
            fwer = False
        if coverages_gt[(min_val, max_val)] >= 0.9:
            denom_9 += 1
            if coverage_bound >= 0.9:
                numer_9 += 1
        if coverages_gt[(min_val, max_val)] >= 0.85:
            denom_85 += 1
            if coverage_bound >= 0.85:
                numer_85 += 1
    print(coverages_gt[(min_val, max_val)], min_val, max_val)
    if denom_9 == 0:
        power_9 = 0
    else:
        power_9 = numer_9 / denom_9
    if denom_85 == 0:
        power_85 = 0
    else:
        power_85 = numer_85 / denom_85
                
    return fwer, (power_9, power_85)

def run_cqr_trial_boolean(audit_trail, boot_params, eps, oos_data, disc):
    auditor = Auditor(
        X=audit_trail['x'].to_numpy().reshape(-1,1),
        Y=audit_trail['y'].to_numpy(),
        Z=audit_trail[['z1', 'z2']].to_numpy(),
        metric=Metric('equalized_coverage')
    )

    intervals_at = get_rectangles(audit_trail['x'].to_numpy().reshape(-1,1), {0:np.arange(0,5,disc)})

    auditor.calibrate_groups(
        alpha=0.1,
        type='upper',
        groups=intervals_at,
        epsilon=eps,
        bootstrap_params=boot_params 
    )

    fwer = True
    true_rejections = 0
    all_rejections = 0
    for interval_dummies in intervals_at.T:
        coverage_boolean = auditor.query_group(interval_dummies)[0][0]
        min_val = auditor.X[interval_dummies].min()
        max_val = auditor.X[interval_dummies].max()
        min_val = round(math.floor(10 * min_val) / 10, 1)
        max_val = round(math.ceil(10 * max_val) / 10, 1)
        cover = coverages_gt[(min_val, max_val)] >= eps
        if coverage_boolean and not cover:
            return False
        if cover and coverage_boolean:
            true_rejections += 1
        if cover:
            all_rejections += 1
            
    if all_rejections == 0:
        power = 1
    else:
        power = true_rejections / all_rejections
    return fwer, true_rejections/all_rejections

def run_trial_boolean(size, seed, boot_params, eps):
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
        alpha=0.1,
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
    parser.add_argument('--studentize', default=False, type=bool)
    parser.add_argument('-s', '--sample_size', nargs='+', type=int)
    parser.add_argument('-e', '--epsilon', default=None, type=float)
    parser.add_argument('-n', '--num_trials', default=1000, type=int)
    parser.add_argument('-d', '--discretization', default=None, type=float)
    return parser.parse_args()


args = parse_args()

fwer_vals = {}
power_vals = {}
disc = args.discretization

for sample_size in args.sample_size:
    fwers = [] 
    powers = []
    rng = np.random.default_rng(seed=1)
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
                fwer, power = run_cqr_trial(audit_trail_sample, boot_params, disc)
            else:
                fwer, power = run_cqr_trial_boolean(audit_trail_sample, boot_params, args.epsilon, oos_data, disc)
        else:
            fwer, power = run_trial_boolean(sample_size, i, boot_params, args.epsilon)
            
        fwers.append(fwer)
        powers.append(power)


    fwer_vals[sample_size] = np.mean(fwers)
    
    print(args, np.mean(fwers), np.mean(powers))
    
with open(f'results_{args.dataset}_{args.sample_size}_{args.epsilon}_{args.discretization}.pkl', 'wb') as fp:
    pickle.dump((fwer_vals, power_vals), fp)
