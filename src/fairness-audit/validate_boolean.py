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
        alpha=0.05,
        type='upper',
        groups=intervals_at,
        bootstrap_params={'seed': seed, 'B': 2000, 'method': 'multinomial'}
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
        return False
    return True

def run_cqr_trial(audit_trail, boot_params, disc):
    intervals_at = get_rectangles(audit_trail['x'].to_numpy().reshape(-1,1), {0:np.arange(0, 5, disc)})
    
    auditor = Auditor(
        X=audit_trail['x'].to_numpy().reshape(-1,1),
        Y=audit_trail['y'].to_numpy(),
        Z=audit_trail[['z1', 'z2']].to_numpy(),
        metric=Metric('equalized_coverage')
    )

    auditor.calibrate_groups(
        alpha=0.05,
        type='upper',
        groups=intervals_at,
        bootstrap_params=boot_params 
    )
    for interval_dummies in intervals_at.T:
        coverage_bound = auditor.query_group(interval_dummies)[0][0]
        min_val = auditor.X[interval_dummies].min()
        max_val = auditor.X[interval_dummies].max()
        min_val = round(math.floor(10 * min_val) / 10, 1)
        max_val = round(math.ceil(10 * max_val) / 10, 1)
        cover = coverage_bound <= coverages_gt[(min_val, max_val)]
        if not cover:
            return False
    return True

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


    for interval_dummies in intervals_at.T:
        coverage_boolean = auditor.query_group(interval_dummies)[0][0]
        if coverage_boolean:
            # print("i issued certificate")
            min_val = auditor.X[interval_dummies].min()
            max_val = auditor.X[interval_dummies].max()
            min_val = round(math.floor(10 * min_val) / 10, 1)
            max_val = round(math.ceil(10 * max_val) / 10, 1)
            cover = coverages_gt[(min_val, max_val)] >= eps
            if not cover:
                return False
    return True

audit_trail = pd.read_csv('../../oos_data.csv', index_col=0)

eps_loose = 0.85
eps = 0.9

fwer_exact = {}
fwer_cqr_exact = {}
fwer_cqr_loose = {}
for sample_size in [100, 200, 400, 800, 1600]:
    fwers_bern = []
    fwers_bern_2 = []
    fwers_cqr = []
    fwers_cqr_2 = []

    rng = np.random.default_rng(seed=10)
    for i in tqdm(np.arange(0, 1000)):
        boot_params = {'seed': i, 'B': 500, 'method': 'multinomial'}
        prob_threshold = (25 / sample_size)

        boot_params = {'seed': i, 'B': 500, 'method': 'multinomial',
                    'student': 'prob_bool', 'student_threshold': prob_threshold**(1/2)}
        
        # fwer_sharp = run_trial_boolean(sample_size, i, boot_params, eps=0.9)
        # fwers_bern.append(fwer_sharp)

        # fwer_sharp_2 = run_trial_boolean(sample_size, i, boot_params, z_star=1.96)
        # fwers_bern_2.append(fwer_sharp_2)

        audit_trail_sample = audit_trail.sample(n=sample_size, replace=True, random_state=rng)

        # validate semi-synthetic boolean certification
        fwer_cqr = run_cqr_trial_boolean(audit_trail_sample, boot_params, eps, audit_trail, disc=0.1)
        fwers_cqr.append(fwer_cqr)

        # fwer_cqr_2 = run_cqr_trial_boolean(audit_trail_sample, boot_params, eps_loose, audit_trail, disc=0.1)
        # fwers_cqr_2.append(fwer_cqr_2)
    
    # fwer_exact[sample_size] = np.mean(fwers_bern)
    # fwer_cqr_exact[sample_size] = np.mean(fwers_cqr)
    fwer_cqr_loose[sample_size] = np.mean(fwers_cqr)
    print(sample_size, np.mean(fwers_cqr), np.mean(fwers_cqr_2))

with open('bernoulli_validation.pkl', 'wb') as fp:
    pickle.dump((fwer_exact, fwer_cqr_exact, fwer_cqr_loose), fp)
