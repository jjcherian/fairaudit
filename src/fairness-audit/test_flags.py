import numpy as np
import pandas as pd
import pickle

from auditor import Auditor
from groups import get_intersections
from metrics import Metric

from tqdm import tqdm

import folktables
from sklearn.linear_model import LogisticRegression, LinearRegression

ACSIncomeReg = folktables.BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    group='RAC1P',
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)



def run_trial(auditor, boot_params, epsilon, alpha, true_values):
    groups = get_intersections(auditor.X)#[:,2].reshape(-1,1))

    # groups = groups[:,np.sum(groups, axis=0) >= 25] # don't normal approx. tiny groups
    flags, values = auditor.flag_groups(
        alpha=alpha,
        epsilon=epsilon,
        type='lower',
        groups=groups,
        bootstrap_params=boot_params
    )

    discoveries = 0
    false_discoveries = 0
    avail_discoveries = 0
    for (age_id, sex_id, race_id), true_value in true_values.items():
        if true_value >= epsilon:
            avail_discoveries += 1

        # get group identity corresponding to group_key
        dummies = np.ones_like(auditor.Y).astype(bool)
        if age_id is not None:
            dummies &= (auditor.X[:,0] == age_id)
        if sex_id is not None:
            dummies &= (auditor.X[:,1] == sex_id)
        if race_id is not None:
            dummies &= (auditor.X[:,2] == race_id)

        # look up group in groups and label that idx
        idx = np.all(dummies[:,None] == groups, axis=0)
        if idx.any() and flags[idx][0]:
            discoveries += 1
            if true_value < epsilon:
                false_discoveries += 1

    fdr = false_discoveries / max(discoveries, 1)
    if avail_discoveries == 0:
        power = 1
    else:
        power = discoveries / avail_discoveries
    return fdr, power

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ACSIncome')
    parser.add_argument('-s', '--sample_size', nargs='+', type=int)
    parser.add_argument('-e', '--epsilon', default=0, type=float)
    parser.add_argument('-n', '--num_trials', default=1000, type=int)
    parser.add_argument('-a', '--alpha', default=0.1, type=float)
    parser.add_argument('-t', '--threshold', default=None, type=float)
    return parser.parse_args()

def eval_misclassify(Z, Y):
    return ~np.isclose(Z, Y)

def mean_misclassify(Z, Y):
    return 1 - np.mean(np.isclose(Z, Y))

def eval_mse(Z, Y):
    return (Z - Y)**2

def mean_mse(Z, Y):
    return np.mean((Z - Y)**2)

args = parse_args()

fdr_vals = {}

data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=["CA"], download=True)

if args.dataset == "ACSIncome":
    features, labels, _ = folktables.ACSIncome.df_to_numpy(ca_data)
    model = LogisticRegression()
    if args.threshold is None:
        metric = Metric("misclassification", eval_misclassify, mean_misclassify)
    else:
        metric = Metric("misclassification", eval_misclassify, lambda z,y : args.threshold)
else:
    features, labels, _ = ACSIncomeReg.df_to_numpy(ca_data)
    model = LinearRegression()
    metric = Metric("mse", eval_mse, mean_mse)

n_train = 1000

x_train = features[0:n_train]
y_train = labels[0:n_train]

x_test = features[n_train:len(features)]
y_test = labels[n_train:len(features)]

group_features = x_test[:,[0,-2,-1]]

# digitize age into <18, 18-34, 35-49, 50-64, 65+
age_bins = [0, 18, 35, 50, 65, 120]
group_features[:,0] = np.digitize(
    group_features[:,0],
    bins = age_bins
)

race_grp = np.zeros_like(group_features[:,2])
white_filter = (group_features[:,2] == 1)
bipoc_filter = (group_features[:,2] == 2) | (group_features[:,2] == 3) | (group_features[:,2] == 4) | (group_features[:,2] == 5)
aapi_filter = (group_features[:,2] == 6) | (group_features[:,2] == 7)
other_filter = (group_features[:,2] == 8) | (group_features[:,2] == 9)

race_grp[white_filter] = 1
race_grp[bipoc_filter] = 2
race_grp[aapi_filter] = 3
race_grp[other_filter] = 4
group_features[:,2] = race_grp

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
if args.dataset == "ACSIncome":
    y_pred = y_pred >= 0.5

all_groups = get_intersections(group_features)
all_groups = np.unique(all_groups, axis=1) # TODO: should this be in get_intersections?

# TODO: subset groups to get rid of the groups that are too low probability?
true_values = {}
pop_threshold = metric.compute_threshold(
    Z=y_pred, Y=y_test
)
for grp in all_groups.T:
    ages = np.unique(group_features[grp,0])
    sexes = np.unique(group_features[grp,1])
    races = np.unique(group_features[grp,2])

    sex_id = sexes[0] if len(sexes) == 1 else None
    race_id = races[0] if len(races) == 1 else None
    age_id = ages[0] if len(ages) == 1 else None

    if args.dataset == "ACSIncome":
        grp_mean = mean_misclassify(y_pred[grp], y_test[grp])
    else:
        grp_mean = mean_mse(y_pred[grp], y_test[grp])

    true_values[(age_id, sex_id, race_id)] = grp_mean - pop_threshold

import IPython
IPython.embed()
fdr_vals = {}
power_vals = {}

for sample_size in args.sample_size:
    fdrs = [] 
    powers = []
    rng = np.random.default_rng(seed=0)
    for i in tqdm(np.arange(0, args.num_trials)):
        sample_indices = rng.integers(0, len(y_test), sample_size)

        boot_params = {'seed': 0, "student": "mad", "student_threshold": 1e-8, "prob_threshold": 0/sample_size}

        auditor = Auditor(
            X=group_features[sample_indices],
            Y=y_test[sample_indices],
            Z=y_pred[sample_indices],
            metric=metric
        )
        fdr, power = run_trial(auditor, boot_params, args.epsilon, args.alpha, true_values)
        fdrs.append(fdr)
        powers.append(power)


    fdr_vals[sample_size] = np.mean(fdrs)
    power_vals[sample_size] = np.mean(np.asarray(powers), axis=0)
    
    print(args, sample_size, np.mean(fdrs), np.mean(np.asarray(powers), axis=0))
    
with open(f'test_results/results_{args.dataset}_{args.sample_size}_{args.epsilon}_{args.alpha}.pkl', 'wb') as fp:
    pickle.dump((fdr_vals, power_vals), fp)
