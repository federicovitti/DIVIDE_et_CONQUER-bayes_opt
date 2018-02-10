import numpy as np
from bo_my import BayesianOptimization
from functions import meno_ackley
import math

# Define search space's dimensions
dx = 2
# Define search space
bounds = np.matlib.repmat([[-32.768], [32.768]], 1, dx)
# Tuning parameters
options = { 'opt_value': 0.,
			'pbounds': bounds,
			'n_iter': 20,
			'B': 10,
			'n_bo': 5,
			'n_percent': 0.7,
			'init_points': 20,
			'acq_fun': 'ei',
			'k': 2.576,
			'xi': 0.05
			}
# List of Divide et Conquer methods
methods = list(['mondrian'])
methods.append('kmeans')
methods.append('dbscan')
methods.append('hdbscan')

# Number of validation's iterations
cv_iter = 10
# Placeholders
y_bests = np.empty((len(methods), cv_iter, options['n_iter']))
errs = np.empty((len(methods), cv_iter, options['n_iter']))
times = np.empty((len(methods), cv_iter, options['n_iter']))

for m in range(len(methods)):
	options['method'] = methods[m]
	for fold in range(cv_iter):
		# Class's construction
		bo = BayesianOptimization(lambda x: meno_ackley(x), options)
		# Method's run
		y_best, tot_err, tot_time = bo.maximize()
		y_bests[m, fold] = y_best
		errs[m, fold] = tot_err
		times[m, fold] = tot_time

		del bo, y_best, tot_err, tot_time
