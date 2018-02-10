import numpy as np
from bo_my import BayesianOptimization
from functions import meno_hartmann3

dx = 3
bounds = np.matlib.repmat([[0.], [1.]], 1, dx)
options = { 'opt_value': -3.86278,
			'pbounds': bounds,
			'n_iter': 2,
			'B': 10,
			'n_bo': 5,
			'n_percent': 0.7,
			'init_points': 20,
			'acq_fun': 'ei',
			'k': 2.576,
			'xi': 0.05
			}
methods = list(['mondrian'])
methods.append('kmeans')
methods.append('dbscan')
methods.append('hdbscan')

cv_iter = 1

y_bests = np.empty((len(methods), cv_iter, options['n_iter']))
errs = np.empty((len(methods), cv_iter, options['n_iter']))
times = np.empty((len(methods), cv_iter, options['n_iter']))
for m in range(len(methods)):
	options['method'] = methods[m]
	for fold in range(cv_iter):
		bo = BayesianOptimization(lambda x: meno_hartmann3(x), options)
		y_best, tot_err, tot_time = bo.maximize()

		y_bests[m, fold] = y_best
		errs[m, fold] = tot_err
		times[m, fold] = tot_time

		del bo, y_best, tot_err, tot_time

