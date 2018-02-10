import numpy as np
from bayesian_optimization import BayesianOptimization
from functions import meno_hartmann3

dx = 3
bounds = np.matlib.repmat([[0.], [1.]], 1, dx)
opt_value = -3.86278
n = 20
cv_iter = 1
n_iter = 2
y_bo = np.empty((1, cv_iter, n_iter))
err_bo = np.empty((1, cv_iter, n_iter))
times_bo = np.empty((1, cv_iter, n_iter))
for fold in range(cv_iter):
	bo = BayesianOptimization(lambda x: meno_hartmann3
	(x), bounds, opt_value)
	y_best, tot_err, tot_time = bo.maximize(init_points=n, n_iter=n_iter, kappa=2)
	y_bo[0, fold] = y_best
	err_bo[0, fold] = tot_err
	times_bo[0, fold] = tot_time
	
	del bo, y_best, tot_err, tot_time

