import numpy as np
from bayesian_optimization import BayesianOptimization
from functions import meno_ackley

# search space's dimesnion
dx = 2
# search space's definition
bounds = np.matlib.repmat([[-32.768], [32.768]], 1, dx)
# number of initial points
n = 20
# real optimum value
opt_value = 0.
# number of validation's iterations
cv_iter = 10
# number of method's iterations
n_iter = 20
# containers
y_bo = np.empty((1, cv_iter, n_iter))
err_bo = np.empty((1, cv_iter, n_iter))
times_bo = np.empty((1, cv_iter, n_iter))

# validations
for fold in range(cv_iter):
	# class's constraction
	bo = BayesianOptimization(lambda x: meno_ackley(x), bounds, opt_value)
	# method's run
	y_best, tot_err, tot_time = bo.maximize(init_points=n, n_iter=n_iter, kappa=2)
	y_bo[0, fold] = y_best
	err_bo[0, fold] = tot_err
	times_bo[0, fold] = tot_time
	
	del bo, y_best, tot_err, tot_time



