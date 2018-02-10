import numpy as np
from ebo_CV import ebo
import test_functions as tf
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)


##################### define branin function ######################
dx = 6
x_range = np.matlib.repmat([[0.], [1.]], 1, dx)
f = tf.sampled_hartmann6_func(x_range, dx)
##################################################################

options = { 'x_range':x_range, 
            'dx':x_range.shape[1],
            'f.min': f.fmax,
            'max_value':None, 
            'T':20, # number of iterations
            'B':10, # number of candidates to be evaluated
            'dim_limit':6, 
            'isplot':0,
            'z':None, 'k':None,
            'alpha':1.,
            'beta':np.array([5.,2.]),
            'opt_n':100, # points randomly sampled to start continuous optimization of acfun
            'pid':'test3',
            'datadir':'tmp_data3/',
            'gibbs_iter':20,
            'useAzure':False,
            'n_add':None, # this should always be None. it makes dim_limit complicated if not None.
            'nlayers': 100,
            'gp_type':'l1', # other choices are l1, sk, sf, dk, df
            'gp_sigma':0.1, # noise standard deviation
            'n_bo':10, # min number of points selected for each partition
            'n_bo_top_percent': 0.7, # percentage of top in bo selections
            'n_top':10, # how many points to look ahead when doing choose Xnew
            'min_leaf_size':20,
            'max_n_leaves':50,
            'func_cheap':True, # if func cheap, we do not use Azure to test functions
            'thresAzure':1, # if > thresAzure, we use Azure
            'save_file_name': 'plotdata/7.pk',
            'heuristic':False
            }

n = 20

cv_iter = 10
y_EBO = np.empty((1, cv_iter, options['T']))
err_EBO = np.empty((1, cv_iter, options['T']))
times_EBO = np.empty((1, cv_iter, options['T']))
for fold in range(cv_iter):
  X = np.random.uniform(x_range[0], x_range[1], size=(n,dx))
  y = np.empty((n,1))
  for i in range(y.shape[0]):
        y[i] = f(X[i].T)
  options['X'] = X
  options['y'] = y
  # run ebo
  e = ebo(f, options)
  y_best, tot_err, tot_time = e.run()
  y_EBO[0, fold] = y_best
  err_EBO[0, fold] = tot_err
  times_EBO[0, fold] = tot_time

  del e, y_best, tot_err, tot_time
