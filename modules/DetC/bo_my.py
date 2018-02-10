import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from helper_my import UtilityFunction, unique_rows, acq_max
import time

class BayesianOptimization(object):

    def __init__(self, f, options, random_state=None):
        """
        :param f:
            Function to be maximized.

        :param options:
            Dictionary with parameters names as keys and their values.

        """
        self.options = options
        self.opt_value = options['opt_value']
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        
        self.method = options['method']
        self.bounds = options['pbounds']
        self.dx = self.bounds.shape[1]

        # Some function to be optimized
        self.f = f

        # Initialization lists --- stores starting points before process begins
        self.init_points = []

        # Numpy array place holders
        self.X = None
        self.Y = None

        # Counter of iterations
        self.i = 0

        # Internal GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            random_state=self.random_state
        )
        # Utility Function placeholder
        self.util = None

        self.n_iter = options['n_iter']
        self.B = options['B']
        self.n_bo = options['n_bo']
        self.n_percent = options['n_percent']
        self.init(options['init_points'])
        self.acq = options['acq_fun']


    def __call__(self, x):
        """
        :param x:
            Point the function needs to be evaluated.
        """
        return self.f(x)


    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """
        # Generate random points
        self.init_points  = self.random_state.uniform(self.bounds[0], self.bounds[1], size=(init_points,self.dx))
        
        # Create empty arrays to store the new points and values of the function.
        self.X = np.empty((0, self.dx))
        self.Y = np.empty(0)
        
        # Evaluate target function at all initialization
        self.X = np.vstack((self.X, self.init_points))
        for y in range(init_points):
            self.Y = np.append(self.Y, self.f(self.X[y]))


    def maximize(self, **gp_params):

        if self.method == 'mondrian':
          from mondrian_classic import Decomposition
        elif self.method == 'dbscan':
          from dbscan import Decomposition
        elif self.method == 'kmeans':
          from kmeans import Decomposition
        else:
          from HDBSCAN import Decomposition

        # Set acquisition function
        self.util = UtilityFunction(kind = self.acq, kappa = self.options['k'], xi = self.options['xi'])
        
        # Initialize x, y and find current y_max
        y_best = self.Y.max()
        x_best = self.X[self.Y.argmax()]
        
        # Set parameters if any was passed
        self.gp.set_params(**gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # Finding argmax of the acquisition function.
        x_new = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_best,
                        bounds=self.bounds,
                        random_state=self.random_state,
                        n_bo = self.n_bo,
                        B = self.B,
                        n_percent = self.n_percent,
                        dx = self.dx)

        # Update optimum value found
        x_max = x_new[0]
        if self.f(x_max) > y_best:
            y_best = self.f(x_max)
            x_best = x_max

        # Placeholders
        simple_regret = np.empty(self.n_iter)
        y_bests = np.empty(self.n_iter)
        tot_time = np.empty(self.n_iter)

        # Start iterations
        for i in range(self.n_iter):
            # Set initial time
            init_time = time.time()
            newX = np.empty((0, self.dx))
            newY = np.empty(0)
            tree = Decomposition(self.X, self.Y, self.bounds)

            # builds tree's leaves
            leaves1 = tree.building()
            tot_eval = np.ceil(2.0*self.B)
            tot_volumn = np.array([n.volumn for n in leaves1]).sum()

            leaves = [e for e in leaves1 if e.X.shape[0] != 0]
            for l in leaves:
                # Define l-th partition data and range
                X = l.X
                Y = l.y
                bounds = l.x_range
                self.n_bo = np.maximum(self.n_bo, np.ceil((tot_eval*l.volumn/tot_volumn)).astype(int))

                # Maximize acquisition function to find next probing point
                ur = unique_rows(X)
                self.gp.fit(X[ur], Y[ur])
                x_new = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_best,
                            bounds=bounds,
                            random_state=self.random_state,
                            n_bo = self.n_bo,
                            B=self.B,
                            n_percent = self.n_percent,
                            dx = self.dx)

                x_max = x_new[0]
                newX = np.vstack((newX, x_new))
                for k in x_new:
                    newY = np.append(newY, self.f(k))

            index = newY.argsort()[::-1][:self.B]
            newX = newX[index]
            newY = newY[index]

            # Update optimum value found
            if newY[0] > y_best:
                y_best = newY[0]
                x_best = newX[0]

            self.X = np.vstack((self.X, newX))
            self.Y = np.hstack((self.Y, newY))

            simple_regret[i] = -y_best - self.opt_value
            y_bests[i] = y_best

            if i == 0: tot_time[i] = time.time() - init_time
            else: tot_time[i] = tot_time[i-1] + time.time()-init_time
        
        return -y_bests, simple_regret, tot_time


 