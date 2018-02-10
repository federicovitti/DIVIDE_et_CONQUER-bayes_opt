import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from helper_bo import UtilityFunction, unique_rows, acq_max
import time


class BayesianOptimization(object):

    def __init__(self, f, pbounds, opt_value, random_state=None):
        """
        :param f:
            Function to be maximized.

        :param pbounds:
            Array with minimum and maximum values for each dimensions' range.

        :param opt_value:
            Function's real optimum value.

        """
        self.opt_value = opt_value
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        self.bounds = pbounds
        self.dx = self.bounds.shape[1]
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


    def __call__(self, x):
        """
        Method to get function's evaluations.

        :param x:
            Point where the function needs to be evaluated.
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

        # Evaluate target function at all initialization points
        self.X = np.vstack((self.X, self.init_points))
        for y in range(init_points):
            self.Y = np.append(self.Y, self.f(self.X[y]))

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """
        Bayesian Optimization method.

        :param init_points:
            Number of random points to probe.
        :param n_iter:
            Number of iterations.
        :param acq:
            Acquisition function to use.
        :param kappa:
            Matérn kernel's parameter.
        :param xi:
            Noise for supporting exploration.
        :param **gp_params:
            Parameters to build Gaussian Process.
        """

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Initialization
        self.init(init_points)

        y_max = self.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)

        # Build Gaussian Process
        self.gp.fit(self.X[ur], self.Y[ur])

        # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds,
                        random_state=self.random_state,
                        dx = self.dx)

        # Placeholders
        simple_regret = np.empty(n_iter)
        y_best = np.empty(n_iter)
        tot_time = np.empty(n_iter)

        # Start iterations
        for i in range(n_iter):
            # Set initial computaitonal time
            init_time = time.time()

            pwarning = False
            if np.any((self.X - x_max).sum(axis=1) == 0):

                x_max = self.random_state.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])

                pwarning = True

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds,
                        random_state=self.random_state,
                        dx = self.dx)

            # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(x_max))

            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]

            # Update placeholders
            y_best[i] = -y_max
            simple_regret[i] = -y_max - self.opt_value
            if i == 0: tot_time[i] = time.time() - init_time
            else: tot_time[i] = tot_time[i-1] + time.time()-init_time

        return y_best, simple_regret, tot_time
    