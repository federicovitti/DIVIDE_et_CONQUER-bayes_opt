import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def acq_max(ac, gp, y_max, bounds, random_state, n_bo, B, n_percent, dx):
    # Warm up with random points
    x_tries = np.random.uniform(bounds[0], bounds[1], size=(1000, dx))
    # Evaluate acquision function
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()
    res =  minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_max.reshape(1, -1),
                       bounds=bounds.T,
                       method="L-BFGS-B")
    max_acq = -res.fun
    x_tries = np.vstack((x_tries, x_max))
    ys = np.hstack((ys, max_acq))
    inds = ys.argsort()[::-1]
    thres = np.ceil(n_bo*n_percent).astype(int)
    inds_of_inds = np.hstack((range(thres), np.random.permutation(range(thres,len(inds)))))
    inds = inds[inds_of_inds[:n_bo]]
    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_tries[inds,:], bounds[0], bounds[1])


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa

        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return norm.cdf(z)


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking
    :param a: array to trim repeated rows from
    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]
