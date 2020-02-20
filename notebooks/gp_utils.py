"""
utilities for gp regression
"""
from sklearn import gaussian_process as gp
from sklearn.gaussian_process.gpr import *
import numpy as np
import emcee
from scipy import stats


# define custom GP class
class GaussianProcess(gp.GaussianProcessRegressor):
    
    def predict(self, X, kernel=None, return_std=False, return_cov=False):
        """Predict using the GP regression model
        
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        kernel : Kernel object, default is self.kernel
            Kernel object to use as K_11 and K_21 in prediction function.

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        X = check_array(X)

        # check kernel
        if kernel is None:
            if self.kernel is None:
                raise ValueError("kernel argument or self.kernel must not be None")
            else:
                kernel = self.kernel

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            K_trans = kernel(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_train_mean + y_mean  # undo normal.
            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = kernel(X) - K_trans.dot(v)  # Line 6
                return y_mean, y_cov
            elif return_std:
                # cache result of K_inv computation
                if self._K_inv is None:
                    # compute inverse K_inv of K based on its Cholesky
                    # decomposition L and its inverse L_inv
                    L_inv = solve_triangular(self.L_.T,
                                             np.eye(self.L_.shape[0]))
                    self._K_inv = L_inv.dot(L_inv.T)

                # Compute variance of predictive distribution
                y_var = kernel.diag(X)
                y_var -= np.einsum("ij,ij->i",
                                   np.dot(K_trans, self._K_inv), K_trans)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean


class DiagPowerLawKernel(gp.kernels.Kernel):
    """
    A diagonal power-law kernel.

    k(x_1, x_2) = (x_1 / anchorx) ** (1 - beta) if x_1 == x_2 else 0

    Parameters
    ----------
    beta : float, default: 1.0
        Parameter controlling power law amplitude w.r.t.
        dependent axis (i.e. x_train)

    beta_bounds : pair of floats, default (1e-10, 3)
        Hard prior bounds on beta parameter

    anchorx : float, default: 100.0
        Anchoring point along X_train for power law
    """

    def __init__(self, beta=1.0, beta_bounds=(1e-10, 3), anchorx=100.0):
        self.beta = beta
        self.beta_bounds = beta_bounds
        self.anchorx = anchorx

    @property
    def hyperparameter_beta(self):
        return gp.kernels.Hyperparameter(
            "beta", "numeric", self.beta_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.eye(X.shape[0]) * (X[:, 0] / self.anchorx)**(1 - self.beta)
            if eval_gradient:
                if not self.hyperparameter_beta.fixed:
                    return (K, ((1 - self.beta) * X[:, 0]**(-self.beta)\
                               * np.eye(X.shape[0]))[:, :, None])
                else:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                return K
        else:
            return np.zeros((X.shape[0], Y.shape[0]))

    def is_stationary(self):
        """
        Returns whether the kernel is stationary.
        """
        return False

    def diag(self, X):
        """
        Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.___call__(X, Y=None, eval_gradient=False)

    def __repr__(self):
        return "{0}(beta={1:.3g})".format(self.__class__.__name__, self.beta)


# define flag log prior
def flat_log_prior(theta, theta_bounds=None):
    if theta_bounds is None:
        return 0
    in_bounds = [(th >= thb[0]) and (th <= thb[1]) for th, thb in zip(theta, theta_bounds)]
    if np.all(in_bounds):
        return 0
    else:
        return -np.inf

# define posterior probability
def log_prob(theta, GP, theta_bounds=None, log=True):
    lml = GP.log_marginal_likelihood(theta, eval_gradient=False)
    if log:
        theta = np.log(theta)
    return flat_log_prior(theta, theta_bounds=theta_bounds) + lml

# regress for hyperparameters w/ emcee
def emcee_train(start_pos, GP, nstep=100, nwalkers=10, a=1.5, log=False):
    nwalkers = 100
    ndim = len(start_pos)
    pos = start_pos + stats.norm.rvs(0, 1, nwalkers * ndim).reshape(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, a=a, args=(GP,), kwargs={'log':log})
    sampler.run_mcmc(pos, nstep)
    flatchain = sampler.flatchain
    
    return flatchain

