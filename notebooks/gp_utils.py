"""
utilities for gp regression
"""
from sklearn import gaussian_process as gp
from sklearn.gaussian_process.gpr import check_X_y, check_array, cholesky, cho_solve, check_random_state, itemgetter
import numpy as np
import emcee
from scipy import stats, special
import functools
import operator
import uvtools as uvt
from scipy import stats, optimize


def setup_gp(kernels, kdict, optimizer='fmin_l_bfgs_b', n_restarts=10, norm_y=False, alpha=1e-10):
    """setup GP
    
    Args:
    kernels : list of str, kernel names found in kdict to include in model
    kdict : dict, keys kernel names and values GP kernels

    Returns:
    G : GaussianProcess model
    Knames : list of kernel names in G.kernel_.get_params() in the order fed by kernels
    theta_labels : list of latex labels
    theta_bounds : list of hyperparameter theta bounds determined by input kdict
    """
    # setup G
    kerns = [kdict[k] for k in kernels]
    kernel = functools.reduce(operator.add, kerns)
    G = GaussianProcess(kernel=kernel, optimizer=optimizer, n_restarts_optimizer=n_restarts,
                        copy_X_train=False, normalize_y=norm_y, alpha=alpha)
   
    # get Knames
    root = ''
    Knames = []
    Nk = len(kernels)
    for i in range(Nk):
        if i == Nk - 1:
            Knames.append(root[:-2])
        else:
            Knames.append(root + 'k2')
            root += 'k1__'
    Knames = Knames[::-1]

    # get theta bounds
    theta_labels = []
    theta_bounds = []
    for kern in kerns:
        theta_bounds.extend([tuple(b) for b in kern.bounds])
        theta_labels.extend(kern.labels)

    return G, Knames, theta_labels, theta_bounds


def get_kernel(kernel, k):
    """
    Given composite sklearn kernel, isolate and return
    a specific component given by k, conforming to Knames
    output of setup_gp

    Parameters
    ----------
    kernel : sklearn kernel object
    k : str, component of kernel to return
        Ex. 'k1__k1__k2' means kernel.k1.k1.k2
    
    Returns
    -------
    sklearn kernel object
    """
    if '__' in k:
        k_opts = k.split('__')
        return get_kernel(getattr(kernel, k_opts[0]), '__'.join(k_opts[1:])) 
    else:
        return getattr(kernel, k)

def gp_fit_predict(data, G, freqs, Kfg_name=None, Keor_name=None):
    """given data and GP, train and predict for FG term
    
    Args:
    data : ndarray
    G : GaussianProcess object
    freqs : frequency array [Hz]
    Kfg_name : name of FG kernel in G.kernel_.get_params()
    Keor_name : name of EoR kernel in G.kernel_.get_params()

    Returns:

    """
    theta_ml = {}
    fg, eor = {}, {}
    fg_cov, eor_cov = {}, {}

    # stack real and imag
    Ntimes, Nfreqs = data.shape
    ydata = G.prep_ydata(freqs[:, None] / 1e6, data.T)

    if G.optimizer is not None:
        # re-optimize kernel_
        G.fit(freqs[:, None] / 1e6, ydata)
        kernel = G.kernel_
    else:
        kernel = G.kernel

    params = kernel.get_params()

    # get FG conditional distribution given trained kernel
    if Kfg_name in params:
        ypred, ycov = G.predict(freqs / 1e6, kernel=params[Kfg_name], return_cov=True)
        ypred = ypred[:, :Ntimes] + 1j * ypred[:, Ntimes:]
        fg = ypred.T
        fg_cov = ycov
    else:
        fg = np.zeros_like(data[k])
        fg_cov = np.eye(Nfreqs)

    # get EoR conditional distribution given trained kernel
    if Keor_name in params:
        ypred, ycov = G.predict(freqs, kernel=params[Keor_name], return_cov=True)
        ypred = ypred[:, :Ntimes] + 1j * ypred[:, Ntimes:]
        eor = ypred.T
        eor_cov = ycov
    else:
        eor = np.zeros_like(data[k])
        eor_cov = np.eye(Nfreqs) 

    return kernel.theta, fg, fg_cov, eor, eor_cov 


# define custom GP class
class GaussianProcess(gp.GaussianProcessRegressor):
    """
    Adapted from sklearn version 0.20.3
    """

    def fit(self, X, y, init_theta=None):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values
        init_theta : array-like, shape = (n_samples,)
            Initial theta. Default is self.kernel.theta
        Returns
        -------
        self : returns an instance of self.
        """
        if init_theta is not None:
            self.kernel_ = self.kernel.clone_with_theta(init_theta)
        else:
            self.kernel_ = self.kernel.clone_with_theta(self.kernel.theta)

        self._rng = check_random_state(self.random_state)

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self

    def predict(self, X, kernel=None, ydata=None, return_std=False, return_cov=False):
        """Predict using the GP regression model
        
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        kernel : Kernel object, default is self.kernel
            Kernel object to use as K_11 and K_21 in prediction function.

        ydata : array-like, shape = (n_samples, n_output_dimms)
            Target-values to condition on. If None use pre-computed
            self.alpha_ which comes from  self.y_train

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
        if X.ndim == 1:
            X = X[:, None]
        X = check_array(X)

        # check kernel
        if kernel is None:
            if self.kernel is None:
                raise ValueError("kernel argument or self.kernel must not be None")
            else:
                kernel = self.kernel

        # check y
        if ydata is not None:
            self.prep_ydata(X, ydata)

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

    def prep_ydata(self, X, ydata):
        """Take complex waterfall and prepare fit matrices
       
        Note: because real and imag components of ydata are
        separated as distinct features, the fitted covariance
        will be a factor of 2 lower than the covariance of
        the full complex data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        ydata : array-like, shape = (n_samples, n_output_dims)
            Target values to condition on

        Returns
        -------
        array-like (n_samples, n_output_dims*2)
        """
        # stack ydata
        y = np.hstack([ydata.real, ydata.imag])

        # Precompute quantities required for predictions which are independent
        # of actual query points
        if X.ndim == 1:
            X = X[:, None]
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # de-mean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        # precompute Kernel matrix inversions
        if hasattr(self, 'kernel_'):
            kernel = self.kernel_
        else:
            kernel = self.kernel
        K = kernel(X)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), y)  # Line 3

        return y + self._y_train_mean

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.
        A log prior is added if self.theta_priors exists.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel = self.kernel.clone_with_theta(theta)

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        # Compute log-prior and add
        log_likelihood += self.log_prior(theta)

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def log_prior(self, theta):
        """
        Compute log prior if self.theta_priors list exists.

        self.theta_priors is a list of callable of len(theta),
        each of which indexes the corresponding element in theta
        and returns its log prior

        Parameters
        ----------
        theta : array-like, shape (n_kernel_params,)
        """
        log_prior = np.zeros_like(theta, dtype=np.float)
        if hasattr(self, 'theta_priors') and self.theta_priors is not None:
            for i, pr in enumerate(self.theta_priors):
                if pr is not None:
                    log_prior[i] += pr(theta[i])
        return np.sum(log_prior, axis=0)

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

# define priors
def log_gamma(a, b, unlog=True):
    """
    Returns a callable log gamma function hard-coded
    with parameters a, b.

    Parameters
    ----------
    a, b : float, gamma function parameters
    unlog : bool, if True, "unlog" input as np.e**x
    """
    def func(x, a=a, b=b, unlog=unlog):
        if unlog:
            x = np.e**x
        return np.log(b**a*x**(a-1)*np.exp(-b*x)/special.gamma(a))
    return gam

def log_gauss(mean, ell, unlog=True):
    """
    Returns a callable log gaussian function hard-coded
    with parameters mean and ell.

    Parameters
    ---------
    mean, ell : float, mean and lengthscale parameter
    unlog : bool, if True, "unlog" input as np.e**x
    """
    def func(x, mean=mean, ell=ell, unlog=unlog):
        if unlog:
            x = np.e**x
        return np.log(np.exp(-0.5*((x-mean)/ell)**2))
    return func

def log_uniform(a, b, unlog=True):
    """
    Returns a callable log uniform function hard-coded
    with parameters a, b.

    Parameters
    ----------
    a,b : float, upper and lower limit in uniform
    unlog : bool, if True, "unlog" input as np.e**x
    """
    def func(x, a=a, b=b, unlog=unlog):
        if unlog:
            x = np.e**x
        return np.log(((x >= a) & (x <= b)).astype(np.float))
    return func

def flat_log_prior(theta, theta_bounds=None):
    if theta_bounds is None:
        return 0
    in_bounds = [(th >= thb[0]) and (th <= thb[1]) for th, thb in zip(theta, theta_bounds)]
    if not np.all(in_bounds) or not np.all(np.isfinite(theta)):
        return -np.inf
    else:
        return 0

# define posterior probability
def log_prob(theta, GP, theta_bounds=None, unlogged=True, return_grad=False, prepend=None):
    """
    Compute log posterior. A prior on theta is computed if GP.theta_priors exits.
    See GP.log_marginal_likelihood for details

    Parameters
    ----------
    theta : 1d numpy array
        Vector of parameter values
    GP : GaussianProcess object
    theta_bounds : list of len-2 tuples
        Hard bounds on parameter prior, enacted through uniform prior
    unlogged : bool
        If True, theta and theta_bounds are taken to be in real space,
        in which case they are logged before passing to GP.log_marginal_like()
    return_grad : bool
        If True, return gradient
    prepend : float or list of float
        List of parameters to prepend to theta before passing to GP.
        theta bounds for parameters in prepend should already be present
        in theta_bounds input.

    Returns
    -------
    float
        log marginal likelihood: log(like) + log(prior)
    float (if return_grad is True)
        log marginal likelihood gradient
    """
    # prepend a parameter if desired
    if prepend is not None:
        theta = np.append(prepend, theta)
    # enact hard bounds if theta_bounds is fed
    pr = flat_log_prior(theta, theta_bounds=theta_bounds)
    if not np.isfinite(pr):
        return -np.inf
    if unlogged:
        theta = np.log(theta)
    if return_grad:
        lml, grad = GP.log_marginal_likelihood(theta, eval_gradient=True)
        return lml + pr, grad
    else:
        lml = GP.log_marginal_likelihood(theta, eval_gradient=False)
        return lml + pr

# explore for hyperparameters w/ emcee
def emcee_run(start_pos, GP, nstep=100, nwalkers=10, unlogged=False, theta_bounds=None,
                prepend=None, moves=None):
    """
    Run emcee EnsembleSampler

    Parameters
    ----------
    start_pos : ndarray, shape (Nwalkers, Nparams)
        Starting position of walkers across parameter space
    GP : GaussianProcess object
    nstep : int
        Number of steps to take. No burn-in. You can truncate the chains
        yourself if burn-in is a concern.
    nwalkers : int
        Number of walkers
    unlogged : bool
        If True, start_pos and theta_bounds are the not in log form,
        which is to say, the parameter space is explored in the unlogged form.
        The GP object only takes parameters in logged form.
        If False, then the parameter space is assumed to be already log-projected.
    theta_bounds : list of len-2 tuples
        Hard bounds on parameter space prior.
    prepend : float or list of float
        Parameter(s) that are not explored by emcee but are necessary when
        the log-likelihood is evaluated by the GP. These parameter(s) are
        prepended to the theta_step vector during the likelihood call.
        Parameters fed as prepend should already be have their bounds set
        in the input theta_bounds array
    moves : emcee move to use
        See emcee.moves for details.

    Returns
    -------
    emcee.EnsembleSampler object
    """
    nwalkers, ndim = start_pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(GP,), moves=moves,
            kwargs={'unlogged':unlogged, 'theta_bounds':theta_bounds, 'prepend':prepend})
    sampler.run_mcmc(start_pos, nstep)
    
    return sampler

def plot_pspec(x, y, ax, comp='real', yerr=None, **kwargs):
    if comp in ['abs', 'abs-real']:
        if comp == 'abs':
            ax.errorbar(x, np.abs(y), yerr=yerr, **kwargs)
        elif comp == 'abs-real':
            ax.errorbar(x, np.abs(np.real(y)), yerr=yerr, **kwargs)
    elif comp == 'real':
        pos = y >= 0.0
        _yerr = None
        if yerr is not None:
            _yerr = yerr[pos]
        ax.errorbar(x[pos], y[pos], yerr=_yerr, **kwargs)
        neg = y <= 0.0
        if yerr is not None:
            _yerr = yerr[neg]
        if 'label' in kwargs:
            kwargs.pop('label')
        ax.errorbar(x[neg], np.abs(y[neg]), yerr=_yerr, markerfacecolor='None', **kwargs)

def get_cov(data):
    cov = np.cov(data)
    for i in range(cov.shape[0]):
        cov[i] = np.roll(np.real(cov[i]), -i, axis=-1)
    cov = np.mean(cov, axis=0)
    cov /= np.abs(cov[0])
    cov[cov < 0] = 0
    return cov

def cov2pspec(K, scalar, window='bh7'):
    """convert a covariance matrix to a power spectrum
    
    Parameters
    ----------
    K : ndarray, square covariance matrix
    window : str, FFT tapering function
    scalar : float, normalization factor

    Returns
    -------
    ndarray : bandpowers
    """
    # get FFT operator
    nfreqs = len(K)
    q = np.fft.ifft(np.eye(K.shape[0]), axis=1) * uvt.dspec.gen_window(window, nfreqs)[None, :]
    # form pspec
    pspec = np.fft.fftshift(np.array([q[i].T.conj().dot(K).dot(q[i]) for i in range(len(q))])) * scalar
    return pspec

def draw_from_cov(cov, Nsamples=1):
    nfreqs = len(cov)
    real = stats.multivariate_normal.rvs(np.zeros(nfreqs), cov, Nsamples)
    imag = stats.multivariate_normal.rvs(np.zeros(nfreqs), cov, Nsamples)
    return real + 1j * imag


