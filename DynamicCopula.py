# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 19:41:23 2022

@author: lfsil
"""


import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy import linalg

# =============================================================================
# The functions below were taken from this source: https://www.arpm.co
# =============================================================================


def meancov_sp(x, p=None):
    """

    Parameters
    ----------
        x : array, shape (j_,n_) if n_>1 or (j_,) if n_=1
        p : array, shape (j_,)

    Returns
    -------
        e_x : array, shape (n_,)
        cv_x : array, shape (n_,n_ )

    """

    if p is None:
        j_ = x.shape[0]
        p = np.ones(j_) / j_  # equal probabilities as default value

    e_x = p @ x
    cv_x = ((x-e_x).T*p) @ (x-e_x)
    return e_x, cv_x

def cov_2_corr(s2):
    """

    Parameters
    ----------
        s2 : array, shape (n_, n_)

    Returns
    -------
        c2 : array, shape (n_, n_)
        s_vol : array, shape (n_,)

    """
    # compute standard deviations
    s_vol = np.sqrt(np.diag(s2))

    diag_inv_svol = np.diag(1/s_vol)
    # compute correlation matrix
    c2 = diag_inv_svol@s2@diag_inv_svol
    return c2, s_vol

def fit_dcc_t(dx, p=None, *, rho2=None, param0=None, g=0.99):
    """

    Parameters
    ----------
        dx : array, shape(t_, i_)
        p : array, optional, shape(t_)
        rho2 : array, shape(i_, i_)
        param0 : list or array, shape(2,)
        g : scalar, optional

    Returns
    -------
        params : list, shape(3,)
        r2_t : array, shape(t_, i_, i_)
        epsi : array, shape(t_, i_)
        q2_t_ : array, shape(i_, i_)

    """

    # Step 0: Setup default values

    t_, i_ = dx.shape

    # flexible probabilities
    if p is None:
        p = np.ones(t_) / t_

    # target correlation
    if rho2 is None:
        _, rho2 = meancov_sp(dx, p)
        rho2, _ = cov_2_corr(rho2)

    # initial parameters
    if param0 is None:
        param0 = [0.01, g - 0.01]  # initial parameters

    # Step 1: Compute negative log-likelihood of GARCH

    def llh(params):
        a, b = params
        mu = np.zeros(i_)
        q2_t = rho2.copy()
        r2_t, _ = cov_2_corr(q2_t)
        llh = 0.0
        for t in range(t_):
            llh = llh - p[t] * multivariate_normal.logpdf(dx[t, :], mu, r2_t)
            q2_t = rho2 * (1 - a - b) + \
                a * np.outer(dx[t, :], dx[t, :]) + b * q2_t
            r2_t, _ = cov_2_corr(q2_t)

        return llh

    # Step 2: Minimize the negative log-likelihood

    # parameter boundaries
    bnds = ((1e-20, 1.), (1e-20, 1.))
    # stationary constraints
    cons = {'type': 'ineq', 'fun': lambda param: g - param[0] - param[1]}
    a, b = minimize(llh, param0, bounds=bnds, constraints=cons)['x']

    # Step 3: Compute realized correlations and residuals

    q2_t = rho2.copy()
    r2_t = np.zeros((t_, i_, i_))
    r2_t[0, :, :], _ = cov_2_corr(q2_t)

    for t in range(t_ - 1):
        q2_t = rho2 * (1 - a - b) + \
            a * np.outer(dx[t, :], dx[t, :]) + b * q2_t
        r2_t[t + 1, :, :], _ = cov_2_corr(q2_t)

    l_t = np.linalg.cholesky(r2_t)
    epsi = np.linalg.solve(l_t, dx)

    return [1. - a - b, a, b], r2_t, epsi, q2_t

def twist_scenarios_mom_match(x, m_, s2_, p=None, method='PCA', d=None):
    """
    Parameters
    ----------
        x : array, shape (j_,n_) if n_>1 or (j_,) for n_=1
        m_ : array, shape (n_,)
        s2_ : array, shape (n_,n_)
        p : array, optional, shape (j_,)
        method : string, optional
        d : array, shape (k_, n_), optional
    Returns
    -------
        x : array, shape (j_, n_) if n_>1 or (j_,) for n_=1
    """
    if np.ndim(m_) == 0:
        m_ = np.reshape(m_, 1).copy()
    else:
        m_ = np.array(m_).copy()
    if np.ndim(s2_) == 0:
        s2_ = np.reshape(s2_, (1, 1))
    else:
        s2_ = np.array(s2_).copy()
    if len(x.shape) == 1:
        x = x.reshape(-1, 1).copy()
    if p is None:
        j_ = x.shape[0]
        p = np.ones(j_) / j_  # uniform probabilities as default value
    # Step 1. Original moments
    m_x, s2_x = meancov_sp(x, p)
    # Step 2. Transpose-square-root of s2_x
    r_x = transpose_square_root(s2_x, method, d)
    # Step 3. Transpose-square-root of s2_
    r_ = transpose_square_root(s2_, method, d)
    # Step 4. Twist matrix
    b = r_ @ np.linalg.inv(r_x)
    # Step 5. Shift vector
    a = m_.reshape(-1, 1) - b @ m_x.reshape(-1, 1)
    # Step 6. Twisted scenarios
    x_ = (a + b @ x.T).T
    return np.squeeze(x_)

def pca_cov(sigma2, k_=None):
    """
    Parameters
    ----------
        sigma2 : array, shape (n_,n_)
        k_ : int, optional
    Returns
    -------
        e : array, shape (n_,k_)
        lambda2 : array, shape (k_,)
    """
    n_ = sigma2.shape[0]
    if k_ is None:
        k_ = n_
    lambda2, e = linalg.eigh(sigma2, eigvals=(n_-k_, n_-1))
    lambda2 = lambda2[::-1]
    e = e[:, ::-1]
    # Enforce a sign convention on the coefficients
    # the largest element in each eigenvector will have a positive sign
    ind = np.argmax(abs(e), axis=0)
    ind = np.diag(e[ind, :]) < 0
    e[:, ind] = -e[:, ind]
    return e, lambda2

def gram_schmidt(sigma2, v=None):
    """
    Parameters
    ----------
        sigma2 : array, shape (n_,n_)
        v      : array, shape (n_,n_), optional
    Returns
    -------
        w : array, shape (n_,n_)
    """
    n_ = sigma2.shape[0]
    # Step 0. Initialization
    w = np.empty_like(sigma2)
    p = np.zeros((n_, n_-1))
    if v is None:
        v = np.eye(n_)
     
    for n in range(n_):
        v_n = v[:, [n]]
        for m in range(n):
        # Step 1. Projection
            p[:, [m]] = (w[:, [m]].T @ sigma2 @ v_n) * w[:, [m]]
        # Step 2. Orthogonalization
        u_n = v_n - p[:, :n].sum(axis=1).reshape(-1, 1)
        # Step 3. Normalization
        w[:, [n]] = u_n/np.sqrt(u_n.T @ sigma2 @ u_n)
    return w

def cpca_cov(sigma2, d, old=False):
    """
    Parameters
    ----------
        sigma2 : array, shape (n_,n_)
        d : array, shape (k_,n_)
    Returns
    -------
        lambda2_d : array, shape (n_,)
        e_d : array, shape (n_,n_)
    """
    n_ = sigma2.shape[0]
    k_ = d.shape[0]
    i_n = np.eye(n_)
    lambda2_d = np.empty((n_, 1))
    e_d = np.empty((n_, n_))
    # Step 0. initialize constraints
    m_ = n_ - k_
    a_n = np.copy(d)
    for n in range(n_):
        # Step 1. orthogonal projection matrix
        p_n = i_n-a_n.T@np.linalg.inv(a_n@a_n.T)@a_n
        # Step 2. conditional dispersion matrix
        s2_n = p_n @ sigma2 @ p_n
        # Step 3. conditional principal directions/variances
        e_d[:, [n]], lambda2_d[n] = pca_cov(s2_n, 1)
        # Step 4. Update augmented constraints matrix
        if n+1 <= m_-1:
            a_n = np.concatenate((a_n.T, sigma2 @ e_d[:, [n]]), axis=1).T
        elif m_ <= n+1 <= n_-1:
            a_n = (sigma2 @ e_d[:, :n+1]).T
    return e_d, lambda2_d.squeeze()

def transpose_square_root(sigma2, method='Riccati', d=None, v=None):
    """
    Parameters
    ----------
        sigma2 : array, shape (n_,n_)
        method : string, optional
        d : array, shape (k_,n_), optional
        v : array, shape (n_,n_), optional
    Returns
    -------
        s : array, shape (n_,n_)
    """
    if np.ndim(sigma2) < 2:
        return np.squeeze(np.sqrt(sigma2))
    n_ = sigma2.shape[0]
    if method == 'CPCA' and d is None:
        method = 'PCA'
       
    # Step 1: Riccati
    if method == 'Riccati':
        e, lambda2 = pca_cov(sigma2)
        s = e * np.sqrt(lambda2) @ e.T
       
    # Step 2: Conditional principal components
    elif method == 'CPCA':
        e_d, lambda2_d = cpca_cov(sigma2, d)
        s = e_d * np.sqrt(lambda2_d)
    # Step 3: Principal components
    elif method == 'PCA':
        e, lambda2 = pca_cov(sigma2)
        s = e * np.sqrt(lambda2) @ e.T
    # Step 4: Gram-Schmidt
    elif method == 'Gram-Schmidt':
        g = gram_schmidt(sigma2, v)
        s = np.linalg.inv(g).T
    # Step 5: Cholesky
    elif method == 'Cholesky':
        s = np.linalg.cholesky(sigma2)
    return s
