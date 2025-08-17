import numpy as np
from scipy.stats import norm, halfnorm


#    Model: log posterior
#    Local-level (random walk) BSTS:
#    mu_t = mu_{t-1} + xi_t,  xi_t ~ N(0, sigma_xi^2)
#    y_t  = mu_t + eps_t,     eps_t ~ N(0, sigma_obs^2)
#    Priors:
#    mu_0 ~ N(50, 10^2); sigma_xi ~ HalfNormal(5); sigma_obs ~ HalfNormal(5)
def log_prior(mu, sigma_xi, sigma_obs):
    if sigma_xi <= 0 or sigma_obs <= 0:
        return -np.inf
    lp = norm.logpdf(mu[0], loc=50.0, scale=10.0)
    lp += halfnorm.logpdf(sigma_xi, scale=5.0)
    lp += halfnorm.logpdf(sigma_obs, scale=5.0)
    # random walk increments
    diffs = mu[1:] - mu[:-1]
    lp += np.sum(norm.logpdf(diffs, loc=0.0, scale=sigma_xi))
    return lp


def log_likelihood(mu, sigma_obs, y):
    return np.sum(norm.logpdf(y, loc=mu, scale=sigma_obs))


def log_posterior(mu, sigma_xi, sigma_obs, y):
    return log_prior(mu, sigma_xi, sigma_obs) + log_likelihood(mu, sigma_obs, y)

# MCMC: Metropolis-Hastings on pre-period


def metropolis_random_walk(y, n_iters=12000, step_mu=0.25, step_sig=0.2, burn=4000, thin=5):
    T = len(y)
    # initialize at data for states, small sigmas
    mu = y.copy()
    sigma_xi = 1.0
    sigma_obs = 1.0
    logp = log_posterior(mu, sigma_xi, sigma_obs, y)

    mu_chain = np.zeros((n_iters, T))
    sig_xi_chain = np.zeros(n_iters)
    sig_obs_chain = np.zeros(n_iters)

    for it in range(n_iters):
        # propose mu (joint RW on entire state vector)
        mu_prop = mu + np.random.normal(0.0, step_mu, size=T)
        logp_prop = log_posterior(mu_prop, sigma_xi, sigma_obs, y)
        if np.log(np.random.rand()) < (logp_prop - logp):
            mu = mu_prop
            logp = logp_prop

        # propose sigma_xi
        sigma_xi_prop = abs(sigma_xi + np.random.normal(0.0, step_sig))
        logp_prop = log_posterior(mu, sigma_xi_prop, sigma_obs, y)
        if np.log(np.random.rand()) < (logp_prop - logp):
            sigma_xi = sigma_xi_prop
            logp = logp_prop

        # propose sigma_obs
        sigma_obs_prop = abs(sigma_obs + np.random.normal(0.0, step_sig))
        logp_prop = log_posterior(mu, sigma_xi, sigma_obs_prop, y)
        if np.log(np.random.rand()) < (logp_prop - logp):
            sigma_obs = sigma_obs_prop
            logp = logp_prop

        mu_chain[it] = mu
        sig_xi_chain[it] = sigma_xi
        sig_obs_chain[it] = sigma_obs

    # burn-in & thin
    mu_post = mu_chain[burn::thin]
    sig_xi_post = sig_xi_chain[burn::thin]
    sig_obs_post = sig_obs_chain[burn::thin]
    return mu_post, sig_xi_post, sig_obs_post
