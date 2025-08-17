import numpy as np


def posterior_predictive_forecast(mu_samples, sig_xi_samples, sig_obs_samples, h):
    n_draws, T = mu_samples.shape
    muT = mu_samples[:, -1]
    mu_future = np.zeros((n_draws, h))
    y_future = np.zeros((n_draws, h))
    for i in range(n_draws):
        cur = muT[i]
        sx = sig_xi_samples[i]
        so = sig_obs_samples[i]
        for t in range(h):
            cur = cur + np.random.normal(0.0, sx)   # evolve latent trend
            mu_future[i, t] = cur
            y_future[i, t] = cur + np.random.normal(0.0, so)  # observation
    return mu_future, y_future
