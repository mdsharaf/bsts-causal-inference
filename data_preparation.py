import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T_total = 100          # total time steps
T_pre = 70             # pre-intervention length
T_post = T_total - T_pre  # post-intervention length
sigma_xi_true = 1.2    # state (trend) noise
sigma_obs_true = 3.0   # observation noise


def generate_data():
    # Latent trend: random walk
    mu_true = np.zeros(T_total)
    mu_true[0] = 50.0
    for t in range(1, T_total):
        mu_true[t] = mu_true[t-1] + np.random.normal(0.0, sigma_xi_true)

    # Observations without treatment
    y_base = mu_true + np.random.normal(0.0, sigma_obs_true, size=T_total)

    # Treatment effect after T_pre (smooth lift)
    treatment_effect = np.zeros(T_total)
    lift_start = 8.0
    for t in range(T_pre, T_total):
        treatment_effect[t] = lift_start * (1 - np.exp(-(t - T_pre) / 10.0))

    y_obs = y_base.copy()
    y_obs[T_pre:] += treatment_effect[T_pre:]

    return y_obs, mu_true, sigma_xi_true


def plot_data(y, mu_true):
    plt.plot(y, label="Observed y")
    plt.plot(mu_true, label="True latent mu", linestyle="--")
    plt.legend()
    plt.title("Synthetic Time Series (Local Level)")
    plt.show()


# y, mu_true, sigma_trend_true = generate_data()
# plot_data(y, mu_true)
