# Bayesian Structural Time Series (BSTS) for Causal Inference

A Python implementation of Bayesian Structural Time Series modeling for causal impact analysis and counterfactual forecasting.

## Overview

This project implements a BSTS model to analyze the causal effect of interventions on time series data. The model uses a local level (random walk) structure to capture the underlying trend and provides uncertainty quantification through Bayesian inference.

## Key Features

- **Bayesian Structural Time Series Model**: Local level model with random walk dynamics
- **MCMC Inference**: Metropolis-Hastings algorithm for parameter estimation
- **Counterfactual Forecasting**: Posterior predictive sampling for what would have happened without intervention
- **Causal Effect Analysis**: Point-wise and cumulative treatment effects with credible intervals
- **Visualization**: Comprehensive plotting of observed data, counterfactuals, and causal effects

## Model Structure

The BSTS model implements a local level specification:

```
State equation:  μₜ = μₜ₋₁ + ξₜ,  ξₜ ~ N(0, σ²ₓᵢ)
Observation:     yₜ = μₜ + εₜ,     εₜ ~ N(0, σ²ₒᵦₛ)
```

**Priors:**
- μ₀ ~ N(50, 10²)
- σₓᵢ ~ HalfNormal(5)
- σₒᵦₓ ~ HalfNormal(5)

## Project Structure

```
├── data_preparation.py      # Synthetic data generation with treatment effect
├── bsts_model.py           # BSTS model implementation and MCMC inference
├── bsts_forecasting.py     # Posterior predictive forecasting
├── forecast_bsts.ipynb     # Main analysis notebook
└── output.png              # Generated visualization
```

## Usage

### Quick Start

1. **Generate synthetic data with treatment effect:**
```python
from data_preparation import generate_data
y_obs, mu_true, sigma_trend_true = generate_data()
```

2. **Fit BSTS model on pre-intervention period:**
```python
from bsts_model import metropolis_random_walk
mu_post, sig_xi_post, sig_obs_post = metropolis_random_walk(
    y_obs[:T_pre], n_iters=12000, step_mu=0.25, step_sig=0.2, burn=4000, thin=5
)
```

3. **Generate counterfactual forecasts:**
```python
from bsts_forecasting import posterior_predictive_forecast
mu_future_samps, y_future_samps = posterior_predictive_forecast(
    mu_post, sig_xi_post, sig_obs_post, h=T_post
)
```

4. **Analyze causal effects:**
```python
# Counterfactual mean and credible intervals
cf_mean = y_future_samps.mean(axis=0)
cf_low = np.percentile(y_future_samps, 5, axis=0)
cf_high = np.percentile(y_future_samps, 95, axis=0)

# Point-wise causal effects
actual_post = y_obs[T_pre:]
point_effect = actual_post - cf_mean

# Cumulative causal effects
cum_effect = np.cumsum(point_effect)
```

### Running the Complete Analysis

Execute the Jupyter notebook `forecast_bsts.ipynb` for a complete end-to-end analysis including:
- Data generation and visualization
- Model fitting and convergence assessment
- Counterfactual forecasting
- Causal effect estimation and visualization

## Data Configuration

**Default Parameters:**
- Total time points: 100
- Pre-intervention period: 70 time points
- Post-intervention period: 30 time points
- Treatment effect: Smooth exponential lift starting at 8.0 units

**Noise Parameters:**
- State noise (σₓᵢ): 1.2
- Observation noise (σₒᵦₛ): 3.0

## Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, halfnorm
```

## Output

The analysis generates three key visualizations:

1. **Observed vs Counterfactual**: Shows the actual time series against the predicted counterfactual with 90% credible intervals
2. **Point-wise Causal Effects**: Period-by-period treatment effects
3. **Cumulative Causal Effects**: Running total of treatment impact over time

## Model Assumptions

- **Structural Stability**: The underlying trend structure remains stable across pre/post periods
- **No Confounding**: The intervention is the only systematic change at the intervention time
- **Local Level Dynamics**: The time series follows a random walk trend

## Applications

This BSTS implementation is suitable for:
- Marketing campaign impact analysis
- Policy intervention assessment
- A/B testing with time series data
- Economic impact studies
- Any scenario requiring counterfactual inference in time series

## References

- Brodersen, K. H., et al. (2015). "Inferring causal impact using Bayesian structural time-series models"
- Scott, S. L., & Varian, H. R. (2014). "Predicting the present with Bayesian structural time series"
