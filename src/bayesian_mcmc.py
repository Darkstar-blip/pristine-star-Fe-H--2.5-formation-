import numpy as np

def log_prior(theta):
    """Physical constraints for the Bayesian model."""
    sfr_eff, alpha = theta
    if 0.0 < sfr_eff < 1.0 and 1.5 < alpha < 3.5:
        return 0.0
    return -np.inf

def log_likelihood(theta, obs_feh, obs_feh_err):
    """Likelihood function comparing SFE model to observed MDF."""
    sfr_eff, alpha = theta
    # Theoretical prediction: higher efficiency shifts [Fe/H] higher
    model_peak = -3.5 + (sfr_eff * 0.5) 
    sigma2 = obs_feh_err**2 
    return -0.5 * np.sum((obs_feh - model_peak)**2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_probability(theta, obs_feh, obs_feh_err):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta, obs_feh, obs_feh_err)