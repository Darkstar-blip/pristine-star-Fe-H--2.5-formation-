import numpy as np
import importlib
import os
from src.bayesian_mcmc import log_probability

emcee = None
try:
    emcee = importlib.import_module("emcee")
except ImportError:
    pass

def run_pipeline():
    print("--- Initializing Galactic Research Pipeline ---")
    
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # 1. Observed Data (Mocked for Sculptor Dwarf Galaxy)
    # Mean [Fe/H] = -3.2, with observational error
    obs_feh = np.random.normal(-3.2, 0.2, 50) 
    obs_feh_err = np.full(50, 0.1)

    # 2. MCMC Setup: [SFR Efficiency, IMF Slope]
    nwalkers, ndim = 32, 2
    initial_guess = [0.1, 2.35] 
    pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

    # 3. Execution
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(obs_feh, obs_feh_err))
    
    print("Running MCMC Inference...")
    sampler.run_mcmc(pos, 1000, progress=True)

    # 4. Save for Analysis
    samples = sampler.get_chain(discard=100, flat=True)
    np.save('data/mcmc_samples.npy', samples)
    print("Success. MCMC results saved to data/mcmc_samples.npy")

if __name__ == "__main__":
    run_pipeline()