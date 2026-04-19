import numpy as np

def salpeter_imf(m, alpha=2.35):
    """Calculates the weight of a mass based on the Initial Mass Function."""
    return m**(-alpha)

def generate_mock_population(n_stars, alpha, feh_mean):
    """Creates a synthetic population for SFE modeling."""
    mass_grid = np.linspace(0.1, 100, n_stars)
    probs = salpeter_imf(mass_grid, alpha) / salpeter_imf(mass_grid, alpha).sum()
    
    masses = np.random.choice(mass_grid, size=n_stars, p=probs)
    metallicities = np.random.normal(feh_mean, 0.4, n_stars)
    return masses, metallicities