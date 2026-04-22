#%% Libraries and data loading
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.special import jn, jvp  # Bessel functions of the first kind
from tqdm import tqdm
import os
import warnings

# --- Physical Constants & System Parameters ---
# Local Dark Matter density in natural units
dm_density = 0.3 * 4.09351499910375e+55 
# Nucleon mass conversion factor
m_nucleon = 1.67492729e-27 * 8.52246714808644e+50 
# Dark Matter self-interaction coupling constant
delta_c = 0.1

# === HULSE-TAYLOR (B1913+16) PARAMETERS ===
# Values sourced from Weisberg & Huang (2016)
pb_sec  = 0.322997448918 * 24 * 3600      # Orbital period in seconds
wb      = 2 * np.pi / pb_sec      # Orbital frequency (2*pi/Pb)
a1      = 2.341782      # Projected semi-major axis in light-seconds
w_rad   = 292.54450 * (np.pi / 180)      # Argument of periastron in radians
ecc     = 0.6171340      # Eccentricity
iota    = np.arcsin(0.7327)      # Inclination angle in radians
m1      = 1.438      # Pulsar mass in Solar Masses
m2      = 1.390      # Companion mass in Solar Masses

# --- Timing & Observation Specs ---
toa_precision = 10e-6  # TOA precision (epsilon) in seconds
t_obs   = (2012 - 1981) * 365.25 * 24 * 3600  # Total observation time in seconds
cadence = 9257 / t_obs  # Observation cadence (ndot) in 1/seconds

# --- Coordinate projections ---
# Alpha_b and Eta_b constants
alpha_b = a1 * np.sin(w_rad)
eta_b   = (1 - ecc**2)**0.5 * a1 * np.cos(w_rad)
# Auxiliary epsilon factor (sqrt(1 - e^2))
eps_factor = np.sqrt(1 - ecc**2)

# --- Variance and Sensitivity Constants ---
# Definition of the Q factor for variance
num_Q = toa_precision**2
den_Q = cadence * (ecc**4 + 4 * ecc**2 * (eps_factor - 2) - 8 * eps_factor + 8) * (eta_b**2 - alpha_b**2 * (ecc**2 - 1))**2
Q_factor = num_Q / den_Q

# Variance of the orbital phase perturbation
var_theta = Q_factor * eps_factor * ecc**2 * (
    2 * alpha_b**2 * (ecc**2 - 1) * (ecc**2 * (eps_factor - 2) - 2 * eps_factor + 2)
    + eta_b**2 * (ecc**2 - 2) * (ecc**2 + 2 * eps_factor - 2)
)

# Mass range in eV and conversion to frequency (1/s)
masses_ev = np.logspace(-19.1, -18, num=400)
m_seconds = masses_ev * 1.51926760385984e+15

# Timing model basis functions (fitting components)
def f1(t): return t - t_obs / 2
def f2(t): return t_obs + t * 0
def f0(t): return t**2 / t_obs - t + t_obs / 6

#%% Sensitivity Calculation
warnings.filterwarnings("ignore")

max_ang_freq = np.max(m_seconds)
max_freq_hz = max_ang_freq / (2 * np.pi)
samples_per_cycle = 10
n_points = int(np.ceil(t_obs * (samples_per_cycle * max_freq_hz)))

n_realizations = 10
n_bessel_terms = 30
# n_points = round(t_obs * cadence)
t_vals = np.linspace(0, t_obs, n_points)

# Gram-Schmidt basis normalization
norm0 = np.trapz(f0(t_vals)**2, t_vals)
norm1 = np.trapz(f1(t_vals)**2, t_vals)
norm2 = np.trapz(f2(t_vals)**2, t_vals)

# Storage for results (masses vs realizations)
g_results = np.zeros((len(m_seconds), n_realizations))

# Universal constants
M_SOL = 2e30
G_CONST = 6.67e-11
m_total = m1 + m2 

for r in range(n_realizations):
    np.random.seed(r)
    # Stochastic parameters for the Dark Matter field
    rf         = np.random.rayleigh(1 / np.sqrt(2))
    gammaf     = np.random.uniform(0, 2 * np.pi)
    phif       = np.random.uniform(0, 2 * np.pi) # Ascending node
    varthetaf  = np.random.uniform(-np.pi, np.pi)
    
    # Semi-major axis in natural units
    a_semi = (G_CONST * M_SOL * m_total * 3.71140109219707e-26 / wb**2)**(1/3)

    for i, m_field in enumerate(tqdm(m_seconds, desc=f"Realization {r+1}/{n_realizations}")):
        # --- Bessel Series Expansion ---
        n_idx = np.arange(1, n_bessel_terms)[:, None]
        t_row = t_vals[None, :]
        phase = n_idx * wb * t_row
        
        jn_vals  = jn(n_idx, n_idx * ecc)
        jnp_vals = jvp(n_idx, n_idx * ecc, n=1)

        # Keplerian orbital components
        costheta  = -ecc + (2.0 * (1.0 - ecc**2) / ecc) * np.sum((jn_vals / n_idx) * np.cos(phase), axis=0)
        sintheta  = 2.0 * np.sqrt(1.0 - ecc**2) * np.sum((jnp_vals / n_idx) * np.sin(phase), axis=0)
        r_over_a  = 1.0 + 0.5 * ecc**2 - 2.0 * ecc * np.sum((jnp_vals / n_idx) * np.cos(phase), axis=0)

        # --- Angular constants & field prefactors ---
        cphi, sphi = np.cos(phif), np.sin(phif)
        F0_field = (delta_c / m_nucleon) * np.sqrt(2.0 * dm_density) * rf * np.sin(m_field * t_vals + gammaf)

        # --- Perturbative equations (adot, Omega_dot, varpi_dot) ---
        a_dot = (2.0 / wb) * F0_field * np.sin(varthetaf) * (
            (ecc / np.sqrt(1.0 - ecc**2)) * (cphi * sintheta * costheta + sphi * sintheta**2)
            + (1.0 / (1.0 - ecc**2))      * (cphi * sintheta - sphi * costheta)
            + (ecc / (1.0 - ecc**2))       * (cphi * sintheta * costheta - sphi * costheta**2)
        )

        Omega_dot = (F0_field * np.cos(varthetaf) / (a_semi * wb)) * (
            (sintheta * np.cos(w_rad) + costheta * np.sin(w_rad)) / (np.sin(iota) * np.sqrt(1.0 - ecc**2))
        ) * r_over_a

        varpi_dot = (np.sqrt(1.0 - ecc**2) * F0_field / (a_semi * ecc * wb)) * np.sin(varthetaf) * (
            - (costheta**2) * cphi - (costheta * sintheta) * sphi
            + sintheta * (sphi * costheta - cphi * sintheta) * (1.0 + r_over_a / (1.0 - ecc**2))
        ) + 2.0 * (np.sin(iota / 2.0)**2) * Omega_dot

        epsilon1_dot = -(2.0 / (a_semi * wb)) * r_over_a * F0_field * np.sin(varthetaf) * (costheta * cphi + sintheta * sphi) \
                       + (1.0 - np.sqrt(1.0 - ecc**2)) * varpi_dot \
                       + 2.0 * np.sqrt(1.0 - ecc**2) * (np.sin(iota / 2.0)**2) * Omega_dot

        # --- Integration for the timing signal h(t) ---
        delta_a_over_a = cumtrapz(a_dot / a_semi, t_vals, initial=0.0)
        int_delta_a    = cumtrapz(delta_a_over_a, t_vals, initial=0.0)
        int_eps1       = cumtrapz(epsilon1_dot, t_vals, initial=0.0)
        int_varpi      = cumtrapz(varpi_dot, t_vals, initial=0.0)

        h_signal = -1.5 * wb * int_delta_a + int_eps1 - int_varpi

        # --- Orthogonal projection and SNR calculation ---
        int0_h = np.trapz(f0(t_vals) * h_signal, t_vals)
        int1_h = np.trapz(f1(t_vals) * h_signal, t_vals)
        int2_h = np.trapz(f2(t_vals) * h_signal, t_vals)

        # Residual signal after removing timing model fit
        Gh_signal = h_signal - f1(t_vals) * int1_h / norm1 - f2(t_vals) * int2_h / norm2 -  f0(t_vals) * int0_h / norm0
        u_snr = np.trapz(Gh_signal**2, t_vals) / (2 * var_theta)

        # Threshold for detection (Bayes Factor 1000)
        g_results[i, r] = (np.log(1000) / u_snr)**0.5

#%%
g_results_stack = np.asarray(g_results, dtype=float)
mass_grid = np.asarray(masses_ev, dtype=float)

np.savez("g_results_eccentric_B1913_16.npz", curves=g_results_stack, masses=mass_grid)

#%% Results Visualization
plt.figure(figsize=(8,6), dpi=300)

# Plot individual realizations
for r in range(n_realizations):
    plt.loglog(masses_ev, g_results[:, r], color="gray", alpha=0.3, lw=0.8)

# Plot median sensitivity curve
g_median = np.median(g_results, axis=1)
plt.loglog(masses_ev, g_median, color="black", lw=2.0, label="Median Sensitivity")

plt.xlabel("m [eV]", fontsize=14)
plt.ylabel("g", fontsize=14)
plt.title("Hulse-Taylor Pulsar (B1913+16) Sensitivity")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.show()