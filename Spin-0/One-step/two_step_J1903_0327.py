import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import quad
from scipy.special import jn, jvp
from tqdm import tqdm
import warnings

# --- Global Configuration ---
warnings.filterwarnings("ignore")

# --- System Parameters: J1903+0327 ---
PB_DAYS = 95.2
pb_sec = PB_DAYS * 24 * 3600             # Orbital period in seconds
wb = 2 * np.pi / pb_sec                  # Orbital angular frequency
a1_ls = 105.6                            # Projected semi-major axis in lt-sec
w_deg = 141.7
w_rad = np.deg2rad(w_deg)                # Argument of periastron in radians
ecc = 0.437                              # Eccentricity
t0_mjd = 57109.4                         # Time of periastron (MJD)
t0_sec = t0_mjd * 24 * 3600              # Time of periastron in seconds
epsilon_toa = 6.24e-6                    # TOA precision (seconds)
tobs_yrs = 10.7
tobs_sec = tobs_yrs * 365 * 24 * 3600    # Observation time in seconds
cadence_yr = 638.8
ndot = cadence_yr / (365 * 24 * 3600)    # Cadence in 1/seconds

# Local Dark Matter density in natural units (GeV/cm^3 to 1/s^4)
dm_density = 0.3 * 4.09351499910375e+55 

# --- Geometric Constants ---
ab = a1_ls * np.sin(w_rad)               # alpha_b
nb = np.sqrt(1 - ecc**2) * a1_ls * np.cos(w_rad) # eta_b
eps_factor = np.sqrt(1 - ecc**2)         # sqrt(1 - e^2)

# Definition of Q factor (noise scaling)
q_num = epsilon_toa**2
q_den = ndot * (ecc**4 + 4 * ecc**2 * (eps_factor - 2) - 8 * eps_factor + 8) * \
        (nb**2 - ab**2 * (ecc**2 - 1))**2
q_factor = q_num / q_den

# Variance of the anomaly variable (delta Theta')
var_theta = q_factor * eps_factor * ecc**2 * (
    2 * ab**2 * (ecc**2 - 1) * (ecc**2 * (eps_factor - 2) - 2 * eps_factor + 2)
    + nb**2 * (ecc**2 - 2) * (ecc**2 + 2 * eps_factor - 2)
)

# --- Mass and Frequency Grids ---
masses_ev = np.logspace(-22, -20, num=400)
m_seconds = masses_ev * 1.51926760385984e+15 # Mass in 1/s

# Basis functions for timing model subtraction (GS orthogonalization)
def f1(t): return t - tobs_sec / 2
def f2(t): return tobs_sec + t * 0
def f0(t): return t**2 / tobs_sec - t + tobs_sec/6

# Simulation Settings
n_realizations = 10
n_terms = 20                             # Number of Bessel terms
n_t = int(round(tobs_sec * ndot))
t_vals = np.linspace(0, tobs_sec, n_t)

# Precompute basis norms
norm1 = np.trapz(f1(t_vals)**2, t_vals)
norm2 = np.trapz(f2(t_vals)**2, t_vals)

# Store results: [masses, realizations]
betas_all = np.zeros((len(m_seconds), n_realizations))

# --- Main Monte Carlo Loop ---
for r in range(n_realizations):
    np.random.seed(r)
    rf = np.random.rayleigh(1 / np.sqrt(2))
    gammaf = np.random.uniform(0, 2 * np.pi)

    for i, m in enumerate(tqdm(m_seconds, desc=f"Realization {r+1}/{n_realizations}")):
        phi0 = np.sqrt(2 * dm_density) * rf / m
        
        # --- Analytical Signal Calculation ---
        n_vec = np.arange(1, n_terms)[:, None]
        t_grid = t_vals[None, :]
        nw = n_vec * wb

        # Resonance denominators
        denom1 = 4 * (nw + 2 * m)
        denom2 = 4 * (nw - 2 * m)
        denom3 = 2 * nw

        # Harmonic components
        cos1 = np.cos((nw + 2 * m) * t_grid - t0_sec * nw + 2 * gammaf)
        cos2 = np.cos((-nw + 2 * m) * t_grid + t0_sec * nw + 2 * gammaf)
        cos3 = np.cos(nw * t_grid - t0_sec * nw)
        cos4 = np.cos(2 * m * t0_sec + 2 * gammaf)

        jn_vals = jn(n_vec, n_vec * ecc)

        term1 = -cos1 / denom1 - cos2 / denom2 - cos3 / denom3 + cos4 / denom1 + cos4 / denom2 - 1 / denom3
        sum1 = np.sum(n_vec * jn_vals * phi0**2 * term1, axis=0)

        cos5 = np.cos((nw - 2 * m) * t_grid - t0_sec * nw - 2 * gammaf)
        cos6 = np.cos((nw + 2 * m) * t_grid - t0_sec * nw + 2 * gammaf)

        term2 = - cos5 / denom2 + cos6 / denom1 + cos4 / denom2 - cos4 / denom1
        sum2 = np.sum(jn_vals * m * phi0**2 * term2, axis=0)

        # Relative change in semi-major axis (a)
        delta_a_over_a = (-2 * wb * sum1
                          + 0.5 * (-2) * (np.cos(m * t_vals + gammaf)**2 - np.cos(m * t0_sec + gammaf)**2)
                          + 4 * (-2) * sum2)

        # Signal components h(t)
        h_a = -1.5 * wb * cumtrapz(delta_a_over_a, t_vals, initial=0)
        h_b = (5 / 4) * wb * phi0**2 / (4 * m) * (np.sin(2 * m * t_vals + 2 * gammaf) - np.sin(2 * m * t0_sec + 2 * gammaf))

        # Summation over Bessel components
        phi_sq = (phi0 * np.cos(m * t_vals + gammaf))**2
        phi_phi_dot = (phi0 * np.cos(m * t_vals + gammaf)) * (-m * phi0 * np.sin(m * t_vals + gammaf))
        
        h_c = np.zeros_like(t_vals)
        for n in range(1, n_terms):
            jn_val = jn(n, n * ecc)
            jn_p = jvp(n, n * ecc)

            an_coeff = 4 * wb * jn_val - 2 * wb * np.sqrt(1 - ecc**2) / ecc * n * jn_p
            bn_coeff = 4 / n * jn_val + 4 / ecc * np.sqrt(1 - ecc**2) * jn_p

            cos_n = np.cos(n * wb * (t_vals - t0_sec))
            sin_n = np.sin(n * wb * (t_vals - t0_sec))

            integrand = an_coeff * phi_sq * cos_n + 2 * bn_coeff * phi_phi_dot * sin_n
            h_c += 0.5 * cumtrapz(integrand, t_vals, initial=0)

        h_d = 5/8 * wb * phi0**2 * (t_vals - t0_sec)
        h_total = h_a + h_b + h_c + h_d

        # Orthogonal projection (Timing model fit subtraction)
        i1_h = np.trapz(f1(t_vals) * h_total, t_vals)
        i2_h = np.trapz(f2(t_vals) * h_total, t_vals)

        gh_vals = h_total - f1(t_vals) * i1_h / norm1 - f2(t_vals) * i2_h / norm2
        
        # Signal-to-noise contribution u
        u_contribution = np.trapz(gh_vals**2, t_vals) / (2 * var_theta)

        # Store coupling strength beta (Lambda^-1) for this realization
        betas_all[i, r] = (np.log(1000) / u_contribution)**0.5

# --- Data Processing and Units ---
unit_conversion = 2.30817405213802e+48  # s^2 to GeV^-2
# Average coupling over realizations
beta_median = np.median(betas_all, axis=1) * unit_conversion

# --- Plotting Results ---
plt.figure(figsize=(8, 6), dpi=100)
plt.loglog(masses_ev, np.sqrt(beta_median), color='black', lw=2, label='Median Sensitivity')

# Plot individual realizations with lower alpha
for r in range(n_realizations):
    plt.loglog(masses_ev, np.sqrt(betas_all[:, r] * unit_conversion), lw=1, alpha=0.3)

plt.xlabel("DM Mass $m$ [eV]", fontsize=12)
plt.ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]", fontsize=12)
plt.title(r"Combined sensitivity ($\Theta'$ variable)", fontsize=13)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- Saving Data ---
output_file = "beta_resonances_delta_prior.txt"
np.savetxt(output_file, np.column_stack((masses_ev, beta_median)),
           header="mass (eV) / beta (GeV^-2)", fmt="%.5e")

print(f"✅ Simulation complete. Results saved to {output_file}")