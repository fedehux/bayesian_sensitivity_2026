import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.special import jn, jvp
from tqdm import tqdm
import warnings

# --- Global Configuration ---
warnings.filterwarnings("ignore")

# --- System Parameters: J1903+0327 ---
PB_DAYS = 95.2
pb_sec = PB_DAYS * 24 * 3600              # Orbital period in seconds
wb = 2 * np.pi / pb_sec                   # Orbital angular frequency
a1_ls = 105.6                             # Projected semi-major axis in light-seconds
w_rad = np.deg2rad(141.7)                 # Argument of periastron in radians
ecc = 0.437                               # Eccentricity
t0_mjd = 57109.4                          # Time of periastron (MJD)
t0_sec = t0_mjd * 24 * 3600               # Time of periastron in seconds
epsilon_toa = 6.24e-6                     # TOA precision (seconds)
tobs_yrs = 10.7
tobs_sec = tobs_yrs * 365 * 24 * 3600     # Observation time in seconds
cadence_yr = 638.8
ndot = cadence_yr / (365 * 24 * 3600)     # Cadence in 1/seconds

# Local Dark Matter density in natural units
dm_density = 0.3 * 4.09351499910375e+55 

# --- Orbital Constants ---
x_ls = a1_ls
ab = x_ls * np.sin(w_rad)                 # alpha_b
nb = (1-ecc**2)**0.5 * a1_ls * np.cos(w_rad) # eta_b
eps_factor = np.sqrt(1 - ecc**2)          # sqrt(1 - e^2)

# --- Mass and Frequency Grid ---
masses_ev = np.logspace(-22, -20.5, num=1200)
m_seconds = masses_ev * 1.51926760385984e+15 # Mass in 1/s

# --- Simulation Settings ---
n_realizations = 10
n_terms = 10
n_t = int(round(tobs_sec * ndot))
# Time grid synchronized with T0
t_vals = np.linspace(t0_sec, t0_sec + tobs_sec, n_t)

# Storage for results
betas_all = np.zeros((len(m_seconds), n_realizations))

# --- Main Monte Carlo Loop ---
for r in range(n_realizations):
    np.random.seed(r)
    rf = np.random.rayleigh(1 / np.sqrt(2))
    gammaf = np.random.uniform(0, 2 * np.pi)

    for i, m in enumerate(tqdm(m_seconds, desc=f"Realization {r+1}/{n_realizations}")):
        phi0 = np.sqrt(2 * dm_density) / m * rf

        phi_vals = phi0 * np.cos(m * t_vals + gammaf)
        phidot_vals = -m * phi0 * np.sin(m * t_vals + gammaf)
        phi_sq = phi_vals**2
        phi_phi_dot = phi_vals * phidot_vals

        n_vec = np.arange(1, n_terms)[:, None]
        t_grid = t_vals[None, :]
        nw = n_vec * wb

        # Resonances
        denom1 = 4 * (nw + 2 * m)
        denom2 = 4 * (nw - 2 * m)
        denom3 = 2 * nw

        cos1 = np.cos((nw + 2 * m) * t_grid - t0_sec * nw + 2 * gammaf)
        cos2 = np.cos((-nw + 2 * m) * t_grid + t0_sec * nw + 2 * gammaf)
        cos3 = np.cos(nw * t_grid - t0_sec * nw)
        cos4 = np.cos(2 * m * t0_sec + 2 * gammaf)

        jn_vals = jn(n_vec, n_vec * ecc)

        term1 = -cos1 / denom1 - cos2 / denom2 - cos3 / denom3 \
                + cos4 / denom1 + cos4 / denom2 - 1 / denom3
        sum1 = np.sum(n_vec * jn_vals * phi0**2 * term1, axis=0)

        cos5 = np.cos((nw - 2 * m) * t_grid - t0_sec * nw - 2 * gammaf)
        cos6 = np.cos((nw + 2 * m) * t_grid - t0_sec * nw + 2 * gammaf)

        term2 = - cos5 / denom2 + cos6 / denom1 + cos4 / denom2 - cos4 / denom1
        sum2 = np.sum(jn_vals * m * phi0**2 * term2, axis=0)

        # delta a / a calculation
        delta_a_a = (-2 * wb * sum1 
                     + 0.5 * (-2) * (np.cos(m * t_vals + gammaf)**2 
                                     - np.cos(m * t0_sec + gammaf)**2)
                     + 4 * (-2) * sum2)

        # --- Kepler Solver: Eccentric Anomaly (E) ---
        mean_anom = wb * (t_vals - t0_sec)
        ecc_anom = mean_anom.copy()
        for _ in range(5):
            f_val = ecc_anom - ecc * np.sin(ecc_anom) - mean_anom
            df_val = 1 - ecc * np.cos(ecc_anom)
            ecc_anom -= f_val / df_val

        # True Anomaly calculation (f)
        true_anom = np.arctan((np.sqrt(1-ecc**2) * np.sin(ecc_anom)) / (np.cos(ecc_anom) - ecc))

        # f_dot components
        p1 = phi_sq/4 * wb / (1-ecc**2)**1.5 * (1 - ecc * np.cos(true_anom))**2
        p2 = -1.5 * wb * delta_a_a / (1-ecc**2)**1.5 * (1 + ecc * np.cos(true_anom))
        p3 = 1/ecc * (2 * phi_phi_dot * np.sin(true_anom) 
                      - wb * (1-ecc**2)**0.5 * phi_sq / 2 * np.cos(true_anom))
        
        f_dot = p1 + p2 + p3
        f_integrated = cumtrapz(f_dot, t_vals, initial=0)

        # Signal template Q(t)
        q_vals = - (ab * np.sin(ecc_anom) - nb * np.cos(ecc_anom)) / (1 - ecc * np.cos(ecc_anom)) \
                 * f_integrated * (1 - ecc * np.cos(ecc_anom)) * (1-ecc**2)**0.5 / (1 + ecc * np.cos(true_anom))

        # Fisher Information / SNR analysis
        q_sq_sum = np.sum(q_vals**2)
        sigma = epsilon_toa / np.sqrt(q_sq_sum)
        betas_all[i, r] = 3 * sigma

# --- Data Processing and Storage ---
unit_conversion = 2.30817405213802e+48 # s^2 to GeV^-2
betas_all *= unit_conversion

output_file = "beta_mass_anomaly_one_step_true_anomaly.txt"
header_str = "mass (eV) " + " ".join([f"beta_real{i}" for i in range(n_realizations)])
np.savetxt(output_file, np.column_stack((masses_ev, betas_all)), 
           header=header_str, fmt="%.5e")

# --- Final Plotting ---
plt.figure(figsize=(8, 6), dpi=100)
for r in range(n_realizations):
    plt.loglog(masses_ev, np.sqrt(betas_all[:, r]), lw=1, alpha=0.5)

plt.xlabel("DM Mass $m$ [eV]", fontsize=12)
plt.ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]", fontsize=12)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig("beta_realizations_true_anomaly.pdf")
plt.show()

print(f"✅ Process complete. Data saved in: {output_file}")