import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from scipy.integrate import IntegrationWarning

def choose_dt(t_obs, p_b, m_dm):
    """
    Dynamically selects a time step (dt) to ensure we don't under-sample 
    either the orbital frequency or the dark matter oscillation frequency.
    """
    w_b = 2 * np.pi / p_b
    # Period of the DM field oscillation (T = 2pi/m)
    t_m = (2 * np.pi / m_dm) if m_dm > 0 else t_obs
    t_2m = t_m / 2.0
    t_wb = 2 * np.pi / w_b
    
    # We want enough points to resolve the fastest oscillation
    dt_freq = min(t_m, t_2m, t_wb) / 200.0
    dt_cap = t_obs / 20000.0
    dt = min(dt_freq, dt_cap)
    # Safety floor to avoid infinite loops/memory issues
    dt = max(dt, t_obs / 400000.0)
    return dt

# --- Physical Constants & Setup ---
# Local Dark Matter density converted to natural units
dm_density = 0.3 * 4.09351499910375e55 

# Range of DM masses to scan (from 10^-23 to 10^-18 eV)
masses_ev = np.logspace(-23, -18, num=40)
# Converting masses to seconds^-1 (natural units)
m_seconds = masses_ev * 1.51926760385984e15

# Pulsar Dataset: (Pb [days], x [lt-s], ecc, Tasc [MJD], precision [us], Tobs [yrs], cadence)
# This list represents the timing properties of the binary pulsars we are analyzing.
pulsars_data = [
    (67.83, 32.34, 7.5e-5, 53420.49, 0.21, 15.47, 1381),
    (0.7, 8.82e-2, 7.1e-6, 55186.11342080, 14.18, 4.5, 663.6),
    (6.8, 1.02e+01, 7.865e-6, 56102.4, 7.86, 15.5, 1201.9),
    (10.2, 2.82e+00, 5.15e-6, 56104.2, 7.94, 15.5, 1663.6),
    (2.9, 2.31e+00, 3.21e-6, 58204.8, 5.25, 15.6, 890.2),
    (4.8, 4.05e+00, 4.75e-6, 58223.9, 4.75, 4.6, 115),
    (4.9, 2.46e+00, 4.5e-6, 58319.2, 12.36, 3.6, 608.7),
    (0.7, 3.72e+00, 3.5e-6, 58431.4, 8.14, 3.5, 1965.9),
    (12.3, 9.23e+00, 3.29e-6, 56211.1, 3.29, 15.6, 497.8),
    (1.2, 1.19e+00, 3.83e-6, 56308.0, 2.68, 8.3, 421.9),
    (8.7, 1.13e+01, 3.23e-6, 56300.8, 3.23, 11.5, 1598.2),
    (32.0, 6.42e+00, 3.11e-6, 56109.4, 5.76, 7.1, 1061.4),
    (6.7, 1.62e+00, 2.94e-6, 57431.5, 1.59, 9.1, 817.2),
    (15.4, 1.22e+01, 3.85e-6, 57785.2, 13.85, 6.3, 1385.5),
    (6.3, 1.56e+00, 3.13e-6, 58312.6, 3.12, 3.5, 1523.3),
    (4.8, 3.98e+00, 4.42e-6, 58312.5, 4.44, 6.3, 2123.5),
    (0.1, 1.82e-03, 15.8e-6, 58314.0, 15.8, 3.4, 1846.1),
    (2.2, 2.32e+00, 3.26e-6, 56196.4, 2.98, 3.6, 686.5),
    (1.2, 1.00e+00, 3.3e-6, 56060.7, 3.26, 15.0, 1138.4),
    (16.3, 1.10e+01, 3.27e-6, 56040.5, 2.4, 11, 506.8),
    (3.3, 3.43e+01, 7.1e-6, 56150.0, 5.06, 10.7, 819),
    (1.5, 1.90e+00, 7.45e-6, 56121.0, 1.18, 15.5, 2261.6),
    (0.3, 7.35e-02, 9.41e-6, 58328.1, 9.41, 3.4, 1451.1)
]

# --- Monte Carlo Realizations ---
# We want to account for the stochastic nature of the DM field (amplitude and phase)
n_realizations = 1
random_params = {}
for i, _ in enumerate(pulsars_data):
    np.random.seed(i) # Ensuring reproducibility per pulsar
    # Raleigh distribution for amplitude, Uniform for phase
    r_vals = np.random.rayleigh(1/np.sqrt(2), size=n_realizations)
    gamma_vals = np.random.uniform(0, 2*np.pi, size=n_realizations)
    random_params[i] = (r_vals, gamma_vals)

warnings.filterwarnings("ignore", category=IntegrationWarning)

# Bayes Factor threshold for sensitivity
log_b = np.log(1000.0) 
all_betas = np.full((n_realizations, len(masses_ev)), np.nan)

# --- Main Calculation Loop ---
# Here we iterate over the DM mass range and calculate the sensitivity 
# for each stochastic realization across the whole pulsar array.
for i_m, m in enumerate(tqdm(m_seconds, desc="Calculating sensitivity (all realizations)")):
    phi0 = np.sqrt(2 * dm_density) / m
    sum_u_r = np.zeros(n_realizations)

    for j, (pb_days, a1, e, t_asc_mjd, eps_us, t_obs_yrs, n_dot) in enumerate(pulsars_data):
        # Unit conversions to SI/Seconds
        pb_sec = pb_days * 24 * 3600.0
        x_sec = a1
        epsilon_sec = eps_us * 1e-6
        t_obs_sec = t_obs_yrs * 365.0 * 24.0 * 3600.0
        n_dot_sec = n_dot / (365.0 * 24.0 * 3600.0)
        t_asc_sec = t_asc_mjd * 24.0 * 3600.0

        w_b = 2 * np.pi / pb_sec
        # Variance associated with the timing noise/precision
        var_psi = 2.0 * epsilon_sec**2 / (n_dot_sec * x_sec**2)

        # Time grid setup for integration
        dt = choose_dt(t_obs_sec, pb_sec, m)
        n_t = int(np.ceil(t_obs_sec / dt)) + 1
        t = np.linspace(0.0, t_obs_sec, n_t)

        # Basis functions for the timing model (polynomial subtraction)
        # We define these to account for the fitting of orbital parameters
        f1 = t - t_obs_sec / 2.0
        f2 = np.full_like(t, t_obs_sec)
        f0 = t**2 / t_obs_sec - t + t_obs_sec / 6.0

        norm0 = np.trapz(f0**2, t)
        norm1 = np.trapz(f1**2, t)
        norm2 = np.trapz(f2**2, t)

        const = (11.0 / 16.0) * w_b * (phi0**2) / m 
        r_vals, gamma_vals = random_params[j]

        for r in range(n_realizations):
            rf = r_vals[r]
            gf = gamma_vals[r]
            
            # The DM-induced signal in the orbital phase
            h_signal = const * (rf**2) * (np.sin(2.0*m*t + 2.0*gf) - np.sin(2.0*m*t_asc_sec + 2.0*gf))

            # Projecting the signal onto the timing model basis to find the residuals
            i0 = np.trapz(f0 * h_signal, t)
            i1 = np.trapz(f1 * h_signal, t)
            i2 = np.trapz(f2 * h_signal, t)

            # Gram-Schmidt-like subtraction of the fitted components
            g_h_psi = h_signal - f0*(i0/norm0) - f1*(i1/norm1) - f2*(i2/norm2)

            # Summing up the signal-to-noise contribution (u_psi)
            u_psi = np.trapz(g_h_psi**2, t) / (2.0 * var_psi)
            sum_u_r[r] += u_psi

    # Solve for the coupling strength beta (Lambda^-1) at the Bayes threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_r = np.sqrt(log_b / sum_u_r)

    # Conversion back to GeV^-2
    conversion_factor = 2.30817405213802e48
    all_betas[:, i_m] = beta_r * conversion_factor

# --- Data Export & Saving ---
# Saving all realizations to an npz file, clarifying variable psi and delta prior
np.savez("beta_psi_delta_all_realizations.npz", masses=masses_ev, betas=all_betas)

median_beta = np.nanmedian(all_betas, axis=0)
np.savetxt(
    "beta_vs_mass_psi_delta_median.txt",
    np.column_stack((masses_ev, median_beta)),
    header="mass (eV) / median_beta (GeV^-2) [Variable: psi, Prior: delta]",
    fmt="%.5e"
)

# --- Plotting Results ---
plt.figure(figsize=(10, 6), dpi=300)
for r in range(n_realizations):
    # Plotting individual realizations with low alpha
    plt.loglog(masses_ev, np.sqrt(all_betas[r, :]), lw=1, alpha=0.3, color='gray')

# Plot the median sensitivity as the primary result
plt.loglog(masses_ev, np.sqrt(median_beta), lw=2, color='blue', label=r'Median Sensitivity ($\Psi$, $\delta$-prior)')

plt.xlabel("$m$ [eV]", fontsize=14)
plt.ylabel(r"$\sqrt{beta}$]", fontsize=14)
plt.title(r"Sensitivity analysis: $\Psi'$ with $\delta$-prior", fontsize=15)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# Save the plot with a descriptive name
plt.savefig("dm_sensitivity_psi_delta_plot.pdf")
plt.show()