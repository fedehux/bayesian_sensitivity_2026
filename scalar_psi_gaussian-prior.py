import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from scipy.integrate import IntegrationWarning

# ============================================================
# Global Parameters
# ============================================================

# Local Dark Matter density in natural units
dm_density = 0.3 * 4.09351499910375e55  
# DM mass range in eV
masses_ev = np.logspace(-23, -18, num=40)
# Mass converted to frequency (1/s)
m_seconds = masses_ev * 1.51926760385984e15 

# Pulsar Dataset: (Pb [days], x [lt-s], ecc, Tasc [MJD], precision [us], Tobs [yrs], cadence)
pulsars_data = [
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

n_realizations = 10

# ============================================================
# Stochastic Realizations (reproducible per pulsar)
# ============================================================

random_params = {}
for i in range(len(pulsars_data)):
    rng = np.random.default_rng(seed=i)
    # Raleigh for amplitude, Uniform for phase
    rf_values = rng.rayleigh(1/np.sqrt(2), size=n_realizations)
    gammaf_values = rng.uniform(0, 2*np.pi, size=n_realizations)
    random_params[i] = (rf_values, gammaf_values)

warnings.filterwarnings("ignore", category=IntegrationWarning)

# ============================================================
# Bayesian Target: detA = 1/B^2
# ============================================================

B = 1000.0
target = 1.0 / (B**2)
log_target = np.log(target)

# Output: beta (GeV^-2) for each realization and mass
all_betas = np.full((n_realizations, len(masses_ev)), np.nan, dtype=float)

# ============================================================
# Helper Functions
# ============================================================

def calculate_log_f_beta(beta_values, ux_list, uy_list, uxy_list):
    """
    Computes log f(beta) = sum_j log( 1 - (ux+uy) beta^2 + (ux*uy - (uxy^2)/4) beta^4 )
    where (ux, uy, uxy) are precomputed projections for each pulsar.
    """
    log_f_beta = np.zeros_like(beta_values, dtype=float)
    for ux, uy, uxy in zip(ux_list, uy_list, uxy_list):
        a = 1.0
        b = -(ux + uy)
        c = (ux * uy - 0.25 * uxy**2)
        term = a + b * beta_values**2 + c * beta_values**4
        # Numerical floor to avoid log of non-positive values due to precision
        term = np.where(term > 0.0, term, 1e-300)  
        log_f_beta += np.log(term)
    return log_f_beta


def choose_dt(t_obs, p_b, m_dm):
    """
    Selects integration time step to resolve both orbital and DM oscillation frequencies.
    """
    w_b = 2 * np.pi / p_b
    t_m = (2 * np.pi / m_dm) if m_dm > 0 else t_obs
    t_2m = t_m / 2.0
    t_wb = 2 * np.pi / w_b

    dt_freq = min(t_m, t_2m, t_wb) / 200.0
    dt_cap = t_obs / 20000.0
    dt = min(dt_freq, dt_cap)
    dt = max(dt, t_obs / 400000.0)
    return dt


# ============================================================
# Main Loop: Scanning Masses and Realizations
# ============================================================

for i_m, m in enumerate(tqdm(m_seconds, desc="Calculating sensitivity (Gaussian, all realizations)")):

    phi0 = np.sqrt(2 * dm_density) / m

    # Coarse grid search for beta_tilde = beta * phi0
    beta_tilde_grid = np.logspace(-90, -60, 120) * phi0

    for r in range(n_realizations):
        ux_list, uy_list, uxy_list = [], [], []

        for j, (pb_days, a1, _e, t_asc_mjd, eps_us, t_obs_yrs, n_dot) in enumerate(pulsars_data):

            # --- Unit conversions ---
            pb_sec = pb_days * 24.0 * 3600.0
            x_sec = a1
            epsilon_sec = eps_us * 1e-6
            t_obs_sec = t_obs_yrs * 365.0 * 24.0 * 3600.0
            n_dot_sec = n_dot / (365.0 * 24.0 * 3600.0)
            t_asc_sec = t_asc_mjd * 24.0 * 3600.0

            # Effective variance for the psi parameter
            var_psi = 2.0 * epsilon_sec**2 / (n_dot_sec * x_sec**2)
            w_b = 2.0 * np.pi / pb_sec

            # --- Adaptive time grid ---
            dt = choose_dt(t_obs_sec, pb_sec, m)
            n_t = int(np.ceil(t_obs_sec / dt)) + 1
            t = np.linspace(0.0, t_obs_sec, n_t)

            # --- Timing Model Basis Functions ---
            f1 = t - t_obs_sec / 2.0
            f2 = np.full_like(t, t_obs_sec)
            f0 = t**2 / t_obs_sec - t + t_obs_sec / 6.0

            norm0 = np.trapz(f0**2, t)
            norm1 = np.trapz(f1**2, t)
            norm2 = np.trapz(f2**2, t)

            # --- Deterministic Signal Components ---
            const = (11.0 / 16.0) * (w_b / m)
            ax_signal =  const * (np.sin(2*m*t) - np.sin(2*m*t_asc_sec))
            ay_signal = -const * (np.sin(2*m*t) - np.sin(2*m*t_asc_sec))
            axy_signal = (11.0 / 8.0) * (w_b / m) * (np.cos(2.0*m*t) - np.cos(2.0*m*t_asc_sec))

            # Projections onto basis for Gram-Schmidt (G operators)
            i0_ax = np.trapz(f0 * ax_signal, t);  i1_ax = np.trapz(f1 * ax_signal, t);  i2_ax = np.trapz(f2 * ax_signal, t)
            i0_ay = np.trapz(f0 * ay_signal, t);  i1_ay = np.trapz(f1 * ay_signal, t);  i2_ay = np.trapz(f2 * ay_signal, t)
            i0_axy = np.trapz(f0 * axy_signal, t); i1_axy = np.trapz(f1 * axy_signal, t); i2_axy = np.trapz(f2 * axy_signal, t)

            # --- Realization specific (X, Y) components ---
            rf_vals, gammaf_vals = random_params[j]
            rf = rf_vals[r]
            gf = gammaf_vals[r]
            X = np.sqrt(2.0) * rf * np.cos(gf)
            Y = np.sqrt(2.0) * rf * np.sin(gf)

            # Linear combination for signal h(t) and its residual Gh
            c0 = (X**2 * i0_ax + Y**2 * i0_ay + (X*Y) * i0_axy)
            c1 = (X**2 * i1_ax + Y**2 * i1_ay + (X*Y) * i1_axy)
            c2 = (X**2 * i2_ax + Y**2 * i2_ay + (X*Y) * i2_axy)

            h_arr = (X**2 * ax_signal + Y**2 * ay_signal + (X*Y) * axy_signal) * (phi0**2 / 2.0)
            g_h = h_arr - f1 * (c1 / norm1) - f2 * (c2 / norm2) - f0 * (c0 / norm0)

            # Precomputing G-residuals for X, Y components
            g_x = ax_signal - f1 * (i1_ax/norm1) - f2 * (i2_ax/norm2) - f0 * (i0_ax/norm0)
            g_y = ay_signal - f1 * (i1_ay/norm1) - f2 * (i2_ay/norm2) - f0 * (i0_ay/norm0)
            g_xy = axy_signal - f1 * (i1_axy/norm1) - f2 * (i2_axy/norm2) - f0 * (i0_axy/norm0)

            # Final integrals for Bayes Factor calculation
            ux = np.trapz(g_x * g_h, t) / (2.0 * var_psi)
            uy = np.trapz(g_y * g_h, t) / (2.0 * var_psi)
            uxy = np.trapz(g_xy * g_h, t) / (2.0 * var_psi)

            ux_list.append(ux)
            uy_list.append(uy)
            uxy_list.append(uxy)

        # --- Matching log_target to find beta_tilde (beta * phi0) ---
        log_f_beta = calculate_log_f_beta(beta_tilde_grid, ux_list, uy_list, uxy_list)
        idx_coarse = np.argmin(np.abs(log_f_beta - log_target))
        beta_coarse = beta_tilde_grid[idx_coarse]

        # Refined grid search around the candidate value
        beta_refined_grid = np.linspace(beta_coarse * 0.8, beta_coarse * 1.2, 100)
        log_f_beta_refined = calculate_log_f_beta(beta_refined_grid, ux_list, uy_list, uxy_list)
        idx_fine = np.argmin(np.abs(log_f_beta_refined - log_target))
        beta_tilde = beta_refined_grid[idx_fine]

        all_betas[r, i_m] = beta_tilde / phi0

# ============================================================
# Final Unit Conversion to GeV^-2
# ============================================================

conversion_factor = (1.51926760385984e24)**2
all_betas *= conversion_factor

# ============================================================
# Data Saving
# ============================================================

# Saving all realizations with variable and prior identifiers
np.savez("beta_psi_gaussian_all_realizations.npz", masses=masses_ev, betas=all_betas)

median_beta = np.nanmedian(all_betas, axis=0)
np.savetxt(
    "beta_vs_mass_psi_gaussian_median.txt",
    np.column_stack((masses_ev, median_beta)),
    header="mass (eV) / median_beta (GeV^-2) [Variable: psi, Prior: Gaussian]",
    fmt="%.5e"
)

# ============================================================
# Plotting Results
# ============================================================

plt.figure(figsize=(10, 6), dpi=300)
for r in range(n_realizations):
    # Plotting individual realizations for sqrt(beta) -> GeV^-1
    plt.loglog(masses_ev, np.sqrt(all_betas[r, :]), lw=1, alpha=0.3, color='gray')

# Plot the median sensitivity
plt.loglog(masses_ev, np.sqrt(median_beta), lw=2, color='red', label=r'Median Sensitivity ($\psi$, Gaussian-prior)')

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$m$ [eV]", fontsize=14)
plt.ylabel(r"$\sqrt{\beta}}$[GeV$^{-1}$]", fontsize=14)
plt.title(r"Sensitivity analysis: $\Psi'$ with Gaussian-prior", fontsize=15)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("dm_sensitivity_psi_gaussian_plot.pdf")
plt.show()