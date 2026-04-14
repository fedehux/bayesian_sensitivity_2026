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

# Pulsar Dataset: (Pb [days], x [s], ecc, Tasc [MJD], precision [us], Tobs [yrs], cadence)
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

n_realizations = 10

# Stochastic Realizations (reproducible per pulsar)
random_params = {}
for i in range(len(pulsars_data)):
    rng = np.random.default_rng(seed=i)
    # Rayleigh for amplitude, Uniform for phase
    rf_values = rng.rayleigh(1/np.sqrt(2), size=n_realizations)
    gammaf_values = rng.uniform(0, 2*np.pi, size=n_realizations)
    random_params[i] = (rf_values, gammaf_values)

warnings.filterwarnings("ignore", category=IntegrationWarning)

# Bayesian Target: BF = 1000 -> detA = 1/B^2
B = 1000.0
TARGET = 1.0 / (B**2)
LOG_TARGET = np.log(TARGET)

# Store results: beta (GeV^-2) for each realization and mass
all_betas = np.full((n_realizations, len(masses_ev)), np.nan, dtype=float)

# ============================================================
# Helper Functions
# ============================================================

def choose_dt(t_obs, p_b, m_dm):
    """Adaptive time step selection for integration."""
    w_b = 2*np.pi / p_b
    t_m = (2*np.pi / m_dm) if m_dm > 0 else t_obs
    t_2m = t_m / 2.0
    t_wb = 2*np.pi / w_b
    dt_freq = min(t_m, t_2m, t_wb) / 200.0
    dt_cap = t_obs / 20000.0
    dt = min(dt_freq, dt_cap)
    return max(dt, t_obs / 400000.0)

def log_prod_term(beta_tilde_grid, ux_list, uy_list, uxy_list):
    """Computes the log-product term for the Gaussian BF calculation."""
    bt2 = beta_tilde_grid**2
    logF = np.zeros_like(beta_tilde_grid, dtype=float)
    for ux, uy, uxy in zip(ux_list, uy_list, uxy_list):
        a = 1.0
        b = -(ux + uy)
        c = (ux * uy - 0.25 * uxy**2)
        term = a + b * bt2 + c * bt2**2
        term = np.where(term > 0.0, term, 1e-300)
        logF += np.log(term)
    return logF

# ============================================================
# Main Calculation Loop
# ============================================================

for i_m, m in enumerate(tqdm(m_seconds, desc="Calculating sensitivity (Gaussian, x)")):

    phi0 = np.sqrt(2.0 * dm_density) / m
    # Coarse search grid for beta_tilde = beta * phi0
    beta_tilde_grid = np.logspace(-80, -67, 100) * phi0 

    # Precomputing pulsar-specific deterministic terms
    precomp = []
    for j, (pb_days, x_val, _e, t_asc_mjd, eps_us, t_obs_yrs, n_dot) in enumerate(pulsars_data):
        pb_sec = pb_days * 24.0 * 3600.0
        eps_sec = eps_us * 1e-6
        t_obs_sec = t_obs_yrs * 365.0 * 24.0 * 3600.0
        ndot_sec = n_dot / (365.0 * 24.0 * 3600.0)
        t_asc_sec = t_asc_mjd * 24.0 * 3600.0

        # Variance for the 'x' parameter
        var_x = 20.0 * eps_sec**2 / (9.0 * ndot_sec)

        dt = choose_dt(t_obs_sec, pb_sec, m)
        n_t = int(np.ceil(t_obs_sec / dt)) + 1
        t = np.linspace(0.0, t_obs_sec, n_t)

        f1 = t - t_obs_sec / 2.0
        f2 = np.full_like(t, t_obs_sec)
        norm1 = np.trapz(f1 * f1, t)
        norm2 = np.trapz(f2 * f2, t)

        # Signal components for x variation
        ax = -x_val * (np.cos(m * t)**2 - np.cos(m * t_asc_sec)**2)
        ay = -x_val * (np.sin(m * t)**2 - np.sin(m * t_asc_sec)**2)
        axy = 2 * x_val * (np.sin(2.0 * m * t) - np.sin(2.0 * m * t_asc_sec))

        # Gram-Schmidt residuals for basis projections
        gx = ax - f1 * (np.trapz(f1 * ax, t) / norm1) - f2 * (np.trapz(f2 * ax, t) / norm2)
        gy = ay - f1 * (np.trapz(f1 * ay, t) / norm1) - f2 * (np.trapz(f2 * ay, t) / norm2)
        gxy = axy - f1 * (np.trapz(f1 * axy, t) / norm1) - f2 * (np.trapz(f2 * axy, t) / norm2)

        precomp.append({
            "t": t, "f1": f1, "f2": f2, "norm1": norm1, "norm2": norm2,
            "ax": ax, "ay": ay, "axy": axy, "GX": gx, "GY": gy, "GXY": gxy, "varx": var_x
        })

    for r in range(n_realizations):
        ux_list, uy_list, uxy_list = [], [], []
        for j, pc in enumerate(precomp):
            rf_vals, gf_vals = random_params[j]
            X = np.sqrt(2.0) * rf_vals[r] * np.cos(gf_vals[r])
            Y = np.sqrt(2.0) * rf_vals[r] * np.sin(gf_vals[r])

            # Resulting DM signal h(t)
            h_arr = (X**2 * pc["ax"] + Y**2 * pc["ay"] + (X * Y) * pc["axy"]) * (phi0**2 / 2.0)

            # Signal projections for residual calculation
            c1 = (X**2 * np.trapz(pc["f1"] * pc["ax"], pc["t"]) + Y**2 * np.trapz(pc["f1"] * pc["ay"], pc["t"]) + (X * Y) * np.trapz(pc["f1"] * pc["axy"], pc["t"]))
            c2 = (X**2 * np.trapz(pc["f2"] * pc["ax"], pc["t"]) + Y**2 * np.trapz(pc["f2"] * pc["ay"], pc["t"]) + (X * Y) * np.trapz(pc["f2"] * pc["axy"], pc["t"]))
            gh = h_arr - pc["f1"] * (c1 / pc["norm1"]) - pc["f2"] * (c2 / pc["norm2"])

            # Bayes factor term integrals
            ux_list.append(np.trapz(pc["GX"] * gh, pc["t"]) / (2.0 * pc["varx"]))
            uy_list.append(np.trapz(pc["GY"] * gh, pc["t"]) / (2.0 * pc["varx"]))
            uxy_list.append(np.trapz(pc["GXY"] * gh, pc["t"]) / (2.0 * pc["varx"]))

        # Match log_target to solve for beta_tilde
        log_f = log_prod_term(beta_tilde_grid, ux_list, uy_list, uxy_list)
        beta_in = beta_tilde_grid[np.argmin(np.abs(log_f - LOG_TARGET))]

        # Refined search around candidate
        beta_fine_grid = np.linspace(beta_in * 0.9, beta_in * 1.1, 100)
        log_f_fine = log_prod_term(beta_fine_grid, ux_list, uy_list, uxy_list)
        beta_tilde = beta_fine_grid[np.argmin(np.abs(log_f_fine - LOG_TARGET))]

        all_betas[r, i_m] = beta_tilde / phi0

# Final Unit Conversion to GeV^-2
conversion = (1.51926760385984e24)**2
all_betas *= conversion

# ============================================================
# Data Export & Saving
# ============================================================

np.savez("beta_x_gaussian_all_realizations.npz", masses=masses_ev, betas=all_betas)

median_beta = np.nanmedian(all_betas, axis=0)
np.savetxt(
    "beta_vs_mass_x_gaussian_median.txt",
    np.column_stack((masses_ev, median_beta)),
    header="mass (eV) / median_beta (GeV^-2) [Variable: x, Prior: Gaussian]",
    fmt="%.5e"
)

# ============================================================
# Plotting Results
# ============================================================

plt.figure(figsize=(10, 6), dpi=150)
for r in range(n_realizations):
    plt.loglog(masses_ev, np.sqrt(all_betas[r, :]), lw=1, alpha=0.35, color='gray')

plt.loglog(masses_ev, np.sqrt(median_beta), lw=3, color="black", label=r"Median Sensitivity ($x$, Gaussian-prior)")
plt.xlabel("$m$ [eV]", fontsize=14)
plt.ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]", fontsize=14)
plt.title(r"Sensitivity analysis: $x$ with Gaussian-prior", fontsize=15)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("dm_sensitivity_x_gaussian_plot.pdf")
plt.show()