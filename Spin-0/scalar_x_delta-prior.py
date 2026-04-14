import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tqdm import tqdm
import warnings
from scipy.integrate import IntegrationWarning

# ============================================================
# Pulsar Dataset
# (Pb [days], x [s], ecc, Tasc [MJD], epsilon [µs], Tobs [yr], ndot [1/yr])
# ============================================================

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

# ============================================================
# Physical Constants
# ============================================================

rho_DM = 0.3 * 4.09351499910375e55        # Local DM density in natural units
eV_to_sinv = 1.51926760385984e15          # conversion: 1 eV -> s^-1
conv_s2_to_GeV2 = 2.30817405213802e48     # conversion: s^2 -> GeV^-2
B_target = 1000.0
logB = np.log(B_target)

# ============================================================
# Numerical Parameters
# ============================================================

n_realizations = 10
masses_ev = np.logspace(-23, -18, 40)
m_seconds = masses_ev * eV_to_sinv

# ============================================================
# Random Realizations (Reproducible per pulsar)
# ============================================================

random_params = {}
for j in range(len(pulsars_data)):
    rng = np.random.default_rng(seed=j)
    # Rayleigh for field amplitude, Uniform for phase
    rf = rng.rayleigh(scale=1 / np.sqrt(2), size=n_realizations)
    gammaf = rng.uniform(0, 2 * np.pi, size=n_realizations)
    random_params[j] = (rf, gammaf)

# ============================================================
# Main Sensitivity Calculation
# ============================================================

warnings.filterwarnings("ignore", category=IntegrationWarning)

# Store results for each realization and mass
all_betas = np.zeros((n_realizations, len(m_seconds)))

for i, m in enumerate(tqdm(m_seconds, desc="Calculating sensitivity (x, delta)")):

    phi0 = np.sqrt(2 * rho_DM) / m
    sum_u = np.zeros(n_realizations)

    for j, (pb_days, x_val, _, t_asc_mjd, eps_us, t_obs_yr, n_dot_yr) in enumerate(pulsars_data):

        # Unit conversions to SI/Seconds
        Tasc = t_asc_mjd * 24 * 3600
        Tobs = t_obs_yr * 365 * 24 * 3600
        eps = eps_us * 1e-6
        ndot = n_dot_yr / (365 * 24 * 3600)

        # Variance for the projected semi-major axis 'x'
        varx = 20 * eps**2 / (9 * ndot)

        # Basis functions for the timing model subtraction
        def f1(t): return t - Tobs / 2
        def f2(t): return Tobs + 0*t
        def f0(t): return t**2 / Tobs - t + Tobs / 6

        # Directional components of the signal
        def ax_func(t): return -x_val * (np.cos(m*t)**2 - np.cos(m*Tasc)**2)
        def ay_func(t): return -x_val * (np.sin(m*t)**2 - np.sin(m*Tasc)**2)
        def axy_func(t): return x_val * (np.sin(2*m*t) - np.sin(2*m*Tasc))

        # Orthogonalization norms
        norm1 = quad(lambda t: f1(t)**2, 0, Tobs)[0]
        norm2 = quad(lambda t: f2(t)**2, 0, Tobs)[0]

        rf_vals, gammaf_vals = random_params[j]

        for r in range(n_realizations):
            # Mapping field realizations to X, Y components
            X = np.sqrt(2) * rf_vals[r] * np.cos(gammaf_vals[r])
            Y = np.sqrt(2) * rf_vals[r] * np.sin(gammaf_vals[r])

            # Resulting DM signal h(t)
            def h(t):
                return (ax_func(t)*X**2 + ay_func(t)*Y**2 + axy_func(t)*X*Y) * phi0**2 / 2

            # Projecting h(t) onto timing model basis
            i1 = quad(lambda t: f1(t)*h(t), 0, Tobs)[0]
            i2 = quad(lambda t: f2(t)*h(t), 0, Tobs)[0]

            # Residual signal Gh(t) after fitting
            def Gh(t):
                return h(t) - f1(t)*i1/norm1 - f2(t)*i2/norm2

            # Signal-to-noise contribution
            u = quad(lambda t: Gh(t)**2, 0, Tobs)[0] / (2 * varx)
            sum_u[r] += u

    # Calculating coupling strength beta (Lambda^-1) that meets BF threshold
    beta_s2 = np.sqrt(logB / sum_u)
    all_betas[:, i] = beta_s2 * conv_s2_to_GeV2

# ============================================================
# Data Export & Saving
# ============================================================

np.savez("beta_x_delta_all_realizations.npz", masses=masses_ev, betas=all_betas)

beta_median = np.median(all_betas, axis=0)
np.savetxt(
    "beta_x_vs_mass_delta_median.txt",
    np.column_stack((masses_ev, beta_median)),
    header="mass (eV) / beta_median (GeV^-2) [Variable: x, Prior: delta]",
    fmt="%.6e"
)

# ============================================================
# Plotting Results
# ============================================================

plt.figure(figsize=(10, 6), dpi=300)

for r in range(n_realizations):
    # Plotting individual realizations
    plt.loglog(masses_ev, np.sqrt(all_betas[r]), color="gray", alpha=0.35)

# Highlight median sensitivity
plt.loglog(masses_ev, np.sqrt(beta_median), color="black", lw=3, label=r"Median Sensitivity ($x$, $\delta$-prior)")

plt.xlabel("$m$ [eV]", fontsize=14)
plt.ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]", fontsize=14)
plt.title(r"Sensitivity analysis: $x$ with $\delta$-prior", fontsize=15)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("beta_x_delta_sensitivity_plot.pdf")
plt.show()