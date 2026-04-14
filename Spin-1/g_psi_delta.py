#%% Libraries and data loading
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import pandas as pd
import math
from scipy.integrate import cumulative_trapezoid as cumtrapz

# === Path to NANOGrav 15yr Table ===
ELL1_TABLE_PATH = "ell1_table.csv"

# --- Physical Constants & Grids ---
# Local Dark Matter density in natural units
dm_density = 0.3 * 4.09351499910375e+55 
# Mass range in eV
masses_ev = np.logspace(-23, -18, num=500)
# Mass converted to frequency (1/s)
m_seconds = masses_ev * 1.51926760385984e+15

# Timing model basis functions (fitting components)
def f1(t, t_obs): return t - t_obs / 2.0
def f2(t, t_obs): return t_obs + 0.0 * t
def f0(t, t_obs): return t**2 / t_obs - t + t_obs / 6.0

# Companion Mass Priors (M_sun)
COMPANION_MASS_PRIORS = {
    "J0406+3039": 0.096246, "J0610-2100": 0.024748, "J1012+5307": 0.124641,
    "J1125+7819": 0.329741, "J1713+0747": 0.324201, "J1719-1438": 0.001304,
    "J1738+0333": 0.103990, "J1745+1017": 0.015826, "J1802-2124": 0.981815,
    "J2145-0750": 0.503270, "J2234+0944": 0.017786, "J2317+1439": 0.201229,
}

IOTA_PRIORS_DEG = {}
PULSAR_MASS_PRIORS = {}

# Default values if missing in CSV or priors
DEFAULT_M1 = 1.35   # Pulsar mass (M_sun)
DEFAULT_M2 = 0.20   # Companion mass (M_sun)
DEFAULT_IOTA_DEG = 60.0  # Inclination (degrees)

# === LOAD NANOGRAV 15YR TABLE ===
df = pd.read_csv(ELL1_TABLE_PATH)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["PB_d", "A1_ltsec"])

# Normalize Pulsar IDs (J/B names)
df["psr_id"] = df["pulsar"].astype(str).str.extract(r'((?:J|B)\d{4}[+-]\d{4})', expand=False)
df["psr_id"] = df["psr_id"].fillna(df["pulsar"].astype(str).str.strip().str.rstrip('.'))

# Calculate Observation Span
if {"MJD_ini", "MJD_fin"}.issubset(df.columns):
    df["span_yrs_raw"] = (df["MJD_fin"] - df["MJD_ini"]) / 365.25
else:
    df["span_yrs_raw"] = np.nan

if "N_TOAs" not in df.columns:
    df["N_TOAs"] = np.nan
if "ndot_per_year" not in df.columns and "ndot_year" in df.columns:
    df["ndot_per_year"] = df["ndot_year"]

# Select the "best" row per Pulsar based on data quality
df["_rank_sort_N"]    = df["N_TOAs"].fillna(-1).astype(float)
df["_rank_sort_span"] = df["span_yrs_raw"].fillna(-1).astype(float)
df["_rank_sort_eps"]  = -df["epsilon_us"].fillna(np.inf).astype(float)
df = (
    df.sort_values(by=["_rank_sort_N", "_rank_sort_span", "_rank_sort_eps"],
                   ascending=[False, False, False])
      .drop_duplicates(subset="psr_id", keep="first")
      .drop(columns=["_rank_sort_N", "_rank_sort_span", "_rank_sort_eps"])
)
df["pulsar"] = df["psr_id"]
df = df.drop(columns=["psr_id"])

# === Robust T_obs calculation ===
def compute_tobs_yrs(row):
    mjd_i, mjd_f = row.get("MJD_ini", np.nan), row.get("MJD_fin", np.nan)
    if np.isfinite(mjd_i) and np.isfinite(mjd_f):
        dt_days = mjd_f - mjd_i
        if 0 < dt_days < 1.0e5:
            return dt_days / 365.25
    n_toas = row.get("N_TOAs", np.nan)
    ndy = row.get("ndot_per_year", row.get("ndot_year", np.nan))
    if np.isfinite(n_toas) and np.isfinite(ndy) and ndy > 0:
        return float(n_toas) / float(ndy)
    return np.nan

df["Tobs_yrs"] = df.apply(compute_tobs_yrs, axis=1)

if "e" not in df.columns:
    df["e"] = np.nan
if "epsilon_us" not in df.columns:
    df["epsilon_us"] = np.nan

# ---------------------------
# Mass/Inclination Helpers
# ---------------------------
def infer_masses_and_iota(psr, row):
    """Returns (M1, M2, sin_i) using: CSV -> Priors -> Defaults."""
    # M1 (Pulsar Mass)
    m1 = row.get("Mp_Msun", np.nan)
    if not np.isfinite(m1):
        m1 = PULSAR_MASS_PRIORS.get(psr, DEFAULT_M1)

    # M2 (Companion Mass)
    m2 = row.get("Mc_Msun", np.nan)
    if not np.isfinite(m2):
        m2 = COMPANION_MASS_PRIORS.get(psr, DEFAULT_M2)

    # sin(iota)
    sin_i = row.get("sin_i", np.nan)
    if not (np.isfinite(sin_i) and (0.0 <= sin_i <= 1.0)):
        iota_rad = row.get("iota_rad", np.nan)
        if np.isfinite(iota_rad):
            sin_i = float(np.sin(iota_rad))
        else:
            iota_deg = row.get("iota_deg", np.nan)
            if np.isfinite(iota_deg):
                sin_i = float(np.sin(np.deg2rad(iota_deg)))
            else:
                iota_deg_prior = IOTA_PRIORS_DEG.get(psr, DEFAULT_IOTA_DEG)
                sin_i = float(np.sin(np.deg2rad(iota_deg_prior)))

    # Safeguard against sin(i) ~ 0
    sin_i = float(np.clip(sin_i, 1e-3, 1.0))
    return float(m1), float(m2), sin_i

# === Assembly of final pulsar data list ===
pulsars_data = []
for _, r in df.iterrows():
    psr = str(r["pulsar"])
    pb_days = float(r["PB_d"])
    a1_ls   = float(r["A1_ltsec"])
    ecc     = float(r["e"]) if np.isfinite(r["e"]) else 0.0
    eps_us  = float(r["epsilon_us"]) if np.isfinite(r["epsilon_us"]) else 1.0
    t_obs   = float(r["Tobs_yrs"]) if np.isfinite(r["Tobs_yrs"]) else 5.0
    ndy     = float(r["ndot_per_year"]) if np.isfinite(r["ndot_per_year"]) else 1000.0
    t_asc   = float(r["MJD_ini"]) if ("MJD_ini" in r and np.isfinite(r["MJD_ini"])) else 0.0

    m1, m2, sin_i = infer_masses_and_iota(psr, r)
    pulsars_data.append((psr, pb_days, a1_ls, ecc, t_asc, eps_us, t_obs, ndy, m1, m2, sin_i))

print(f"✅ Table cleaned: {len(df)} unique pulsars processed.")

#%% DELTA PSI SENSITIVITY CALCULATION
warnings.filterwarnings("ignore")

all_g_curves = []
n_realizations = 10

for seed in tqdm(range(n_realizations), desc="Global Progress", position=0):
    np.random.seed(seed)

    # Stochastic parameters for the Dark Matter field per pulsar
    pulsars_random = []
    for (psr, pb_days, a1, ecc, t_asc_d, eps_us, t_obs_yrs, ndot_yr, m1, m2, sin_i) in pulsars_data:
        rf        = np.random.rayleigh(1/np.sqrt(2))
        gammaf    = np.random.uniform(0, 2*np.pi)
        phif      = np.random.uniform(0, 2*np.pi)
        varthetaf = np.random.uniform(-np.pi, np.pi)
        pulsars_random.append((psr, pb_days, a1, ecc, t_asc_d, eps_us, t_obs_yrs,
                               ndot_yr, m1, m2, sin_i, rf, gammaf, phif, varthetaf))

    g_mass_sweep = []
    for m in tqdm(m_seconds, desc=f"Realization {seed+1}/{n_realizations}", position=1, leave=False):
        sum_u = 0.0

        for (psr, pb_days, a1, ecc, t_asc_d, eps_us, t_obs_yrs, ndot_yr,
             m1, m2, sin_i, rf, gammaf, phif, varthetaf) in pulsars_random:

            # Unit conversion to SI/Seconds
            pb_sec  = pb_days * 24.0 * 3600.0
            x_ls    = a1
            epsilon = eps_us * 1e-6
            t_obs   = t_obs_yrs * 365.0 * 24.0 * 3600.0
            ndot    = ndot_yr / (365.0 * 24.0 * 3600.0)
            t_asc   = t_asc_d * 24.0 * 3600.0
            wb      = 2.0 * np.pi / pb_sec

            m_sun = 2e30
            g_const = 6.67e-11
            mtot = m1 + m2
            # Semi-major axis calculation
            a_semi = (g_const * m_sun * mtot * 3.71140109219707e-26 / wb**2)**(1/3)

            # Timing variance for the psi parameter
            varpsi = 2.0 * epsilon**2 / (ndot * x_ls**2)

            delta_c = 0.1
            m_nucleon = 1.67492729e-27 * 8.52246714808644e+50

            # Time discretization
            n_points = max(2, int(np.round(t_obs * ndot)))
            t_grid = np.linspace(0.0, t_obs, n_points * 10)

            # Basis function values
            f0_vals = f0(t_grid, t_obs)
            f1_vals = f1(t_grid, t_obs)
            f2_vals = f2(t_grid, t_obs)

            # Gram-Schmidt norms
            norm0 = np.trapz(f0_vals**2, t_grid)
            norm1 = np.trapz(f1_vals**2, t_grid)
            norm2 = np.trapz(f2_vals**2, t_grid)

            m_plus  = m + wb
            m_minus = m - wb

            # Signal prefactor
            prefactor = -(1.5) * delta_c / (a_semi * m_nucleon) * np.sqrt(2.0 * dm_density) * rf * np.sin(varthetaf)

            # Analytical signal components for resonance and orbital terms
            h_vals_wb = prefactor * (
                ((1.0/m_plus + 1.0/m_minus) * np.sin(phif - gammaf) * t_grid)
                + np.sin(gammaf) * (
                    (np.sin(m_plus * t_grid - phif) + np.sin(phif)) / (m_plus**2)
                  - (np.sin(m_minus * t_grid + phif) - np.sin(phif)) / (m_minus**2)
                )
                + np.cos(gammaf) * (
                    -(np.cos(m_plus * t_grid - phif) - np.cos(phif)) / (m_plus**2)
                  + (np.cos(m_minus * t_grid + phif) - np.cos(phif)) / (m_minus**2)
                )
            )

            cos_theta = np.cos(wb * t_grid)
            sin_theta = np.sin(wb * t_grid)
            
            f0_field = (delta_c / m_nucleon) * np.sqrt(2.0 * dm_density) * rf * np.sin(m * t_grid + gammaf)

            # Perturbative changes in orbital parameters (Omega and epsilon)
            omega_dot = (f0_field * np.cos(varthetaf) / (a_semi * wb)) * (sin_theta / sin_i)
            iota = np.arcsin(sin_i)
            epsilon_dot = -(2.0 / (a_semi * wb)) * f0_field * np.sin(varthetaf) * (cos_theta * np.cos(phif) + sin_theta * np.sin(phif)) \
                          + 2.0 * (np.sin(iota / 2.0)**2) * omega_dot

            # Integration of the perturbations
            int_eps   = cumtrapz(epsilon_dot, t_grid, initial=0.0)
            int_omega = cumtrapz(omega_dot,   t_grid, initial=0.0)

            # Combined timing signal
            h_total = h_vals_wb + int_eps - int_omega

            # Residual signal (G operator) after fitting timing model
            int0_h = np.trapz(f0_vals * h_total, t_grid)
            int1_h = np.trapz(f1_vals * h_total, t_grid)
            int2_h = np.trapz(f2_vals * h_total, t_grid)

            gh_signal = h_total \
                - f0_vals * int0_h / (norm0 if norm0 != 0 else 1.0) \
                - f1_vals * int1_h / (norm1 if norm1 != 0 else 1.0) \
                - f2_vals * int2_h / (norm2 if norm2 != 0 else 1.0)

            # Accumulate signal-to-noise contribution
            u_stat = np.trapz(gh_signal**2, t_grid) / (2.0 * varpsi)
            sum_u += u_stat

        # Solve for the coupling constant g matching the Bayes factor threshold
        g_val = (np.log(1000.0) / sum_u)**0.5
        g_mass_sweep.append(g_val)

    all_g_curves.append(g_mass_sweep)

# === Save Results ===
g_results_stack = np.asarray(all_g_curves, dtype=float)
mass_grid       = np.asarray(masses_ev, dtype=float)

np.savez("g_stack_results.npz", curves=g_results_stack, masses=mass_grid)

#%%
loaded_full = np.load("g_stack_results.npz", allow_pickle=False)
g_full = np.asarray(loaded_full["curves"], dtype=float)
m_full = np.asarray(loaded_full["masses"], dtype=float)

g_full[g_full <= 0] = np.nan

plt.figure(figsize=(8, 6), dpi=100)

for i in range(g_full.shape[0]):
    plt.plot(m_full, g_full[i, :], color="gray", lw=0.5, alpha=0.3)

g_med_full = np.nanmedian(g_full, axis=0)
plt.plot(m_full, g_med_full, color="black", lw=2, label="Median (all terms)")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("m [eV]", fontsize=12)
plt.ylabel("g(m)", fontsize=12)
plt.xlim(1e-23, 5e-19)

plt.ylim(np.nanmin(g_full)*0.1, np.nanmax(g_full)*10)

plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()