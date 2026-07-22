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

# --- Physical Constants ---
# Local Dark Matter density in natural units
dm_density = 0.3 * 4.09351499910375e+55
# Nucleon mass conversion factor
m_nucleon  = 1.67492729e-27 * 8.52246714808644e+50
# B-L charge asymmetry between binary components
delta_c    = 0.1
# eV to 1/s conversion factor
EV_TO_INVS = 1.51926760385984e+15

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

IOTA_PRIORS_DEG    = {}
PULSAR_MASS_PRIORS = {}

# Default values if missing in CSV or priors
DEFAULT_M1       = 1.35    # Pulsar mass (M_sun)
DEFAULT_M2       = 0.20    # Companion mass (M_sun)
DEFAULT_IOTA_DEG = 60.0    # Inclination (degrees)

# === LOAD NANOGRAV 15YR TABLE ===
df = pd.read_csv(ELL1_TABLE_PATH)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["PB_d", "A1_ltsec"])

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
    psr     = str(r["pulsar"])
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

# Exact resonant masses for all pulsars (N=1 only for circular systems: m = wb)
# Each system contributes one resonance at m = 2*pi / Pb within the mass range
resonances_ev = np.array([
    (2.0 * np.pi / (pb_days * 86400.0)) / EV_TO_INVS
    for (_, pb_days, *_) in pulsars_data
    if 1e-23 <= (2.0 * np.pi / (pb_days * 86400.0)) / EV_TO_INVS <= 1e-18
])

# Mass range in eV: base log-grid plus exact resonance points
masses_ev = np.unique(np.concatenate([
    np.logspace(-23, -18, num=400),
    resonances_ev
]))
# Mass converted to frequency (1/s)
m_seconds = masses_ev * EV_TO_INVS

#%% DELTA PSI SENSITIVITY CALCULATION
warnings.filterwarnings("ignore")

all_g_curves = []
n_realizations = 10

for seed in tqdm(range(n_realizations), desc="Global Progress", position=0):
    np.random.seed(seed)

    # --- Stochastic parameters and mass-independent precomputations per pulsar ---
    # Everything that depends only on pulsar properties (not on m) is computed
    # here once per realization, avoiding redundant recalculation inside the mass loop.
    pulsars_precomp = []
    for (psr, pb_days, a1, ecc, t_asc_d, eps_us, t_obs_yrs, ndot_yr, m1, m2, sin_i) in pulsars_data:

        # Stochastic DM field parameters; varthetaf sampled isotropically on the sphere
        # via arccos(Uniform[-1,1]) to avoid polar oversampling
        rf        = np.random.rayleigh(1/np.sqrt(2))
        gammaf    = np.random.uniform(0, 2*np.pi)
        phif      = np.random.uniform(0, 2*np.pi)
        varthetaf = np.arccos(np.random.uniform(-1.0, 1.0))

        # Unit conversions
        pb_sec  = pb_days * 24.0 * 3600.0
        epsilon = eps_us * 1e-6
        t_obs   = t_obs_yrs * 365.0 * 24.0 * 3600.0
        ndot    = ndot_yr / (365.0 * 24.0 * 3600.0)
        wb      = 2.0 * np.pi / pb_sec
        iota    = np.arcsin(sin_i)

        # Semi-major axis in natural units
        m_sun   = 2e30
        g_const = 6.67e-11
        a_semi  = (g_const * m_sun * (m1+m2) * 3.71140109219707e-26 / wb**2)**(1/3)

        # Timing variance for the Psi' parameter (mass-independent)
        varpsi = 2.0 * epsilon**2 / (ndot * a1**2)

        # Time grid (fixed per pulsar, independent of m)
        n_points = max(2, int(np.round(t_obs * ndot)))
        t_grid   = np.linspace(0.0, t_obs, n_points * 30)

        # Basis function values and Gram-Schmidt norms (mass-independent)
        f0_vals = f0(t_grid, t_obs)
        f1_vals = f1(t_grid, t_obs)
        f2_vals = f2(t_grid, t_obs)
        norm0   = np.trapz(f0_vals**2, t_grid)
        norm1   = np.trapz(f1_vals**2, t_grid)
        norm2   = np.trapz(f2_vals**2, t_grid)

        # Orbital phase terms (depend on wb only, not on m)
        cos_theta = np.cos(wb * t_grid)
        sin_theta = np.sin(wb * t_grid)

        # Signal prefactor (mass-independent part); the full prefactor is
        # prefactor * sin(m*t + gammaf), where sin(m*t + gammaf) is added inside
        # the mass loop via f0_field
        prefactor_const = -(1.5) * delta_c / (a_semi * m_nucleon) \
                          * np.sqrt(2.0 * dm_density) * rf * np.sin(varthetaf)

        # F0 amplitude (mass-independent; the sin(m*t + gammaf) factor is inside mass loop)
        F0_amp = (delta_c / m_nucleon) * np.sqrt(2.0 * dm_density) * rf

        pulsars_precomp.append(dict(
            wb=wb, a_semi=a_semi, varpsi=varpsi, t_grid=t_grid,
            f0_vals=f0_vals, f1_vals=f1_vals, f2_vals=f2_vals,
            norm0=norm0, norm1=norm1, norm2=norm2,
            cos_theta=cos_theta, sin_theta=sin_theta,
            prefactor_const=prefactor_const, F0_amp=F0_amp,
            gammaf=gammaf, phif=phif, rf=rf, varthetaf=varthetaf,
            sin_i=sin_i, iota=iota,
        ))

    # --- Mass sweep ---
    g_mass_sweep = []
    for m in tqdm(m_seconds, desc=f"Realization {seed+1}/{n_realizations}", position=1, leave=False):
        sum_u = 0.0

        for pc in pulsars_precomp:
            # Unpack precomputed pulsar quantities
            wb              = pc["wb"]
            a_semi          = pc["a_semi"]
            varpsi          = pc["varpsi"]
            t_grid          = pc["t_grid"]
            f0_vals, f1_vals, f2_vals = pc["f0_vals"], pc["f1_vals"], pc["f2_vals"]
            norm1, norm2    = pc["norm1"], pc["norm2"]
            cos_theta       = pc["cos_theta"]
            sin_theta       = pc["sin_theta"]
            F0_amp          = pc["F0_amp"]
            gammaf          = pc["gammaf"] # DM field phase
            phif            = pc["phif"]   # Ascending node longitude (Upsilon_asc)
            varthetaf       = pc["varthetaf"]
            iota            = pc["iota"]
            sin_i           = pc["sin_i"]
            rf              = pc["rf"]

            # 1. Resonance frequencies
            m_plus  = m + wb
            m_minus = m - wb

            tol = 1e-18

            # 2. S(a) and C(a)
            t_phase = m * t_grid + gammaf
            
            Sa_plus = -(np.sin(m_plus * t_grid + gammaf - phif) + np.sin(phif - gammaf)) / m_plus
            Ca_plus =  -(np.cos(m_plus * t_grid + gammaf - phif) - np.cos(phif - gammaf)) / m_plus
            
            if abs(m_minus) < tol:
                # Analytical limit (L'Hôpital / Taylor)
                Sa_minus = t_grid * np.cos(gammaf + phif)
                Ca_minus = -t_grid * np.sin(gammaf + phif)
            else:
                # Normal case
                Sa_minus = (np.sin(m_minus * t_grid + gammaf + phif) - np.sin(phif + gammaf)) / m_minus
                Ca_minus = (np.cos(m_minus * t_grid + gammaf + phif) - np.cos(phif + gammaf)) / m_minus

            Sa = Sa_plus + Sa_minus
            Ca = Ca_plus + Ca_minus
            
            # S_a = - [sin(m+t + gamma - phi) + sin(phi - gamma)]/m+ + [sin(m-t + gamma + phi) - sin(phi + gamma)]/m-
            # Sa = -(np.sin(m_plus * t_grid + gammaf - phif) + np.sin(phif - gammaf)) / m_plus \
            #      +(np.sin(m_minus * t_grid + gammaf + phif) - np.sin(phif + gammaf)) / m_minus
            # Ca =  (np.cos(m_plus * t_grid + gammaf - phif) - np.cos(phif - gammaf)) / m_plus \
            #      +(np.cos(m_minus * t_grid + gammaf + phif) - np.cos(phif + gammaf)) / m_minus


            prefactor_da = (delta_c / (wb * m_nucleon * a_semi)) * np.sqrt(2.0 * dm_density) * rf * np.sin(varthetaf)

            da_over_a = prefactor_da * Sa

            h_vals_orbit = -1.5 * wb * cumtrapz(da_over_a, t_grid, initial=0.0)

            f0_field = F0_amp * np.sin(m * t_grid + gammaf)
            omega_dot   = -(f0_field * np.cos(varthetaf) / (a_semi * wb)) * (sin_theta / sin_i)
            epsilon_dot = (2.0 / (a_semi * wb)) * f0_field * np.sin(varthetaf) \
                          * (cos_theta * np.cos(phif) + sin_theta * np.sin(phif)) \
                          + 2.0 * (np.sin(iota / 2.0)**2) * omega_dot

            int_eps   = cumtrapz(epsilon_dot, t_grid, initial=0.0)
            int_omega = cumtrapz(omega_dot,   t_grid, initial=0.0)

            h_total = h_vals_orbit + int_eps - int_omega

            int1_h = np.trapz(f1_vals * h_total, t_grid)
            int2_h = np.trapz(f2_vals * h_total, t_grid)

            gh_signal = h_total \
                - f1_vals * int1_h / (norm1 if norm1 != 0 else 1.0) \
                - f2_vals * int2_h / (norm2 if norm2 != 0 else 1.0)

            u_stat = np.trapz(gh_signal**2, t_grid) / (2.0 * varpsi)
            sum_u += u_stat

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
plt.xlim(1e-23, 1e-18)

plt.ylim(np.nanmin(g_full)*0.1, np.nanmax(g_full)*10)

plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()