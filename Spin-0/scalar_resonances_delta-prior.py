import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import quad
from scipy.special import jn, jvp
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings("ignore")

# --------------------------
# Pulsar Parameters
# --------------------------
# Name, Pb[d], x[ls], w[deg], e, T0[MJD], precision[us], Tobs[yr], cadence[yr^-1]
pulsars = [
    ("J1903+0327", 95.2, 105.6, 141.7, 0.437, 57109.4, 6.24, 10.7, 638.8),
    ("J1946+3417", 27.0, 13.9, 223.4, 0.134, 58008.3, 3.47, 5.7, 827.2),
    ("J2234+0611", 32.0, 13.9, 277.2, 0.129, 57850.1, 3.25, 6.5, 545.5),
]

# --- Physical Constants ---
dm_density = 0.3 * 4.09351499910375e+55 # GeV/cm^3 in natural units (1/s^4)
unit_conversion = 2.30817405213802e+48  # Conversion factor for output units
n_terms = 10                            # Number of terms in the summation
n_realizations = 10                      # Number of stochastic field realizations

# DM mass range scan (from 10^-22 to 10^-20 eV)
masses_ev = np.logspace(-22, -20, num=400)
m_seconds = masses_ev * 1.51926760385984e+15 # Mass converted to 1/s

# --------------------------
# Basis Functions f0, f1, f2
# --------------------------
# These functions represent the timing model components subtracted during fitting
def f0(t, t_obs): return t**2 / t_obs - t + t_obs/6
def f1(t, t_obs): return t - t_obs/2
def f2(t, t_obs): return t_obs + 0*t

# --------------------------
# Sensitivity Routine for a Single Pulsar
# --------------------------
def calc_u_for_pulsar(pb_days, a1, w_deg, ecc, t0_mjd, precision_us, t_obs_yrs, cadence_yr, m_dm, rf, gammaf):
    # Unit conversions to SI/Seconds
    wb = 2 * np.pi / (pb_days * 24 * 3600)
    w_rad = np.deg2rad(w_deg)
    t_obs_sec = t_obs_yrs * 365 * 24 * 3600
    n_dot = cadence_yr / (365 * 24 * 3600)  # Cadence in 1/s
    t0_sec = t0_mjd * 24 * 3600

    # Orbital geometry constants
    ab = a1 * np.sin(w_rad)
    nb = np.sqrt(1 - ecc** ecc) * a1 * np.cos(w_rad)
    eps = np.sqrt(1 - ecc**2)

    # Q and vartheta definitions (Noise and scaling parameters)
    q_numerator = (precision_us * 1e-6)**2
    q_denominator = n_dot * (ecc**4 + 4 * ecc**2 * (eps - 2) - 8 * eps + 8) * (nb**2 - ab**2 * (ecc**2 - 1))**2
    q_const = q_numerator / q_denominator
    vartheta = q_const * eps * ecc**2 * (
        2 * ab**2 * (ecc**2 - 1) * (ecc**2 * (eps - 2) - 2 * eps + 2)
        + nb**2 * (ecc**2 - 2) * (ecc**2 + 2 * eps - 2)
    )

    # Sampling and grid
    n_t = int(round(t_obs_sec * n_dot))
    t_vals = np.linspace(0, t_obs_sec, n_t)

    # Orthogonalization norms
    norm1 = np.trapz(f1(t_vals, t_obs_sec)**2, t_vals)
    norm2 = np.trapz(f2(t_vals, t_obs_sec)**2, t_vals)

    # ULDM Field Amplitude and values
    phi0 = np.sqrt(2 * dm_density) * rf / m_dm
    phi_vals = phi0 * np.cos(m_dm * t_vals + gammaf)
    phidot_vals = -m_dm * phi0 * np.sin(m_dm * t_vals + gammaf)
    phi_squared = phi_vals**2
    phi_phi_dot = phi_vals * phidot_vals

    # Signal calculation terms
    # n_array for vectorized summation
    n_vec = np.arange(1, n_terms)[:, None]
    t_grid = t_vals[None, :]
    nw = n_vec * wb

    # Resonance denominators
    denom1 = 4 * (nw + 2*m_dm)
    denom2 = 4 * (nw - 2*m_dm)
    denom3 = 2 * nw

    cos_p = np.cos((nw + 2*m_dm)*t_grid - t0_sec*nw + 2*gammaf)
    cos_m = np.cos((nw - 2*m_dm)*t_grid - t0_sec*nw - 2*gammaf)
    cos_wtilde = np.cos(nw*t_grid - t0_sec*nw)
    cos_const = np.cos(2*m_dm*t0_sec + 2*gammaf)

    # Bessel function values
    jn_vals = jn(n_vec, n_vec * ecc)

    term1 = (1/denom3 - cos_wtilde/denom3 - cos_p/denom1 - cos_m/denom2 + 
             cos_const/denom1 + cos_const/denom2)
    
    sum1 = np.sum(n_vec * jn_vals * phi0**2 * term1, axis=0)

    # Denominators for the second summation
    denom_a = 4 * (nw + 2*m_dm)
    denom_b = 4 * (2*m_dm - nw)
    cos_2m_minus = np.cos((2*m_dm - nw)*t_grid + t0_sec*nw + 2*gammaf)

    term2 = (cos_p/denom_a + cos_2m_minus/denom_b - cos_const/denom_a - cos_const/denom_b)
    sum2 = np.sum(jn_vals * m_dm * phi0**2 * term2, axis=0)

    # Relative change in semi-major axis
    delta_a_over_a = (
        -2*wb*sum1 
        + 0.5*(-2)*(np.cos(m_dm*t_vals + gammaf)**2 - np.cos(m_dm*t0_sec + gammaf)**2)
        + 4*(-2)*sum2
    )

    # h(t) components: Secular and periodic terms
    h_a = -1.5 * wb * cumtrapz(delta_a_over_a, t_vals, initial=0)
    h_b = (5/4) * wb * phi0**2 / (4*m_dm) * (np.sin(2*m_dm*t_vals + 2*gammaf) - np.sin(2*m_dm*t0_sec + 2*gammaf))

    h_c = np.zeros_like(t_vals)
    for n in range(1, n_terms):
        jn_val = jn(n, n*ecc)
        jn_prime = jvp(n, n*ecc)
        a_n_coeff = 4*wb*jn_val - 2*wb*np.sqrt(1-ecc**2)/ecc * n * jn_prime
        b_n_coeff = 4/n*jn_val + 4/ecc*np.sqrt(1-ecc**ecc)*jn_prime
        
        cos_term = np.cos(n*wb*(t_vals - t0_sec))
        sin_term = np.sin(n*wb*(t_vals - t0_sec))
        
        integrand = a_n_coeff*phi_squared*cos_term + 2*b_n_coeff*phi_phi_dot*sin_term
        h_c += 0.5 * cumtrapz(integrand, t_vals, initial=0)

    h_d = 5/8 * wb * phi0**2 * (t_vals - t0_sec)
    h_total = h_a + h_b + h_c + h_d

    # Orthogonal Projection (Gram-Schmidt subtraction of fitted parameters)
    int1_h = np.trapz(f1(t_vals, t_obs_sec)*h_total, t_vals)
    int2_h = np.trapz(f2(t_vals, t_obs_sec)*h_total, t_vals)
    gh_vals = h_total - f1(t_vals, t_obs_sec)*int1_h/norm1 - f2(t_vals, t_obs_sec)*int2_h/norm2

    # Signal-to-noise contribution u
    u_contribution = np.trapz(gh_vals**2, t_vals) / (2*vartheta)
    return u_contribution

# --------------------------
# Combined Sensitivity Calculation
# --------------------------
betas_all = np.zeros((len(m_seconds), n_realizations))

for r in range(n_realizations):
    np.random.seed(r)
    rf_stochastic = np.random.rayleigh(1/np.sqrt(2))
    gammaf_stochastic = np.random.uniform(0, 2*np.pi)

    for i, m in enumerate(tqdm(m_seconds, desc=f"Realization {r+1}/{n_realizations}")):
        u_combined = 0
        for (name, pb, a1, w, ecc, t0, prec, t_obs, nc) in pulsars:
            u_combined += calc_u_for_pulsar(pb, a1, w, ecc, t0, prec, t_obs, nc, m, rf_stochastic, gammaf_stochastic)
        
        # Bayes factor threshold (B=1000)
        betas_all[i, r] = np.sqrt(np.log(1000) / u_combined)

# Final median coupling strength calculation
beta_combined_median = np.median(betas_all, axis=1) * unit_conversion

# --------------------------
# Plotting and Export
# --------------------------
# Saving results
np.savez("resonance_combined_betas.npz", betas=betas_all, masses=masses_ev)

plt.figure(figsize=(8, 6), dpi=150)

# Plot each stochastic realization
for r in range(betas_all.shape[1]):
    plt.loglog(masses_ev, np.sqrt(betas_all[:, r] * unit_conversion),
               lw=1, alpha=0.4, label=f"Realization {r+1}")

# Plot the median sensitivity curve
plt.loglog(masses_ev, np.sqrt(beta_combined_median), "k", lw=3, label="Median Sensitivity")

plt.xlabel("$m$ [eV]", fontsize=14)
plt.ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]", fontsize=14)
plt.title("Combined sensitivity with orbital resonances", fontsize=15)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.xlim(1e-22, 3e-21)
plt.show()