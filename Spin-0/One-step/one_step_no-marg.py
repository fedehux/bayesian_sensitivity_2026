import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.special import jn, jvp
from tqdm import tqdm
import warnings

# --- Global Configuration ---
warnings.filterwarnings("ignore")

# --- System Parameters: J1903+0327 ---
Pb = 95.2 * 24 * 3600              # Orbital period in seconds
wb = 2 * np.pi / Pb
a1 = 105.6                         # Projected semi-major axis in lt-sec
w_rad = np.deg2rad(141.7)          # Argument of periastron in radians
ecc = 0.437                        # Eccentricity
T0 = 57109.4                       # Time of periastron (MJD)
epsilon_toa = 6.24e-6              # TOA precision (seconds)
Tobs = 10.7 * 365 * 24 * 3600      # Observation time in seconds
ndot = 638.8 / (365 * 24 * 3600)   # Cadence in 1/seconds
dm_density = 0.3 * 4.09351499910375e+55 # Local DM density in natural units

# --- Orbital Constants ---
ab = a1 * np.sin(w_rad)             # alpha_b
nb = np.sqrt(1 - ecc**2) * a1 * np.cos(w_rad) # eta_b
sqrt1me2 = np.sqrt(1 - ecc**2)

# --- Mass and Frequency Grid ---
masses_ev = np.logspace(-22, -20.5, num=800)
m_seconds = masses_ev * 1.51926760385984e+15 # Mass in 1/s

# --- Simulation Configuration ---
n_realizations = 10
n_terms = 10                       # Number of Bessel terms
N_t = int(round(Tobs * ndot))
t_vals = np.linspace(0, Tobs, N_t)

beta_m_theta = []

# --- Main Calculation Loop ---
for m in tqdm(m_seconds, desc="Calculating sensitivity (Full one-step)"):
    betas_realizations = []

    for r in range(n_realizations):
        np.random.seed(r)
        rf = np.random.rayleigh(1 / np.sqrt(2))
        gammaf = np.random.uniform(0, 2 * np.pi)
        phi0 = np.sqrt(2 * dm_density) / m * rf 

        # Field and orbital quantities
        phi_vals = phi0 * np.cos(m * t_vals + gammaf)
        phidot_vals = -m * phi0 * np.sin(m * t_vals + gammaf)
        phi_sq = phi_vals**2
        phi_phi_dot = phi_vals * phidot_vals

        n_array = np.arange(1, n_terms)[:, None]
        t_grid = t_vals[None, :]
        nw = n_array * wb

        # Resonances
        denom1 = 4 * (nw + 2 * m)
        denom2 = 4 * (nw - 2 * m)
        denom3 = 2 * nw

        cos1 = np.cos((nw + 2 * m) * t_grid - T0 * nw + 2 * gammaf)
        cos2 = np.cos((-nw + 2 * m) * t_grid + T0 * nw + 2 * gammaf)
        cos3 = np.cos(nw * t_grid - T0 * nw)
        cos4 = np.cos(2 * m * T0 + 2 * gammaf)

        jn_vals = jn(n_array, n_array * ecc)

        term1 = -cos1 / denom1 - cos2 / denom2 - cos3 / denom3 + cos4 / denom1 + cos4 / denom2 - 1 / denom3
        sum1 = np.sum(n_array * jn_vals * phi0**2 * term1, axis=0)

        cos5 = np.cos((nw - 2 * m) * t_grid - T0 * nw - 2 * gammaf)
        cos6 = np.cos((nw + 2 * m) * t_grid - T0 * nw + 2 * gammaf)

        term2 = -cos5 / denom2 + cos6 / denom1 + cos4 / denom2 - cos4 / denom1
        sum2 = np.sum(jn_vals * m * phi0**2 * term2, axis=0)

        # --- delta a / a ---
        delta_a_over_a = -2 * wb * sum1 + 0.5 * (-2) * (np.cos(m * t_vals + gammaf)**2 - np.cos(m * T0 + gammaf)**2) + 4 * (-2) * sum2
        integral_da_a = cumulative_trapezoid(delta_a_over_a, t_vals, initial=0)

        # --- delta e(t) ---
        t_tilde = (t_vals - T0)[None, :]
        cos2T0 = np.cos(2 * m * T0 + 2 * gammaf)
        A1 = (1 - np.cos(nw * t_tilde)) / (2 * nw)
        A2 = 0.25 * ((cos2T0 - np.cos((nw + 2 * m) * t_tilde + 2 * m * T0 + 2 * gammaf)) / (nw + 2 * m) + 
                     (cos2T0 - np.cos((nw - 2 * m) * t_tilde + 2 * m * T0 + 2 * gammaf)) / (nw - 2 * m))
        sum_e1 = -wb * np.sum(n_array * jn_vals * (A1 + A2), axis=0)
        
        B_sum = (cos2T0 - np.cos((2 * m + nw) * t_tilde + 2 * m * T0 + 2 * gammaf)) / (2 * m + nw) + \
                (cos2T0 - np.cos((2 * m - nw) * t_tilde + 2 * m * T0 + 2 * gammaf)) / (2 * m - nw)
        sum_e2 = m * np.sum(jn_vals * B_sum, axis=0)
        delta_e = phi0**2 * (1 - ecc**2) / ecc * (sum_e1 + sum_e2)

        # --- delta omega(t) ---
        jnp_vals = jvp(n_array, n_array * ecc)
        sum_w1 = np.sum(jnp_vals * 0.5 * (np.sin(nw * t_tilde + nw * T0) - np.sin(nw * T0)), axis=0)
        
        C_den_p, C_den_m = (nw + 2 * m), (nw - 2 * m)
        C_sum = (cos2T0 - np.cos(C_den_p * t_tilde + 2 * m * T0 + 2 * gammaf)) / C_den_p + \
                (cos2T0 - np.cos(C_den_m * t_tilde - 2 * m * T0 - 2 * gammaf)) / C_den_m
        sum_w2 = (wb / 4) * np.sum(n_array * jnp_vals * C_sum, axis=0)
        
        W_den_m, W_den_p = (2 * m - nw), (2 * m + nw)
        W_sum = (cos2T0 - np.cos(W_den_m * t_tilde + 2 * m * T0 + 2 * gammaf)) / W_den_m - \
                (cos2T0 - np.cos(W_den_p * t_tilde + 2 * m * T0 + 2 * gammaf)) / W_den_p
        sum_w3 = m * np.sum(jnp_vals * W_sum, axis=0)
        delta_omega = phi0**2 * sqrt1me2 / ecc * (sum_w1 + sum_w2 + sum_w3)

        # --- delta Theta' component ---
        h_a = -1.5 * wb * integral_da_a
        h_b = (5/4) * wb * phi0**2 / (4 * m) * (np.sin(2 * m * t_vals + 2 * gammaf) - np.sin(2 * m * T0 + 2 * gammaf))
        
        h_c = np.zeros_like(t_vals)
        for n in range(1, n_terms):
            jn_val, jnp_val = jn(n, n * ecc), jvp(n, n * ecc)
            an = 4 * wb * jn_val - 2 * wb * sqrt1me2 / ecc * n * jnp_val
            bn = 4 / n * jn_val + 4 / ecc * sqrt1me2 * jnp_val
            integrand = an * phi_sq * np.cos(n * wb * (t_vals - T0)) + 2 * bn * phi_phi_dot * np.sin(n * wb * (t_vals - T0))
            h_c += 0.5 * cumulative_trapezoid(integrand, t_vals, initial=0)
            
        h_d = (5/8) * wb * phi0**2 * (t_vals - T0)
        h_total = h_a + h_b + h_c + h_d

        # Kepler Equation
        mean_anom = wb * (t_vals - T0)
        E = mean_anom.copy()
        for _ in range(5):
            E -= (E - ecc * np.sin(E) - mean_anom) / (1 - ecc * np.cos(E))

        # BT Geometry terms
        cosE, sinE = np.cos(E), np.sin(E)
        geom_a = ab * (cosE - ecc) + nb * sinE
        geom_e = ab + (sinE * (ab * sinE - nb * cosE)) / (1 - ecc * cosE) + ecc * nb / (1 - ecc**2) * sinE
        geom_th = (ab * sinE - nb * cosE) / (1 - ecc * cosE)
        geom_om = (nb / sqrt1me2) * (cosE - ecc) - sqrt1me2 * ab * sinE

        # Combine for total timing residual Q(t)
        Q_total = (delta_a_over_a * geom_a) - (delta_e * geom_e) - (h_total * geom_th) + (delta_omega * geom_om)

        sigma = epsilon_toa / np.sqrt(np.sum(Q_total**2))
        betas_realizations.append(3 * sigma)

    beta_m_theta.append(betas_realizations)

# --- Save Results ---
unit_conversion = 2.30817405213802e+48
beta_m_theta = np.array(beta_m_theta) * unit_conversion
header_str = "mass (eV) " + " ".join([f"beta_real{i}" for i in range(n_realizations)])
np.savetxt("results_one_step_complete.txt", np.column_stack((masses_ev, beta_m_theta)), header=header_str, fmt="%.5e")

# --- Plotting ---
plt.figure(figsize=(7, 5), dpi=130)
beta_full_median = np.median(beta_m_theta, axis=1)
plt.loglog(masses_ev, np.sqrt(beta_full_median), color="red", lw=2, label="Median (Full)")

plt.xlabel("Mass [eV]", fontsize=11)
plt.ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]", fontsize=11)
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()