import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.special import jn, jvp  # Bessel functions of the first kind
from tqdm import tqdm
import os
from scipy.integrate import IntegrationWarning
import warnings

# --- Global Configuration ---
warnings.filterwarnings("ignore")

# --- System Parameters: J1903+0327 ---
Pb = 95.2 * 24 * 3600              # Orbital period in seconds
wb = 2 * np.pi / Pb
a1 = 105.6                         # Projected semi-major axis in light-seconds
w = 141.7 / 180 * np.pi            # Argument of periastron in radians
e = 0.437                          # Eccentricity
T0 = 57109.4                       # Time of periastron (MJD)
epsilon = 6.24e-6                  # TOA precision
Tobs = 10.7 * 365 * 24 * 3600      # Observation time in seconds
ndot = 638.8 / (365 * 24 * 3600)   # Cadence in 1/second
dm_density = 0.3 * 4.09351499910375e+55 # GeV / cm^3 to 1/ s^4 (natural units)

# --- Orbital Constants ---
x = a1
ab = x * np.sin(w)                 # alpha_b
nb = (1-e**2)**0.5 * a1 * np.cos(w) # eta_b
eps_factor = np.sqrt(1 - e**2)     # sqrt(1 - e^2)

# Q factor definition
q_num = epsilon**2
q_den = ndot * (e**4 + 4 * e**2 * (eps_factor - 2) - 8 * eps_factor + 8) * (nb**2 - ab**2 * (e**2 - 1))**2
Q = q_num / q_den

# var(delta Theta')
vartheta = Q * eps_factor * e**2 * (
    2 * ab**2 * (e**2 - 1) * (e**2 * (eps_factor - 2) - 2 * eps_factor + 2)
    + nb**2 * (e**2 - 2) * (e**2 + 2 * eps_factor - 2)
)

masses_ev = np.logspace(-22, -20.5, num=1200)
m_s = masses_ev * 1.51926760385984e+15 # Mass in 1/s

# --- Simulation Settings ---
n_realizations = 5
n_terms = 10
N_t = round(Tobs * ndot)
t_vals = np.linspace(T0, T0 + Tobs, N_t)

# Storage for betas per mass and realization
betas_all = np.zeros((len(m_s), n_realizations))

# --- Main Calculation Loop ---
for r in range(n_realizations):
    np.random.seed(r)  # Fixed seed per realization
    rf = np.random.rayleigh(1 / np.sqrt(2))
    gammaf = np.random.uniform(0, 2 * np.pi)

    for i, m in enumerate(tqdm(m_s, desc=f"Realization {r+1}/{n_realizations}")):
        phi0 = np.sqrt(2 * dm_density) / m * rf

        phi_vals = phi0 * np.cos(m * t_vals + gammaf)
        phidot_vals = -m * phi0 * np.sin(m * t_vals + gammaf)
        phi_squared = phi_vals**2
        phi_phi_dot = phi_vals * phidot_vals

        n_array = np.arange(1, n_terms)[:, None]
        t = t_vals[None, :]
        nw = n_array * wb

        denom1 = 4 * (nw + 2 * m)
        denom2 = 4 * (nw - 2 * m)
        denom3 = 2 * nw

        cos1 = np.cos((nw + 2 * m) * t - T0 * nw + 2 * gammaf)
        cos2 = np.cos((-nw + 2 * m) * t + T0 * nw + 2 * gammaf)
        cos3 = np.cos(nw * t - T0 * nw)
        cos4 = np.cos(2 * m * T0 + 2 * gammaf)

        jn_vals = jn(n_array, n_array * e)

        term1 = -cos1 / denom1 - cos2 / denom2 - cos3 / denom3 \
                + cos4 / denom1 + cos4 / denom2 - 1 / denom3
        sum1 = np.sum(n_array * jn_vals * phi0**2 * term1, axis=0)

        cos5 = np.cos((nw - 2 * m) * t - T0 * nw - 2 * gammaf)
        cos6 = np.cos((nw + 2 * m) * t - T0 * nw + 2 * gammaf)

        term2 = - cos5 / denom2 + cos6 / denom1 + cos4 / denom2 - cos4 / denom1
        sum2 = np.sum(jn_vals * m * phi0**2 * term2, axis=0)

        delta_a_over_a = (-2 * wb * sum1
                          + 0.5 * (-2) * (np.cos(m * t_vals + gammaf)**2
                                          - np.cos(m * T0 + gammaf)**2)
                          + 4 * (-2) * sum2)
        integral_delta_a_a = cumtrapz(delta_a_over_a, t_vals, initial=0)

        # 1) Mean anomaly
        M_vals = wb * (t_vals - T0)

        # 2) Initialize E = M
        E = M_vals.copy()

        # 3) Newton solver for Kepler's equation
        for _ in range(5):
            f_kepler = E - e * np.sin(E) - M_vals
            df_kepler = 1 - e * np.cos(E)
            E -= f_kepler / df_kepler

        # Conversion to true anomaly
        true_anom = np.arctan((np.sqrt(1-e**2) * np.sin(E)) / (np.cos(E) - e))

        # f_dot components
        p1 = phi_squared/4 * wb / (1-e**2)**1.5 * (1 - e * np.cos(true_anom))**2
        p2 = -1.5 * wb * delta_a_over_a / (1-e**2)**1.5 * (1 + e * np.cos(true_anom))
        p3 = 1/e * (2 * phi_phi_dot * np.sin(true_anom)
                    - wb * (1-e**2)**0.5 * phi_squared / 2 * np.cos(true_anom))
        
        f_dot = p1 + p2 + p3
        f_vals = cumtrapz(f_dot, t_vals, initial=0)

        # Signal template Q(t)
        q_vals = - (ab * np.sin(E) - nb * np.cos(E)) / (1 - e * np.cos(E)) \
                 * f_vals * (1 - e * np.cos(E)) * (1-e**2)**0.5 / (1 + e * np.cos(true_anom))

        q_sq_sum = np.sum(q_vals**2)

        sigma = epsilon / np.sqrt(q_sq_sum)
        beta = 3 * sigma

        betas_all[i, r] = beta

# --- Unit Conversion ---
unit_conversion = 2.30817405213802e+48
betas_all = betas_all * unit_conversion  # shape (n_masses, n_realizations)

# Save data: first column = masses, rest = realizations
output_data = np.column_stack((masses_ev, betas_all))
np.savetxt("beta_mass_anomaly_one_step_true_anomaly.txt", output_data,
           header="mass (eV) " + " ".join([f"beta_real{i}" for i in range(n_realizations)]),
           fmt="%.5e")

# --- Final Plotting ---
plt.figure(dpi=100)

for i in range(n_realizations):
    plt.loglog(masses_ev, np.sqrt(betas_all[:, i]), label=f"Realization {i+1}", linestyle="-")

plt.xlabel("Mass [eV]")
plt.ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]")
plt.tight_layout()
plt.savefig("beta_realizations_true_anomaly.pdf")
plt.show()