import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
import matplotlib.pyplot as plt
from scipy.special import jn, jvp  # Bessel functions of the first kind
from tqdm import tqdm
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
epsilon = 6.3e-6                   # TOA precision (seconds)
Tobs = 15 * 365 * 24 * 3600        # Observation time in seconds
ndot = 638.8 / (365 * 24 * 3600)   # Cadence in 1/seconds
dm_density = 0.3 * 4.09351499910375e+55 # GeV / cm^3 to 1/s^4 (natural units)

# --- Orbital Constants ---
x_ls = a1
ab = x_ls * np.sin(w)               # alpha_b
nb = (1-e**2)**0.5 * a1 * np.cos(w) # eta_b
eps_factor = np.sqrt(1 - e**2)      # sqrt(1 - e^2) factor

# --- Mass and Frequency Grid ---
masses_ev = np.logspace(-22, -20.5, num=800)
m_s = masses_ev * 1.51926760385984e+15 # Mass in 1/s

# --- Simulation Configuration ---
n_realizations = 10
beta_m_theta = []  # Storage for results
n_terms = 10       # Number of Bessel terms
N_t = round(Tobs * ndot)
t_vals = np.linspace(0, Tobs, N_t)

# --- Main Calculation Loop ---
for m in tqdm(m_s, desc="Calculating sensitivity (marginalized one-step)"):
    betas_realizations = []

    for r in range(n_realizations):
        np.random.seed(r)
        rf = np.random.rayleigh(1 / np.sqrt(2))
        gammaf = np.random.uniform(0, 2 * np.pi)
        phi0 = np.sqrt(2 * dm_density) / m * rf 

        # --- Signal Calculation ---
        phi_vals = phi0 * np.cos(m * t_vals + gammaf)
        phidot_vals = -m * phi0 * np.sin(m * t_vals + gammaf)
        phi_squared = phi_vals**2
        phi_phi_dot = phi_vals * phidot_vals

        # Resonant component calculation (Bessel)
        n_array = np.arange(1, n_terms)[:, None]
        t_grid = t_vals[None, :]
        nw = n_array * wb

        denom1 = 4 * (nw + 2 * m)
        denom2 = 4 * (nw - 2 * m)
        denom3 = 2 * nw

        cos1 = np.cos((nw + 2 * m) * t_grid - T0 * nw + 2 * gammaf)
        cos2 = np.cos((-nw + 2 * m) * t_grid + T0 * nw + 2 * gammaf)
        cos3 = np.cos(nw * t_grid - T0 * nw)
        cos4 = np.cos(2 * m * T0 + 2 * gammaf)

        jn_vals = jn(n_array, n_array * e)

        term1 = -cos1 / denom1 - cos2 / denom2 - cos3 / denom3 + cos4 / denom1 + cos4 / denom2 - 1 / denom3
        sum1 = np.sum(n_array * jn_vals * phi0**2 * term1, axis=0)

        cos5 = np.cos((nw - 2 * m) * t_grid - T0 * nw - 2 * gammaf)
        cos6 = np.cos((nw + 2 * m) * t_grid - T0 * nw + 2 * gammaf)

        term2 = - cos5 / denom2 + cos6 / denom1 + cos4 / denom2 - cos4 / denom1
        sum2 = np.sum(jn_vals * m * phi0**2 * term2, axis=0)

        # h(t) Signal components
        delta_a_over_a_vals = -2 * wb * sum1 + 0.5 * (-2) * (np.cos(m * t_vals + gammaf)**2 - np.cos(m * T0 + gammaf)**2) + 4 * (-2) * sum2
        integral_da_a = cumulative_trapezoid(delta_a_over_a_vals, t_vals, initial=0)

        h_a = -1.5 * wb * integral_da_a
        h_b = (5 / 4) * wb * phi0**2 / (4 * m) * (np.sin(2 * m * t_vals + 2 * gammaf) - np.sin(2 * m * T0 + 2 * gammaf))

        h_c = np.zeros_like(t_vals)
        for n in range(1, n_terms):
            jn_val = jn(n, n * e)
            jn_prime = jvp(n, n * e)
            an_coeff = 4 * wb * jn_val - 2 * wb * np.sqrt(1 - e**2) / e * n * jn_prime
            bn_coeff = 4 / n * jn_val + 4 / e * np.sqrt(1 - e**2) * jn_prime
            
            integrand = an_coeff * phi_squared * np.cos(n * wb * (t_vals - T0)) + 2 * bn_coeff * phi_phi_dot * np.sin(n * wb * (t_vals - T0))
            h_c += 0.5 * cumulative_trapezoid(integrand, t_vals, initial=0)
        
        h_d = 5/8 * wb * phi0**2 * (t_vals - T0)
        h_total = h_a + h_b + h_c + h_d

        # --- Kepler Solver (E) ---
        mean_anomaly = wb * (t_vals - T0)
        E = mean_anomaly.copy()
        for _ in range(5):  # Newton-Raphson method
            f_val = E - e * np.sin(E) - mean_anomaly
            df_val = 1 - e * np.cos(E)
            E -= f_val / df_val
            
        # --- Fisher Information Matrix (Marginalization) ---
        common_factor = (ab * np.sin(E) - nb * np.cos(E)) / (1 - e * np.cos(E))
        h_template = - common_factor * h_total
        g_template = 1.5 * wb / a1 * (t_vals) * common_factor

        F11 = np.sum(2 * h_template * h_template) / epsilon**2
        F12 = np.sum(2 * h_template * g_template) / epsilon**2
        F22 = np.sum(2 * g_template * g_template) / epsilon**2
        
        F_inv_11 = F22 / (F11 * F22 - F12**2)
        sigma_beta = np.sqrt(F_inv_11)
        betas_realizations.append(3 * sigma_beta) # 3-sigma sensitivity level

    beta_m_theta.append(betas_realizations)

# --- Data Processing and Storage ---
unit_conversion = 2.30817405213802e+48  # s^2 to GeV^-2 conversion
beta_m_theta = np.array(beta_m_theta) * unit_conversion

# Save results
output_filename = "beta_mass_theta_one_step_marginalized.txt"
header_str = "mass (eV) " + " ".join([f"beta_real{i}" for i in range(n_realizations)])
np.savetxt(output_filename, np.column_stack((masses_ev, beta_m_theta)), 
           header=header_str, fmt="%.5e")

# --- Final Visualization ---
plt.figure(figsize=(8, 6), dpi=100)

for i in range(n_realizations):
    plt.loglog(masses_ev, np.sqrt(beta_m_theta[:, i]), label=f"Realization {i+1}", lw=1.5)

plt.xlabel("Mass (eV)", fontsize=11)
plt.ylabel(r"$\sqrt{\beta}$ (GeV$^{-1}$)", fontsize=11)
plt.title(r"Combined sensitivity ($\Theta'$ variable - marginalized one-step)", fontsize=12)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()
# plt.savefig("beta_mass_theta_one_step_plot.pdf")
plt.show()

print(f"✅ Process complete. Data saved in: {output_filename}")