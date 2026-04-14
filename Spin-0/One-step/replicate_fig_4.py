import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
})

datos_resonancia_comp = np.loadtxt("beta_resonances_delta_prior.txt")
m_res_comp = datos_resonancia_comp[:,0]
sqrt_beta_comp = np.sqrt(datos_resonancia_comp[:,1])

datos_one_step_theta = np.loadtxt("results_one_step_complete.txt")
m_one_theta = datos_one_step_theta[:,0]
sqrt_beta_one_theta = np.sqrt(datos_one_step_theta[:,1])

datos_one_step_theta_cortado = np.loadtxt("beta_mass_theta_one_step_marginalized.txt")
m_one_theta_cort = datos_one_step_theta_cortado[:,0]
sqrt_beta_one_theta_cort = np.sqrt(datos_one_step_theta_cortado[:,1])

fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)

ax.plot(m_one_theta, sqrt_beta_one_theta, color="C0", lw=1, 
        label="One-step, without marginalization")

ax.plot(m_res_comp, sqrt_beta_comp, color="C1", lw=1.2, 
        label="Two-step", zorder=10)

ax.plot(m_one_theta_cort, sqrt_beta_one_theta_cort, color="C2", lw=1, 
        label="One-step, marginalized")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-22, 3e-21)

ax.set_xlabel(r"$m$ [eV]", fontsize=10)
ax.set_ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]", fontsize=10)

ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))
ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.tick_params(which='both', direction='out')

ax.tick_params(axis='both', which='major', length=4, width=0.7, labelsize=8)
ax.tick_params(axis='both', which='minor', length=2, width=0.5)

ax.legend(fontsize=6.5, frameon=True, loc='best')

plt.tight_layout()
plt.savefig("Comparison_methods.pdf", bbox_inches='tight')
plt.show()