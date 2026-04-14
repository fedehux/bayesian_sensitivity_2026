import numpy as np 
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

def load_npz_curves(fname):
    """Returns (masses, beta_realizations [R x M], beta_median [M]) or None"""
    if not os.path.exists(fname):
        return None
    data = np.load(fname)
    masas = data["masas"]
    betas = data["betas"]  # shape: (n_realizations, n_masses)
    beta_med = np.nanmedian(betas, axis=0)
    return masas, betas, beta_med

def plot_all_and_median(ax, masas, betas, beta_med, label_median, color,
                        lw_all=1.5, alpha_all=0.25, lw_med=3.0,
                        ls_all='-', ls_med='-'):
    for r in range(betas.shape[0]):
        ax.loglog(masas, np.sqrt(betas[r, :]),
                  lw=lw_all, alpha=alpha_all, color=color, ls=ls_all)
    ax.loglog(masas, np.sqrt(beta_med),
              lw=lw_med, color=color, ls=ls_med, label=label_median, alpha=1)

# δψ (delta prior)
psi_npz = load_npz_curves("beta_psi_delta_all_realizations.npz")
# δx (delta prior)
x_npz = load_npz_curves("beta_x_delta_all_realizations.npz")
# Gaussiano (ψ)
gauss_psi_npz = load_npz_curves("beta_psi_gaussian_all_realizations.npz")               # (psi gaussiano)
# Gaussiano (x)
gauss_x_npz   = load_npz_curves("beta_x_gaussian_all_realizations.npz")       # (x gaussiano)

# -------------------- external references: Cassini & PTA --------------------
# Cassini
datos_cas = np.loadtxt("cassini_betas.txt")
m_cas = datos_cas[:, 0]
b_cas = datos_cas[:, 1]
# PTA
dat_pta = np.loadtxt("PTA.txt", delimiter=",")
m_pta = dat_pta[:, 0]
b_pta = dat_pta[:, 1]

# -------------------- style --------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
})

fig, ax = plt.subplots(figsize=(3.3, 2.5), dpi=300)

color_psi = "steelblue"
color_x = "darkorange"

if psi_npz is not None:
    m_psi, betas_psi, beta_med_psi = psi_npz
    plot_all_and_median(ax, m_psi, betas_psi, beta_med_psi,
                        r"$\delta \Psi'$ | $\delta$-prior",
                        color=color_psi, ls_all='--', ls_med='--',
                        lw_all=0.3, lw_med=0.8)

if x_npz is not None:
    m_x, betas_x, beta_med_x = x_npz
    plot_all_and_median(ax, m_x, betas_x, beta_med_x,
                        r"$\delta x$ | $\delta$-prior",
                        color=color_x, ls_all='--', ls_med='--',
                        lw_all=0.3, lw_med=0.8)

if gauss_psi_npz is not None:
    m_gp, betas_gp, beta_med_gp = gauss_psi_npz
    plot_all_and_median(ax, m_gp, betas_gp, beta_med_gp,
                        r"$\delta \Psi'$ | Gaussian prior",
                        color=color_psi, ls_all='-', ls_med='-',
                        lw_all=0.3, lw_med=0.8)

if gauss_x_npz is not None:
    m_gx, betas_gx, beta_med_gx = gauss_x_npz
    plot_all_and_median(ax, m_gx, betas_gx, beta_med_gx,
                        r"$\delta x$ | Gaussian prior",
                        color=color_x, ls_all='-', ls_med='-',
                        lw_all=0.3, lw_med=0.8)


resonance_npz = np.load("resonance_combined_betas.npz")

if resonance_npz is not None:
    m_res = resonance_npz["masses"]
    betas_res = (resonance_npz["betas"] * 2.30817405213802e+48).T 
    
    beta_med_res = np.median(betas_res, axis=0)
    
    mask = m_res < 3e-21
    
    plot_all_and_median(ax, 
                        m_res[mask], 
                        betas_res[:, mask], 
                        beta_med_res[mask],
                        r"$\delta \Theta '$ | Eccentric",
                        color="black", 
                        ls_all='-', 
                        ls_med='-',
                        lw_all=0.3, 
                        lw_med=1.0)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-23, 1e-18)
ax.set_ylim(1e-19, 1e-9)

ax.set_xlabel(r"$m$ [eV]", fontsize=10)
ax.set_ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]", fontsize=10, labelpad=2)
ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))
ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.tick_params(axis="both", which="major", length=4, width=0.7, labelsize=8)
ax.tick_params(axis="both", which="minor", length=2, width=0.5)

ax.fill_between(m_pta, b_pta, y2=ax.get_ylim()[1], color="green", alpha=0.15, zorder=1)
ax.plot(m_pta, b_pta, color="green", lw=0.8, zorder=1.1)
ax.fill_between(m_cas, b_cas, y2=ax.get_ylim()[1], color="red", alpha=0.15, zorder=1)
ax.plot(m_cas, b_cas, color="red", lw=0.8, zorder=1.1)

ax.text(3.5e-23, 1.5e-18, "PTA", color="white", fontsize=7, va="bottom", ha="right", weight="bold",
        path_effects=[pe.withStroke(linewidth=1.0, foreground="black")])
ax.text(1.3e-20, 1.0e-15, "Cassini", color="white", fontsize=7, va="bottom", ha="right", weight="bold",
        path_effects=[pe.withStroke(linewidth=1.0, foreground="black")])

ax.legend(fontsize=6, loc="lower right", frameon=False, ncol=1, handlelength=1.2)
plt.tight_layout()
plt.savefig("beta_mass_constraints.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()