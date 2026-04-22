import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
})

constraints_dir = "other_constraints"
main_data_file = "g_stack_results.npz"

data = np.load(main_data_file, allow_pickle=True)
all_g_curves = np.asarray(data["curves"])
mass_data = np.asarray(data["masses"])

fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)

# Main realizations and median
for r in range(all_g_curves.shape[0]):
    ax.plot(mass_data, all_g_curves[r], alpha=0.25, lw=0.5) 

g_median_main = np.median(all_g_curves, axis=0)
ax.plot(mass_data, g_median_main, color="black", lw=1.5, label="Combined circular orbits", zorder=10)

# Axis scaling and limits
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-23, 1e-18)
ax.set_ylim(1e-27, 1e-14)

ax.set_xlabel(r"$m$ [eV]", fontsize=10)
ax.set_ylabel("g", fontsize=10, rotation=0, labelpad=10)

# Ticks configuration
ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))
ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))
ax.tick_params(which='both', direction='out')
ax.tick_params(axis='both', which='major', length=4, width=0.7, labelsize=8)
ax.tick_params(axis='both', which='minor', length=2, width=0.5)

# External constraints setup
external_files = [
    ("PPTA", "PPTA", "brown"),
    ("Eot-Wash-DM", "Eöt-Wash (DM)", "darkred"),
    ("Eot-Wash-EP", "Eöt-Wash (EP)", "green"),
    ("LISAPathfinder", "LISA Pathfinder", "purple"),
    ("LISA-Path-relative-acc", "LISA Pathfinder (rel. acc.)", "indigo"),
]

masses_dict, gs_dict, colors_dict = {}, {}, {}

for filename, label, color in external_files:
    path = os.path.join(constraints_dir, filename + ".txt")
    if os.path.exists(path):
        arr = np.loadtxt(path, skiprows=3, usecols=(0, 1))
        masses_dict[filename], gs_dict[filename], colors_dict[filename] = arr[:, 0], arr[:, 1], color

for key in masses_dict:
    if key in ["LISAPathfinder", "Eot-Wash-DM"]: 
        continue

    m, g = masses_dict[key], gs_dict[key]
    z_order, alpha_val = (3, 0.7) if key in ["PPTA", "LISA-Path-relative-acc"] else (2, 0.3) if key == "Eot-Wash-EP" else (1, 0.1)

    ax.fill_between(m, g, ax.get_ylim()[1], color=colors_dict[key], alpha=alpha_val, zorder=z_order)
    ax.plot(m, g, color=colors_dict[key], lw=1, zorder=z_order + 0.1)

# --- Pulsar B1913+16 Data ---
data_B = np.load("g_results_eccentric_B1913_16.npz")
m_B, curves_B = data_B["masses"], data_B["curves"]

for r in range(curves_B.shape[1]):
    ax.loglog(m_B, curves_B[:, r], color="orange", alpha=0.3, lw=0.8, zorder=8)

ax.loglog(m_B, np.median(curves_B, axis=1), color="orange", lw=1.2, label="B1913+16", zorder=9)

# --- Pulsar J1903+0327 Data ---
data_J = np.load("g_results_eccentric_J1903_0327.npz")
m_J, curves_J = data_J["masses"], data_J["curves"]

for r in range(curves_J.shape[1]):
    ax.loglog(m_J, curves_J[:, r], color="blue", alpha=0.1, lw=0.5, zorder=3)

ax.loglog(m_J, np.median(curves_J, axis=1), color="blue", lw=1.0, label="J1903+0327", zorder=4)

# --- Annotations ---
text_effects = [pe.withStroke(linewidth=0.8, foreground="black")]
ax.text(1e-19, 1e-25, "MICROSCOPE", color="white", fontsize=7, va="bottom", ha="right", weight="bold", path_effects=text_effects)
ax.text(5e-23, 1e-25, "PPTA", color="white", fontsize=7, va="bottom", ha="right", weight="bold", path_effects=text_effects)
ax.text(2e-19, 0.6e-23, 'Eöt-Wash (EP)', color="white", fontsize=7, va="bottom", ha="right", weight="bold", path_effects=text_effects)
ax.text(0.7e-18, 2e-25, 'LISA Pathfinder', color="white", fontsize=7, va="bottom", ha="right", weight="bold", rotation=90, path_effects=text_effects)

ax.legend(fontsize=6, loc="upper left", frameon=True, ncol=1, handlelength=1.2, facecolor='white', framealpha=1, edgecolor="white")

# Background shading for MICROSCOPE
if "Eot-Wash-EP" in masses_dict:
    ax.fill_between(masses_dict["Eot-Wash-EP"], 7e-26, ax.get_ylim()[1], color="grey", alpha=0.4)
ax.axhline(y=7e-26, color="grey", lw=0.8)

plt.tight_layout()
plt.savefig("constraints_on_g.pdf", bbox_inches='tight')
plt.show()