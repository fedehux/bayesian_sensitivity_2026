#%% LIBRERÍAS 
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

otras = "other_constraints"

datafile = "g_stack_results.npz"

data = np.load(datafile, allow_pickle=True)

all_g_curves = np.asarray(data["curves"])
masas_data = np.asarray(data["masses"])

fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)

for r in range(all_g_curves.shape[0]):
    ax.plot(masas_data, all_g_curves[r], alpha=0.25, lw=0.5) 

g_med = np.median(all_g_curves, axis=0)
ax.plot(masas_data, g_med, color="black", lw=1.5, label="Mediana realizaciones", zorder=10)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-23, 5.5e-19)
ax.set_ylim(1e-27, 1e-16)

ax.set_xlabel(r"$m$ [eV]", fontsize=10)
ax.set_ylabel("g", fontsize=10, rotation=0, labelpad=10)

ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))
ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.tick_params(which='both', direction='out')

ax.tick_params(axis='both', which='major', length=4, width=0.7, labelsize=8)
ax.tick_params(axis='both', which='minor', length=2, width=0.5)

files_labels_colors = [
    ("PPTA",                   "PPTA",                         "brown"),
    ("Eot-Wash-DM",            "Eöt-Wash (DM)",                "darkred"),
    ("Eot-Wash-EP",            "Eöt-Wash (EP)",                "green"),
    ("LISAPathfinder",         "LISA Pathfinder",              "purple"),
    ("LISA-Path-relative-acc", "LISA Pathfinder (rel. acc.)",  "indigo"),
]

masas = {}
gs = {}
labels = {}
colores = {}

for fname, label, color in files_labels_colors:
    fpath = otras + "/" + fname + ".txt"
    if not os.path.exists(fpath):
        print(f"[WARNING]: Non-existent {fpath}")
        continue

    arr = np.loadtxt(fpath, skiprows=3, usecols=(0, 1))
    m = arr[:, 0].astype(float)
    g = arr[:, 1].astype(float)

    masas[fname] = m
    gs[fname] = g
    labels[fname] = label
    colores[fname] = color

for fname in masas:
    if fname == "LISAPathfinder":
        continue
    if fname == "Eot-Wash-DM": continue

    m = np.asarray(masas[fname])
    g = np.asarray(gs[fname])

    if fname in ["PPTA", "Eot-Wash-DM", "LISA-Path-relative-acc"]:
        z = 3; alpha = 0.7
    elif fname in ["Eot-Wash-EP"]:
        z = 2; alpha = 0.3
    else:
        z = 1; alpha = 0.1

    ax.fill_between(m, g, ax.get_ylim()[1],
                    color=colores[fname], alpha=alpha, zorder=z)

    ax.plot(m, g, color=colores[fname], lw=1, zorder=z+0.1)
    

# AJUSTE DE FUENTES (7 para textos de anotación)
ax.text(1e-19, 1e-25, "MICROSCOPE",
        color="white", fontsize=7, va="bottom", ha="right",
        weight="bold",
        path_effects=[pe.withStroke(linewidth=0.8, foreground="black")])

ax.text(5e-23, 1e-25, "PPTA",
        color="white", fontsize=7, va="bottom", ha="right",
        weight="bold",
        path_effects=[pe.withStroke(linewidth=0.8, foreground="black")])

ax.text(2e-19, 0.6e-23, 'Eöt-Wash (EP)',
        color="white", fontsize=7, va="bottom", ha="right",
        weight="bold",
        path_effects=[pe.withStroke(linewidth=0.8, foreground="black")])

ax.text(0.55e-18, 2e-25, 'LISA Pathfinder (acc.)',
        color="white", fontsize=5, va="bottom", ha="right",
        weight="bold", rotation=90,
        path_effects=[pe.withStroke(linewidth=0.8, foreground="black")])

ax.fill_between(masas["Eot-Wash-EP"], 7e-26, ax.get_ylim()[1], color="grey", alpha=0.4)

ax.axhline(y=7e-26, xmin=0, xmax=1, label="MICROSCOPE (Final)", color="grey", lw=0.8)

plt.tight_layout()
plt.savefig("constraints_on_g.pdf", bbox_inches='tight')
plt.show()
