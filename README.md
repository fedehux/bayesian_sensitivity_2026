# bayesian_sensitivity_2026

# Sensitivity of binary pulsar timing to spin-0 and spin-1 ultralight dark matter

This repository contains the Python implementation used to derive the
sensitivity curves presented in *Sensitivity of binary pulsar timing to
spin-0 and spin-1 ultralight dark matter*.

The code is organized in two top-level folders, one for each scenario
(spin-0 and spin-1), each containing the analysis scripts, the
numerical results (`.npz`), the external constraints used as
comparison, and a `Reported bounds/` subfolder with the final
sensitivity curves in plain text format.

---

## Spin-0

The provided scripts compute sensitivities for the `x` and `Ψ′`
variables under both delta and Gaussian priors. We additionally include
a specialized treatment for eccentric orbits using the `Θ′` variable,
applied both to a combination of three eccentric millisecond pulsars
and to the Hulse–Taylor binary B1913+16.

### Main scripts

- `scalar_x_delta-prior.py`, `scalar_x_gaussian-prior.py` — sensitivity
  for the `x` variable under each prior choice.
- `scalar_psi_delta-prior.py`, `scalar_psi_gaussian-prior.py` —
  sensitivity for the `Ψ′` variable under each prior choice.
- `scalar_resonances_delta-prior.py` — eccentric-orbit treatment with
  the `Θ′` variable, combining three eccentric millisecond pulsars
  (J1903+0327, J1946+3417, J2234+0611).
- `scalar_resonances_delta-prior_B1913-16.py` — same eccentric-orbit
  treatment specialized to the Hulse–Taylor binary B1913+16.
- `recreate_fig_2.py` — combines the outputs of all of the above to
  reproduce Figure 2 of the paper (`beta_mass_constraints.pdf`).

After running the scripts whose names begin with `scalar_` (which
generate the corresponding `*.npz` result files), executing
`recreate_fig_2.py` from the local directory reproduces the quadratic
coupling constant constraints shown in Figure 2.

### One-step (Appendix)

The `One-step/` subfolder provides a specialized treatment for
eccentric orbits implementing several distinct numerical approaches to
the sensitivity calculation. These implementations allow for a detailed
comparison between marginalized and non-marginalized methods,
alongside a dedicated script that uses the true anomaly as the
integration variable. The included `replicate_fig_4.py` script
integrates these techniques and reproduces Figure 4 of the Appendix.

### External constraints

The files `PTA.txt` and `cassini_betas.txt` collect the external bounds
used as comparison curves in Figure 2.

### Reported bounds

The `Reported bounds/` subfolder contains the final sensitivity curves
of this work as plain `.txt` files (`mass [eV]`, `sqrt(beta) [GeV^-1]`),
one per analysis variant. These are the curves we report as bounds in
the paper and are intended for direct reuse by readers.

---

## Spin-1

For the spin-1 case the only included Python code corresponds to the
`Ψ′` variable. We also provide the analogous eccentric-orbit treatment
specialized to two systems (B1913+16 and J1903+0327), and we collect
the constraints from other tests of the equivalence principle so that
readers can reproduce Figure 3 of the paper.

### Main scripts

- `g_psi_delta.py` — sensitivity for the `Ψ′` variable, computed across
  the NANOGrav 15-yr `ELL1`-binary sample (table provided in
  `ell1_table.csv`).
- `g_theta_B1913-16.py` — eccentric-orbit treatment specialized to the
  Hulse–Taylor binary B1913+16.
- `g_theta_J1903-0327.py` — eccentric-orbit treatment specialized to
  J1903+0327.
- `recreate_fig_3.py` — combines the outputs of the above with the
  external equivalence-principle bounds to reproduce Figure 3 of the
  paper (`constraints_on_g.pdf`).

After running the three sensitivity scripts (which generate the
corresponding `g_*.npz` result files), executing `recreate_fig_3.py`
from the local directory reproduces Figure 3.

### External constraints

The `other_constraints/` subfolder collects external equivalence-
principle bounds (PPTA, Eöt-Wash, LISA Pathfinder) used as comparison
curves in Figure 3. The MICROSCOPE bound is drawn directly inside
`recreate_fig_3.py`.

### Reported bounds

The `Reported bounds/` subfolder contains the final sensitivity curves
of this work as plain `.txt` files (`mass [eV]`, `g_median`), one per
system. These are the curves we report as bounds in the paper.

---

## Requirements

The scripts rely on a standard scientific Python stack: `numpy`,
`scipy`, `matplotlib`, `pandas`, and `tqdm`.

## Citation

If you use these codes or the reported bounds, please cite the
accompanying paper (citation details to be added upon publication).
