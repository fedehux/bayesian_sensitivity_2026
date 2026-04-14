# bayesian_sensitivity_2026

This repository contains the Python implementation used to derive the sensitivity curves presented in 'Sensitivity of binary pulsar timing to spin-0 and spin-1 ultralight dark matter'. In the Spin-0 folder, the provided scripts compute sensitivities for the $x$ and $\Psi'$ variables under both delta and Gaussian priors. Additionally, we include the specialized treatment for eccentric orbits using the $\Theta'$ variable. By executing "recreate_fig_2.py" within the local directory (after executing the files whose names begin with scalar_), the quadratic coupling constant constraints shown in Figure 2 can be reproduced.

We provide a specialized treatment for eccentric orbits within the One-step subfolder, which includes several distinct numerical approaches for calculating sensitivity. These implementations allow for a detailed comparison between marginalized and non-marginalized methods, alongside a dedicated script utilizing the true anomaly. The included replicate_fig_4.py script serves to integrate these various techniques, enabling the direct reproduction of Figure 4 from the Appendix.

As for the spin-1 case, the only included Python code corresponds to the $\Psi'$ variable. The constraints for other tests of the equivalence principle are collected as well, in order for readers to recreate Figure 3 from the paper. 
