# EchoStateNetwork

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23266679-blue.svg)](https://doi.org/10.5281/zenodo.23266679)
[![PyPI](https://img.shields.io/pypi/v/echostatenetwork)](https://pypi.org/project/echostatenetwork/)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://andreanovoa.github.io/EchoStateNetwork/)

Echo state networks / reservoir computing in pure numpy. One class,
`EchoStateNetwork`: ridge-regression training with chaotic recycle validation
(contiguous runs or ragged dwell segments), parametric inputs, optional Bayesian
hyperparameter search (`scikit-optimize`), closed-loop prediction, and Jacobians
for data assimilation. For spatially extended systems, the parallel layout
(`patch_size`, `halo`, `periodic`) runs one reservoir per patch of neighbouring
sites (Pathak et al. 2018), with shared matrices and a position for every
reservoir row, which covariance localization requires; see the
[parallel ESN page](https://andreanovoa.github.io/EchoStateNetwork/parallel/).

This is the single shared reservoir core behind
[`qlroms`](https://andreanovoa.github.io/qlroms-docs/) (quantized-local ESN families) and
[`romda`](https://github.com/andreanovoa/real-time-bias-aware-DA) (real-time ESN forecasting and
bias-aware data assimilation).

Documentation: <https://andreanovoa.github.io/EchoStateNetwork/>

## Tutorials

- [`tutorials/01_echo_state_network.ipynb`](tutorials/01_echo_state_network.ipynb)
  — train an ESN on the Lorenz 63 system, with full and partial observability.
- [`tutorials/02_parametric_esn.ipynb`](tutorials/02_parametric_esn.ipynb)
  — one reservoir conditioned on a physical parameter, forecasting at unseen
  parameter values.
- [`tutorials/05_parallel_esn_kuramoto_sivashinsky.ipynb`](tutorials/05_parallel_esn_kuramoto_sivashinsky.ipynb)
  — global, parallel (shared matrices) and local (independent patches) networks
  on the Kuramoto–Sivashinsky equation.
- [`tutorials/06_esn_architectures.ipynb`](tutorials/06_esn_architectures.ipynb)
  — drawings of the architectures that the constructor keywords build, a gallery
  and an interactive explorer (`ipywidgets`).

## Install

```bash
pip install echostatenetwork          # core (numpy/scipy/matplotlib)
pip install "echostatenetwork[opt]"   # + scikit-optimize for Bayesian hyperparameter search
pip install -e ".[dev]"               # development
```

## Quickstart

```python
import numpy as np
from echostatenetwork import EchoStateNetwork

y = my_time_series                       # (Nt, N_dim)
esn = EchoStateNetwork(y[:1].T, dt=0.01, N_units=200, t_train=40.0, t_val=4.0)   # y[:1].T sets N_dim
esn.train(y)
u_wash, r = ...                          # see docstrings: washout then closed loop
```

## Citation

If this package contributes to your work, please cite the software archive on Zenodo:

```bibtex
@software{novoa_echostatenetwork,
  author = {Nóvoa, Andrea},
  title = {echostatenetwork: echo state networks in pure numpy},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23266679},
  url = {https://doi.org/10.5281/zenodo.23266679},
}
```

and the paper that introduced chaotic recycle validation, which `train()` employs by
default:

```bibtex
@article{racca2021robust,
  author = {Racca, Alberto and Magri, Luca},
  title = {Robust optimization and validation of echo state networks for learning chaotic dynamics},
  journal = {Neural Networks},
  volume = {142},
  pages = {252--268},
  year = {2021},
  doi = {10.1016/j.neunet.2021.05.004},
  url = {https://doi.org/10.1016/j.neunet.2021.05.004},
}
```

## Contributors

- [Andrea Nóvoa](https://github.com/andreanovoa) (maintainer)
- [Alberto Racca](https://github.com/alberacca): the recycle-validation echo state network
  ([alberacca/Echo-State-Networks](https://github.com/alberacca/Echo-State-Networks),
  [MagriLab/Robust-Validation-ESN](https://github.com/MagriLab/Robust-Validation-ESN)), on which this package is based
- [Defne Ozan](https://github.com/defneozan): the parametric and adjoint echo state network
  ([MagriLab/Adjoint-ESN](https://github.com/MagriLab/Adjoint-ESN))

## Acknowledgements

This repository is based on
[alberacca/Echo-State-Networks](https://github.com/alberacca/Echo-State-Networks),
the reference implementation of the recycle-validation echo state network by Racca &
Magri (2021). The parallel layout follows Pathak et al. (2018) and Vlachas et al. (2020).

- Pathak, J., Hunt, B., Girvan, M., Lu, Z., & Ott, E. (2018). Model-free prediction of
  large spatiotemporally chaotic systems from data: a reservoir computing approach.
  *Physical Review Letters*, 120, 024102.
  [doi:10.1103/PhysRevLett.120.024102](https://doi.org/10.1103/PhysRevLett.120.024102)
- Vlachas, P. R., Pathak, J., Hunt, B. R., Sapsis, T. P., Girvan, M., Ott, E., &
  Koumoutsakos, P. (2020). Backpropagation algorithms and reservoir computing in
  recurrent neural networks for the forecasting of complex spatiotemporal dynamics.
  *Neural Networks*, 126, 191-217.
  [doi:10.1016/j.neunet.2020.02.016](https://doi.org/10.1016/j.neunet.2020.02.016)
