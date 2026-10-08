# EchoStateNetwork

[![PyPI](https://img.shields.io/pypi/v/echostatenetwork)](https://pypi.org/project/echostatenetwork/)

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
[`qlrom`](https://github.com/andreanovoa/qlrom) (quantized-local ESN families) and
[`romda`](https://github.com/andreanovoa/romda) (real-time ESN forecasting and
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
esn = EchoStateNetwork(y, dt=0.01, N_units=200, t_train=40.0, t_val=4.0)
esn.train([y])
u_wash, r = ...                          # see docstrings: washout then closed loop
```

## Acknowledgements

This repository is based on
[alberacca/Echo-State-Networks](https://github.com/alberacca/Echo-State-Networks),
the reference implementation of the recycle-validation ESN:

> Racca, A., & Magri, L. (2021). Robust optimization and validation of echo state
> networks for learning chaotic dynamics. *Neural Networks*, 142, 252-268.
> [doi:10.1016/j.neunet.2021.05.004](https://doi.org/10.1016/j.neunet.2021.05.004)
