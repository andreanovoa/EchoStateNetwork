# echostatenetwork

Echo state networks / reservoir computing in pure numpy. One class,
[`EchoStateNetwork`](api/esn.md): ridge-regression training with chaotic recycle
validation (contiguous runs or ragged dwell segments), parametric inputs, optional
Bayesian hyperparameter search (`scikit-optimize`), closed-loop prediction, and
Jacobians for data assimilation. For spatially extended systems, the
[parallel layout](parallel.md) runs one reservoir per patch of neighbouring
sites, with shared matrices and a position for every reservoir row, which
covariance localization requires.

This is the single shared reservoir core behind
[`qlrom`](https://github.com/andreanovoa/qlrom) (quantized-local ESN families) and
[`romda`](https://github.com/andreanovoa/romda) (real-time ESN forecasting and
bias-aware data assimilation).

## Tutorials

- [The EchoStateNetwork class](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/01_echo_state_network.ipynb)
  — train an ESN on the Lorenz 63 system, with full and partial observability.
- [Parametric ESN](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/02_parametric_esn.ipynb)
  — one reservoir conditioned on a physical parameter, forecasting at unseen
  parameter values.
- [Leaky-integrator ESN](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/03_leaky_esn.ipynb)
  — the leak-rate hyperparameter and when it helps.
- [Validation strategies](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/04_validation_strategies.ipynb)
  — SSV/WFV/KFV/recycle validation compared live on Lorenz 63, including
  seeded ensembles with `train(n_seeds=...)`; see also
  [Validation strategies](validation.md).
- [Parallel ESN on Kuramoto–Sivashinsky](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/05_parallel_esn_kuramoto_sivashinsky.ipynb)
  — a global network, a parallel network with shared matrices and a local
  network with independent patches, compared on closed-loop forecasts,
  long-term statistics and training cost; see also
  [Parallel echo state network](parallel.md).
- [Architectures of the ESN](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/06_esn_architectures.ipynb)
  — drawings of the architectures that the constructor keywords build (inputs,
  parameters, input skip, leak rate, parallel layout, hyperparameter search),
  as a gallery and as an interactive explorer.

## Acknowledgements

This repository is based on
[alberacca/Echo-State-Networks](https://github.com/alberacca/Echo-State-Networks),
the reference implementation of the recycle-validation ESN:

> Racca, A., & Magri, L. (2021). Robust optimization and validation of echo state
> networks for learning chaotic dynamics. *Neural Networks*, 142, 252-268.
> [doi:10.1016/j.neunet.2021.05.004](https://doi.org/10.1016/j.neunet.2021.05.004)
