# EchoStateNetwork

Echo state networks / reservoir computing in pure numpy. One class,
`EchoStateNetwork`: ridge-regression training with chaotic recycle validation
(contiguous runs or ragged dwell segments), parametric inputs, optional Bayesian
hyperparameter search (`scikit-optimize`), closed-loop prediction, and Jacobians
for data assimilation.

This is the single shared reservoir core behind
[`qlrom`](https://github.com/andreanovoa/qlrom) (quantized-local ESN families) and
[`romda`](https://github.com/andreanovoa/romda) (real-time ESN forecasting and
bias-aware data assimilation).

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
