# Getting started

## Install

```bash
pip install echostatenetwork          # core (numpy/scipy/matplotlib)
pip install "echostatenetwork[opt]"   # + scikit-optimize for Bayesian hyperparameter search
pip install -e ".[dev]"               # development
```

## Minimal workflow

```python
import numpy as np
from echostatenetwork import EchoStateNetwork

y = my_time_series                       # (Nt, N_dim)

esn = EchoStateNetwork(y=y[:1].T,        # sets the state dimension
                       dt=0.01,          # time step of the data
                       N_units=200,      # reservoir size
                       t_train=40.0,     # training window
                       t_val=4.0)        # validation window
esn.train(y)                             # ridge regression + Bayesian hyperparameter search
```

Training accepts a single trajectory `(Nt, N_dim)`, a batch of segments
`(L, Nt, N_dim)`, or a ragged list of segments of different lengths. After
training, `esn.step(u, r)` advances the reservoir one step; washout (open loop on
observed data) followed by closed-loop prediction is shown end to end in the
[tutorials](index.md#tutorials).

Every option of `EchoStateNetwork`, i.e. a class attribute such as `N_units` or
`rho_range`, can be passed as a keyword to the constructor or to `esn.train`. A
keyword that is not an option, e.g. the misspelt `rho_rnage`, a method such as
`copy`, or a read-only property such as `trained`, is ignored with a warning,
which names the keyword and, for a typo, suggests the closest name (e.g.
`rho_range`). The warning points at the line that calls the package; to silence
it, filter it by its message or category, not by the module `echostatenetwork`.

The package is silent by default. `echostatenetwork.set_verbose()` prints its
reports on stderr: the hyperparameter search and the training summary of
`esn.train`, and diagnostic messages, e.g. a training option changed by `train`;
`set_verbose(False)` silences them again (see the
[API reference](api/esn.md#echostatenetwork.set_verbose)). The keyword `verbose`
of earlier versions is removed: it gives a `FutureWarning` and has no effect. The
summary of the last training is also returned by `esn.training_summary()`. A
trained network, its copy (`esn.copy()`) and a network loaded with its matrices
are trained again only with `esn.train(..., force=True)`, which keeps the
reservoir while the options that define it are unchanged (see
[Training](training.md#training)).

## Parametric ESN

Pass `input_parameters` (shape `(N_param, L)`, one parameter vector per training
segment) to condition the reservoir on a physical parameter — see the
[parametric ESN tutorial](tutorials/02_parametric_esn.ipynb).

## Parallel ESN

For a spatially extended system, such as Lorenz-96 on a ring of sites, pass
`patch_size` (the sites of a patch), `halo` (the sites read on either side of a
patch) and `periodic` to run one reservoir of `N_units` units per patch. By
default, the patches share the input, reservoir and readout matrices and the
hyperparameters; `shared=False` gives every patch its own network (`patches`).
The reservoir state stacks the patches (`N_r = N_patches * N_units` rows), and
`state_positions` gives the position of every state row for covariance
localization — see [Parallel echo state network](parallel.md).

```python
esn = EchoStateNetwork(y=y[:1].T, dt=0.02, N_units=300,
                       patch_size=2, halo=3, periodic=True)       # shared=False: independent patches
```
