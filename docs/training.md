# Training and forecasting

One `EchoStateNetwork` holds three matrices: a fixed random input matrix
$\mathbf{W}_\mathrm{in}$, a fixed random reservoir matrix $\mathbf{W}$
(sparse, rescaled to unit spectral radius when generated), and the trained
readout $\mathbf{W}_\mathrm{out}$ — the only matrix that learning touches.

## The model

One `step(u, r)` advances the reservoir open loop and reads out the physical
state:

$$
\mathbf{r}_{n+1} = (1-\alpha)\,\mathbf{r}_n + \alpha \tanh\!\big(
\sigma_\mathrm{in} \mathbf{W}_\mathrm{in} [\tilde{\mathbf{u}}_n;\, b_\mathrm{in}]
+ \rho\, \mathbf{W} \mathbf{r}_n \big), \qquad
\mathbf{u}_{n+1} = \mathbf{W}_\mathrm{out}^\top [\mathbf{r}_{n+1};\, b_\mathrm{out}]
$$

where $\tilde{\mathbf{u}} = (\mathbf{u} - \texttt{shift}) / \texttt{norm}$ is
the shift-and-scale normalized input (set from the training data range by
default), $b_\mathrm{in}, b_\mathrm{out}$ are symmetry-breaking biases, and
$\alpha$ = `leak_rate` (the default $\alpha = 1$ is the plain tanh update; see
the [leaky-ESN tutorial](tutorials/03_leaky_esn.ipynb)).
The hyperparameters that matter are the effective spectral radius `rho`, the
input scaling `sigma_in`, and the Tikhonov regularization `tikh` — all three
selected by validation (below).

For a *parametric* ESN, `input_parameters` appends parameter columns to the
input, densely connected through $\mathbf{W}_\mathrm{in}$ — one reservoir
conditioned on a physical parameter
([tutorial 02](tutorials/02_parametric_esn.ipynb)).
With partial observability, `observed_idx` selects which components of the
full state are fed back as inputs.

For a spatially extended system, the *parallel* layout (`patch_size`, `halo`,
`periodic`) advances one reservoir per patch of neighbouring sites with the
same update, and the patches share the three matrices
([parallel echo state network](parallel.md)).

## Training

`train(train_data)` accepts a single trajectory `(Nt, N_dim)`, a batch of
segments `(L, Nt, N_dim)`, or a ragged list of segments of different lengths,
and runs four steps:

1. **Split** the data into washout (`N_wash` steps, inside `t_train`),
   training + validation (`t_train`, `t_val` — inferred from the data when
   omitted), and an optional held-out test tail (`t_test`). Gaussian noise
   (`noise`) is added to the training *inputs* only — targets stay clean — as
   regularization.
2. **Generate** $\mathbf{W}_\mathrm{in}$ and $\mathbf{W}$ from `seed`, if not
   already set, or when an option that defines them changed (see below).
3. **Select hyperparameters** by Bayesian optimization over
   `hyperparameters_to_optimize` (skip by passing `[]`), scoring each
   candidate with a [validation strategy](validation.md) — closed-loop error
   on folds carved from the training series. The search trace is kept in
   `bo_results`.

    Every evaluation of the validation strategy selects the Tikhonov factor,
    `tikh`, from the grid `tikh_range`, so `tikh` is not a dimension of the
    Bayesian optimization. A search over `'tikh'` alone is therefore a grid
    search: `train` evaluates the strategy once at the fixed `rho`, `sigma_in`
    and `leak_rate`, without Bayesian optimization and without
    scikit-optimize. `bo_results` then holds that evaluation: one value in
    `func_vals`, `x = []` and `hp_names = []`. A parametric network is the
    exception: with `optimize_parameter_normalization`, which is on by default
    with `input_parameters`, `train` adds the shift and the scale of every
    parameter column (`param_shift_i`, `param_norm_i`) to the search, which is
    then a Bayesian optimization with the budget `N_func_evals`. A search
    without `'tikh'` keeps the fixed `tikh` at every evaluation, and leaves
    `tikh_range` as it was.

    ![Bayesian hyperparameter search landscape](img/bho_landscape.png)

    *Validation-error landscape over `(rho, sigma_in)` from
    [tutorial 01](tutorials/01_echo_state_network.ipynb):
    markers are the evaluated points (initial grid + gp-hedge acquisitions),
    the star the selected optimum. `plot_training=True` produces this figure.*
4. **Solve** the ridge regression for $\mathbf{W}_\mathrm{out}$ on the full
   train + validation data with the selected `tikh`.

The keywords of `train` set attributes of the network before these steps, e.g.
`esn.train(y, rho_range=(0.5, 1.0))`. A keyword that is not an option, e.g. a
misspelt name, a method or a read-only property, is ignored, and one warning
per call names it, also with `n_seeds > 1` and with independent patches. A call
of `train` that raises, or that is interrupted, leaves the network as it was,
with the options and the matrices that it had before the call.

`training_summary()` returns the realized split, folds, and selected
hyperparameters of the last call; with `echostatenetwork.set_verbose()`, `train`
also logs it, with the table of the hyperparameter search.

A trained network is not trained again: a second `train` call warns and leaves
the network unchanged, unless it passes `force=True`. A copy of a trained
network (`esn.copy()`) and a network loaded with its matrices (`from_arrays`,
`pickle`) are trained, so that training them on other data also needs
`force=True`. With `force=True`, `train` computes the hyperparameter search and
$\mathbf{W}_\mathrm{out}$ again, and

- keeps the reservoir ($\mathbf{W}_\mathrm{in}$, $\mathbf{W}$) while the
  options that define it are unchanged: `N_units`, `connect`, `Win_type`, the
  number of inputs (`observed_idx`, `input_parameters`), `param_in_reservoir`,
  `patch_size` and `halo`, as keywords of `train` or as attributes. When one of
  these options changes, `train` draws a new reservoir and a diagnostic message
  names the option. `readout_input` changes $\mathbf{W}_\mathrm{out}$ only;
- draws the new reservoir from the seed given to `train`, else from the seed of
  the reservoir, with a new generator: it is the reservoir that a new network
  with the same options draws with `train(seed=...)`. The first call of `train`
  draws the reservoir from the generator `rng` of the attribute `seed` after the
  noise on the inputs, so this reservoir has no seed that reproduces it. A
  reservoir that `train` draws again then comes from the attribute `seed`, and
  an option changed and then changed back does not restore the reservoir of the
  first call;
- keeps the reservoir whatever the seed given to `train`, which is not an
  option of the reservoir: `train(seed=s, force=True)` without a changed option
  keeps the reservoir, and warns that `seed` is ignored unless `s` is the seed
  of the reservoir. A new network trained with `train(seed=s)`, or
  `train(seed=s, n_seeds=m, force=True)`, draws the reservoir of `s`. Setting the attribute `seed`, e.g. `esn.seed = 7`,
  starts the noise on the inputs from a new generator and keeps the reservoir;
- treats a reservoir set by hand (`W`, `Win`, `from_arrays`), or held by a
  network pickled by echostatenetwork 0.2.3 or earlier, as a reservoir drawn
  with the options that the network has at the start of the call: an option
  that the call changes draws a new reservoir from the attribute `seed`, so
  that networks that share one reservoir draw the same new one. When only one
  matrix fits the options, e.g. after `esn.N_units = 20; esn.W = W20`, `train`
  keeps that matrix and draws the other one alone;
- keeps a `t_train` or `t_val` that was given (to the constructor, to `train`
  or as an attribute), and infers again from the new data a `t_train` or
  `t_val` that the previous call inferred. The same data therefore give the
  same split, and shorter data still train. An attribute set to the value that
  the previous call inferred, e.g. `esn.t_val = esn.t_val`, cannot be told
  from the inferred value and is inferred again: to keep it, pass it to `train`,
  e.g. `esn.train(y, t_val=esn.t_val, force=True)`;
- adds again to `hyperparameters_to_optimize` the names `param_shift_i` and
  `param_norm_i` that the previous call added, only while the network has
  parameters and `optimize_parameter_normalization`, unless the list was set
  since. Removing the parameters (`input_parameters=None`) or setting
  `optimize_parameter_normalization=False` therefore removes them from the
  search. The ranges `param_shift_range` and `param_norm_range` that the first
  call sized from the parameters are kept;
- keeps the time step of the data, `dt`: `dt_ESN = dt * upsample` follows a
  new `upsample`, and `t_train`, `t_val` remain times.

Training a single network again on the same data with the same options does
not reproduce it: the noise on the inputs continues the generator `rng`, so the
selected hyperparameters and $\mathbf{W}_\mathrm{out}$ may change. With
`n_seeds > 1`, every realization draws its noise from its own seed, and training
again reproduces the realizations.

With independent patches (`shared=False`), the network of every patch has its
own seed, `patches[p].seed`, which `train` spawns from the seed of the parallel
network when it builds the networks of the patches. The network of a patch
follows the rule above: an option that defines its reservoir draws a new one
from the seed of the patch, which a new single network with the options of the
patch reproduces with `train(seed=patches[p].seed)`, and a new parallel network
does not. `train` builds the networks again, with seeds spawned from the seed
given to `train`, else from the attribute `seed`, only when `patch_size` or
`halo` changes; a seed alone keeps them, with a warning. Setting the attribute
`seed` of the parallel network therefore changes neither the seeds nor the
noise of the patches until `train` builds the networks again. The networks of
the patches take the split and the time step of the parallel network. An
option changed on the parallel network, as a keyword of `train` or as an
attribute (e.g. `esn.N_units = 10`), reaches the network of every patch; the
other options of a patch, e.g. the `rho` that its own search selected, are
kept.

### Seeded ensembles

The reservoir is a random draw, so `train(n_seeds=m)` trains `m` realizations
(seeds `base, base+1, ...`) in parallel processes, keeps the network with the
best validation score in place, and stores every score in `seed_scores` —
selection by validation score consistently lands in the ensemble's best half
on test error (see the
[validation strategies](validation.md#ensembles-of-reservoir-seeds) page).
The score of a realization is the minimum of its search (`bo_results['fun']`)
or, without a search, one evaluation of the validation strategy at the fixed
hyperparameters of the realization, `tikh` included. The first seed, `base`, is
the `seed` given to `train`. Without it, `base` is the first seed of the
previous call with `n_seeds > 1` while the attribute `seed` is one of the seeds
of that call, as after it (the kept seed), so that training again with
`force=True` on the same data and options selects the same realization;
otherwise, e.g. on the first call, `base` is the attribute `seed`. A call with
`n_seeds = 1` leaves `seed_scores`, which then hold the scores of the last call
with `n_seeds > 1`. A warning raised while the realizations train, e.g.
when fewer validation folds than `N_folds` fit, appears once, at the line that
calls `train`, and follows the warnings filters of the calling process. That is
also where the warning points when it comes from code of the user that the
package calls back, e.g. a validation strategy that calls `validation.WFV`; a
single network points at that code instead.

## Forecasting

The ESN runs in two modes:

![Open-loop and closed-loop configurations](img/esn_open_closed_loop.png)

*(a) open loop: the reservoir is driven by data $\phi(t_i)$ and its readouts
$\hat{\phi}(t_{i+1})$ are one-step-ahead predictions; (b) closed loop: each
prediction is fed back as the next input and the network runs autonomously.*

- **Open loop**: `step()` is fed observed data; used for
  training and for *washout* — driving the reservoir from
  $\mathbf{r}_0 = \mathbf{0}$ through a short window of true history so it
  synchronizes with the physical state before a forecast.
- **Closed loop**: each prediction is fed back as the next input
  (`outputs_to_inputs` selects the observed components and re-appends the
  parameter columns); the model runs autonomously.

```python
r = np.zeros((esn.N_units, 1))
u = np.zeros((esn.N_dim, 1))
for u_in in wash:                      # open-loop washout on true history
    u, r = esn.step(u_in[:, None], r)

forecast = []
for _ in range(n_steps):               # closed-loop forecast
    u, r = esn.step(esn.outputs_to_inputs(u), r)
    forecast.append(u[:, 0])
```

Alignment: the washout's last output is the one-step-ahead prediction of the
frame *after* `wash`, so closed-loop prediction `i` corresponds to data row
`len(wash) + 1 + i`.

`run_test()` packages this — open-loop washout, closed-loop forecast, error
against truth — over the held-out test tail, with plots. For a fair error
metric across components use `compute_nMAE(Y_true, Y_pred, norm=...)`, the
range-normalized mean absolute error used throughout training and validation.

See the
[EchoStateNetwork tutorial](tutorials/01_echo_state_network.ipynb)
for the full workflow on Lorenz 63, and the
[API reference](api/esn.md) for every attribute and method.
