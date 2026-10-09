# Parallel echo state network

A spatially extended system, such as the Lorenz-96 model on a ring of 40 sites
or a discretized partial differential equation, has a state with many
components that interact locally. A single reservoir that reads the whole state
needs more units as the number of sites grows, and its readout couples every
unit with every site. The parallel layout of Pathak et al. (2018) replaces the
single reservoir with one reservoir per patch of neighbouring sites. Each
reservoir reads its patch and a halo of sites on either side, and predicts only
its patch. The term comes from domain decomposition, in which the halo, or ghost
region, is the copy of the data of the neighbouring subdomains that each process
stores locally, so that it can compute independently; here, the halo is the copy
of the neighbouring sites that each reservoir reads. Pathak et al. (2018) call it
a buffer region, of $l$ sites. Vlachas et al. (2020) employ the same layout for reservoirs and for
recurrent networks trained by backpropagation, on the Lorenz-96 and
Kuramoto–Sivashinsky systems.

In `EchoStateNetwork`, the layout is an option of the constructor: `patch_size`
sets the number of sites of a patch, `halo` the number of sites read on either
side, `periodic` the treatment of the domain boundaries, and `shared` whether the
patches share their matrices and hyperparameters (the default) or are
independent. Without `patch_size` (the default, `None`), the network has one
reservoir. The
[Kuramoto–Sivashinsky tutorial](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/05_parallel_esn_kuramoto_sivashinsky.ipynb)
compares a global network, a parallel network with shared matrices and a local
network with independent patches.

## The model

The state $\mathbf{u}$ has $N$ sites (`N_dim`). With patches of $G$ sites
(`patch_size`), the domain splits into $P = N/G$ patches (`N_patches`),
numbered $p = 0, \dots, P-1$ as in the code. Patch $p$ predicts the sites
$\mathcal{P}_p = \{pG, \dots, pG + G - 1\}$ and reads the window
$\mathcal{I}_p = \{pG - H, \dots, pG + G + H - 1\}$, which adds a halo of $H$
sites (`halo`) on either side of the patch. One step advances the reservoir of
every patch and reads out the sites of the patch,

$$
\mathbf{r}_{p,n+1} = (1-\alpha)\,\mathbf{r}_{p,n} + \alpha \tanh\!\big(
\sigma_\mathrm{in} \mathbf{W}_\mathrm{in} [\tilde{\mathbf{u}}_n[\mathcal{I}_p];\, b_\mathrm{in}]
+ \rho\, \mathbf{W} \mathbf{r}_{p,n} \big), \qquad
\mathbf{u}_{n+1}[\mathcal{P}_p] = \mathbf{W}_\mathrm{out}^\top [\mathbf{r}_{p,n+1};\, b_\mathrm{out}],
$$

where $\mathbf{r}_{p,n}$ is the state of the reservoir of patch $p$ at step
$n$, with $N_\mathrm{units}$ units (`N_units`); $\tilde{\mathbf{u}}_n[\mathcal{I}_p]$
is the normalized input at the sites of the window; and the other symbols are
those of the [single reservoir](training.md#the-model). The input matrix
$\mathbf{W}_\mathrm{in}$ has $G + 2H + 1$ columns (the sites of a window and
the bias), and the readout $\mathbf{W}_\mathrm{out}$ has $G$ columns (the
sites of a patch). The state of the network stacks the reservoirs in patch
order, $\mathbf{r} = [\mathbf{r}_0; \dots; \mathbf{r}_{P-1}]$, with
$N_r = P N_\mathrm{units}$ rows (`N_r`). This is the array that `step(u, r)`
advances, for one member or for an ensemble of members stored as columns.

One patch that covers the whole domain without a halo ($G = N$, $H = 0$) is
the single reservoir.

## How the halo couples the patches

Without a halo ($H = 0$), each reservoir reads only its own patch, so the
patches evolve independently, and the network cannot represent the coupling
between neighbouring sites. With a halo, the windows of neighbouring patches
overlap. In closed loop, the forecast of each patch enters the windows of its
neighbours at the next step, so information crosses the domain at up to
$G + H - 1$ sites per step. The halo needs to cover at least the range of the
coupling in the governing equations: otherwise, the one-step forecast of a site
at the edge of a patch misses inputs on which its time derivative depends. For
example, in Lorenz-96 the time derivative of site $i$ depends on the sites
$i-2$ to $i+1$, so $H \geq 2$.

## Shared matrices

All patches share $\mathbf{W}_\mathrm{in}$, $\mathbf{W}$ and
$\mathbf{W}_\mathrm{out}$, as the kernel of a convolution is shared across
positions. This is valid for a system that is invariant under a shift of the
sites, such as Lorenz-96 with uniform forcing or the Kuramoto–Sivashinsky
equation with periodic boundary conditions, because the map from the window of
a patch to the next state of the patch is then the same for every patch. The
reservoirs still predict different values, because each one reads a different
window and carries a different state. On a ring, shifting the input and the
reservoir state by one patch shifts the forecast by one patch.

The shared readout is fitted by the ridge regression of
[`train()`](training.md#training) on all the patches at once: each time step
provides $P$ samples, one per patch, each mapping the state of a reservoir to
the sites of its patch. The readout has as many weights as the readout of one
patch, $(N_\mathrm{units} + 1) G$, but $P$ times more samples. The input
normalization (`norm`, `shift`) takes one value for all the sites, computed
from the samples of all the sites together, because the patches share
$\mathbf{W}_\mathrm{in}$ and must see their windows in the same units.

Hyperparameter selection works as for a single reservoir: every
[validation strategy](validation.md) of this package scores closed-loop
forecasts of the whole network, and the strategies that retrain the readout per
fold (`SSV`, `WFV`, `KFV`) pool the patches in the same way.

Shared matrices are the default because, for a system that is invariant under a
shift of the sites, they lose no generality and save data. First, by
symmetry, the optimal hyperparameters are the same for every patch, so one
search on all the patches estimates them, whereas $P$ independent searches cost
$P$ times more and each one sees $P$ times fewer samples. Second, the shared
readout is fitted on $P$ samples per time step, i.e. on $P$ times more samples
per readout weight than the readout of one independent patch.

## Independent patches

With `shared=False`, every patch has its own network, in `patches`: a list of
$P$ `EchoStateNetwork` objects, one per patch, each of which reads and forecasts
the window of its patch. Each patch has its own random realization of
$\mathbf{W}_\mathrm{in}$ and $\mathbf{W}$, its own readout, its own
hyperparameters ($\rho$, $\sigma_\mathrm{in}$, the Tikhonov factor and
`leak_rate`) and its own normalization. The matrices and the hyperparameters
always go per patch together; there is no partly shared combination. One step
of the parallel network advances the network of every patch with the update in
[the model](#the-model), in which $\mathbf{W}_\mathrm{in}$, $\mathbf{W}$,
$\mathbf{W}_\mathrm{out}$, $\sigma_\mathrm{in}$, $\rho$ and $\alpha$ take the
values of patch $p$.

`train()` trains the network of every patch on the record of its own window,
with the validation strategy and the Bayesian optimization of the package: one
hyperparameter search and one ridge regression per patch, each on the samples of
its own patch. The network of a patch forecasts its whole window, so that the
record of the window trains it as a single reservoir; the parallel network keeps
the readout columns of the sites of the patch, which are the ridge solution for
these sites alone, because the ridge regression decouples the output columns.
The validation forecasts of a patch feed back only the sites of the patch and
take the halo from the data, and the validation error scores only the sites of
the patch, so that the search of one patch does not depend on the other
patches. With `train(n_seeds=m)`, every patch keeps the best of its own $m$
reservoir realizations.

The normalization is per patch: each network normalizes every position of its
window with its own shift and scale, because its input matrix is its own and
nothing requires a common scale. With shared matrices, by contrast, one shift
and one scale serve all the sites.

Independent patches suit systems whose local dynamics vary in space, e.g. with
an inhomogeneous forcing, walls or inlets, or sensors of different quality, in
which the map from the window to the next state of the patch differs between
patches. The price is $P$ hyperparameter searches, $P$ times more readout
weights, and $P$ times fewer samples per readout weight than with shared
matrices.

## Domain boundaries

On a periodic domain (`periodic=True`, the default), the sites lie on a ring and
the windows of the first and last patches wrap around. The window of a patch
must not exceed the ring, $G + 2H \leq N$.

On a non-periodic domain (`periodic=False`), the windows of the patches at the
boundaries extend outside the domain. These positions are padded with zero in
normalized units, which is the training mean of the data, and they are marked
with $-1$ in `patch_sites`. The padding keeps the windows of equal width, which
the shared input matrix requires. A non-periodic domain is not invariant under
a shift of the sites near its boundaries, so the shared matrices are an
approximation there, and independent patches (`shared=False`) let the patches at
the boundaries learn their own dynamics.

In both cases, `patch_size` divides the number of sites.

## Positions for data assimilation

Covariance localization in data assimilation tapers the covariance between two
state components with the distance between them, which requires a position for
every component. The rows of a single reservoir have no position: through the
reservoir matrix, every unit depends on every site, and the readout maps every
unit to every site. With the parallel layout, every reservoir row belongs to
one patch, which reads only its window. `state_positions`
returns the position, in units of sites, of every row of the state
$[\mathbf{u}; \mathbf{r}]$: output $i$ lies at site $i$, and every row of
$\mathbf{r}_p$ lies at the centre of patch $p$, $pG + (G-1)/2$. On a ring, the
positions are periodic with period $N$. These are the positions that a local
ensemble transform Kalman filter requires, e.g. the `state_positions` argument
of biasda, with `loc_period` equal to $N$ on a ring.

## When to employ the parallel layout

- High-dimensional, spatially extended systems with local coupling, in which
  the sites of the state are ordered in space. At a fixed reservoir size per
  patch, the cost of one step and of the training grows linearly with the
  number of sites.
- Data assimilation with covariance localization, which needs the positions of
  the reservoir rows.

The layout does not suit a state without a spatial order, such as the three
variables of Lorenz-63 or the coefficients of a proper orthogonal
decomposition.

## Usage

```python
import numpy as np
from echostatenetwork import EchoStateNetwork

X = lorenz96_record                    # (Nt, 40): 40 sites on a ring, time step 0.02

esn = EchoStateNetwork(X[:1].T, dt=0.02, upsample=1,
                       N_units=300,    # units of each reservoir
                       patch_size=2,   # G: sites that each reservoir predicts
                       halo=3,         # H: sites read on either side of the patch
                       periodic=True,  # the sites lie on a ring
                       t_train=200.0, t_val=40.0)
esn.train(X)                           # one readout, fitted on the 20 patches at once

r = np.zeros((esn.N_r, 1))             # 20 reservoirs of 300 units, stacked: 6000 rows
for u_in in wash:                      # open-loop washout on observed data
    u, r = esn.step(u_in[:, None], r)

forecast = []
for _ in range(n_steps):               # closed-loop forecast
    u, r = esn.step(esn.outputs_to_inputs(u), r)
    forecast.append(u[:, 0])

positions = esn.state_positions        # (40 + 6000,): sites of the rows of [u; r]
```

For an ensemble, `u` has shape `(N_dim, N_ens)` and `r` has shape
`(N_r, N_ens)`; `Jacobian(u, r)` returns the open-loop Jacobian, whose rows of
patch $p$ are non-zero only in the columns of its window.

Independent patches take one more keyword; the state, the positions, `step`,
the closed loop, the Jacobian and `to_arrays`/`from_arrays` work as with shared
matrices.

```python
local = EchoStateNetwork(X[:1].T, dt=0.02, upsample=1, N_units=300,
                         patch_size=2, halo=3, shared=False,
                         t_train=200.0, t_val=40.0)
local.train(X)                         # 20 searches and 20 ridge regressions, one per patch
local.patches[0].rho, local.patches[0].Win     # hyperparameters and matrices of patch 0
```

| Keyword or attribute | Meaning |
|---|---|
| `patch_size` | sites $G$ of a patch; `None` (default) for one reservoir |
| `halo` | sites $H$ read on either side of a patch (default 0) |
| `periodic` | the sites lie on a ring (default `True`); `False` pads the windows |
| `shared` | the patches share matrices and hyperparameters (default `True`); `False` for independent patches |
| `N_units` | units of each reservoir |
| `N_patches` | number of patches, $P = N/G$ |
| `N_r` | rows of the reservoir state, $P N_\mathrm{units}$ |
| `patch_sites` | `(N_patches, G + 2H)` sites of each window, $-1$ outside a non-periodic domain |
| `state_positions` | `(N_dim + N_r,)` position of every row of $[\mathbf{u}; \mathbf{r}]$ |
| `Win`, `W`, `Wout` | shared matrices, `(N_units, G + 2H + 1)`, `(N_units, N_units)`, `(N_units + 1, G)` |
| `patches` | with `shared=False`: one `EchoStateNetwork` per patch, with its own matrices and hyperparameters |

## Limitations

- The layout raises a `ValueError` when `patch_size` does not divide the number
  of sites, when the window exceeds a periodic domain, when it is combined with
  `input_parameters`, `readout_input` or partial observability (`observed_idx`),
  and when `shared=False` is set without `patch_size`.
- With independent patches, `Win`, `W` and `Wout` of the parallel network raise
  an `AttributeError` that points to `patches[p]`, and its `rho`, `sigma_in` and
  `tikh` are the starting values passed to every patch: the selected values live
  in `patches[p]`. Training plots are not drawn, and one step loops over the
  patches, so it is slower than with shared matrices when there are many
  patches.
- Validation strategies outside this package that do not call the closed-loop
  hook of the package (`_closed_loop_input`) run the validation forecast of an
  independent patch closed on its whole window, halo included.
- Code outside this package that sizes the reservoir state with `N_units`
  (e.g. a custom validation strategy) needs `N_r` instead; `step` raises a
  `ValueError` that says so.

## References

Pathak, J., Hunt, B., Girvan, M., Lu, Z., & Ott, E. (2018). Model-free
prediction of large spatiotemporally chaotic systems from data: a reservoir
computing approach. *Physical Review Letters*, 120, 024102.
[doi:10.1103/PhysRevLett.120.024102](https://doi.org/10.1103/PhysRevLett.120.024102)

Vlachas, P. R., Pathak, J., Hunt, B. R., Sapsis, T. P., Girvan, M., Ott, E., &
Koumoutsakos, P. (2020). Backpropagation algorithms and reservoir computing in
recurrent neural networks for the forecasting of complex spatiotemporal
dynamics. *Neural Networks*, 126, 191-217.
[doi:10.1016/j.neunet.2020.02.016](https://doi.org/10.1016/j.neunet.2020.02.016)
