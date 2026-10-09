---
hide:
  - navigation
  - toc
---

# Live demo: building an echo state network

An echo state network is built from a few choices: the data it trains on and how they
are split, which components of the state the reservoir reads, what the readout reads,
whether one reservoir serves the whole state or one reservoir serves each patch of
neighbouring sites, which hyperparameters the Bayesian optimization searches, and how
the validation folds are placed. This page lets you make these choices with the controls
and shows, as they change, the architecture of the network, the equations that `train()`
and the forecast solve, the training record with its validation folds, and the code that
builds and trains the network. Nothing is trained in the browser: copy the code and run
it locally on your data.

<style>.md-grid { max-width: 1600px; }</style>

<iframe id="esn-builder" src="../esn-builder.html" title="Interactive echo state network builder" style="width: 100%; height: 2200px; border: 0;"></iframe>

<script>
(() => {
  const frame = document.getElementById("esn-builder");
  const theme = () => (document.body.dataset.mdColorScheme === "slate" ? "dark" : "light");
  const send = () => frame.contentWindow.postMessage({ esnBuilderTheme: theme() }, "*");
  addEventListener("message", (e) => {
    const h = e.data && e.data.esnBuilderHeight;
    if (h) frame.style.height = h + "px";
  });
  frame.addEventListener("load", send);
  new MutationObserver(send).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
})();
</script>

## The controls

Every control sets one keyword of `EchoStateNetwork` or of `train()`, and the defaults are
those of the class.

**Data.** The number of sites of the state (`N_dim`), the time step of the data (`dt`),
the number of data steps per network step (`upsample`), the lengths of the training and
validation records (`t_train`, `t_val`), and the number of washout steps (`N_wash`). The
tiles show the resulting numbers of washout, training and validation steps.

**Layout.** One reservoir for the whole state, or one reservoir per patch of sites, with
shared matrices or with independent patches (`patch_size`, `halo`, `periodic`, `shared`).
The patch sizes on offer divide the number of sites, and the halo stops where the window
of a patch would exceed a periodic domain. The parallel layout reads and forecasts every
site, so it disables the controls of the inputs and the readout.

**Inputs and readout.** The number of observed sites (`observed_idx`, spread evenly over
the state), the number of physical parameters appended to the input
(`input_parameters`), whether the parameters enter the reservoir
(`param_in_reservoir`), what the readout reads besides the reservoir state
(`readout_input`), and the level of the noise added to the training inputs (`noise`).

**Reservoir.** The number of units of each reservoir (`N_units`), the number of
connections per unit (`connect`), whether the input matrix is sparse or dense
(`Win_type`), and the number of realizations of the random matrices,
$\mathbf{W}_\mathrm{in}$ and $\mathbf{W}$, with the seed of the first one
(`train(n_seeds=...)`, `seed`). With more than one realization, `train()` trains every
realization in its own process, with its own hyperparameter search, and keeps the
realization with the lowest validation score.

**Hyperparameters.** For each of the spectral radius, $\rho$, the input scaling,
$\sigma_\mathrm{in}$, the Tikhonov factor, $\beta$, and the leak rate, $\alpha$: either a
search range (`rho_range`, `sigma_in_range` in $\log_{10}$, the grid `tikh_range`,
`leak_rate_range`) or a fixed value (`rho`, `sigma_in`, `tikh`, `leak_rate`), together
with the budget of the Bayesian optimization (`N_func_evals`, `N_grid`). The ticked
hyperparameters form `hyperparameters_to_optimize`.

**Validation.** The strategy that scores every evaluation of the search: chaotic recycle
validation (`RVC_Noise`, the default), single shot (`SSV`), walk forward (`WFV`) or
K-fold (`KFV`) validation, with the number of folds (`N_folds`), the step between folds
(`val_fold_step`) and the metric (`validation_metric`). The
[validation page](../validation.md) compares the strategies.

## The panels

**Architecture.** Two views of the same network. The folded view draws the network
once; the view unfolded in time draws one column per step $n$ of a forecast, with the
input at the bottom, the reservoir in the middle and the output at the top. In the
unfolded view, the reservoir state passes from each step to the next, the washout
drives the reservoir with the data for `N_wash` steps and discards its outputs except
the last one, which is the first forecast, and from then on every forecast is fed back
as the input of the next step. In the folded view, the input strip (left) and the output strip (right) have one cell per
site. Each brace marks the window that a reservoir reads. With the parallel layout, every
cell is hatched with the pattern of each reservoir that reads or forecasts it, and the
windows of the first and last patches wrap around a periodic domain (dashed copies of the
sites) or read zeros outside a non-periodic one (grey cells). Unobserved sites are grey.
Parameters enter from above, in orange, at a junction of the input arrow or directly into
the readout arrow. The arrow that leaves the input arrow downwards and enters the readout
arrow from below is the input skip, the loop above a reservoir marks a leak rate below one,
and independent patches draw their networks in different colours.

**Equations.** The normalization of the input and the input noise, the reservoir update,
the readout, the washout, the ridge regression for the readout, the hyperparameter
selection with the validation objective of the chosen strategy, and the closed-loop
forecast. A fixed hyperparameter appears as its value, so that each equation reads as the
code solves it; for example, a fixed leak rate of one removes the leak from the reservoir
update. The patch index is a superscript, $\bm{r}^{(p)}$.

**Data split and validation folds.** The training record, to scale, with its washout,
its training and its validation steps, and one row per validation fold, placed as
`echostatenetwork.validation` places them: the rows of the ridge regression that train
the readout of the fold, the washout before each closed-loop validation run (recycle
validation), and the validation interval.

**Code.** The constructor and the training call that build the network, with the
keywords that differ from the class defaults, followed by a washout and a closed-loop
forecast.

The tiles count the rows of the reservoir state, the readout weights that the ridge
regression trains, the dimension and budget of the search, and the closed-loop
validation runs per evaluation and in total, which set the cost of `train()`.

## Where to go next

The tutorials train each architecture on data:
[tutorial 01](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/01_echo_state_network.ipynb)
a single reservoir with full and partial observation,
[tutorial 02](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/02_parametric_esn.ipynb)
a parametric network,
[tutorial 03](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/03_leaky_esn.ipynb)
a leaky integrator,
[tutorial 04](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/04_validation_strategies.ipynb)
the validation strategies, and
[tutorial 05](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/05_parallel_esn_kuramoto_sivashinsky.ipynb)
the parallel layout on the Kuramoto–Sivashinsky equation.

## References

- Pathak, J., Hunt, B., Girvan, M., Lu, Z., & Ott, E. (2018). Model-free prediction of
  large spatiotemporally chaotic systems from data: a reservoir computing approach.
  *Physical Review Letters*, 120, 024102.
- Racca, A., & Magri, L. (2021). Robust optimization and validation of echo state
  networks for learning chaotic dynamics. *Neural Networks*, 142, 252–268.
