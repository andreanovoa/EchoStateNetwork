"""Benchmark of `echostatenetwork` against the reference code of Racca & Magri (2021).

Reference: github.com/MagriLab/Robust-Validation-ESN (Jupyter notebooks in ``source/``),
the code of Racca & Magri (2021), *Robust optimization and validation of echo state
networks for learning chaotic dynamics*, Neural Networks 142, 252-268 (arXiv:2103.03174).
The script reads the code cells of the reference notebooks (``Functions.ipynb``,
``Val_Functions.ipynb``, the search cell of ``Validation_Short_Lorenz.ipynb`` and the test
cell of ``Test_Short_Lorenz.ipynb``) and executes them in a private namespace, so that the
reference runs as published, on its own 'short' Lorenz 63 dataset (forward Euler,
dt = 0.01; washout 1 LT, training 8 LT, validation 3 LT; one LT = 111 steps).

Four comparisons, in order:

1. Model equivalence. With the same data, W_in, W, rho, sigma_in, input and output bias,
   washout and Tikhonov factor, both codes give the same W_out and closed-loop forecast
   once the conventions are mapped (``mapped_esn``): input bias 1 (package default 0.1),
   normalization by the reference's fixed range without shift (package: mean shift and the
   range of the training inputs), no input noise, ``upsample=1``, the reference's matrices
   transposed into the package (the reference multiplies row vectors from the left).
2. Validation objective. For one network and the 5 x 5 initial grid, the fold placement
   (the data rows each fold scores), the number of folds and the objective of every
   strategy of the reference (SSV, WFV, WFC, KFV, KFC, RV, RVC) against the package
   (``_SSV``, ``_WFV``, ``_KFV``, ``_RVC_Noise``; the chaotic variants with
   ``val_fold_step`` = 1 LT), in both metrics (the reference's log10 MSE and the package's
   log10 nMAE).
3. End to end. For every strategy and reservoir seed, the hyperparameter search of each code
   (5 x 5 grid + 24 gp-hedge points over rho in (0.1, 1), sigma_in in (0.5, 5), tikh =
   1e-11), then the test of the reference (100 intervals from 24 LT, spaced 3 LT, washout
   1 LT): log10 MSE over 3 LT and prediction horizon (threshold 0.2, at most 15 LT), over 50
   seeds (the paper's ensemble). Four arms: ``ref`` (the reference as published), ``pkg``
   (the package on the reference's network, conventions mapped as in step 1), ``pkg_lin``
   (``pkg`` searching sigma_in on the reference's linear axis and scoring the folds with its
   log10 MSE, which leaves the fold conventions and the Gaussian-process kernel as the only
   differences) and ``native`` (the package with its own conventions and matrices of the same
   seed; only data, budget, ranges and Tikhonov factor shared). ``--quick``: 5 seeds, a
   3 x 3 grid + 3 points and 20 test intervals (about 30 s on 20 workers).
4. Runtime of one training, one validation evaluation and one full search, both codes.

Outputs (written next to this script): ``benchmark_robust_validation.npz`` (all numbers,
the stdout report included) and ``benchmark_robust_validation.{pdf,png}`` (step 3:
prediction horizon per strategy, both codes, seeds as points).

Run:  git clone --depth 1 https://github.com/MagriLab/Robust-Validation-ESN <dir>
      python scripts/benchmark_robust_validation.py --reference <dir> [--quick]
      [--report-only]   # reload the npz and regenerate the report and the figure
"""
import os

# Workers must not oversubscribe BLAS (the linear algebra is tiny); set before numpy.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

from echostatenetwork import EchoStateNetwork, validation  # noqa: E402

# ______________________________________________________________ experiment constants
# (the values of Validation_Short_Lorenz.ipynb and Test_Short_Lorenz.ipynb)

DT = 0.01
N_LYAP = int(0.9 ** -1 / DT)          # 111 steps in one Lyapunov time
N_WASH, N_TRAIN, N_VAL = 1 * N_LYAP, 8 * N_LYAP, 3 * N_LYAP
N_WTV = N_WASH + N_TRAIN + N_VAL      # 1332 rows of washout, training and validation data
N_TSTART = 24 * N_LYAP                # first test interval
N_DATA = N_WTV + 500 * N_LYAP         # the length the reference generates
U0 = np.array([7.432487609628195, 10.02071718705213, 29.62297428638419])
NORM_REF = np.array([35.23020746, 45.09776766, 36.07598481])   # hard-coded in the reference

N_UNITS = 100
TIKH = 1e-11
RHO_RANGE = (0.1, 1.0)
SIGMA_IN_RANGE = (0.5, 5.0)           # linear in the reference; the package searches log10(sigma_in)
N_GRID, N_FUNC_EVALS = 5, 49          # 5 x 5 grid + 24 gp-hedge points (files *_50_5.h5)
THRESHOLD = 0.2                       # PH threshold; the published notebooks never define it
KK = 5                                # the PH runs over at most KK * N_VAL steps (15 LT)
HP_PROBE = (0.7, 1.25)                # (rho, sigma_in) of steps 1 and 2

# name -> (reference function, package strategy, N_folds, val_fold_step)
STRATEGIES = {
    'SSV': ('SSV', '_SSV', 1, None),
    'WFV': ('WFV', '_WFV', 2, None),
    'WFC': ('WFC', '_WFV', 3, N_LYAP),
    'KFV': ('KFV', '_KFV', 3, None),
    'KFC': ('KFC', '_KFV', 9, N_LYAP),
    'RV': ('RV', '_RVC_Noise', 3, None),
    'RVC': ('RVC', '_RVC_Noise', 9, None),
}
E2E_STRATEGIES = ['SSV', 'WFV', 'WFC', 'KFV', 'KFC', 'RVC']   # step 3 (as compare_validation_strategies)
ARMS = ['ref', 'pkg', 'pkg_lin', 'native']

FULL = dict(seeds=list(range(1, 51)), n_test=100, n_grid=N_GRID, n_func_evals=N_FUNC_EVALS)
QUICK = dict(seeds=list(range(1, 6)), n_test=20, n_grid=3, n_func_evals=12)

OUT_DIR = Path(__file__).resolve().parent
STEM = 'benchmark_robust_validation'

# ______________________________________________________________ the reference code


def _code_cells(path, contains=None):
    """Code cells of a notebook, magics (``%...``) dropped; with `contains`, only the cells
    holding that text."""
    nb = json.loads(Path(path).read_text())
    cells = []
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        if cell['cell_type'] != 'code' or (contains is not None and contains not in src):
            continue
        cells.append('\n'.join(line for line in src.splitlines() if not line.lstrip().startswith('%')))
    return cells


_REF = {}   # one namespace per process


def reference(ref_dir):
    """The reference code, executed in a private namespace: its ESN (Functions.ipynb), its
    strategies (Val_Functions.ipynb), its search ``g`` (Validation_Short_Lorenz.ipynb) and
    its test ``f`` (Test_Short_Lorenz.ipynb), on its own data. The functions read their
    settings as globals of the namespace, which `set_network` and `set_search` write."""
    key = str(ref_dir)
    if key in _REF:
        return _REF[key]
    src = Path(ref_dir)
    import skopt
    from skopt.learning import GaussianProcessRegressor as GPR
    from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
    from skopt.space import Real
    ns = dict(np=np, skopt=skopt, GPR=GPR, ConstantKernel=ConstantKernel, Matern=Matern, Real=Real)
    for cell in _code_cells(src / 'Functions.ipynb') + _code_cells(src / 'Val_Functions.ipynb'):
        exec(cell, ns)
    (search_cell,) = _code_cells(src / 'Validation_Short_Lorenz.ipynb', contains='def g(val)')
    (test_cell,) = _code_cells(src / 'Test_Short_Lorenz.ipynb', contains='def f(x)')
    exec(search_cell, ns)
    exec(test_cell, ns)

    # the globals of Validation_Short_Lorenz.ipynb (cells 2 and 4)
    U = ns['solve_ode'](N_DATA, DT, U0)
    ns.update(U=U, dt=DT, N_lyap=N_LYAP, N_washout=N_WASH, N_train=N_TRAIN, N_val=N_VAL,
              N_tstart=N_TSTART, norm=NORM_REF,
              U_washout=U[:N_WASH], U_t=U[N_WASH:N_WASH + N_TRAIN - 1],
              Y_t=U[N_WASH + 1:N_WASH + N_TRAIN], U_tv=U[N_WASH:N_WTV - 1], Y_tv=U[N_WASH + 1:N_WTV],
              Y_v=U[N_WASH + N_TRAIN:N_WTV],
              bias_in=1.0, bias_out=1.0, N_units=N_UNITS, dim=3, connectivity=3,
              sigma_in=1.0, rho=1.0, sparseness=1 - 3 / (N_UNITS - 1), tikh=TIKH,
              threshold=THRESHOLD)
    _REF[key] = ns
    return ns


def reference_matrices(seed):
    """W_in (dim+1, N_units) and W (N_units, N_units) of realization `seed`, drawn as in
    cell 8 of Validation_Short_Lorenz.ipynb (one input per unit, input bias included; W
    uniform with 3 connections per unit on average, scaled to unit spectral radius)."""
    rnd = np.random.RandomState(seed)
    Win = np.zeros((4, N_UNITS))
    for j in range(N_UNITS):
        Win[rnd.randint(0, 4), j] = rnd.uniform(-1, 1)
    W = rnd.uniform(-1, 1, (N_UNITS, N_UNITS)) * (rnd.rand(N_UNITS, N_UNITS) < 3 / (N_UNITS - 1))
    W /= np.max(np.abs(np.linalg.eigvals(W)))
    return Win, W


def set_network(ns, seed, rho=None, sigma_in=None):
    ns['Win'], ns['W'] = reference_matrices(seed)
    if rho is not None:
        ns['rho'], ns['sigma_in'] = rho, sigma_in


def set_search(ns, n_grid, n_func_evals):
    """The initial grid of cell 6 of Validation_Short_Lorenz.ipynb for `n_grid` points per
    axis (that cell hard-codes 7), and the budget ``n_tot``."""
    lo, hi, slo, shi = ns['spec_in'], ns['spec_end'], ns['in_scal_in'], ns['in_scal_end']
    ns['n_grid'], ns['n_tot'] = n_grid, n_func_evals
    ns['x1'] = [[lo + (hi - lo) / (n_grid - 1) * i, slo + (shi - slo) / (n_grid - 1) * j]
                for i in range(n_grid) for j in range(n_grid)]


# ______________________________________________________________ the package


def log10_mse(case, Y_true, Y_pred, norm=None):
    """The reference's fold error, log10 of the MSE in physical units (diverged: 10)."""
    m = np.log10(np.mean((Y_pred - Y_true) ** 2))
    return float(m) if np.isfinite(m) else 10.


class MappedESN(EchoStateNetwork):
    """`EchoStateNetwork` with two of the reference's conventions the package has no option
    for: a fixed input normalization without shift (`fixed_norm`), and sigma_in searched on a
    linear axis (`linear_sigma_in`; the package searches log10 sigma_in). Neither changes the
    model equations."""
    fixed_norm = None
    linear_sigma_in = False

    def _split_and_format_data(self, data=None, add_noise=True):
        out = super()._split_and_format_data(data, add_noise=add_noise)
        if self.fixed_norm is not None:
            self.norm = np.asarray(self.fixed_norm, dtype=float).copy()
            self.shift = np.zeros_like(self.norm)
        return out

    def _reset_hyperparams(self, params, names, tikhonov=None):
        super()._reset_hyperparams(params, names, tikhonov=tikhonov)
        if self.linear_sigma_in and 'sigma_in' in names:
            self.sigma_in = params[list(names).index('sigma_in')]


def package_esn(arm, seed, name=None, n_grid=N_GRID, n_func_evals=N_FUNC_EVALS, search=True,
                rho=HP_PROBE[0], sigma_in=HP_PROBE[1], metric=None, tikh=TIKH):
    """The package network of an arm: 'pkg' maps the reference's conventions and takes its
    matrices of realization `seed`; 'pkg_lin' is 'pkg' searching sigma_in on the linear axis
    of the reference with its metric (log10 MSE); 'native' keeps the package's own conventions
    and draws its matrices from `seed`. All share the split, the budget, the ranges and tikh."""
    n_folds, step = (STRATEGIES[name][2:] if name else (4, None))
    kwargs = dict(dt=DT, upsample=1, seed=int(seed), N_units=N_UNITS, N_wash=N_WASH,
                  t_train=(N_WASH + N_TRAIN) * DT, t_val=N_VAL * DT, t_test=0.,
                  N_folds=n_folds, val_fold_step=step, N_grid=n_grid, N_func_evals=n_func_evals,
                  hyperparameters_to_optimize=['rho', 'sigma_in'] if search else [],
                  rho_range=RHO_RANGE, sigma_in_range=tuple(np.log10(SIGMA_IN_RANGE)),
                  rho=rho, sigma_in=sigma_in, tikh=tikh, tikh_range=[tikh],
                  validation_metric=metric)
    if arm in ('pkg', 'pkg_lin'):
        kwargs.update(bias_in=np.array([1.0]), bias_out=np.array([1.0]), noise=0.0)
    if arm == 'pkg_lin':
        kwargs.update(sigma_in_range=SIGMA_IN_RANGE, validation_metric=metric or log10_mse)
    esn = MappedESN(np.zeros((3, 1)), **kwargs)
    if arm == 'pkg_lin':
        esn.linear_sigma_in = True
    if arm in ('pkg', 'pkg_lin'):
        esn.fixed_norm = NORM_REF
        Win, W = reference_matrices(seed)
        esn.Win, esn.W = Win.T, W.T          # the reference computes u @ Win and r @ W
    return esn


def train_package(esn, data_wtv, arm, name=None):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='(WFV|KFV): only', category=UserWarning)
    strategy = getattr(EchoStateNetwork, STRATEGIES[name][1]) if name else None
    with redirect_stdout(io.StringIO()):
        esn.train(data_wtv, add_noise=(arm == 'native'), plot_training=False,
                  validation_strategy=strategy)
    return esn


def package_forecast(esn, U, start, n):
    """Washout from r = 0 on the N_WASH rows before `start`, then `n` predictions of rows
    ``start ...``: the first is the open-loop readout of the washout, the others closed loop
    (the convention of the reference's ``closed_loop`` and ``predictability_horizon``)."""
    r = np.zeros((esn.N_r, 1))
    for u_in in U[start - N_WASH:start]:
        u, r = esn.step(u_in, r)
    Y = np.empty((n, 3))
    Y[0] = u[:, 0]
    for i in range(1, n):
        u, r = esn.step(esn.outputs_to_inputs(u), r)
        Y[i] = u[:, 0]
    return Y


def horizon(Yh, Y):
    """The prediction horizon of the reference's ``predictability_horizon``, in LT: the first
    step i >= 1 whose error norm, over the root mean square norm of Y, exceeds THRESHOLD
    (len(Y) - 1 when none does)."""
    E = np.sqrt(np.sum((Yh - Y) ** 2, axis=1) / np.mean(np.sum(Y ** 2, axis=1)))
    above = np.flatnonzero(E[1:] > THRESHOLD)
    return ((above[0] + 1) if above.size else len(Y) - 1) / N_LYAP


def test_package(esn, U, n_test):
    """The test of the reference (``f`` of Test_Short_Lorenz.ipynb) on a package network:
    mean log10 MSE over the first N_VAL steps and mean prediction horizon, over the `n_test`
    intervals that start at N_TSTART + i N_VAL."""
    mse, ph = [], []
    for i in range(n_test):
        s = N_TSTART + i * N_VAL
        Y = U[s:s + KK * N_VAL]
        Yh = package_forecast(esn, U, s, KK * N_VAL)
        m = np.log10(np.mean((Yh[:N_VAL] - Y[:N_VAL]) ** 2))
        mse.append(m if np.isfinite(m) else 10.)
        ph.append(horizon(Yh, Y))
    return float(np.mean(mse)), float(np.mean(ph))


def test_reference(ns, x, n_test):
    """``f`` of Test_Short_Lorenz.ipynb, which hard-codes 100 intervals; `n_test` < 100 runs
    the same loop on the first `n_test` (quick mode)."""
    if n_test == 100:
        return ns['f'](list(x))
    ns['rho'], ns['sigma_in'] = x
    Wout = ns['train'](ns['U_washout'], ns['U_tv'], ns['Y_tv'], TIKH)[1]
    U, mse, ph = ns['U'], 0., 0.
    for i in range(n_test):
        s = N_TSTART + i * N_VAL
        xa1 = ns['open_loop'](U[s - N_WASH:s], np.zeros(N_UNITS))[-1]
        Yh = ns['closed_loop'](N_VAL - 1, xa1, Wout)[0]
        m = np.log10(np.mean((Yh - U[s:s + N_VAL]) ** 2))
        mse += 10. if np.isnan(m) else m
        ph += ns['predictability_horizon'](xa1, U[s:s + KK * N_VAL], Wout)
    return mse / n_test, ph / n_test


# ______________________________________________________________ step 1: model equivalence


def step1(ref_dir, seed=1):
    ns = reference(ref_dir)
    U = ns['U']
    rho, sigma_in = HP_PROBE
    set_network(ns, seed, rho, sigma_in)
    s = N_TSTART                                       # closed loop from the first test interval
    xa1 = ns['open_loop'](U[s - N_WASH:s], np.zeros(N_UNITS))[-1]

    out = {}
    for label, arm, tikh, extra in [
            ('mapped', 'pkg', TIKH, {}),
            ('mapped, bias row not regularized', 'pkg', TIKH, dict(regularize_bias_out=False)),
            ('mapped, tikh = 1e-6', 'pkg', 1e-6, {}),
            ('package conventions, same W and W_in', 'native', TIKH, {})]:
        Xa, Wout_ref, LHS = ns['train'](ns['U_washout'], ns['U_tv'], ns['Y_tv'], tikh)[:3]
        esn = package_esn(arm, seed, search=False, tikh=tikh)
        for k, v in extra.items():
            setattr(esn, k, v)
        if arm == 'native':      # the reference's matrices, the package's normalization and bias
            Win, W = reference_matrices(seed)
            esn.Win, esn.W = Win.T, W.T
            esn.noise = 0.
        train_package(esn, U[:N_WTV], 'pkg')
        dW = np.abs(esn.Wout - Wout_ref).max() / np.abs(Wout_ref).max()
        Yh_ref = ns['closed_loop'](KK * N_VAL - 1, xa1, Wout_ref)[0]
        Yh_pkg = package_forecast(esn, U, s, KK * N_VAL)
        err = np.linalg.norm(Yh_pkg - Yh_ref, axis=1) / np.sqrt(np.mean(np.sum(Yh_ref ** 2, axis=1)))
        sep = np.flatnonzero(err > 1e-3)
        ph_ref = ns['predictability_horizon'](xa1, U[s:s + KK * N_VAL], Wout_ref)
        out[label] = dict(dWout=float(dW), err_1LT=float(err[:N_LYAP].max()),
                          sep_LT=float(sep[0] / N_LYAP) if sep.size else np.inf,
                          ph_ref=float(ph_ref), ph_pkg=float(horizon(Yh_pkg, U[s:s + KK * N_VAL])),
                          cond=float(np.linalg.cond(LHS)), norm=esn.norm.copy(), shift=esn.shift.copy())
    # the reservoir states of the open-loop pass: the package's _compute_RR_terms vs open_loop
    Xa, Wout_ref = ns['train'](ns['U_washout'], ns['U_tv'], ns['Y_tv'], TIKH)[:2]
    esn = package_esn('pkg', seed, search=False)
    train_package(esn, U[:N_WTV], 'pkg')
    R = esn._compute_RR_terms(U[np.newaxis, :N_WTV - 1], U[np.newaxis, 1:N_WTV])[3][0]
    out['dR'] = float(np.abs(R - Xa[1:, :N_UNITS]).max())
    # one-step predictions on the training rows
    out['dY_open'] = float(np.abs(np.hstack([R, np.ones((len(R), 1))]) @ esn.Wout - Xa[1:] @ Wout_ref).max())
    return out


def report_step1(s1):
    lines = ['STEP 1 - model equivalence (seed 1, rho = {:.2f}, sigma_in = {:.2f}, tikh = {:g})'.format(*HP_PROBE, TIKH),
             f'  open-loop reservoir states, max |R_pkg - R_ref|          : {s1["dR"]:.2e}',
             f'  open-loop readout on the training rows, max |dY|          : {s1["dY_open"]:.2e}', '',
             f'  {"package configuration":<40} {"cond(LHS)":>9} {"|dWout|/|Wout|":>15} {"dY 1 LT":>9} '
             f'{"split [LT]":>10} {"PH ref":>7} {"PH pkg":>7}']
    for label, d in s1.items():
        if not isinstance(d, dict):
            continue
        lines.append(f'  {label:<40} {d["cond"]:9.1e} {d["dWout"]:15.2e} {d["err_1LT"]:9.1e} {d["sep_LT"]:10.2f} '
                     f'{d["ph_ref"]:7.2f} {d["ph_pkg"]:7.2f}')
    lines.append('  (dY 1 LT: largest closed-loop difference over the first LT, relative to the RMS state; '
                 'split: first time it exceeds 1e-3)')
    return '\n'.join(lines)


# ______________________________________________________________ step 2: validation objective


class FoldRecorder:
    """A package validation metric that records, per fold, the data row of the first scored
    target, the number of scored steps and both metrics; it returns `use` (the log10 MSE of
    the reference or the package's log10 nMAE)."""

    def __init__(self, U, use):
        self.U, self.use, self.folds = U, use, []

    def __call__(self, case, Y_true, Y_pred, norm=1.0):
        t0 = int(np.flatnonzero((self.U[:N_WTV] == Y_true[0]).all(axis=1))[0])
        mse = float(np.log10(np.mean((Y_pred - Y_true) ** 2)))
        nmae = validation.log_nMAE(case, Y_true, Y_pred, norm)
        self.folds.append((t0, len(Y_true), mse, nmae))
        return mse if self.use == 'mse' else nmae


def reference_folds(ns, name, x):
    """Evaluate the reference strategy `name` at x, recording per fold the data row of the
    first scored target, the number of scored steps, whether the first scored step is an
    open-loop readout, and both metrics."""
    U = ns['U']
    ns['rho'], ns['sigma_in'] = x                 # the strategy sets them too; the states below need them
    Xa_full = ns['open_loop'](U[:N_WTV], np.zeros(N_UNITS))[:, :N_UNITS]
    norm_l = U[1:N_WTV].max(axis=0) - U[1:N_WTV].min(axis=0)       # the package's nMAE norm
    folds, original = [], ns['closed_loop']

    def recording(N, x0, Wout):
        Yh, xa = original(N, x0, Wout)
        k = int(np.argmin(np.abs(Xa_full - x0[:N_UNITS]).max(axis=1)))   # x0 = state after row k - 1
        skip = 1 if N == N_VAL else 0                                    # SSV drops its first output
        Y, Yp = U[k + skip:k + skip + N_VAL], Yh[skip:]
        folds.append((k + skip, len(Yp), skip == 0, float(np.log10(np.mean((Yp - Y) ** 2))),
                      float(np.log10(np.mean(np.abs(Yp - Y)) / np.mean(norm_l)))))
        return Yh, xa

    ns['closed_loop'] = recording
    try:
        value = float(ns[STRATEGIES[name][0]](list(x)))
    finally:
        ns['closed_loop'] = original
    return value, folds


def _evaluate_pair(ns, U, seed, name, x, tikh=TIKH):
    """One evaluation of strategy `name` at x by both codes: the reference's value and its
    folds, and the package's folds (`FoldRecorder`, scoring the log10 MSE)."""
    ns['tikh'] = tikh
    try:
        v, f_ref = reference_folds(ns, name, x)
    finally:
        ns['tikh'] = TIKH
    rec = FoldRecorder(U, 'mse')
    esn = package_esn('pkg', seed, name, search=False, rho=x[0], sigma_in=x[1], metric=rec, tikh=tikh)
    esn.hyperparameters_to_optimize = ['tikh']      # one evaluation of the strategy, no search
    train_package(esn, U[:N_WTV], 'pkg', name)
    return v, f_ref, rec.folds


def step2(ref_dir, seed=1, n_grid=N_GRID):
    from scipy.stats import spearmanr
    ns = reference(ref_dir)
    U = ns['U']
    set_network(ns, seed)
    grid = [(r, s) for r in np.linspace(*RHO_RANGE, n_grid) for s in np.linspace(*SIGMA_IN_RANGE, n_grid)]
    out = {}
    for name in STRATEGIES:
        v, f_ref, f_pkg = _evaluate_pair(ns, U, seed, name, HP_PROBE)
        v6, _, f_pkg6 = _evaluate_pair(ns, U, seed, name, HP_PROBE, tikh=1e-6)
        d = dict(ref_folds=f_ref, pkg_folds=f_pkg, ref_value=v,
                 pkg_mse=float(np.mean([f[2] for f in f_pkg])),
                 pkg_nmae=float(np.mean([f[3] for f in f_pkg])),
                 ref_nmae=float(np.mean([f[4] for f in f_ref])),
                 ref_value_t6=v6, pkg_mse_t6=float(np.mean([f[2] for f in f_pkg6])))
        vals = np.array([[val,
                          np.mean([f[2] for f in fp]), np.mean([f[3] for f in fp])]
                         for val, fr, fp in (_evaluate_pair(ns, U, seed, name, x) for x in grid)])
        ref_vals, pkg_mse, pkg_nmae = vals.T
        d.update(grid_ref=ref_vals, grid_pkg_mse=pkg_mse, grid_pkg_nmae=pkg_nmae,
                 max_dmse=float(np.abs(pkg_mse - ref_vals).max()),
                 rs_mse=float(spearmanr(ref_vals, pkg_mse).statistic),
                 rs_nmae=float(spearmanr(ref_vals, pkg_nmae).statistic),
                 argmin_ref=grid[int(np.argmin(ref_vals))], argmin_pkg=grid[int(np.argmin(pkg_nmae))])
        out[name] = d
    return out


def _spans(folds, n):
    """Compact list of the first scored rows (and the scored length) of the folds."""
    starts = [f[0] for f in folds]
    return f'{len(folds)} x {folds[0][1]}: ' + ','.join(map(str, starts[:n])) + (',...' if len(starts) > n else '')


def report_step2(s2):
    lines = ['STEP 2 - validation objective (seed 1; fold rows are data rows of the first scored target; '
             f'the validation data are rows {N_WASH + N_TRAIN}..{N_WTV - 1})', '',
             f'  {"":4} {"reference folds":<46} {"package folds":<46} {"OL 1st":>6}',
             ]
    for name, d in s2.items():
        ol = 'yes' if d['ref_folds'][0][2] else 'no'
        lines.append(f'  {name:<4} {_spans(d["ref_folds"], 9):<46} {_spans(d["pkg_folds"], 9):<46} {ol:>6}')
    lines += ['  (OL 1st: the first scored step of the reference is the open-loop readout of the washed-out '
              'state; every scored step of the package is closed loop)', '',
              f'  at (rho, sigma_in) = {HP_PROBE}:  objective of each code, both metrics',
              f'  {"":4} {"ref log10MSE":>13} {"pkg log10MSE":>13} {"ref log10nMAE":>14} {"pkg log10nMAE":>14}'
              f' {"tikh=1e-6: ref":>15} {"pkg":>8}',
              ]
    for name, d in s2.items():
        lines.append(f'  {name:<4} {d["ref_value"]:13.4f} {d["pkg_mse"]:13.4f} {d["ref_nmae"]:14.4f} '
                     f'{d["pkg_nmae"]:14.4f} {d["ref_value_t6"]:15.4f} {d["pkg_mse_t6"]:8.4f}')
    lines += ['', f'  over the {len(s2["SSV"]["grid_ref"])}-point initial grid (tikh = {TIKH:g}):',
              f'  {"":4} {"max |dlog10MSE|":>16} {"rS(MSE)":>8} {"rS(nMAE)":>9}   argmin ref    argmin pkg (nMAE)']
    for name, d in s2.items():
        lines.append(f'  {name:<4} {d["max_dmse"]:16.3f} {d["rs_mse"]:8.3f} {d["rs_nmae"]:9.3f}   '
                     f'({d["argmin_ref"][0]:.3g}, {d["argmin_ref"][1]:.3g})  '
                     f'({d["argmin_pkg"][0]:.3g}, {d["argmin_pkg"][1]:.3g})')
    lines.append('  (rS: Spearman correlation over the grid between the reference objective and the package '
                 'objective in each metric)')
    return '\n'.join(lines)



def echo_state_check(ref_dir, seed=1):
    """The package's RVC washes out from r = 0 on the N_WASH rows before every fold; the
    reference starts every fold from the state of one open-loop pass over the whole series.
    Largest |r_washout - r_pass| over the reference's RVC fold starts, per rho (sigma_in of
    HP_PROBE): how far a 1 LT washout is from forgetting the initial state."""
    ns = reference(ref_dir)
    U = ns['U']
    set_network(ns, seed)
    out = {}
    for rho in np.linspace(*RHO_RANGE, N_GRID):
        ns['rho'], ns['sigma_in'] = rho, HP_PROBE[1]
        Xa = ns['open_loop'](U[:N_WTV], np.zeros(N_UNITS))[:, :N_UNITS]
        d = [np.abs(ns['open_loop'](U[t - N_WASH:t], np.zeros(N_UNITS))[-1, :N_UNITS] - Xa[t]).max()
             for t in range(2 * N_WASH, N_WTV - N_VAL + 1, N_LYAP)]
        out[float(rho)] = float(max(d))
    return out


# ______________________________________________________________ step 3: end to end


def member_task(task):
    """One (arm, strategy, seed): the hyperparameter search of that code, then the test of
    the reference on the selected network."""
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    ns = reference(task['ref_dir'])
    U = ns['U']
    name, seed, arm = task['strategy'], task['seed'], task['arm']
    t0 = time.perf_counter()
    extra = {}
    with redirect_stdout(io.StringIO()):
        if arm == 'ref':
            set_network(ns, seed)
            set_search(ns, task['n_grid'], task['n_func_evals'])
            res = ns['g'](ns[STRATEGIES[name][0]])
            x, fun = (float(res.x[0]), float(res.x[1])), float(res.fun)
            extra['grid_vals'] = [float(v) for v in res.func_vals[:task['n_grid'] ** 2]]
            t_search = time.perf_counter() - t0
            mse, ph = test_reference(ns, x, task['n_test'])
        else:
            esn = package_esn(arm, seed, name, n_grid=task['n_grid'], n_func_evals=task['n_func_evals'])
            train_package(esn, U[:N_WTV], arm, name)
            x, fun = (float(esn.rho), float(esn.sigma_in)), float(esn.bo_results['fun'])
            extra['grid_vals'] = [float(v) for v in esn.bo_results['func_vals'][:task['n_grid'] ** 2]]
            t_search = time.perf_counter() - t0
            mse, ph = test_package(esn, U, task['n_test'])
            if arm in ('pkg', 'pkg_lin'):   # the same network in the reference: its objective and its test
                set_network(ns, seed)
                extra['val_ref_metric'] = float(ns[STRATEGIES[name][0]](list(x)))
                extra['mse_reftest'], extra['ph_reftest'] = test_reference(ns, x, task['n_test'])
    return dict(arm=arm, strategy=name, seed=int(seed), rho=x[0], sigma_in=x[1], fun=fun,
                mse=float(mse), ph=float(ph), t_search=t_search, wall=time.perf_counter() - t0, **extra)


def run_step3(ref_dir, cfg):
    tasks = [dict(arm=arm, strategy=name, seed=seed, ref_dir=str(ref_dir), **{k: cfg[k] for k in
                  ('n_grid', 'n_func_evals', 'n_test')})
             for arm in ARMS for name in E2E_STRATEGIES for seed in cfg['seeds']]
    n_workers = max(1, min(len(tasks), os.cpu_count() - 4))
    print(f'{len(tasks)} searches on {n_workers} workers ...', flush=True)
    results, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context('spawn')) as ex:
        for r in ex.map(member_task, tasks, chunksize=1):
            results.append(r)
            if len(results) % 30 == 0 or len(results) == len(tasks):
                print(f'  {len(results)}/{len(tasks)} done ({time.perf_counter() - t0:.0f} s)', flush=True)
    return results


def _cell(results, arm, name, key):
    return np.array([r[key] for r in results if r['arm'] == arm and r['strategy'] == name
                     and key in r])


def report_step3(results, cfg):
    n = len(cfg['seeds'])
    lines = [f'STEP 3 - end to end ({n} seeds {cfg["seeds"][0]}..{cfg["seeds"][-1]}; '
             f'{cfg["n_grid"]}x{cfg["n_grid"]} grid + {cfg["n_func_evals"] - cfg["n_grid"] ** 2} gp-hedge points; '
             f'{cfg["n_test"]} test intervals; mean +- std over the seeds)', '',
             f'  {"":4} {"arm":<7} {"rho":>12} {"sigma_in":>12} {"val. objective":>16} '
             f'{"test log10MSE":>15} {"PH [LT]":>13} {"median PH":>9}']
    for name in E2E_STRATEGIES:
        for arm in ARMS:
            c = {k: _cell(results, arm, name, k) for k in ('rho', 'sigma_in', 'fun', 'mse', 'ph')}
            lines.append(f'  {name if arm == "ref" else "":4} {arm:<7} '
                         + ' '.join(f'{c[k].mean():6.2f}+-{c[k].std():4.2f}' if k != 'fun' else
                                    f'{c[k].mean():8.2f}+-{c[k].std():5.2f}' for k in ('rho', 'sigma_in', 'fun'))
                         + f' {c["mse"].mean():8.2f}+-{c["mse"].std():4.2f} {c["ph"].mean():7.2f}+-'
                           f'{c["ph"].std():4.2f} {np.median(c["ph"]):9.2f}')
    lines += ['  (val. objective: log10 MSE for ref and pkg_lin, log10 nMAE for pkg and native; pkg = the '
              'reference network with mapped conventions; pkg_lin = pkg searching sigma_in on a linear axis with '
              'the log10 MSE; native = package conventions)']
    for arm in ('pkg', 'pkg_lin'):
        lines += _paired(results, arm, n)
    lines += ['', '  initial grid, pkg_lin vs ref (same grid points, same objective; the searches differ '
              'only after it):']
    for name in E2E_STRATEGIES:
        ref = sorted((r for r in results if r['arm'] == 'ref' and r['strategy'] == name), key=lambda r: r['seed'])
        lin = sorted((r for r in results if r['arm'] == 'pkg_lin' and r['strategy'] == name),
                     key=lambda r: r['seed'])
        d = np.array([np.abs(np.subtract(a['grid_vals'], b['grid_vals'])).max() for a, b in zip(lin, ref)])
        same = sum(int(np.argmin(a['grid_vals']) == np.argmin(b['grid_vals'])) for a, b in zip(lin, ref))
        lines.append(f'  {name:4} max |d log10 MSE| over the grid: median {np.median(d):.3f}, max {d.max():.3f};'
                     f' same grid argmin in {same}/{n} seeds')
    return '\n'.join(lines)


def _paired(results, arm, n):
    lines = ['', f'  paired, same network ({arm} - ref):',
              f'  {"":4} {"dPH mean":>9} {"dPH std":>8} {"|dPH|<0.05":>10} {"same x":>7} '
              f'{"ref obj. at pkg x - ref optimum":>32} {"PH pkg: pkg test - ref test":>28}']
    for name in E2E_STRATEGIES:
        ref = sorted((r for r in results if r['arm'] == 'ref' and r['strategy'] == name), key=lambda r: r['seed'])
        pkg = sorted((r for r in results if r['arm'] == arm and r['strategy'] == name), key=lambda r: r['seed'])
        dph = np.array([p['ph'] - r['ph'] for p, r in zip(pkg, ref)])
        same = sum(abs(p['rho'] - r['rho']) < 1e-3 and abs(p['sigma_in'] - r['sigma_in']) < 1e-3
                   for p, r in zip(pkg, ref))
        dval = np.array([p['val_ref_metric'] - r['fun'] for p, r in zip(pkg, ref)])
        dtest = np.array([p['ph'] - p['ph_reftest'] for p in pkg])
        lines.append(f'  {name:4} {dph.mean():9.2f} {dph.std():8.2f} {np.sum(np.abs(dph) < .05):>7d}/{n:<2d} '
                     f'{same:>4d}/{n:<2d} {dval.mean():21.2f} +- {dval.std():5.2f} '
                     f'{np.abs(dtest).max():20.3f} (max)')
    lines.append('  (ref obj. at pkg x: the reference objective at the hyperparameters the package selects, '
                 'minus the reference optimum, in log10 MSE; same x: both within 1e-3)')
    return lines


# ______________________________________________________________ step 4: runtime


def _best_of(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def step4(ref_dir, results, seed=1):
    """Serial wall times on this machine (one process, one BLAS thread)."""
    warnings.filterwarnings('ignore', category=UserWarning)
    ns = reference(ref_dir)
    U = ns['U']
    set_network(ns, seed, *HP_PROBE)
    rows = {}
    rows['one training (washout + open loop + ridge)'] = (
        _best_of(lambda: ns['train'](ns['U_washout'], ns['U_tv'], ns['Y_tv'], TIKH)),
        _best_of(lambda: train_package(package_esn('pkg', seed, search=False), U[:N_WTV], 'pkg')))
    for name in ('SSV', 'KFC', 'RVC'):
        def pkg_eval(name=name):
            esn = package_esn('pkg', seed, name, search=False)
            esn.hyperparameters_to_optimize = ['tikh']
            train_package(esn, U[:N_WTV], 'pkg', name)
        rows[f'one validation evaluation, {name} (incl. training)'] = (
            _best_of(lambda name=name: ns[STRATEGIES[name][0]](list(HP_PROBE))), _best_of(pkg_eval))
    esn = package_esn('pkg', seed, search=False)
    train_package(esn, U[:N_WTV], 'pkg')
    r = np.zeros((N_UNITS, 1))
    u = U[0]
    xa = np.hstack([np.zeros(N_UNITS), 1.])
    rows['one step [us]'] = (1e6 * _best_of(lambda: [ns['step'](xa[:N_UNITS], u) for _ in range(2000)]) / 2000,
                             1e6 * _best_of(lambda: [esn.step(u, r) for _ in range(2000)]) / 2000)
    for name in E2E_STRATEGIES:
        rows[f'full search, {name} (median over seeds, in parallel workers)'] = tuple(
            float(np.median(_cell(results, arm, name, 't_search'))) for arm in ('ref', 'pkg'))
    rows['test of one network (median, in parallel workers)'] = tuple(
        float(np.median(_cell(results, arm, 'RVC', 'wall') - _cell(results, arm, 'RVC', 't_search')))
        for arm in ('ref', 'native'))
    return rows


def report_step4(rows):
    lines = ['STEP 4 - runtime on this machine [s unless stated]', '',
             f'  {"":<62} {"reference":>10} {"package":>10} {"ratio":>6}']
    for k, (a, b) in rows.items():
        lines.append(f'  {k:<62} {a:10.4g} {b:10.4g} {b / a:6.2f}')
    lines.append('  (the pkg test row of the package arm also runs the reference test, so it is read from the '
                 'native arm)')
    return '\n'.join(lines)


# ______________________________________________________________ figure


COLOURS = {'ref': '#2a78d6', 'pkg': '#eb6834', 'pkg_lin': '#e87ba4', 'native': '#1baf7a'}
MARKERS = {'ref': 'o', 'pkg': 's', 'pkg_lin': 'v', 'native': 'D'}
LABELS = {'ref': 'reference (Racca & Magri 2021)', 'pkg': 'package, reference network and conventions',
          'pkg_lin': 'as previous, linear $\\sigma_\\mathrm{in}$ axis and log$_{10}$ MSE',
          'native': 'package, own conventions'}


def make_figure(results, cfg, paths):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(9.0, 6.6), layout='constrained', sharex=True,
                            gridspec_kw=dict(height_ratios=(2.2, 1)))
    rng = np.random.default_rng(0)
    width = 0.2
    for k, name in enumerate(E2E_STRATEGIES):
        for a, arm in enumerate(ARMS):
            ph = _cell(results, arm, name, 'ph')
            xc = k + (a - (len(ARMS) - 1) / 2) * width
            axs[0].plot(xc + rng.uniform(-0.05, 0.05, len(ph)), ph, MARKERS[arm], ms=4.5,
                        mfc='none', mec=COLOURS[arm], mew=1.0, alpha=0.8,
                        label=LABELS[arm] if k == 0 else None)
            axs[0].errorbar(xc, ph.mean(), yerr=ph.std(), fmt='_', ms=14, mew=2, color='k', lw=1.2,
                            capsize=0, zorder=3)
        ref = sorted((r for r in results if r['arm'] == 'ref' and r['strategy'] == name), key=lambda r: r['seed'])
        for a, arm in enumerate(('pkg', 'pkg_lin')):
            pkg = sorted((r for r in results if r['arm'] == arm and r['strategy'] == name),
                         key=lambda r: r['seed'])
            d = np.array([p['ph'] - r['ph'] for p, r in zip(pkg, ref)])
            xc = k + (a - 0.5) * 0.3
            axs[1].plot(xc + rng.uniform(-0.06, 0.06, len(d)), d, MARKERS[arm], ms=4.5, mfc='none',
                        mec=COLOURS[arm], alpha=0.8)
            axs[1].errorbar(xc, d.mean(), yerr=d.std(), fmt='_', ms=14, mew=2, color='k', lw=1.2, zorder=3)
    axs[0].set_ylabel('test prediction horizon [LT]')
    axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False, fontsize=8)
    fig.suptitle(f'Lorenz 63, short dataset (12 LT): {len(cfg["seeds"])} reservoir seeds, '
                     f'{cfg["n_test"]} test intervals each (bar: mean $\\pm$ std)', fontsize=10)
    axs[1].axhline(0, color='0.6', lw=0.8)
    axs[1].set_ylabel('PH package $-$ reference\n(same network) [LT]')
    axs[1].set_xticks(range(len(E2E_STRATEGIES)), E2E_STRATEGIES)
    axs[1].set_xlabel('validation strategy')
    for ax in axs:
        ax.grid(axis='y', color='0.9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    for path in paths:
        fig.savefig(path, dpi=200)
    plt.close(fig)


# ______________________________________________________________ main


def _reference_commit(ref_dir):
    try:
        return subprocess.run(['git', '-C', str(Path(ref_dir).parent), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _flatten(prefix, obj, npz):
    """Store a nested dict of numbers/arrays/tuples as flat npz keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f'{prefix}{"_" if prefix else ""}{str(k).replace(" ", "_")}', v, npz)
    else:
        npz[prefix] = np.asarray(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reference', default=os.environ.get('ROBUST_VALIDATION_ESN'),
                        help='the clone of MagriLab/Robust-Validation-ESN (or its source/ folder)')
    parser.add_argument('--quick', action='store_true', help='5 seeds, 3x3 grid + 3 points, 20 test intervals')
    parser.add_argument('--report-only', action='store_true', help='reload the npz, redo the report and figure')
    args = parser.parse_args()
    matplotlib.use('Agg')
    cfg = QUICK if args.quick else FULL
    stem = STEM + ('_quick' if args.quick else '')
    npz_path = OUT_DIR / f'{stem}.npz'
    fig_paths = [OUT_DIR / f'{stem}.pdf', OUT_DIR / f'{stem}.png']

    if args.report_only:
        z = np.load(npz_path, allow_pickle=False)
        results = json.loads(str(z['step3_results_json']))
        print(str(z['report']))
        make_figure(results, cfg, fig_paths)
        print(f'Saved {fig_paths[0]} and {fig_paths[1]}')
        return

    if args.reference is None:
        parser.error('--reference: clone https://github.com/MagriLab/Robust-Validation-ESN and pass its path')
    ref_dir = Path(args.reference).resolve()
    if (ref_dir / 'source').is_dir():
        ref_dir = ref_dir / 'source'
    t_start = time.perf_counter()
    import echostatenetwork
    header = (f'echostatenetwork {echostatenetwork.__version__} vs MagriLab/Robust-Validation-ESN '
              f'{_reference_commit(ref_dir)[:10]} ({"quick" if args.quick else "full"} mode)')
    print(header, flush=True)

    s1 = step1(ref_dir)
    print('\n' + report_step1(s1), flush=True)
    s2 = step2(ref_dir, n_grid=cfg['n_grid'] if args.quick else N_GRID)
    esc = echo_state_check(ref_dir)
    rep2 = report_step2(s2) + '\n\n  RVC washout from r = 0 over 1 LT vs the state of the open-loop pass, ' \
        'max |dr| over the fold starts: ' + ', '.join(f'rho {k:.3g}: {v:.1e}' for k, v in esc.items())
    print('\n' + rep2, flush=True)
    results = run_step3(ref_dir, cfg)
    rep3 = report_step3(results, cfg)
    print('\n' + rep3, flush=True)
    s4 = step4(ref_dir, results)
    rep4 = report_step4(s4)
    print('\n' + rep4, flush=True)

    report = '\n\n'.join([header, report_step1(s1), rep2, rep3, rep4])
    npz = dict(report=report, seeds=np.asarray(cfg['seeds']), strategies=np.asarray(E2E_STRATEGIES),
               arms=np.asarray(ARMS), step3_results_json=json.dumps(results),
               threshold=THRESHOLD, tikh=TIKH, dt=DT, n_lyap=N_LYAP)
    _flatten('step1', {k: v for k, v in s1.items()}, npz)
    _flatten('step2', {name: {k: v for k, v in d.items() if not k.endswith('folds')} for name, d in s2.items()}, npz)
    for name, d in s2.items():
        npz[f'step2_{name}_ref_folds'] = np.asarray([f[:2] for f in d['ref_folds']])
        npz[f'step2_{name}_pkg_folds'] = np.asarray([f[:2] for f in d['pkg_folds']])
    _flatten('step2_echo_state', {f'rho{k:.3g}': v for k, v in esc.items()}, npz)
    for arm in ARMS:
        for name in E2E_STRATEGIES:
            for key in ('rho', 'sigma_in', 'fun', 'mse', 'ph', 't_search', 'wall'):
                npz[f'step3_{arm}_{name}_{key}'] = np.array(
                    [r[key] for r in sorted(results, key=lambda r: r['seed'])
                     if r['arm'] == arm and r['strategy'] == name])
    npz['step4_rows'] = np.asarray(list(s4))
    npz['step4_seconds'] = np.asarray(list(s4.values()))
    np.savez(npz_path, **npz)
    make_figure(results, cfg, fig_paths)
    print(f'\nSaved {npz_path}\nSaved {fig_paths[0]} and {fig_paths[1]}')
    print(f'Total runtime: {time.perf_counter() - t_start:.1f} s')


if __name__ == '__main__':
    main()
