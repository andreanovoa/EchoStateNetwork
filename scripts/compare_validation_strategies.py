"""Table-1 / Figure-8 comparison of validation strategies (Racca & Magri 2021) on Lorenz 63.

Reproduces, at reduced scale, the Sec. 5.1 model-free-ESN experiment of Racca &
Magri (2021), *Robust optimization and validation of echo state networks for
learning chaotic dynamics*, Neural Networks 142, 252-268 (arXiv:2103.03174):
how well does each validation strategy's selected-hyperparameter validation
error rank-predict the test error, over an ensemble of reservoir seeds?

Strategies (columns of Table 1):

- SSV:  single shot validation           (``EchoStateNetwork._SSV``),
- WFV:  walk forward validation          (``EchoStateNetwork._WFV``),
- WFVc: chaotic WFV                      (``_WFV`` with ``val_fold_step`` = 1 LT),
- KFV:  K-fold cross validation          (``EchoStateNetwork._KFV``),
- KFVc: chaotic KFV                      (``_KFV`` with ``val_fold_step`` = 1 LT),
- RVC:  chaotic recycle validation       (``_RVC_Noise``, the package default;
        its folds are spaced ~1 LT apart, the paper's RVc analogue).

The chaotic versions advance consecutive validation intervals by ONE LYAPUNOV
TIME instead of the validation-interval length, so the intervals overlap and
the fold count multiplies; the regular versions get their natural
non-overlapping fold count (requesting ``N_folds`` = 100 and letting the fold
geometry auto-reduce keeps the budget philosophy identical for all). WFVc is
the one exception to the open budget: ``_WFV`` derives its sliding
training-window length as what the folds leave over, so its chaotic fold
count is chosen to KEEP the regular WFV's training window (see
``wfv_chaotic_n_folds``) -- an open budget would collapse the window to a
couple of steps and the objective to noise. The paper's supplementary WFVc*
variant is not implemented: its Table-1 column is printed as '-'.

Datasets, in Lyapunov times (LT = 1/0.906, dt = 0.02 ~ 0.018 LT):

- short: washout 1 LT | train 8 LT | validation 3 LT   (12 LT total),
- long:  washout 2 LT | train 16 LT | validation 6 LT  (24 LT, double the
  train+val budget at the same 8:3 proportions).

Both are prefixes of one Lorenz 63 trajectory whose tail (past the long
dataset) is a SEPARATE common test set with ``N_TEST_WINDOWS`` starting points.

Per ensemble member (reservoir seed) and strategy, Bayesian hyperparameter
optimization (5x5 grid + 24 gp-hedge points) selects (rho, sigma_in) with
tikh fixed at 1e-11, and two numbers are recorded:

- m_Val:  the strategy's objective at the selected point, mapped back from its
  log10 scale (``10**bo_results['fun']``, the geometric-mean closed-loop
  normalized validation error over the folds);
- m_Test: closed-loop test error of the SAME hyperparameters with Wout
  RETRAINED on the full train+val series (a fresh ESN with
  ``hyperparameters_to_optimize=[]``, same seed and identical W/Win), scored
  as the geometric mean (log10-mean) over the test windows of the MSE of the
  range-normalized error over the first 2 LT of each closed-loop forecast.

m_Val is an MAE-like and m_Test an MSE-like quantity on different data, but
the Table-1 metric -- the Spearman rank correlation r_S between {m_Val} and
{m_Test} over the ensemble -- is invariant to such monotone rescalings.

Outputs (written next to this script):
- stdout: Table-1-style table of r_S per dataset x strategy ('*' marks each
  row's maximum),
- ``compare_validation_strategies.npz``: the table + all per-member data,
- ``compare_validation_strategies.pdf``: page 1 = the Figure-8 scatter grid
  (2x3, long dataset, -log10(m_Test) vs -log10(m_Val), regression line, r_S
  annotated); page 2 = box plots of test log10(MSE) and prediction horizon
  per strategy (long dataset, separate axes).

Run:  conda run -n qlrom python scripts/compare_validation_strategies.py
"""
import os

# Workers must not oversubscribe BLAS (the linear algebra is tiny); set before numpy.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import io  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from echostatenetwork import EchoStateNetwork  # noqa: E402

# ______________________________________________________________ experiment constants

LT = 1.0 / 0.906        # Lorenz 63 Lyapunov time
DT = 0.02               # ESN time step, ~0.018 LT per step
N_LT1 = int(round(LT / DT))  # one Lyapunov time in ESN steps (= 55)

N_UNITS = 100
TIKH = 1e-11            # fixed Tikhonov parameter
RHO_RANGE = (0.1, 1.0)  # spectral radius search range (paper Sec. 5.1)
# sigma_in is set as 10**x by the package (_reset_hyperparams), so the range is
# given in log10 to cover the paper's (0.5, 5) after the transform:
SIGMA_IN_RANGE = (np.log10(0.5), np.log10(5.0))
N_GRID = 5              # BHO: 5x5 initial grid ...
N_FUNC_EVALS = 49       # ... + 24 gp-hedge points
NOISE = 1e-3            # input noise regularization, identical for all strategies
N_FOLDS_MAX = 100       # requested folds; the fold geometry auto-reduces to what fits

SEEDS = np.arange(1, 97)  # reservoir-seed ensemble (paper: 50 networks; more
#                           seeds halve the sampling error of r_S, ~1/sqrt(n))

# Datasets: washout is inside t_train (the package's convention); t_val is the
# validation budget the strategies carve their folds from.
DATASETS = {
    'short': dict(t_wash=1.0 * LT, t_train=9.0 * LT, t_val=3.0 * LT),   # 12 LT
    'long': dict(t_wash=2.0 * LT, t_train=18.0 * LT, t_val=6.0 * LT),   # 24 LT
}

# name -> (EchoStateNetwork method, chaotic fold advance of 1 LT?)
STRATEGIES = {
    'SSV': ('_SSV', False),
    'WFV': ('_WFV', False),
    'WFVc': ('_WFV', True),
    'KFV': ('_KFV', False),
    'KFVc': ('_KFV', True),
    'RVC': ('_RVC_Noise', True),
}
TABLE_COLUMNS = ['SSV', 'WFV', 'WFVc', 'WFVc*', 'KFV', 'KFVc', 'RVC']

N_TEST_WINDOWS = 50               # test starting points on the common test tail
TEST_SPACING_LT = 2.0             # spacing between test starting points
T_WASH_TEST_LT = 1.0              # open-loop washout before each test forecast
T_FORECAST_LT = 5.0               # closed-loop forecast length per window
T_MSE_LT = 2.0                    # MSE window within each forecast
PH_THRESHOLD = 0.2                # prediction-horizon relative-error threshold

OUT_DIR = Path(__file__).resolve().parent

# ______________________________________________________________ data & geometry


def lorenz_time_series(n_steps, dt=DT, transient=20.0):
    """Integrate Lorenz 63 (sigma=10, rho=28, beta=8/3) and return ``n_steps``
    samples at spacing ``dt`` after discarding an on-attractor transient."""

    def rhs(_t, u):
        x, y, z = u
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z]

    t_eval = transient + np.arange(n_steps) * dt
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), [-8.0, 8.0, 27.0],
                    t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol.y.T  # (n_steps, 3)


def dataset_steps(cfg):
    """(n_wash, n_train, n_val) in ESN steps, with the class's own rounding."""
    return (int(round(cfg['t_wash'] / DT)), int(round(cfg['t_train'] / DT)),
            int(round(cfg['t_val'] / DT)))


def wfv_chaotic_n_folds(cfg):
    """N_folds for the chaotic WFV: keep the REGULAR WFV's sliding
    training-window length and advance it by 1 LT instead of N_val -- the
    paper's definition of the subscript-c variant. _WFV derives the window as
    ``m = n_post - (K-1)*step - N_val``, so an open fold budget (N_FOLDS_MAX)
    would max K and collapse the training window to a couple of steps; K must
    instead follow from the fixed window."""
    n_wash, n_train, n_val = dataset_steps(cfg)
    n_post = (n_train + n_val - 1) - n_wash
    k_reg = 1 + (n_post - 1 - n_val) // n_val          # regular WFV fold count
    m_reg = n_post - (k_reg - 1) * n_val - n_val       # its training window
    return 1 + (n_post - m_reg - n_val) // N_LT1


def rvc_n_folds(cfg):
    """N_folds giving _RVC_Noise a fold spacing of ~1 LT (the paper's RVc):
    its folds are evenly spaced over usable_span = Nt - N_wash - N_val with
    spacing usable_span // (N_folds - 1)."""
    n_wash, n_train, n_val = dataset_steps(cfg)
    usable_span = (n_train + n_val - 1) - n_wash - n_val
    return usable_span // N_LT1 + 1


def fold_count(name, cfg):
    """Fold count each strategy realizes on a dataset (for the report header)."""
    n_wash, n_train, n_val = dataset_steps(cfg)
    n_post = (n_train + n_val - 1) - n_wash    # post-washout teacher-forced rows
    method, chaotic = STRATEGIES[name]
    step = N_LT1 if chaotic else n_val
    if method == '_SSV':
        return 1
    if method == '_WFV':
        return wfv_chaotic_n_folds(cfg) if chaotic else \
            min(N_FOLDS_MAX, 1 + (n_post - 1 - n_val) // step)
    if method == '_KFV':
        return min(N_FOLDS_MAX, 1 + (n_post - n_val) // step)
    return rvc_n_folds(cfg)                     # _RVC_Noise


# ______________________________________________________________ per-member task


def make_esn(data_wtv, seed, cfg, n_folds, val_fold_step, **overrides):
    """ESN configured per the (scaled) Sec. 5.1 setup; fresh reservoir per seed."""
    n_wash = dataset_steps(cfg)[0]
    kwargs = dict(
        dt=DT, upsample=1, seed=int(seed),
        N_units=N_UNITS, N_wash=n_wash,
        t_train=cfg['t_train'], t_val=cfg['t_val'],
        N_folds=n_folds, val_fold_step=val_fold_step,
        N_grid=N_GRID, N_func_evals=N_FUNC_EVALS,
        hyperparameters_to_optimize=['rho', 'sigma_in'],
        rho_range=RHO_RANGE, sigma_in_range=SIGMA_IN_RANGE,
        tikh=TIKH, tikh_range=[TIKH], noise=NOISE,
    )
    kwargs.update(overrides)
    return EchoStateNetwork(data_wtv.T, **kwargs)


def closed_loop_window(esn, test_data, start, n_wash, n_fc):
    """Wash out open-loop on ``test_data[start:start+n_wash]``, then forecast
    ``n_fc`` steps closed-loop. Returns (Y_pred, Y_true), aligned as in the
    package's validation strategies (prediction i <-> row start+n_wash+1+i)."""
    r = np.zeros((esn.N_units, 1))
    u = np.zeros((esn.N_dim, 1))
    for u_in in test_data[start:start + n_wash]:
        u, r = esn.step(u_in, r)
    y_pred = np.zeros((n_fc, esn.N_dim))
    for i in range(n_fc):
        u_in = esn.outputs_to_inputs(full_state=u)
        u, r = esn.step(u_in, r)
        y_pred[i] = u[:, 0]
    y_true = test_data[start + n_wash + 1:start + n_wash + 1 + n_fc]
    return y_pred, y_true


def score_member(esn, test_data, starts, comp_range, ph_norm):
    """Closed-loop MSE (range-normalized components, first T_MSE_LT of each
    forecast) and prediction horizon (threshold PH_THRESHOLD, in LT) per test
    window."""
    n_wash = int(round(T_WASH_TEST_LT * LT / DT))
    n_fc = int(round(T_FORECAST_LT * LT / DT))
    n_mse = int(round(T_MSE_LT * LT / DT))

    mse_w, horizon_w = [], []
    for s in starts:
        y_pred, y_true = closed_loop_window(esn, test_data, s, n_wash, n_fc)
        err = (y_pred - y_true) / comp_range  # range-normalized error
        mse_w.append(np.mean(err[:n_mse] ** 2))
        rel = np.linalg.norm(err, axis=1) / ph_norm
        above = np.flatnonzero(rel > PH_THRESHOLD)
        n_ph = above[0] + 1 if above.size else n_fc  # cap at the forecast length
        horizon_w.append(n_ph * DT / LT)
    return np.array(mse_w), np.array(horizon_w)


def member_task(task):
    """One (dataset, strategy, seed) member: BHO with the strategy -> m_Val,
    retrain Wout on the full train+val series with the selected
    hyperparameters -> m_Test + horizon on the common test tail."""
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    cfg = DATASETS[task['dataset']]
    method, chaotic = STRATEGIES[task['strategy']]
    if method == '_RVC_Noise':
        n_folds = rvc_n_folds(cfg)
    elif method == '_WFV' and chaotic:
        n_folds = wfv_chaotic_n_folds(cfg)  # keep WFV's training-window length
    else:
        n_folds = N_FOLDS_MAX
    val_fold_step = N_LT1 if (chaotic and method != '_RVC_Noise') else None

    t0 = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        # ---- validation phase: BHO of (rho, sigma_in) with this strategy
        esn = make_esn(task['data_wtv'], task['seed'], cfg, n_folds, val_fold_step)
        esn.train(task['data_wtv'], plot_training=False,
                  validation_strategy=getattr(EchoStateNetwork, method))
        # the strategy's objective at the selected point is a log10 error;
        # map back so m_Val is a positive MSE-like quantity (rank-preserving)
        m_val = 10.0 ** esn.bo_results['fun']

        # ---- test phase: fresh ESN, same seed and IDENTICAL W/Win, selected
        # hyperparameters fixed (no BHO), Wout retrained on full train+val
        esn_t = make_esn(task['data_wtv'], task['seed'], cfg, n_folds, None,
                         hyperparameters_to_optimize=[],
                         rho=float(esn.rho), sigma_in=float(esn.sigma_in),
                         tikh=float(esn.tikh))
        esn_t.W = esn.W.copy()
        esn_t.Win = esn.Win.copy()
        esn_t.train(task['data_wtv'], plot_training=False)

        mse_w, horizon_w = score_member(esn_t, task['data_test'], task['starts'],
                                        task['comp_range'], task['ph_norm'])
    log10_mse_w = np.log10(mse_w)
    m_test = 10.0 ** log10_mse_w.mean()  # geometric mean over the test windows

    return dict(dataset=task['dataset'], strategy=task['strategy'],
                seed=int(task['seed']), m_val=float(m_val), m_test=float(m_test),
                log10_mse_windows=log10_mse_w, horizon_windows=horizon_w,
                horizon_mean=float(horizon_w.mean()), rho=float(esn.rho),
                sigma_in=float(esn.sigma_in), tikh=float(esn.tikh),
                wall=time.perf_counter() - t0)


# ______________________________________________________________ reporting


def rs_table(r_s):
    """Table-1-style stdout table: rows = datasets, columns = strategies (with
    the unimplemented WFVc* as '-'), '*' marks each row's maximum."""
    lines = ['Spearman rank correlation r_S between m_Val and m_Test '
             f'({len(SEEDS)} reservoir seeds):', '']
    header = f'{"dataset":>8} |' + ''.join(f'{c:>8}' for c in TABLE_COLUMNS)
    lines += [header, '-' * len(header)]
    for ds in DATASETS:
        best = max(r_s[ds].values())
        row = f'{ds:>8} |'
        for col in TABLE_COLUMNS:
            if col not in r_s[ds]:
                row += f'{"-":>8}'
            else:
                mark = '*' if r_s[ds][col] == best else ''
                row += f'{r_s[ds][col]:.2f}{mark}'.rjust(8)
        lines.append(row)
    lines.append("('*' marks the row maximum; WFVc* is the paper's "
                 'supplementary variant, not implemented)')
    return '\n'.join(lines)


def make_figure8(results, r_s, pdf):
    """Page 1: Figure-8 scatter grid for the long dataset -- one panel per
    strategy, -log10(m_Test) vs -log10(m_Val), regression line, r_S annotated."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(11.0, 7.0), layout='constrained',
                            sharey=True)
    for ax, name in zip(axs.flat, STRATEGIES):
        x = -np.log10([r['m_val'] for r in results['long'][name]])
        y = -np.log10([r['m_test'] for r in results['long'][name]])
        ax.plot(x, y, 'o', ms=5, color='tab:blue', alpha=0.75)
        a, b = np.polyfit(x, y, 1)
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, a * xx + b, '-', color='tab:red', lw=1.5)
        ax.set_title(name)
        ax.text(0.04, 0.92, f'$r_S = {r_s["long"][name]:.2f}$',
                transform=ax.transAxes, va='top')
        ax.set_xlabel(r'$-\log_{10}(m_\mathrm{Val})$')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'$-\log_{10}(m_\mathrm{Test})$')
    fig.suptitle(f'Lorenz 63, long dataset (24 LT), {len(SEEDS)} reservoir seeds '
                 '-- cf. Racca & Magri (2021) Fig. 8')
    pdf.savefig(fig)
    plt.close(fig)


def make_boxplots(results, pdf):
    """Page 2: box plots (separate axes) of test log10(MSE) and prediction
    horizon per strategy, long dataset."""
    import matplotlib.pyplot as plt

    names = list(STRATEGIES)
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.2), layout='constrained')
    per_strategy = {
        'log10_mse': [[np.log10(r['m_test']) for r in results['long'][n]] for n in names],
        'horizon': [[r['horizon_mean'] for r in results['long'][n]] for n in names],
    }
    for ax, key, label in [(axs[0], 'log10_mse', r'test $\log_{10}$(MSE)'),
                           (axs[1], 'horizon', 'prediction horizon [LT]')]:
        vals = per_strategy[key]
        ax.boxplot(vals, tick_labels=names, whis=(0, 100))
        rng = np.random.default_rng(0)
        for k, v in enumerate(vals):  # individual ensemble members
            ax.plot(k + 1 + rng.uniform(-0.12, 0.12, size=len(v)), v,
                    'o', ms=4, color='tab:gray', alpha=0.6)
        ax.set_xlabel('validation strategy')
        ax.set_ylabel(label)
    fig.suptitle(f'Lorenz 63, long dataset, ensemble of {len(SEEDS)} reservoir '
                 f'seeds ({N_TEST_WINDOWS} test windows each)')
    pdf.savefig(fig)
    plt.close(fig)


# ______________________________________________________________ main


def build_tasks():
    """Generate the common trajectory and the full (dataset x strategy x seed)
    task list. Both datasets are prefixes of one trajectory; the test tail
    starts after the LONG dataset and is common to both."""
    n_wtv = {ds: sum(dataset_steps(cfg)[1:]) for ds, cfg in DATASETS.items()}
    n_wash_test = int(round(T_WASH_TEST_LT * LT / DT))
    n_fc = int(round(T_FORECAST_LT * LT / DT))
    spacing = int(round(TEST_SPACING_LT * LT / DT))
    starts = [w * spacing for w in range(N_TEST_WINDOWS)]
    n_test = starts[-1] + n_wash_test + 1 + n_fc

    data = lorenz_time_series(max(n_wtv.values()) + n_test)
    data_test = data[max(n_wtv.values()):]

    tasks = []
    for ds, cfg in DATASETS.items():
        data_wtv = data[:n_wtv[ds]]
        # normalization shared by every member of this dataset: component range
        # of the train+val window (the paper normalizes by the max variation)
        # and the time-averaged normalized state magnitude (prediction horizon)
        comp_range = data_wtv.max(axis=0) - data_wtv.min(axis=0)
        ph_norm = np.sqrt(np.mean(np.sum((data_test / comp_range) ** 2, axis=1)))
        for name in STRATEGIES:
            for seed in SEEDS:
                tasks.append(dict(dataset=ds, strategy=name, seed=int(seed),
                                  data_wtv=data_wtv, data_test=data_test,
                                  starts=starts, comp_range=comp_range,
                                  ph_norm=ph_norm))
    return tasks


def main():
    matplotlib.use('Agg')
    t_start = time.perf_counter()

    counts = {ds: {name: fold_count(name, cfg) for name in STRATEGIES}
              for ds, cfg in DATASETS.items()}
    print('Racca & Magri (2021) Table-1/Figure-8 protocol on Lorenz 63')
    print(f'ensemble: {len(SEEDS)} reservoir seeds | BHO: {N_GRID}x{N_GRID} grid '
          f'+ {N_FUNC_EVALS - N_GRID ** 2} gp-hedge points | N_units={N_UNITS}, '
          f'tikh={TIKH:g}')
    print('folds realized (short/long): ' + ', '.join(
        f'{name} {counts["short"][name]}/{counts["long"][name]}' for name in STRATEGIES))

    tasks = build_tasks()
    n_workers = min(len(tasks), os.cpu_count() - 4)
    print(f'{len(tasks)} member trainings on {n_workers} workers ...', flush=True)

    results = {ds: {name: [None] * len(SEEDS) for name in STRATEGIES} for ds in DATASETS}
    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context('spawn')) as ex:
        for r in ex.map(member_task, tasks, chunksize=1):
            results[r['dataset']][r['strategy']][np.flatnonzero(SEEDS == r['seed'])[0]] = r
            n_done += 1
            if n_done % 50 == 0 or n_done == len(tasks):
                print(f'  {n_done}/{len(tasks)} members done '
                      f'({time.perf_counter() - t_start:.0f} s)', flush=True)

    # ------------------------------------------------------------ r_S per cell
    r_s, nan_report = {}, []
    for ds in DATASETS:
        r_s[ds] = {}
        for name in STRATEGIES:
            m_val = np.array([r['m_val'] for r in results[ds][name]])
            m_test = np.array([r['m_test'] for r in results[ds][name]])
            if not (np.all(np.isfinite(m_val)) and np.all(np.isfinite(m_test))):
                nan_report.append((ds, name))
            r_s[ds][name] = float(spearmanr(m_val, m_test).statistic)

    table = rs_table(r_s)
    print('\n' + table)
    print('\nNaN check: ' + ('all m_Val/m_Test finite' if not nan_report else
                             f'NON-FINITE values in {nan_report}'))

    # ------------------------------------------------------------------ outputs
    npz = {'seeds': np.asarray(SEEDS), 'dt': DT, 'lyapunov_time': LT,
           'ph_threshold': PH_THRESHOLD, 'datasets': list(DATASETS),
           'strategies': list(STRATEGIES), 'table': table,
           'r_s': np.array([[r_s[ds][n] for n in STRATEGIES] for ds in DATASETS])}
    for ds in DATASETS:
        for name in STRATEGIES:
            for key in ['m_val', 'm_test', 'horizon_mean', 'rho', 'sigma_in',
                        'tikh', 'wall']:
                npz[f'{ds}_{name}_{key}'] = np.array([r[key] for r in results[ds][name]])
            for key in ['log10_mse_windows', 'horizon_windows']:
                npz[f'{ds}_{name}_{key}'] = np.stack([r[key] for r in results[ds][name]])
    np.savez(OUT_DIR / 'compare_validation_strategies.npz', **npz)

    with PdfPages(OUT_DIR / 'compare_validation_strategies.pdf') as pdf:
        make_figure8(results, r_s, pdf)
        make_boxplots(results, pdf)

    print(f'\nSaved {OUT_DIR / "compare_validation_strategies.npz"}')
    print(f'Saved {OUT_DIR / "compare_validation_strategies.pdf"}')
    print(f'Total runtime: {time.perf_counter() - t_start:.1f} s')


if __name__ == '__main__':
    main()
