"""EchoStateNetwork: a pure-numpy echo state network / reservoir computer.

Shared ESN implementation for qlrom, romda and related projects.
"""

import difflib
import inspect
import io
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, suppress
from copy import deepcopy

# Validation methods
from functools import cached_property, lru_cache, partial, partialmethod
from itertools import product

import matplotlib.backends.backend_pdf as plt_pdf
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, issparse, lil_matrix
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import eigs as sparse_eigs

from . import validation
from .verbose import _collected_warnings, _warn, _warnings_once

logger = logging.getLogger(__name__)

#: `Win_type` options; 'sparse_dense_bias' gives every neuron the bias column (see `EchoStateNetwork.Win`)
WIN_TYPES = ('sparse', 'sparse_dense_bias', 'dense')


def add_pdf_page(pdf, fig):
    """Save one matplotlib figure as a page in an open PdfPages, then close it."""
    pdf.savefig(fig)
    plt.close(fig)


def _train_one_seed(case, seed, train_data, add_noise, validation_strategy):
    """Worker for `EchoStateNetwork.train(n_seeds=...)` (module-level so it pickles
    into worker processes): regenerate the reservoir for this seed, train fully,
    and return ``(case, score, log, warnings)`` with one validation score comparable across seeds
    -- the minimum of the search when a search ran (``bo_results['fun']``; with 'tikh' alone, its one
    evaluation), otherwise one evaluation of the validation strategy at the fixed hyperparameters of the
    network, `tikh` included, on a sacrificial copy (the strategy overwrites `Wout`). The reports of the
    package (logger 'echostatenetwork', info and above) and any print are captured and returned instead of
    written: worker output interleaves across processes, and a forked Jupyter kernel's ZMQ stream is not
    fork-safe -- the parent logs them in order. The warnings of the package are collected as well
    (``(message, category)`` pairs, each once): the parent issues them, at the line of its caller and under its
    warnings filters, which the process of a seed cannot reach. When the seed fails, the error carries the warnings
    collected before it (`_SEED_WARNINGS`), which the parent issues before it raises the error."""
    log = io.StringIO()
    package = logging.getLogger('echostatenetwork')
    saved = package.handlers[:], package.level, package.propagate
    capture = logging.StreamHandler(log)
    capture.setFormatter(logging.Formatter('%(message)s'))
    package.handlers, package.propagate = [capture], False
    package.setLevel(logging.INFO)
    try:
        return _train_seed_captured(case, seed, train_data, add_noise, validation_strategy, log)
    finally:
        package.handlers, package.propagate = saved[0], saved[2]
        package.setLevel(saved[1])


def _train_seed_captured(case, seed, train_data, add_noise, validation_strategy, log):
    """Body of `_train_one_seed`, with the logs already captured into `log`."""
    with redirect_stdout(log), _collected_warnings() as caught:        # the parent issues them
        try:
            for attr in ('_W', '_Win', '_Wout'):
                if hasattr(case, attr):
                    delattr(case, attr)
            case.seed = seed
            case.train(train_data, add_noise=add_noise, plot_training=False,
                       validation_strategy=validation_strategy, seed=seed)
            if case.bo_results is not None:
                score = case.bo_results['fun']
            else:
                strategy = validation_strategy or validation.RVC_Noise
                # the data of the copy: the kept network keeps the norm that its Wout was fitted with, its generator
                # and its split_summary, in which an inferred t_val stays inferred
                probe = case.copy()
                U_wtv, Y_wtv = probe._split_and_format_data(train_data, add_noise=add_noise)[:2]
                # the score of the network that is kept: its fixed tikh, not the best value of tikh_range
                probe.val_k, probe.tikh_range = 0, [case.tikh]
                score = strategy([], probe, U_wtv, Y_wtv, np.zeros(1), [],
                                 print_convergence=False)
        except Exception as error:
            # a warning may explain the failure: the error carries the warnings to the parent (an exception pickles
            # its attributes), which issues them before it raises the error
            with suppress(AttributeError, TypeError):
                setattr(error, _SEED_WARNINGS, list(caught))
            raise
    return case, float(score), log.getvalue(), caught


#: attribute of the error of a failed seed of ``train(n_seeds > 1)`` that holds the warnings the seed collected
_SEED_WARNINGS = '_echostatenetwork_warnings'


def _pop_verbose(kwargs, where):
    """Remove the keyword 'verbose', which no longer exists, from `kwargs`, with a FutureWarning at the line of the
    caller (`where` names the call in the message)."""
    if kwargs.pop('verbose', None) is not None:
        _warn(f"{where}: the keyword 'verbose' is removed and has no effect; the reports of train() go to the logger "
              "'echostatenetwork', which echostatenetwork.set_verbose() shows.", FutureWarning)


#: a missing value (an attribute that is not set, a getter that is not called). Ellipsis, a builtin singleton: a
#: record that holds it (`_record_split`) pickles and copies as itself, also for the processes of ``n_seeds > 1``,
#: and loads in versions of the package without this name
_UNSET = Ellipsis


def _option_names(cls, network=None, extra=()) -> list:
    """The public names that a keyword can set: the attributes of `cls` and its properties with a setter (not its
    methods or read-only properties, which a keyword cannot set either), with `network` the public attributes of that
    network, and the names of `extra` (e.g. the parameters of `train`). They are the candidates of the hint of
    `_warn_ignored`."""
    names = {n for n in dir(cls) if not n.startswith('_') and _keyword_kind(cls, n) == 'option'}
    if network is not None:
        names.update(n for n in vars(network) if not n.startswith('_'))
    return sorted(names.union(extra))


@lru_cache(maxsize=4096)
def _class_kind(cls, key) -> tuple:
    """The kind of the attribute `key` of the class `cls`, for `_keyword_kind`, and the getter of a property with a
    setter (else None). The kind is 'absent', 'property' (a property with a setter), 'read-only', 'method' or
    'option'. Cached per class and name, because `inspect.getattr_static` costs microseconds and the constructor
    classifies every keyword: an attribute added to a class, or turned from a value into a method or a property,
    after a keyword of that name was classified keeps its first kind."""
    attr = inspect.getattr_static(cls, key, _UNSET)
    if attr is _UNSET:
        return 'absent', None
    if isinstance(attr, property):
        return ('read-only', None) if attr.fset is None else ('property', attr.fget)
    if isinstance(attr, (staticmethod, classmethod, partialmethod)) or \
            (callable(attr) and hasattr(type(attr), '__get__')):
        return 'method', None
    return 'option', None


def _keyword_kind(cls, key, network=None) -> str:
    """What a keyword of the constructor (`cls`) or of `train` (`cls` and `network`) names, without calling a getter
    that may fail (e.g. `N_train` before training, `patch_sites` without patches):

    - 'option': an attribute that the keyword sets (a class attribute, a property with a setter, or an attribute
      set on `network`);
    - 'unset': a property with a setter that `network` does not hold yet (its getter raises AttributeError, e.g.
      `Win` before training);
    - 'read-only': a property without a setter (e.g. `trained`, `sparsity`, `N_train`);
    - 'method': a method, which the keyword would replace (e.g. `copy`, `step`);
    - 'unknown': none of these.

    A method is a staticmethod, a classmethod, a partialmethod or a callable that binds to the instance, as in
    dynamodels; a callable value that does not bind (e.g. a `functools.partial`) is an option."""
    kind, getter = _class_kind(cls, key)
    if kind == 'absent':
        return 'option' if network is not None and key in vars(network) else 'unknown'
    if kind == 'property':
        if network is not None:
            try:
                getter(network)
            except AttributeError:
                return 'unset'
            except Exception:       # e.g. a ValueError: the setter decides
                pass
        return 'option'
    return kind


#: keywords that are not options of the network, with the option that the hint of `_warn_ignored` suggests and a
#: note on it: `dt` is a parameter of the constructor only, and `train` changes the time step with `dt_ESN`
_HINTS = {'dt': ('dt_ESN', 'train sets the time step of the data, dt, with dt_ESN = dt * upsample')}


def _hint(key, names):
    """The hint of `_warn_ignored` for the unknown keyword `key`, among the option names `names`: `_HINTS`, else
    the closest name, ``(name, note)``, or None. The cutoff 0.8 keeps typos ('N_unit', 'rho_rnage') and drops loose
    matches: the default cutoff, 0.6, suggests 'plot_Wout' for 'plot'."""
    if key in _HINTS and _HINTS[key][0] in names:
        return _HINTS[key]
    match = difflib.get_close_matches(key, names, n=1, cutoff=0.8)
    return (match[0], None) if match else None


def _warn_ignored(where, kinds, owner, cls, network=None, extra=(), independent=False):
    """Warn once, at the line of the caller, that keywords are ignored. `kinds` maps a kind of `_keyword_kind` to
    the keywords of that kind: 'unknown' keywords are not attributes of `owner` (each with a hint from
    `_option_names(cls, network, extra)`, see `_hint`, e.g. 'rho_range' for 'rho_rnage', 'dt_ESN' for 'dt'),
    'method' keywords name a method, 'read-only' keywords a property that cannot be set, 'computed' keywords a
    result of `train` (`Wout`), and 'unset' keywords an attribute that the network does not hold yet (e.g. `Win`
    before training; with `independent` patches, the network of every patch holds its own). Nothing when no keyword
    is ignored."""
    parts = []
    if unknown := sorted(kinds.get('unknown', ())):
        names = _option_names(cls, network, extra)   # only when a warning is due: dir() of the class
        hints = [(key, hint) for key in unknown if (hint := _hint(key, names))]
        notes = ''.join(f' {note}.' for _, (_, note) in hints if note)
        parts.append(f'keyword(s) {unknown} are not attributes of {owner} and are ignored'
                     + (f" (did you mean {', '.join(f'{name!r} for {key!r}' for key, (name, _) in hints)}?{notes})"
                        if hints else ''))
    if methods := sorted(kinds.get('method', ())):
        parts.append(f'keyword(s) {methods} name methods of {owner}, not options, and are ignored')
    if read_only := sorted(kinds.get('read-only', ())):
        parts.append(f'keyword(s) {read_only} are read-only properties of {owner} and are ignored')
    if computed := sorted(kinds.get('computed', ())):
        parts.append(f'keyword(s) {computed} are computed by train and are ignored')
    if unset := sorted(kinds.get('unset', ())):
        # train keeps a reservoir set on the network by hand, but computes Wout again
        reservoir = sorted({'W', 'Win'}.intersection(unset))
        if independent and reservoir:
            advice = ' (with independent patches, the network of every patch holds its own, in patches[p])'
        elif reservoir:
            advice = f" (set {' and '.join(reservoir)} on the network before calling train, which keeps them)"
        else:
            advice = ''
        parts.append(f'keyword(s) {unset} are attributes that the network does not hold yet and are ignored{advice}')
    if parts:
        _warn(f"{where}: {'; '.join(parts)}")


def _same(a, b):
    """Whether two option values are equal (arrays, lists and scalars): for the log of `train`, and for what changed
    since the previous call (the inferred split, the options of independent patches)."""
    try:
        return bool(np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object)))
    except (TypeError, ValueError):
        return a is b


#: options that the network of an independent patch takes neither from the parallel network (`_init_patches`) nor
#: from the keywords of `train` (`_train_patches`): the layout, the results of the last search, and the validation
#: metric, which the network of a patch wraps to score the sites of its patch only (`_patch_metric`)
_PATCH_EXCLUDED = frozenset({'patch_size', 'halo', 'periodic', 'shared', 'patches', 'observed_idx',
                             'input_parameters', 'readout_input', 'bo_results', 'split_summary',
                             'validation_metric'})
#: attributes that the split of `train` may infer from its data, which the next call of `train` infers again from
#: its own data (`_restore_split`): the split, and the names of the search over the normalization of the parameters.
#: Not `param_shift_range` and `param_norm_range`: sized from the parameters of the first call, they are kept, so that
#: a call on data in which a parameter takes fewer values (e.g. a cluster id of qlroms) searches the same range
_INFERRED = ('t_train', 't_val', 'hyperparameters_to_optimize')
#: keywords of `train` that `_set_options` sets after the others, in this order
_SET_LAST = ('Win', 'W', 'dt_ESN')
#: keywords of `train` after which the layout is checked again, as by the constructor (`_check_layout`)
_LAYOUT_OPTIONS = frozenset({'patch_size', 'halo', 'periodic', 'shared', 'observed_idx', 'input_parameters',
                             'readout_input'})


def _patch_metric(case, Y_true, Y_pred, norm=1.0, metric=None, sites=None):
    """Validation metric of the network of an independent patch: `metric` (default
    `validation.log_nMAE`) on the columns `sites` of the patch within its window, the
    only outputs that the parallel network keeps. Module-level, so that it pickles
    into the worker processes of ``train(n_seeds=...)``."""
    metric = metric or validation.log_nMAE
    norm = norm[sites] if np.ndim(norm) else norm
    return metric(case, Y_true[:, sites], Y_pred[:, sites], norm)

# XDG_RUNTIME_DIR = 'tmp/'


class EchoStateNetwork:
    """
    The EchoStateNetwork class implements a reservoir computing model for time series prediction. It is based on
    the Echo State Network (ESN) approach, which uses a randomly connected reservoir of neurons to map inputs
    into high-dimensional space. This allows the model to capture complex dynamics with efficient training.

    The attributes of a network group into:

    - reservoir and network hyperparameters (e.g. N_units, rho, sigma_in, tikh, and leak_rate, the rate of the
      leaky integrator, with 1.0 for no leak);
    - the configuration of training, validation and test (e.g. t_train, t_val, t_test, N_wash);
    - the settings of the hyperparameter search (e.g. hyperparameters_to_optimize, rho_range);
    - the output of the search of the last train() call (bo_results; None when no search ran, see train);
    - the input and output matrices (Win, Wout) and the reservoir matrix (W);
    - the parallel layout (patch_size, halo, periodic, shared): one reservoir per patch of sites, see below.

    Parallel layout:
        With ``patch_size=G``, the ``N_dim`` sites are split into
        ``N_patches = N_dim / G`` patches of ``G`` neighbouring sites. Each patch has
        a reservoir of ``N_units`` units, which reads the ``G + 2H`` sites of its
        window (the patch and a halo of ``H = halo`` sites on either side) and
        predicts the ``G`` sites of its patch (Pathak et al. 2018). By default
        (``shared=True``), the patches share `Win`, `W`, `Wout` and the
        hyperparameters, as a convolution does, so the layout suits systems that
        are invariant under a shift of the sites. The reservoirs still predict
        different values, because each reads a different window, and they are
        coupled through the overlapping halos. With ``shared=False``, every patch
        has its own network (`patches`): its own random `Win` and `W`, readout,
        hyperparameters and normalization, trained and selected on the record of its
        own window. The reservoir state stacks the patches,
        ``r = [r_0; ...; r_{P-1}]`` with ``N_r = N_patches * N_units`` rows, and
        every row has a position (`state_positions`), which covariance localization
        in data assimilation requires.

    References:
        Based on https://github.com/alberacca/Echo-State-Networks, which implements
        Racca & Magri (2021). Robust optimization and validation of echo state
        networks for learning chaotic dynamics. Neural Networks, 142, 252-268
        (arXiv:2103.03174).

        Parallel layout: Pathak, Hunt, Girvan, Lu & Ott (2018). Model-free
        prediction of large spatiotemporally chaotic systems from data: a reservoir
        computing approach. Physical Review Letters, 120, 024102.
    """

    bias_in = np.array([0.1])
    bias_in_range = (0.0, 2.0)   # searched when 'bias_in' is in hyperparameters_to_optimize
    bias_out = np.array([1.0])
    regularize_bias_out = True   # False: the Tikhonov factor leaves the bias row of the readout out

    # Slim copy of the last skopt result; None before training or when no search ran (hyperparameters_to_optimize
    # empty, and no search over the normalization of the parameters, see train()). With 'tikh' alone, the one
    # evaluation of the validation strategy (hp_names = []).
    bo_results: dict | None = None

    # Split summary from the last train() call; source for training_summary().
    split_summary: dict | None = None

    connect = 3  # neuron connectivity
    figs_folder = './figs_ESN/'
    filename = 'my_ESN'

    # Parametric input: (N_param, L) for constant per-segment parameters,
    # or list of (Nt_l, N_param) arrays when the parameter varies within a segment.
    input_parameters: np.ndarray | None = None

    # Readout on [r; u; bias] instead of [r; bias]: a direct linear input-to-output
    # term (u = the normalised ESN input, parameter columns included), the common
    # "input skip" ESN variant. Wout then has N_units + N_dim_in + 1 rows.
    # 'params': the readout reads only the parameter rows, Wout on [r; p; bias].
    readout_input = False
    # False: Win's parameter columns are zero, so the parameter reaches the output only
    # through the readout (with readout_input='params'), never the reservoir.
    param_in_reservoir = True

    # Parallel layout (Pathak et al. 2018): one reservoir of N_units units per patch of
    # `patch_size` neighbouring sites, which reads its patch and `halo` sites on either
    # side and predicts its patch. None: one reservoir for the whole state.
    # `periodic`: the sites lie on a ring; False pads the windows that cross the domain
    # boundary with zeros (see `patch_sites`). `shared`: the patches share Win, W, Wout
    # and the hyperparameters; False gives every patch its own network in `patches`.
    patch_size: int | None = None
    halo = 0
    periodic = True
    shared = True
    # shared=False: one EchoStateNetwork per patch, built by train()/_generate_W_Win
    patches: list | None = None

    N_folds = 4
    val_fold_step = None
    # optional probe metric shared by every validation strategy:
    # metric(case, Y_true, Y_pred, norm) -> float; None = each strategy's default
    # (validation.log_nMAE for the recycle family). See validation.py.
    validation_metric = None
    # optional guardrail shared by the closed-loop validation strategies: weight of the
    # saturation penalty (0 = off). See qlroms' segment strategies for the rationale --
    # a saturated reservoir scores well over a short probe and fails over a long forecast.
    saturation_penalty = 0.0
    # stability gate of the recycle validation (RVC_Noise): with stability_steps > 0, every
    # Tikhonov candidate also runs closed loops of that many steps, and its fold-mean score
    # rises by stability_penalty when a loop leaves the attractor (non-finite, or beyond 3x
    # the largest |y| of the data) or its standard deviation (the RMS over the outputs) leaves
    # stability_band times that of its record. The short folds score the accuracy, the long loop
    # the stability (two-stage validation, as in Lu et al. 2018; Platt et al. 2022).
    # stability_starts: the number of washouts, spread over the training record, from which the loop runs; the
    # candidate fails if any loop fails (1: the start of the first segment only)
    # stability_tail: the band must also hold on the last stability_tail of the loop (a fraction; 0 tests the
    # whole loop only), because a loop that follows the attractor for most of its length and collapses near the
    # end keeps the standard deviation of the whole loop in the band
    # stability_penalty: the score of a diverged probe of log_nMAE (10), so that a candidate that fails the gate
    # loses to every candidate that passes it, as a diverged probe does; at 1.0 (one decade), a failing
    # candidate still won with a short-fold score better by more than one decade
    stability_steps = 0
    stability_starts = 1
    stability_band = (0.8, 1.2)
    stability_tail = 0.5
    stability_penalty = 10.0
    N_func_evals = 20
    N_grid = 4
    # Bayesian optimization: bho_noise is the GP noise ('gaussian' learns a white-noise level,
    # which suits the noisy validation score); grid_midpoints puts the initial grid at the
    # midpoints of N_grid equal cells, so that no initial point lies on the boundary
    bho_noise = 1e-10
    grid_midpoints = False
    N_initial_rand = 0
    N_units = 100
    N_wash = 50

    max_L_tests = 10
    max_short_tests = 10
    perform_test = True

    observed_idx = None

    # t_train/t_val can be inferred from the data inside train().
    t_val = None
    t_train = None
    t_test = 0.5
    upsample = 5
    Win_type = 'sparse'
    norm_method = 'range'

    # Default hyperparameters and optimization ranges.
    noise = 1e-10
    noise_type = 'gauss'
    hyperparameters_to_optimize = ['rho', 'sigma_in', 'tikh']
    rho = 0.9
    rho_range = (.8, 1.05)
    sigma_in = 10 ** -3
    sigma_in_range = (-5, -1)
    # sigma_param: the input scaling of the input_parameters columns of Win, searched (log10) over sigma_param_range
    # like sigma_in when listed in hyperparameters_to_optimize; None scales them with sigma_in
    sigma_param = None
    sigma_param_range = (-3, 0.477)
    tikh = 1e-12
    tikh_range = [1e-8, 1e-10, 1e-12, 1e-16]
    leak_rate = 1.0
    leak_rate_range = (0.1, 1.0)

    # Dense parameter columns can swamp the reservoir if the raw values are not rescaled.
    # optimize_parameter_normalization: True searches the shift and scale of every parameter column with the other
    # hyperparameters (param_shift_range, param_norm_range); 'data' normalizes them like the state (norm_method over
    # the training values), before the search; False leaves them raw.
    optimize_parameter_normalization = False
    param_shift_range = (-1.0, 1.0)
    param_norm_range = (0.1, 10.0)

    def __init__(self, y, dt=None, **kwargs):
        """Initialize the reservoir dimensions and time step; matrices are built at
        training time (see `train` / `_generate_W_Win`), not here.

        Parameters
        ----------
        y : np.ndarray
            Sample physical state used only to infer `N_dim`, shape ``(N_dim, N_samples)``
            (or 1D, treated as ``N_samples=1``).
        dt : float, optional
            Time step of the input data (default 1), such that ``dt_ESN = dt * upsample`` (`dt_ESN` follows a
            later change of `upsample`).
        **kwargs
            Any `EchoStateNetwork` class attribute to override (e.g. ``N_units``,
            ``rho``, ``observed_idx``, ``input_parameters``, ``optimize_parameter_normalization``,
            or ``patch_size``, ``halo``, ``periodic`` and ``shared`` for the parallel layout).
            The keyword ``dt_ESN``, the time step of the network, sets ``dt = dt_ESN / upsample``, with a
            UserWarning when `dt` is also given and disagrees. A keyword that is not a class attribute is
            ignored with a UserWarning, which names it and, for a typo, suggests the closest attribute (e.g.
            ``N_units`` for ``N_unit``); so is a keyword that names a method or a read-only property. The
            removed keyword ``verbose`` gives a FutureWarning.

        Raises
        ------
        AssertionError
            If `y` has more than two dimensions.
        ValueError
            If the parallel layout cannot work: `patch_size` does not divide `N_dim`,
            the window of a patch exceeds a periodic domain, `patch_size` is combined
            with `readout_input` or partial observability (`observed_idx`),
            ``shared=False`` with `input_parameters`, or ``shared=False`` is set
            without `patch_size`.
        """

        if y.ndim == 1:
            y = y[:, np.newaxis]
        elif y.ndim > 2:
            raise AssertionError(f'y.shape={y.shape}. The input y must have 2 dimension')


        #   Initialise state dimensions and reservoir state to zeros ------------ #
        self.N_dim = y.shape[0] # Dimension of the physical system i.e., the output dimension
        self.observed_idx = kwargs.pop('observed_idx', np.arange(self.N_dim)) # Default to full observability

        # Set provided input parameters ------------------------- #
        _pop_verbose(kwargs, type(self).__name__)
        dt_ESN = kwargs.pop('dt_ESN', None)         # sets the time step of the data below, once upsample is set
        ignored = {}
        for key in list(kwargs):
            kind = _keyword_kind(EchoStateNetwork, key)
            if kind == 'option':
                setattr(self, key, kwargs.pop(key))
            else:                                   # not an option of the class: never set
                ignored.setdefault(kind, []).append(key)
        _warn_ignored(type(self).__name__, ignored, 'EchoStateNetwork', EchoStateNetwork)

        # Default to rescaling parametric inputs unless the user explicitly disables it.
        # Dense parameter columns can otherwise dominate the reservoir dynamics.
        if self.input_parameters is not None and 'optimize_parameter_normalization' not in vars(self):
            self.optimize_parameter_normalization = True

        self._check_layout()

        # Time step of the data; the network steps at dt_ESN = dt * upsample, which follows upsample
        self._dt_data = 1. if dt is None else dt
        if dt_ESN is not None:
            if dt is not None and not np.isclose(dt * self.upsample, dt_ESN):
                _warn(f'{type(self).__name__}: dt_ESN={dt_ESN} and dt={dt} disagree (dt * upsample = '
                      f'{dt * self.upsample}); the time step of the data is dt_ESN / upsample = '
                      f'{dt_ESN / self.upsample}.')
            self.dt_ESN = dt_ESN

        # Initialize ESN matrices.
        self.val_k = kwargs.get('val_k', 0)
        self.initialised = False

    @property
    def _n_readout(self) -> int:
        """int: rows of `Wout` -- reservoir units, the input rows `readout_input` reads, and the bias."""
        rows = self._readout_rows
        return self.N_units + 1 + rows.stop - rows.start

    @property
    def _readout_rows(self) -> slice:
        """slice: rows of the normalised input the readout reads -- all with
        `readout_input` True, the parameter rows with 'params', none when False."""
        if isinstance(self.readout_input, str):
            if self.readout_input != 'params' or self.input_parameters is None:
                raise ValueError(f"readout_input={self.readout_input!r}: only 'params' is "
                                 "accepted, and it needs input_parameters")
            return slice(self.N_dim_in - self._n_param(self.input_parameters), self.N_dim_in)
        return slice(0, self.N_dim_in if self.readout_input else 0)

    @property
    def trained(self):
        """Flag to check if the model has been trained (with independent patches,
        every network of `patches`)"""
        if self._independent:
            return self.patches is not None and all(net.trained for net in self.patches)
        return hasattr(self, '_Win') and hasattr(self, '_Wout') and hasattr(self, '_W')

    @property
    def _independent(self) -> bool:
        """bool: parallel layout with independent patches (``shared=False``)."""
        return self.patch_size is not None and not self.shared

    def _per_patch_error(self, name):
        """AttributeError for a matrix that independent patches hold per patch."""
        return AttributeError(f'with independent patches (shared=False), every patch has its '
                              f'own {name}: see patches[p].{name}')

    @property
    def W(self) -> csr_matrix:
        """The reservoir (recurrent) connectivity matrix, shape ``(N_units, N_units)``,
        stored in CSR format. Rescaled to unit spectral radius when generated (see
        `_generate_W_Win`), so `rho` is the *effective* spectral radius used in `step`.
        With independent patches (``shared=False``), every patch has its own `W`, in
        ``patches[p].W``.
        """
        if self._independent:
            raise self._per_patch_error('W')
        return self._W


    @property
    def rng(self):
        if not hasattr(self, '_rng'):
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    @property
    def seed(self):
        if not hasattr(self, '_seed'):
            self._seed = 0
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value
        if hasattr(self, '_rng'):
            del self._rng

    @W.setter
    def W(self, value):
        """
        Setter for the reservoir state matrix (W). Converts the input to CSR format.
        """
        if self._independent:
            raise self._per_patch_error('W')
        if not isinstance(value, csr_matrix):
            value = csr_matrix(value)

        # Ensure the matrix is square and has the correct dimensions
        assert value.shape == (self.N_units, self.N_units), \
            f'W must be a square matrix of shape ({self.N_units}, {self.N_units}), but got {value.shape}'

        # Set the reservoir state matrix; a matrix set by hand has no record of the options that drew it
        self._W = value
        self.__dict__.pop('_reservoir_key', None)

    @property
    def Win(self) -> np.ndarray | csr_matrix:
        """The input matrix, shape ``(N_units, N_dim_in + 1)`` (the last column
        multiplies the input bias `bias_in`). Sparse (``Win_type='sparse'``, one
        random connection per neuron to a state or bias column, but densely
        connected to any `input_parameters` columns), sparse with a dense bias
        column (``Win_type='sparse_dense_bias'``: one state column per neuron, and
        every neuron sees the bias, which breaks the odd symmetry of the tanh
        reservoir in all units; Herteux & Räth 2020) or dense
        (``Win_type='dense'``); see `_generate_W_Win`. With the parallel layout
        (`patch_size`), the input matrix that the patches share, shape
        ``(N_units, patch_size + 2 * halo + 1)``: its columns are the sites of a
        window (`patch_sites`) and the bias. With independent patches
        (``shared=False``), every patch has its own `Win`, in ``patches[p].Win``.
        """
        if self._independent:
            raise self._per_patch_error('Win')
        return self._Win

    @Win.setter
    def Win(self, value):
        """
        Setter for the input matrix (Win). Converts the input to CSR format if sparse.
        """
        if self._independent:
            raise self._per_patch_error('Win')

        assert self.Win_type in WIN_TYPES, f"Win type {self.Win_type} not implemented {WIN_TYPES}"

        if self.Win_type != 'dense' and not isinstance(value, csr_matrix):
            value = csr_matrix(value)
        elif self.Win_type == 'dense' and hasattr(value, 'toarray'):
            value = value.toarray()


        # Ensure the matrix has the correct dimensions
        assert value.shape ==  (self.N_units, self._n_in + 1), \
            f'Win must be a matrix of shape ({self.N_units}, {self._n_in + 1}), but got {value.shape}'

        # Set the input matrix; a matrix set by hand has no record of the options that drew it
        self._Win = value
        self.__dict__.pop('_reservoir_key', None)
        self._invalidate_jacobian_cache()


    def _invalidate_jacobian_cache(self):
        self.__dict__.pop('dr_di', None)


    @property
    def Wout(self) -> np.ndarray:
        """The trained (ridge-regression) read-out matrix, shape
        ``(N_units + 1, N_dim)``, or ``(N_units + N_dim_in + 1, N_dim)`` with
        `readout_input` (rows [r; u; bias]) -- the last row multiplies the output bias
        `bias_out`. Used in `reservoir_to_physical`. With the parallel layout
        (`patch_size`), the readout that the patches share, shape
        ``(N_units + 1, patch_size)``: it maps the reservoir of a patch to the sites
        of that patch. With independent patches (``shared=False``), every patch has
        its own readout, in ``patches[p].Wout``.
        """
        if self._independent:
            raise self._per_patch_error('Wout')
        return self._Wout

    @Wout.setter
    def Wout(self, value: np.ndarray):
        """
        Setter for the reservoir state matrix (W).
        """
        if self._independent:
            raise self._per_patch_error('Wout')
        # Ensure the matrix has the correct dimensions
        assert value.shape == (self._n_readout, self._n_out), \
            f'Wout must be a matrix of shape ({self._n_readout}, {self._n_out}), but got {value.shape}'
        # Set the output matrix
        self._Wout = value

    @property
    def val_k(self):
        """int: Number of Bayesian-optimization validation evaluations performed so
        far (reset to 0 at the start of each `train` call). Defaults to 0.
        """
        if not hasattr(self, '_val_k'):
            return 0
        return self._val_k

    @val_k.setter
    def val_k(self, value):
        """
        Setter for the validation counter.
        """
        if not isinstance(value, int):
            raise TypeError('val_k must be an integer')
        self._val_k = value

    @property
    def dt_physical(self):
        """float: Time step of the underlying physical data, the `dt` of the constructor (``dt_ESN / upsample``)."""
        return self._data_time_step()

    @property
    def dt_ESN(self):
        """float: Time step of the network, ``dt * upsample``, with `dt` the time step of the data (`dt_physical`).
        It follows `upsample`: ``train(upsample=k)``, or ``esn.upsample = k``, changes `dt_ESN` and leaves `dt`.
        Setting `dt_ESN` sets the time step of the data to ``dt_ESN / upsample``."""
        return self._data_time_step() * self.upsample

    @dt_ESN.setter
    def dt_ESN(self, value):
        self._dt_data = value / self.upsample

    def _data_time_step(self):
        """The time step of the data, `_dt_data`. A network pickled while `dt_ESN` was stored, before it followed
        `upsample`, holds `dt_ESN` instead, which the property hides: it becomes ``_dt_data = dt_ESN / upsample``
        here, on the first use, also when a subclass unpickles the network with a `__setstate__` of its own that
        does not call this class's (e.g. romda's `LatentROM`)."""
        state = self.__dict__
        if '_dt_data' not in state and 'dt_ESN' in state:
            self._dt_data = state.pop('dt_ESN') / self.upsample
        try:
            return state['_dt_data']
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '_dt_data'") from None

    def __getstate__(self):
        """Pickle (also `copy`): the attributes of the network, and `dt_ESN`, which echostatenetwork 0.2.3 and
        earlier store and read, so that they load a network pickled by this version."""
        state = self.__dict__.copy()
        if '_dt_data' in state:
            state['dt_ESN'] = self.dt_ESN
        return state

    def __setstate__(self, state):
        """Unpickle (also `copy`). A network pickled while `dt_ESN` was stored, before it followed `upsample`,
        keeps its `dt_ESN`; for a network pickled by this version, `dt_ESN` is only the copy that `__getstate__`
        stores for older versions."""
        self.__dict__.update(state)
        legacy = self.__dict__.pop('dt_ESN', None)
        if legacy is not None and '_dt_data' not in self.__dict__:
            self.dt_ESN = legacy

    @property
    def N_train(self):
        """int: Number of training steps, ``round(t_train / dt_ESN)``.

        Raises ValueError if `t_train` is not yet set (when omitted it is
        inferred from the data at `train` time -- see `_split_and_format_data`).
        """
        if self.t_train is None:
            raise ValueError("t_train is not set; it is inferred from the data at train() "
                             "time when omitted (see _split_and_format_data).")
        return int(round(self.t_train / self.dt_ESN))

    @property
    def N_val(self):
        """int: Number of validation steps, ``round(t_val / dt_ESN)``.

        Raises ValueError if `t_val` is not yet set (when omitted it is
        inferred from the data at `train` time -- see `_split_and_format_data`).
        """
        if self.t_val is None:
            raise ValueError("t_val is not set; it is inferred from the data at train() "
                             "time when omitted (see _split_and_format_data).")
        return int(round(self.t_val / self.dt_ESN))

    @property
    def N_test(self):
        """int: Number of test steps, ``round(t_test / dt_ESN)``."""
        return int(round(self.t_test / self.dt_ESN))

    @property
    def WCout(self):
        r"""Least-squares fit of `W` from `Wout`'s state block,
        $\arg\min_{\mathbf{X}} \lVert \mathbf{W}_\mathrm{out}[:N_\mathrm{units}]\,\mathbf{X}
        - \mathbf{W} \rVert_F$, shape ``(N_dim, N_units)``. Sketched (commented-out) in
        `Jacobian` as a building block for a closed-loop Jacobian; not currently used
        anywhere, since the closed-loop Jacobian is unimplemented.
        """
        # if not hasattr(self, '_WCout'):
        #     return None
        return self._WCout

    @WCout.setter
    def WCout(self, value=None):
        """
        Setter for the closed-loop reservoir weight matrix (W Cout).
        """
        assert self.trained, 'ESN must be trained with washout before calling step method. Call ESN.train() first.'
        if value is None:
            self._WCout = np.linalg.lstsq(self.Wout[:self.N_units], self.W.toarray(), rcond=None)[0]
        else:
            self._WCout = value


    @property
    def sparsity(self):
        r"""float: Fraction of possible reservoir connections that are zero,
        $1 - \texttt{connect}/(N_\mathrm{units}-1)$ (each neuron connects, on
        average, to `connect` others out of the $N_\mathrm{units}-1$ possible).
        """
        return 1. - self.connect / (self.N_units - 1)


    @staticmethod
    def _n_param(input_parameters):
        """Number of parameter columns, whether input_parameters is a constant-per-segment
        ndarray (N_param, L) or a list of L per-timestep (Nt_l, N_param) arrays."""
        return input_parameters[0].shape[-1] if isinstance(input_parameters, list) else input_parameters.shape[0]

    @staticmethod
    def _param_range(input_parameters):
        """(min, max) of input_parameters, whether ndarray or list-of-arrays (see _n_param)."""
        if isinstance(input_parameters, list):
            return (min(np.asarray(p).min() for p in input_parameters),
                    max(np.asarray(p).max() for p in input_parameters))
        return float(input_parameters.min()), float(input_parameters.max())

    @property
    def N_dim_in(self):
        """int: Number of ESN input dimensions -- the observed state components
        (``len(observed_idx)``) plus, for a parametric ESN, the parameter count of `input_parameters`
        (rows of the ``(N_param, L)`` array, or columns of each per-timestep
        ``(Nt_l, N_param)`` segment).
        """
        if self.input_parameters is None:
            return len(self.observed_idx)
        return len(self.observed_idx) + self._n_param(self.input_parameters)

    # _______________________________________________________________________________________ PARALLEL LAYOUT
    @property
    def N_patches(self) -> int:
        """int: Number of patches, ``N_dim // patch_size``, each with its own
        reservoir; 1 without the parallel layout (``patch_size=None``)."""
        return 1 if self.patch_size is None else self.N_dim // self.patch_size

    @property
    def N_r(self) -> int:
        """int: Number of rows of the reservoir state `r` that `step` advances,
        ``N_patches * N_units``: the reservoirs of the patches stacked in patch
        order, ``r = [r_0; ...; r_{P-1}]``. Equal to `N_units` for a single reservoir."""
        return self.N_patches * self.N_units

    @property
    def patch_sites(self) -> np.ndarray:
        """np.ndarray: Sites that each reservoir reads, shape
        ``(N_patches, patch_size + 2 * halo)``. Patch ``p`` predicts the sites
        ``p * patch_size, ..., (p + 1) * patch_size - 1`` and reads them together
        with `halo` sites on either side. On a periodic domain (`periodic`), the
        windows wrap around the ring. On a non-periodic domain, the windows of the
        patches at the boundaries extend outside the domain; -1 marks these
        positions, which read zero, i.e. the training mean of the data in the
        normalized units of `normalize_input` (of the network of the patch, with
        independent patches).

        Raises
        ------
        ValueError
            Without the parallel layout (``patch_size=None``).
        """
        if self.patch_size is None:
            raise ValueError('patch_sites requires the parallel layout: set patch_size.')
        sites = (np.arange(self.N_patches)[:, np.newaxis] * self.patch_size
                 + np.arange(-self.halo, self.patch_size + self.halo)[np.newaxis, :])
        if self.periodic:
            return sites % self.N_dim
        return np.where((sites >= 0) & (sites < self.N_dim), sites, -1)

    @property
    def state_positions(self) -> np.ndarray:
        """np.ndarray: Position, in units of sites, of every row of the state
        ``[u; r]`` with ``u`` the ``N_dim`` outputs and ``r`` the ``N_r`` reservoir
        rows, shape ``(N_dim + N_r,)``. Output ``i`` lies at site ``i`` and every
        row of the reservoir of patch ``p`` lies at the centre of the patch,
        ``p * patch_size + (patch_size - 1) / 2``. These are the positions that
        covariance localization in data assimilation requires (e.g. the
        ``state_positions`` of a local ensemble transform Kalman filter); on a
        periodic domain the period is ``N_dim`` sites. The reservoir rows alone are
        ``state_positions[N_dim:]``.

        Raises
        ------
        ValueError
            Without the parallel layout: a single reservoir reads every site, so
            its rows have no position.
        """
        if self.patch_size is None:
            raise ValueError('a single reservoir reads every site, so its rows have no position: '
                             'set patch_size for the parallel layout.')
        centres = np.arange(self.N_patches) * self.patch_size + (self.patch_size - 1) / 2
        return np.concatenate([np.arange(self.N_dim, dtype=float), np.repeat(centres, self.N_units)])

    @property
    def _n_in(self) -> int:
        """int: columns of `Win` before the bias column -- `N_dim_in`, or the window
        width ``patch_size + 2 * halo`` with the parallel layout."""
        if self.patch_size is None:
            return self.N_dim_in
        n_param = 0 if self.input_parameters is None else self._n_param(self.input_parameters)
        return self.patch_size + 2 * self.halo + n_param   # the window, then the parameters

    @property
    def _n_out(self) -> int:
        """int: columns of `Wout` -- `N_dim`, or `patch_size` with the parallel layout."""
        return self.N_dim if self.patch_size is None else self.patch_size

    def _check_layout(self):
        """Raise a ValueError if the layout cannot work with the other options: see `_check_patch_layout`, and
        ``shared=False`` requires `patch_size`. The constructor checks it, and `train` again after a keyword of the
        layout (`_set_options`)."""
        if self.patch_size is not None:
            self._check_patch_layout()
        elif not self.shared:
            raise ValueError('shared=False requires the parallel layout: set patch_size.')

    def _check_patch_layout(self):
        """Raise a ValueError if the parallel layout (`patch_size`, `halo`,
        `periodic`) cannot work with the other options."""
        G, H, N = self.patch_size, self.halo, self.N_dim
        if not isinstance(G, (int, np.integer)) or G < 1 or N % G:
            raise ValueError(f'patch_size={G!r} must be a positive integer that divides the '
                             f'N_dim={N} sites into patches of equal size.')
        if not isinstance(H, (int, np.integer)) or H < 0:
            raise ValueError(f'halo={H!r} must be a non-negative integer.')
        if self.periodic and G + 2 * H > N:
            raise ValueError(f'on a periodic domain, the window of a patch (patch_size + 2 * halo '
                             f'= {G + 2 * H} sites) must not exceed the N_dim={N} sites.')
        if self.input_parameters is not None and not self.shared:
            raise ValueError('input_parameters is not implemented with independent patches (shared=False).')
        if self.readout_input:
            raise ValueError('readout_input is not implemented with the parallel layout (patch_size).')
        if not np.array_equal(self.observed_idx, np.arange(N)):
            raise ValueError('with the parallel layout (patch_size), each reservoir reads its window '
                             'from the full state: observed_idx must list all the N_dim sites in order.')

    def _to_columns(self, x):
        """``(N_patches, n, N_ens)`` -> ``(n, N_patches * N_ens)``: the patches become
        columns of the shared reservoir, column ``p * N_ens + j`` for patch ``p``
        of member ``j``."""
        return x.transpose(1, 0, 2).reshape(x.shape[1], -1)

    def _from_columns(self, x):
        """Inverse of `_to_columns`, stacked: ``(n, N_patches * N_ens)`` ->
        ``(N_patches * n, N_ens)``, rows ``p * n, ..., (p + 1) * n - 1`` for patch ``p``."""
        n = x.shape[0]
        return x.reshape(n, self.N_patches, -1).transpose(1, 0, 2).reshape(self.N_patches * n, -1)

    # ---------------------------------------------------------------- independent patches
    def _init_patches(self, seed=None):
        """Build `patches` for ``shared=False``: one untrained `EchoStateNetwork` per
        patch, whose state is the window of the patch (``patch_size + 2 * halo``
        sites). Each network copies the options of this one (reservoir size,
        hyperparameters and their search ranges, validation settings, ...) and takes
        its own seed, spawned from `seed` (default `self.seed`), so that the patches
        are independent random realizations. Its validation probes feed back only the
        sites of the patch and take the halo from the data, and its validation metric
        scores only the sites of the patch (see `_closed_loop_input`, `_patch_metric`)."""
        options = self._patch_options()
        # what the networks took from this network: train() passes them a value changed since (_train_patches)
        self._patch_options_seen = deepcopy(options)
        options = deepcopy(options)
        sites = slice(self.halo, self.halo + self.patch_size)
        root = self.seed if seed is None else seed
        seeds = np.random.SeedSequence(root).generate_state(self.N_patches)
        # what defines the networks of the patches: train() builds them again when it changes (_stale_patches)
        self._patches_key = dict(patch_size=self.patch_size, halo=self.halo, seed=root)
        self.patches = []
        for p, seed_p in enumerate(seeds):
            net = EchoStateNetwork(np.zeros((self._n_in, 1)), dt=self.dt_physical, **options)
            net.seed = int(seed_p)
            net._patch = p                      # the parallel network reports the summary of every patch
            net._forecast_sites = sites
            net.validation_metric = partial(_patch_metric, metric=self.validation_metric, sites=sites)
            self.patches.append(net)

    def _patch_options(self) -> dict:
        """The options of this network that the network of every independent patch takes: the public class
        attributes of `EchoStateNetwork` that are values (not methods or properties), except `_PATCH_EXCLUDED`."""
        return {k: getattr(self, k) for k, v in vars(EchoStateNetwork).items()
                if not k.startswith('_') and k not in _PATCH_EXCLUDED and not callable(v)
                and not isinstance(v, (property, cached_property, staticmethod, classmethod))}

    def _patch_record(self, data, p):
        """Record of the window of patch `p`: the columns `patch_sites[p]` of `data`
        (``(..., N_dim)`` array or ragged list of segments), with zeros at the
        positions outside a non-periodic domain."""
        if isinstance(data, (list, tuple)):
            return [self._patch_record(segment, p) for segment in data]
        data = np.asarray(data)
        padded = np.concatenate([data, np.zeros(data.shape[:-1] + (1,))], axis=-1)
        return padded[..., self.patch_sites[p]]

    def _patch_windows(self, u):
        """Raw input windows of the independent patches, ``(N_patches, n_window, N_ens)``,
        from ``u`` ``(N_dim, N_ens)``. A position outside a non-periodic domain takes
        the `shift` of the network of its patch, so that it reads zero once
        normalized, as with shared patches."""
        sites = self.patch_sites
        windows = u[sites]                      # -1 positions are overwritten below
        for p, net in enumerate(self.patches):
            outside = sites[p] < 0
            windows[p, outside] = net.shift[outside, np.newaxis]
        return windows

    def _step_independent(self, u, r):
        """`step` with independent patches: every network of `patches` advances its
        own reservoir from its window, and the parallel network keeps the forecast of
        the sites of each patch."""
        n, sites = self.N_units, slice(self.halo, self.halo + self.patch_size)
        u_out, r_out = [], []
        for p, (net, window) in enumerate(zip(self.patches, self._patch_windows(u))):
            u_p, r_p = net.step(window, r[p * n:(p + 1) * n])
            u_out.append(u_p[sites])
            r_out.append(r_p)
        return np.concatenate(u_out), np.concatenate(r_out)

    def _train_patches(self, train_data, add_noise=True, validation_strategy=None, seed=None,
                       n_seeds=1, **options):
        """Backend of `train` with independent patches (``shared=False``): train the
        network of every patch on the record of its own window, with its own
        hyperparameter search (the validation strategy and Bayesian optimization of
        `train`, run per patch), its own normalization and its own ridge regression;
        `n_seeds` selects the reservoir realization per patch. Training plots are not
        drawn. The validation probes of a patch forecast the sites of the patch in
        closed loop with the halo taken from the data, so that the search of one patch
        does not depend on the other patches.

        `options` are the keywords of `train`, which `train` has already set on this network. The network of
        every patch receives the options of `_patch_options` that are keywords of this call or that changed on this
        network since the networks took them (e.g. ``esn.N_units = 10`` before ``train(force=True)``); an option
        that this network holds but a plain `EchoStateNetwork` does not, e.g. an attribute of a subclass, stays on
        this network. A new `validation_metric`, given to `train` or set on this network, is wrapped to score the
        sites of the patch only. The networks of the patches are built again, with seeds spawned from the given
        `seed`, else from the attribute `seed`, when `patch_size` or `halo` changes (`_stale_patches`); a given `seed`
        alone does not build them again (a warning says so). Otherwise, every network keeps its reservoir by the
        rule of `train`, which draws a new one from the seed of the patch when an option that defines it changes,
        and its own results of the search (e.g. `rho`) unless they changed on this network. Every network takes the
        time step of the data and the split of this network: the `t_train` and `t_val` that were given, or None,
        which infers them again from the record of the window. A warning that the networks of the patches raise,
        e.g. fewer validation folds than `N_folds`, is issued once per call, at the line that calls `train`."""
        stale = self._stale_patches()
        root = self.__dict__.get('_patches_key', {}).get('seed')
        kept = not stale and self.patches is not None
        if stale:
            logger.debug('train: new networks of the patches, because %s changed.', ', '.join(stale))
            self.patches = None
        for attr in ('_W', '_Win', '_Wout', '_reservoir_key'):     # the matrices of a shared layout trained before
            self.__dict__.pop(attr, None)
        current = self._patch_options()
        if self.patches is None:
            self._init_patches(seed)
            changed = {}
        else:
            seen = self.__dict__.get('_patch_options_seen')
            if seen is None:        # networks built before this record: the options that define the reservoir
                names = ('N_units', 'connect', 'Win_type', 'param_in_reservoir')
            else:
                names = [k for k, v in current.items() if not _same(v, seen.get(k, _UNSET))]
            changed = {k: current[k] for k in names if k not in ('t_train', 't_val')}   # the split: below
            for net in self.patches:        # a metric given to train, or set on this network since
                wrapped = net.validation_metric
                if 'validation_metric' in options or (getattr(wrapped, 'func', None) is _patch_metric
                                                      and wrapped.keywords['metric'] is not self.validation_metric):
                    net.validation_metric = partial(_patch_metric, metric=self.validation_metric,
                                                    sites=net._forecast_sites)
        options = deepcopy({**changed, **{k: v for k, v in options.items() if k in current}})
        if kept and seed is not None and seed != root:
            self._warn_patch_seed(seed, root, options)
        for p, net in enumerate(self.patches):
            net._inferred_split = {}            # the split of this network decides, not the record of the patch
            net.t_train, net.t_val = self.t_train, self.t_val
            net._dt_data = self.dt_physical
            net.train(self._patch_record(train_data, p), add_noise=add_noise, plot_training=False,
                      validation_strategy=validation_strategy, n_seeds=n_seeds, force=True, **options)
        self._patch_options_seen = deepcopy(current)
        # the split is the same for every patch (same record length)
        before = self._inferred_state()
        self.t_train, self.t_val = self.patches[0].t_train, self.patches[0].t_val
        self._record_split(before)
        logger.info('%s', self.training_summary())

    def _warn_patch_seed(self, seed, root, options):
        """Warn that `train` ignores `seed` for the networks of independent patches that it keeps, built from the
        seed `root` (None: unknown, e.g. networks pickled by echostatenetwork 0.2.3, which kept no record of it).
        `options` are the options that the networks receive (`_train_patches`): when one of them changes a
        reservoir (`N_units`, `connect`, `Win_type`, as `_stale_reservoir` decides), the networks keep their seeds
        but draw new reservoirs from them."""
        def changes(net):
            record = net.__dict__.get('_reservoir_key') or {}       # none: the options at the start of the call
            return any(not _same(options[name], record.get(name, getattr(net, name)))
                       for name in ('N_units', 'connect', 'Win_type') if name in options)
        spawned = 'patches[p].seed' if root is None else f'spawned from seed {root}'
        kept = (f'keep their seeds ({spawned}), from which they draw their new reservoirs'
                if any(changes(net) for net in self.patches) else
                f'keep their reservoirs and their seeds ({spawned})')
        _warn(f'{type(self).__name__}.train: seed={seed} is ignored: the networks of the patches {kept}; train '
              'spawns new seeds only when patch_size or halo changes (see force). A new network trained with '
              f'seed={seed} draws the networks of that seed.')

    def _stale_patches(self) -> list:
        """Why the networks of the independent patches no longer fit (empty: `train` keeps them): `patch_size` or
        `halo` changed since `_init_patches` built them. A seed given to `train` does not rebuild them, as for the
        reservoir of a single network (`_stale_reservoir`)."""
        if self.patches is None:
            return []
        key = self.__dict__.get('_patches_key', {})
        stale = [f'{name} ({key[name]} -> {getattr(self, name)})' for name in ('patch_size', 'halo')
                 if name in key and key[name] != getattr(self, name)]
        if not stale and (len(self.patches) != self.N_patches or self.patches[0].N_dim != self._n_in):
            stale.append('the window of the patches')
        return stale


    @property
    def norm(self):
        """np.ndarray: Per-component scale factor used by `normalize_input`, shape
        ``(N_dim_in,)``. Defaults to ones (no scaling) until set by
        `_split_and_format_data`/`_set_norm`.
        """
        if not hasattr(self, '_norm'):
            return np.ones((self.N_dim_in,))
        return self._norm

    @norm.setter
    def norm(self, value):
        """
        Setter for the normalization factor. Ensures it is N_dim_in.
        """
        if hasattr(self, 'N_dim_in'):
            assert value.size == self.N_dim_in, \
                f'Normalization factor must be dimension Ndim={self.N_dim_in}, got {value.shape}'
        self._norm = value.flatten()


    @cached_property
    def dr_di(self) -> csr_matrix | np.ndarray:
        r"""Linear (pre-activation) part of the input-to-reservoir Jacobian,
        $\sigma_\mathrm{in}\,\mathbf{W}_\mathrm{in,1}\,\mathrm{diag}(1/\texttt{norm})$,
        shape ``(N_units, N_dim_in)``, where $\mathbf{W}_\mathrm{in,1}$ is `Win` with
        the bias column dropped. This is *not* the full $\partial\mathbf{r}/\partial\mathbf{u}$:
        `Jacobian` additionally applies the $\mathrm{diag}(1-\mathbf{r}^2)$ factor from
        differentiating $\tanh$. Cached via `functools.cached_property` and
        invalidated whenever `Win` is reassigned. Not defined for the parallel
        layout, whose `Jacobian` assembles the blocks of the patches instead.
        """
        if self.patch_size is not None:
            raise ValueError('dr_di is not defined for the parallel layout (patch_size); '
                             'Jacobian assembles the blocks of the patches.')
        norm = self.norm.copy()

        Win_1 = self.Win[:, :self.N_dim_in]  # type: Union[csr_matrix, np.ndarray]
        g = self.sigma_in * 1.0 / norm
        if self.sigma_param is not None and self.input_parameters is not None:
            n_par = self._n_param(self.input_parameters)
            g[self.N_dim_in - n_par:] = self.sigma_param / norm[self.N_dim_in - n_par:]

        if issparse(Win_1):
            # .multiply returns a COO matrix: convert back to CSR for efficient products
            return csr_matrix(Win_1.multiply(g[np.newaxis, :]))
        else:
            return Win_1 * g[np.newaxis, :]


    @property
    def shift(self):
        """np.ndarray: Per-component offset used by `normalize_input`, shape
        ``(N_dim_in,)``. Defaults to zeros (no shift) until set by
        `_split_and_format_data`/`_set_norm`.
        """
        if not hasattr(self, '_shift'):
            return np.zeros((self.N_dim_in,))
        return self._shift


    @shift.setter
    def shift(self, value):
        """
        Setter for the shift factor. Ensures it is N_dim_in (nb. after initialization).
        """

        if hasattr(self, 'N_dim_in'):
            assert value.size == self.N_dim_in, \
                f'Shift factor must be dimension Ndim={self.N_dim_in}, got {value.shape}'
        self._shift = value.flatten()

    # _______________________________________________________________________________________________________ STEP & JACOBIAN
    def step(self, u, r):
        r"""Advance the reservoir by one open-loop time step,
        $\mathbf{r}_{n+1} = (1-\alpha)\,\mathbf{r}_n +
        \alpha\tanh(\sigma_\mathrm{in}\mathbf{W}_\mathrm{in}[\mathbf{u}_n; b_\mathrm{in}]
        + \rho\mathbf{W}\mathbf{r}_n)$ with $\alpha$ = `leak_rate` (the default
        $\alpha=1$ is the plain tanh update, no leak), and read out the corresponding
        physical state (see the class docstring for the full formulation).

        With the parallel layout (`patch_size`), the same update advances the
        reservoir of every patch $p$, with $\mathbf{u}_n$ replaced by its values at
        the sites of the window of the patch, ``patch_sites[p]``, and $\mathbf{r}_n$
        by the rows of the patch, $\mathbf{r}_{p,n}$. The shared readout maps
        $\mathbf{r}_{p,n+1}$ to the sites of patch $p$. Internally, the patches are
        extra columns of one shared reservoir, so an ensemble of ``N_ens`` members
        advances ``N_patches * N_ens`` columns at once. With independent patches
        (``shared=False``), the network of every patch (`patches`) advances its own
        reservoir from its window with its own matrices and hyperparameters.

        Parameters
        ----------
        u : np.ndarray
            Input state at the current time step, shape ``(N_dim_in, N_ens)``.
        r : np.ndarray
            Reservoir state at the current time step, shape ``(N_r, N_ens)``, with
            ``N_r = N_units`` for a single reservoir and ``N_r = N_patches * N_units``
            (the patches stacked in patch order) with the parallel layout.

        Returns
        -------
        u_out : np.ndarray
            Physical output at the next time step, shape ``(N_dim, N_ens)``.
        r_out : np.ndarray
            Updated reservoir state, shape ``(N_r, N_ens)``.

        Raises
        ------
        ValueError
            With the parallel layout, if `r` does not have `N_r` rows.
        """
        # Normalise input data and augment with input bias (ESN symmetry parameter)

        # assert self.trained, 'ESN must be trained with washout before calling step method. Call ESN.train() first.'

        if u.ndim == 1:
            u = np.expand_dims(u, axis=-1)
        elif u.ndim == 3:
            assert u.shape[0] == 1, f'Input u has shape {u.shape}, only 1 sample at a time is allowed'
            u = u[0]
        if r.ndim == 1:
            r = np.expand_dims(r, axis=-1)
        elif r.ndim == 3:
            assert r.shape[0] == 1, f'Input r has shape {r.shape}, only 1 sample at a time is allowed'
            r = r[0]

        if self.patch_size is not None and r.shape[0] != self.N_r:
            raise ValueError(f'r has {r.shape[0]} rows, but the reservoir state stacks the '
                             f'{self.N_patches} patches: N_r = N_patches * N_units = {self.N_r} rows.')
        if self._independent:
            return self._step_independent(u, r)

        # Normalize input
        u_norm = self.normalize_input(u)

        if self.patch_size is not None:
            # parallel layout: the patches become columns of the shared reservoir --
            # u_norm -> the windows (n_window, N_patches * N_ens), r -> (N_units, N_patches * N_ens);
            # the appended zero row is what the -1 entries of patch_sites read
            # input_parameters: the rows after the N_dim sites, read by every patch
            params = np.tile(u_norm[self.N_dim:], (1, self.N_patches))
            u_norm = self._to_columns(np.vstack([u_norm[:self.N_dim], np.zeros((1, u.shape[-1]))])[self.patch_sites])
            u_norm = np.vstack([u_norm, params])
            r = self._to_columns(r.reshape(self.N_patches, self.N_units, -1))

        # Augment input with bias
        bias_in = self.bias_in * np.ones((1, u_norm.shape[-1]))
        u_aug = np.concatenate((u_norm, bias_in))
        if self.sigma_param is not None and self.input_parameters is not None:
            # the parameter rows scaled by sigma_param instead of sigma_in (the product below applies sigma_in)
            n_par = self._n_param(self.input_parameters)
            first = self._n_in - n_par
            ratio = np.asarray(self.sigma_param, dtype=float) / np.asarray(self.sigma_in, dtype=float)
            if ratio.ndim and self.patch_size is not None:       # per-member values: the patches are columns
                ratio = np.tile(ratio, self.N_patches)
            u_aug[first:first + n_par] = u_aug[first:first + n_par] * ratio

        # Forecast the reservoir state (leaky-integrator; leak_rate=1 -> plain tanh)
        x_tanh = np.tanh(self.sigma_in * self.Win.dot(u_aug) + self.rho * self.W.dot(r))
        r_out = x_tanh if self.leak_rate == 1.0 else \
            (1.0 - self.leak_rate) * r + self.leak_rate * x_tanh

        # compute output from ESN if not during training
        u_out = self._readout(r_out, u_norm)
        if self.patch_size is not None:   # columns -> sites and stacked patches
            return self._from_columns(u_out), self._from_columns(r_out)
        return u_out, r_out



    def reservoir_to_physical(self, r, u=None):
        """Convert the reservoir state to the physical state via the output matrix.

        Parameters
        ----------
        r : np.ndarray
            Reservoir state, shape ``(N_r, N_ens)`` (the output bias row is
            appended internally); ``N_r = N_units`` for a single reservoir.
        u : np.ndarray, optional
            The (raw, un-normalised) ESN input that produced `r`, shape
            ``(N_dim_in, N_ens)``. Required when `readout_input` is True, since the
            readout then also reads the input; ignored otherwise.

        Returns
        -------
        np.ndarray
            Physical state, shape ``(N_dim, N_ens)``.
        """
        if self._independent:   # the network of each patch reads out its window
            r = r[:, np.newaxis] if r.ndim == 1 else r
            n, sites = self.N_units, slice(self.halo, self.halo + self.patch_size)
            return np.concatenate([net.reservoir_to_physical(r[p * n:(p + 1) * n])[sites]
                                   for p, net in enumerate(self.patches)])
        if self.patch_size is not None:   # each patch reads out its own sites
            r = r[:, np.newaxis] if r.ndim == 1 else r
            return self._from_columns(self._readout(
                self._to_columns(r.reshape(self.N_patches, self.N_units, -1)), None))
        if not self.readout_input:
            return self._readout(r, None)
        if u is None:
            raise ValueError("readout_input=True: the readout also reads the input that "
                             "produced r; pass u= (the raw ESN input of that step).")
        u = u[:, np.newaxis] if u.ndim == 1 else u
        return self._readout(r, self.normalize_input(u))

    def _readout(self, r, u_norm):
        """Wout^T [r; u_norm; bias_out] (u_norm only when `readout_input`)."""
        bias_out = self.bias_out * np.ones((1, r.shape[-1]))
        parts = (r, u_norm[self._readout_rows], bias_out) if self.readout_input else (r, bias_out)
        return np.dot(self.Wout.T, np.concatenate(parts))

    def normalize_input(self, data):
        r"""Shift-and-scale the input, $(\mathbf{u} - \texttt{shift}) / \texttt{norm}$
        (see `shift`, `norm` and `_set_norm`), before it is fed to `Win`.

        Parameters
        ----------
        data : np.ndarray
            Input data to be normalized, shape ``(N_dim_in, N_ens)``.

        Returns
        -------
        np.ndarray
            Normalized input data, same shape as `data`.
        """
        return (data - self.shift[:, np.newaxis]) / self.norm[:, np.newaxis]


    def _closed_loop_input(self, u_out, u_data):
        """Input of the next step of a closed-loop validation probe: the forecast
        `u_out` mapped back to the input space (`outputs_to_inputs`). The network of an
        independent patch (see `patches`) feeds back only the sites of its patch and
        takes its halo from `u_data`, the data at that step, so that its probes do not
        depend on the other patches."""
        sites = getattr(self, '_forecast_sites', None)
        if sites is None:
            return self.outputs_to_inputs(full_state=u_out)
        u_in = np.array(u_data, dtype=float).reshape(-1, 1)
        u_in[sites] = u_out[sites]
        return u_in

    def outputs_to_inputs(self, full_state):
        """Map a full physical state (e.g. a closed-loop prediction) back to the
        ESN's input space: selects the observed components (`observed_idx`) and, for
        a parametric ESN, appends `input_parameters`.

        Parameters
        ----------
        full_state : np.ndarray
            Full physical state vector, shape ``(N_dim, N_ens)``.

        Returns
        -------
        np.ndarray
            Input state vector, shape ``(N_dim_in, N_ens)``.
        """
        assert full_state.shape[0] == self.N_dim, f'full_state has shape {full_state.shape}, expected first dim to be {self.N_dim}'

        observed_state = full_state[self.observed_idx]

        assert observed_state.shape[0] == len(self.observed_idx), f'observed_state has shape {observed_state.shape}, expected first dim to be {len(self.observed_idx)}'

        if self.input_parameters is None:
            return observed_state
        else:
            return np.concatenate([observed_state, self.input_parameters], axis=0)


    def Jacobian(self, u_in, r_in, open_loop_J=True):
        r"""Analytical Jacobian of the one-step map, $\mathbf{J} = \partial\mathbf{u}_{n+1}/\partial\mathbf{u}_n$,
        obtained by differentiating `step` through $\tanh$:

        $$
        \mathbf{J} = \mathbf{W}_\mathrm{out,1}^\mathrm{T}\,
        \mathrm{diag}\!\left(1-\mathbf{r}_{n+1}^{\,2}\right)\,
        \sigma_\mathrm{in}\,\mathbf{W}_\mathrm{in,1}\,\mathrm{diag}(1/\texttt{norm}),
        $$

        where $\mathbf{W}_\mathrm{out,1}$ and $\mathbf{W}_\mathrm{in,1}$ are `Wout`/`Win`
        with the bias row/column dropped, and $\mathbf{r}_{n+1}$ is the reservoir state
        obtained by stepping from (`u_in`, `r_in`). The $1/\texttt{norm}$ factor comes
        from the chain rule through `normalize_input`. With a leaky reservoir
        (`leak_rate` $\alpha<1$) the middle factor becomes
        $\alpha\,\mathrm{diag}(1-\tilde{\mathbf{x}}^{\,2})$ with $\tilde{\mathbf{x}}$
        the tanh pre-leak value, recovered from the step as
        $(\mathbf{r}_{n+1} - (1-\alpha)\mathbf{r}_n)/\alpha$.

        With the parallel layout (`patch_size`), the Jacobian is sparse: the rows of
        the sites of patch $p$ depend only on the columns of its window,
        ``patch_sites[p]``, through the block above evaluated with the reservoir
        state of the patch, $\mathbf{r}_{p,n+1}$. A position outside a non-periodic
        domain reads no input and contributes no column. With independent patches
        (``shared=False``), each block is the Jacobian of the network of the patch.

        Parameters
        ----------
        u_in : np.ndarray
            Input state, shape ``(N_dim_in, N_ens)``.
        r_in : np.ndarray
            Reservoir state, shape ``(N_r, N_ens)`` (``N_r = N_units`` for a single
            reservoir).
        open_loop_J : bool
            If True (default), compute the open-loop Jacobian above. The closed-loop
            variant (linearizing through the feedback of `u_out` back into the next
            input) is not implemented -- see Raises.

        Returns
        -------
        np.ndarray
            Jacobian $\partial\mathbf{u}_\mathrm{out}/\partial\mathbf{u}_\mathrm{in}$,
            shape ``(N_dim, N_dim_in)`` if ``N_ens == 1``, else ``(N_dim, N_dim_in, N_ens)``.

        Raises
        ------
        NotImplementedError
            If `open_loop_J` is False (the closed-loop Jacobian is unimplemented; a
            numerical check of the sketched derivation did not pass).
        """
        assert self.trained, 'ESN must be trained before computing the Jacobian. Call ESN.train() first.'
        if self._independent:
            if not open_loop_J:
                raise NotImplementedError('Numerical test of closed-loop Jacobian did not pass')
            return self._independent_jacobian(u_in, r_in)


        Wout_1 = self.Wout[:self.N_units, :].T

        # # Option(i) rin function of bin:
        rout = self.step(u_in, r_in)[1]

        if self.leak_rate == 1.0:
            tt = 1. - rout ** 2
        else:
            # d(r_out)/d(pre-activation) = leak_rate * (1 - x_tanh^2), with the
            # tanh pre-leak value recovered from the leaky update
            r_prev = r_in[:, np.newaxis] if r_in.ndim == 1 else \
                (r_in[0] if r_in.ndim == 3 else r_in)
            x_tanh = (rout - (1. - self.leak_rate) * r_prev) / self.leak_rate
            tt = self.leak_rate * (1. - x_tanh ** 2)
        if not open_loop_J:
            # u_aug = np.concatenate((u_in / self.norm, self.bias_in))
            # rout = np.tanh(self.sigma_in * self.Win.dot(u_aug) + self.rho * np.dot(self.WCout.T, u_in))
            # dr_di = self.sigma_in * Win_1 / self.norm + self.rho * self.WCout.T
            #  Win_G += dr_di ......
            raise NotImplementedError('Numerical test of closed-loop Jacobian did not pass')
        if self.patch_size is not None:
            return self._patch_jacobian(tt)
        dr_di = self.dr_di

        # readout_input adds the direct term Wout_u^T diag(1/norm) on the rows it reads
        rows = self._readout_rows
        J_u = np.zeros((self.N_dim, self.N_dim_in))
        J_u[:, rows] = self.Wout[self.N_units:self._n_readout - 1].T / self.norm[np.newaxis, rows]
        N_ens = tt.shape[-1]
        if N_ens == 1:
            if issparse(dr_di):
                RHS = dr_di.T.multiply(tt[:, 0][np.newaxis, :])
            else:
                RHS = dr_di.T * tt[:, 0][np.newaxis, :]
            return RHS.dot(Wout_1.T).T + J_u

        J = np.zeros((self.N_dim, self.N_dim_in, N_ens))
        for ens_i in range(N_ens):
            if issparse(dr_di):
                RHS = dr_di.T.multiply(tt[:, ens_i][np.newaxis, :])
            else:
                RHS = dr_di.T * tt[:, ens_i][np.newaxis, :]
            J[:, :, ens_i] = RHS.dot(Wout_1.T).T + J_u

        return J

    def _independent_jacobian(self, u_in, r_in):
        """Backend of `Jacobian` with independent patches: the Jacobian of the network
        of every patch, rows of the sites of the patch, scattered into the columns of
        its window."""
        u = u_in[:, np.newaxis] if u_in.ndim == 1 else u_in
        r = r_in[:, np.newaxis] if r_in.ndim == 1 else r_in
        N_ens, n, H, G = u.shape[-1], self.N_units, self.halo, self.patch_size
        J = np.zeros((self.N_dim, self.N_dim_in, N_ens))
        for p, (net, window) in enumerate(zip(self.patches, self._patch_windows(u))):
            J_p = net.Jacobian(window, r[p * n:(p + 1) * n]).reshape(self._n_in, self._n_in, N_ens)
            sites = self.patch_sites[p]
            inside = sites >= 0
            J[p * G:(p + 1) * G][:, sites[inside]] += J_p[H:H + G][:, inside]
        return J[..., 0] if N_ens == 1 else J

    def _patch_jacobian(self, tt):
        """Backend of `Jacobian` for the parallel layout: scatter the block
        ``Wout_1^T diag(tt_p) sigma_in Win_1 diag(1/norm[window])`` of every patch p
        into the rows of its sites and the columns of its window. `tt` is the
        derivative of the reservoir update with respect to its pre-activation,
        shape ``(N_r, N_ens)``."""
        N_ens = tt.shape[-1]
        sites = self.patch_sites                                       # (P, n_window)
        Win_1 = self.Win[:, :self._n_in]
        Win_1 = Win_1.toarray() if issparse(Win_1) else np.asarray(Win_1)
        n_window = sites.shape[1]
        inv_norm = np.where(sites >= 0, 1. / self.norm[sites], 0.)      # padded positions: no input
        tt = tt.reshape(self.N_patches, self.N_units, N_ens)
        blocks = np.einsum('jg,pjm,jk,pk->pgkm', self.Wout[:self.N_units], tt,
                           self.sigma_in * Win_1[:, :n_window], inv_norm, optimize=True)   # (P, G, n_window, N_ens)
        J = np.zeros((self.N_dim, self.N_dim_in, N_ens))
        rows = np.arange(self.N_dim).reshape(self.N_patches, self.patch_size, 1)
        np.add.at(J, (rows, np.maximum(sites, 0)[:, np.newaxis, :]), blocks)
        if n_window < self._n_in:          # input_parameters: every patch reads them
            J[:, self.N_dim:] = np.einsum('jg,pjm,jk,k->pgkm', self.Wout[:self.N_units], tt,
                                          (self.sigma_in if self.sigma_param is None else self.sigma_param)
                                          * Win_1[:, n_window:], 1. / self.norm[self.N_dim:],
                                          optimize=True).reshape(self.N_dim, -1, N_ens)
        return J[..., 0] if N_ens == 1 else J




    # _______________________________________________________________________________________ TRAIN & VALIDATE THE ESN
    def train(self,
              train_data,
              add_noise=True,
              plot_training=True,
              save_ESN_training=False,
              folder=None,
              validation_strategy=None,
              seed=None,
              n_seeds=1,
              force=False,
              **kwargs
              ):
        """Train the ESN: format the data into washout/train/validation/test sets,
        (re)generate `Win`/`W` if not already set or if the options that define them
        changed (see `force`), select hyperparameters, and fit `Wout` by ridge
        regression on the selected hyperparameters.

        The hyperparameters of `hyperparameters_to_optimize` are selected by Bayesian
        optimization, which scores every candidate with the validation strategy. A
        network with `input_parameters` searches the normalization of its parameters
        as well: `optimize_parameter_normalization`, True by default with
        `input_parameters`, adds the shift and the scale of every parameter column
        (``param_shift_i``, ``param_norm_i``) to the search, also to an empty
        `hyperparameters_to_optimize`. No search runs only when the list is empty and
        nothing is added to it, i.e. without `input_parameters` or with
        ``optimize_parameter_normalization=False``.

        With 'tikh' in `hyperparameters_to_optimize`, every evaluation of the
        validation strategy selects `tikh` from the grid `tikh_range`, so 'tikh' is
        not a dimension of the Bayesian optimization. Without it, every evaluation
        keeps the fixed `tikh`. A search over 'tikh' alone is therefore a grid
        search: one evaluation of the strategy at the fixed `rho`, `sigma_in` and
        `leak_rate`, without Bayesian optimization and without scikit-optimize
        (`N_func_evals`, `N_grid` and `N_initial_rand` do not apply). Dimensions that
        the network adds to the search, e.g. the normalization of the parameters,
        make it a Bayesian optimization again.

        With the parallel layout (`patch_size`), the patches share `Wout`, which the
        ridge regression fits on all the patches at once: every time step provides
        `N_patches` samples, one per patch, each mapping the reservoir of the patch
        to its sites. The input `norm` and `shift` take one value for all sites,
        pooled over the sites, because the patches share `Win`. With independent
        patches (``shared=False``), the network of every patch trains on the record of
        its own window, with its own hyperparameter search, normalization and ridge
        regression (see `_train_patches`); `n_seeds` then applies per patch.

        A call that raises, e.g. a ValueError of the layout or data too short for the split, or that is
        interrupted, leaves the network as it was: its options, matrices, split and generator `rng`. A warning
        of the package is issued once per call, at the line that called the package (`echostatenetwork.verbose`).

        Parameters
        ----------
        train_data : np.ndarray
            Training data, shape ``(L, Nt, N_dim)``, or a ragged list of L
            segments of shape ``(Nt_l, N_dim)``.
        add_noise : bool
            If True, add Gaussian noise (scaled by `noise`) to the training input.
        plot_training : bool
            If True, visualize the training process (BO convergence, `Wout`, and
            post-training test forecasts).
        save_ESN_training : bool
            If True, save the training plots to a PDF (in `folder`).
        folder : str, optional
            Directory to save training plots to. Defaults to `figs_folder`.
        validation_strategy : callable, optional
            Validation function for hyperparameter tuning. Defaults to `_RVC_Noise`.
        seed : int, optional
            Seed of the random `Win` and `W` when `train` draws them, with a new generator: on the first call
            (unless the network holds a reservoir drawn by `_generate_W_Win` or set by hand before it), and when
            an option that defines the reservoir changes (see `force`). By default, the first call draws them
            from the generator `rng` of the attribute `seed`, after the noise of the inputs, so that no given seed
            reproduces them. A seed alone does not replace a reservoir that the network holds: `train` keeps the
            reservoir and warns that `seed` is ignored, unless it is the seed of that reservoir. The keyword does
            not change the attribute `seed`, which seeds the noise of the inputs, and setting the attribute does
            not draw a new reservoir. With ``n_seeds > 1``, the first seed of the realizations: by default, the
            first seed of the previous ``n_seeds`` call while the attribute `seed` is one of its seeds (as after
            that call), so that training again draws the same realizations; otherwise the attribute `seed`.
        n_seeds : int
            If > 1, train this many reservoir realizations (seeds ``base, base+1,
            ...``, with `base` given by `seed`) on the same data and settings,
            in parallel processes, and keep the one with the best validation score;
            per-seed scores are stored in `seed_scores` for statistical comparison
            (a later call with ``n_seeds = 1`` leaves them, and they set the first
            seed of the next ``n_seeds`` call, see `seed`).
            The score of a seed is ``bo_results['fun']`` after a search, otherwise
            one evaluation of the validation strategy at the fixed hyperparameters
            of the seed, `tikh` included. Training plots are skipped in this mode.
            A warning raised while the seeds train, e.g. fewer validation folds than
            `N_folds`, is issued once, by the calling process, at the line that calls
            `train`, also when the warning comes from code of the user that the package
            calls back (e.g. a validation strategy that calls `validation.WFV`), which a
            single seed points at.
        force : bool
            A trained network is not trained again unless `force` is True (a warning says so). A copy of a
            trained network (`copy`) and a network loaded with its matrices (`from_arrays`, `pickle`) are
            trained. With `force`, `train` keeps

            - the reservoir (`W`, `Win`) while `N_units`, `connect`, `Win_type`, the number of inputs
              (`observed_idx`) and of parameter columns (`input_parameters`), `param_in_reservoir`, `patch_size`
              and `halo` are unchanged, as keywords or as attributes, whatever the given `seed`;
            - a `t_train` or `t_val` that was given, to the constructor, to `train` or as an attribute;

            and computes again

            - the hyperparameter search and `Wout`;
            - the reservoir when one of these options changes, from the given `seed`, else from the seed of
              the reservoir;
            - a `t_train` or `t_val` that the previous call inferred, from the new data, so that the same data
              give the same split;
            - `dt_ESN` from a new `upsample`, with the time step of the data unchanged.

            The Notes give the edge cases: the seed of a new reservoir, independent patches, matrices set by
            hand, networks pickled by echostatenetwork 0.2.3, attributes set to an inferred value, and how to
            change the time step of the data.
        **kwargs
            Any option of the network to override before training (e.g. `N_units`, `rho_range`), set in the
            order given, except `W`, `Win` and `dt_ESN`, which are set last. The keywords of the layout
            (`patch_size`, `halo`, `periodic`, `shared`, `observed_idx`, `input_parameters`, `readout_input`)
            are checked as by the constructor. A keyword that is not an option, e.g. a misspelt name, a method
            or a read-only property such as `trained`, is ignored with a UserWarning, which names it and, for a
            typo, suggests the closest option or parameter of `train` (e.g. `rho_range` for `rho_rnage`,
            `n_seeds` for `n_seed`, and `dt_ESN` for `dt`, which is a parameter of the constructor only), once
            per call, also with ``n_seeds > 1`` and independent patches, and also when a trained network is not
            trained again. `Wout`, which `train` computes, and an attribute that the network does not hold yet
            (e.g. `W` or `Win` before training) are ignored with the same warning; `W` and `Win` set on the
            network before `train` are kept (with independent patches, on the network of a patch). The removed
            keyword ``verbose`` gives a FutureWarning.

        Returns
        -------
        None
            Sets `Wout` (and `Win`/`W` when they are not present or no longer fit, see `force`) in place. Also
            sets `bo_results`: a dict with keys ``'func_vals'`` (the per-evaluation
            validation-loss trace), ``'x_iters'`` (evaluated points), ``'x'``/``'fun'``
            (the selected point and its loss), ``'hp_names'``, and ``'n_grid_points'``
            -- or None when no search ran: `hyperparameters_to_optimize` is empty, and the
            network adds no normalization of parameters to it (see above). This
            is a slimmed copy of the `skopt` result: the raw ``OptimizeResult``
            retains the training corpus and the fitted GP models, which would bloat
            every pickle/deepcopy of a trained ESN. With 'tikh' alone, `bo_results`
            holds the one evaluation: ``'func_vals'`` of length one, ``'x' = []``,
            ``'x_iters' = [[]]``, ``'hp_names' = []`` (the selected Tikhonov factor
            is `tikh`, as after a Bayesian optimization) and ``'n_grid_points' = 1``;
            ``'fun'`` is the validation score at the selected `tikh`, which also
            scores a seed of ``n_seeds > 1``.

        Notes
        -----
        Training again, ``train(force=True)``, in detail:

        - The seed of a new reservoir. A new reservoir comes from the given `seed`, else from the seed of the
          reservoir, with a new generator: it is the reservoir that a new network with these options draws with
          ``train(seed=...)``, and a diagnostic message names the option that changed. The first call draws the
          reservoir from the generator `rng` of the attribute `seed`, after the noise of the inputs, so no seed
          reproduces it: a reservoir drawn again then comes from the attribute `seed`, and an option that
          changes and changes back does not restore the reservoir of the first call. A seed alone does not
          replace the reservoir (see `seed`), but ``train(seed=s, n_seeds=m, force=True)`` trains the
          realizations of the seeds ``s, s+1, ..., s+m-1``, each with its own reservoir, and keeps the best one.
          Setting the attribute `seed` starts the noise of the inputs from a new generator and keeps the
          reservoir. `readout_input` changes `Wout` only.
        - Independent patches. The networks of the patches are built again, with seeds spawned from the given
          `seed`, else from the attribute `seed`, only when `patch_size` or `halo` changes. Otherwise, every
          network takes the options changed on this network (keywords or attributes) and keeps its reservoir by
          the rule above, which draws a new one from the seed of its patch, ``patches[p].seed``: a new single
          network with the options of the patch draws it with ``train(seed=patches[p].seed)``, a new parallel
          network does not. Otherwise, a seed given to `train` is ignored, with a warning: the networks keep their
          seeds, and their reservoirs unless an option changes them.
        - Matrices set by hand and networks pickled by echostatenetwork 0.2.3 or earlier. A reservoir set by
          hand (`W`, `Win`, `from_arrays`), or held by a 0.2.3 network, takes the options of the network at the
          start of the call as its own: an option that the call changes draws a new reservoir from the attribute
          `seed`, so that networks that share one reservoir draw the same new one. A matrix set by hand whose
          shape no longer fits the options (e.g. ``esn.N_units = 20; esn.W = W20`` keeps the old `Win`) is drawn
          again alone.
        - The split of a network pickled by echostatenetwork 0.2.3, which has no record of what it inferred.
          On one series, `t_train` and `t_val` are taken as inferred when they are the split that `train`
          infers (the training and validation steps fill 80% of the series, and `t_val` is 20% of these) and
          `split_summary` says that `t_val` was inferred. For a network trained with ``n_seeds > 1`` (it holds
          `seed_scores`), whose summary may say that an inferred `t_val` was given, the split alone decides: a
          given split equal to the inferred one is then inferred again from the new data. Otherwise, `t_val` is
          inferred again when the summary says so, and `t_train` is kept. Pass `t_train` and `t_val` to `train`
          to keep them.
        - Attributes set to an inferred value. An attribute set to the value that the previous call inferred,
          e.g. ``esn.t_val = esn.t_val``, cannot be told from the inferred value, and is inferred again: pass it
          to `train` to keep it, e.g. ``esn.train(y, t_val=esn.t_val, force=True)``.
        - The search over the normalization of the parameters. The names ``param_shift_i``, ``param_norm_i``
          that the previous call added to `hyperparameters_to_optimize` are added again only while the network
          has parameters and `optimize_parameter_normalization`, unless the list was set since; the
          `param_shift_range`, `param_norm_range` that the first call sized from the parameters are kept.
        - The time step. `dt_ESN` follows `upsample`, and `t_train`, `t_val` remain times. The time step of the
          data, `dt`, is a parameter of the constructor, not an option: ``train(dt=...)`` ignores it with a
          warning. To change it, pass ``dt_ESN = dt * upsample`` to `train`, or set the attribute `dt_ESN`;
          for ``dt = 0.5``: ``esn.train(y, dt_ESN=0.5 * esn.upsample, force=True)``.
        - The noise of the inputs continues the generator `rng`, so that a single network trained again on the
          same data does not reproduce its previous training; the seeds of ``n_seeds > 1`` do.
        """
        # once, here, before the dispatch: the networks of independent patches and the seeds of n_seeds > 1 train
        # with the keywords that are kept, so they do not warn again
        _pop_verbose(kwargs, f'{type(self).__name__}.train')
        ignored = {}
        for key in list(kwargs):
            # Wout: train computes it, also on a trained network, which holds it
            kind = 'computed' if key == 'Wout' else _keyword_kind(type(self), key, network=self)
            if kind == 'method' and _keyword_kind(EchoStateNetwork, key) == 'option':
                kind = 'option'         # an option that a subclass holds as a callable, e.g. a staticmethod metric
            if kind != 'option':
                ignored.setdefault(kind, []).append(key)
                del kwargs[key]
        if ignored:                     # the parameters of train are candidates of the hint as well
            _warn_ignored(f'{type(self).__name__}.train', ignored, 'the network', type(self), network=self,
                          extra=[name for name in inspect.signature(EchoStateNetwork.train).parameters
                                 if name not in ('self', 'kwargs')], independent=self._independent)
        if self.trained and not force:
            _warn(f'{type(self).__name__}.train: the network is already trained and is not trained again; pass '
                  'force=True to train it again.')
            return None
        if self.trained and getattr(self, '_patch', None) is None:
            logger.debug('train(force=True): training the trained network again.')
        saved = self._saved_state()
        try:
            with _warnings_once():      # each warning once per call, however many evaluations, seeds or patches
                return self._train_network(train_data, add_noise, plot_training, save_ESN_training, folder,
                                           validation_strategy, seed, n_seeds, kwargs)
        except BaseException:           # an error, or an interruption: the network is left as it was
            self._restore_state(saved)
            raise

    def _train_network(self, train_data, add_noise, plot_training, save_ESN_training, folder, validation_strategy,
                       seed, n_seeds, kwargs):
        """Body of `train`, after the check of its keywords (`kwargs`, the options to set). Not `_train`, which
        romda's autoencoders define, and which a latent ROM (e.g. AE_ESN) inherits with this class."""
        self._record_reservoir()            # before the keywords, which _stale_reservoir compares with this record
        self._set_options(kwargs)           # before the dispatch: the keywords may change the layout
        self._restore_split(keep=kwargs)    # what the previous call inferred from its data is inferred again
        if self._independent:
            return self._train_patches(train_data, add_noise=add_noise,
                                       validation_strategy=validation_strategy, seed=seed,
                                       n_seeds=n_seeds, **kwargs)
        if self.patches is not None:        # the networks of independent patches, from a layout trained before
            self.patches = None
        if n_seeds > 1:
            return self._train_multi_seed(train_data, n_seeds, add_noise=add_noise,
                                          validation_strategy=validation_strategy, seed=seed)

        # ========================== STEP 1: DATA FORMATTING ==========================
        # Format data into washout, train/validation, and test sets; remember what the split inferred
        before = self._inferred_state()
        try:
            U_wtv, Y_wtv, U_test, Y_test = self._split_and_format_data(train_data, add_noise=add_noise)
        finally:
            self._record_split(before)

        # W and Win: kept while the options that define them are unchanged, whatever the seed, drawn again otherwise
        stale, drop = self._stale_reservoir()
        key = self.__dict__.get('_reservoir_key') or {}
        if stale:
            if seed is None:                # the seed of the reservoir, else the attribute seed, with a new generator
                seed = self.seed if key.get('seed') is None else key['seed']
            logger.debug('%strain: new %s (seed %s), because %s changed.',
                         '' if getattr(self, '_patch', None) is None else f'patch {self._patch}: ',
                         ' and '.join(attr[1:] for attr in drop), seed, ', '.join(stale))
            for attr in (*drop, '_reservoir_key'):
                self.__dict__.pop(attr, None)
        elif seed is not None and seed != key.get('seed') and hasattr(self, '_W') and hasattr(self, '_Win'):
            _warn(f"{type(self).__name__}.train: seed={seed} is ignored: the network keeps its reservoir (W, Win)"
                  + ('' if key.get('seed') is None else f", drawn from seed {key['seed']}") + ', which train draws '
                  'again only when an option that defines it changes (see force). A new network trained with '
                  f'seed={seed}, or n_seeds > 1, draws the reservoir of that seed.')
        if not hasattr(self, '_W') or not hasattr(self, '_Win'):
            self._generate_W_Win(seed=seed)

        self.Wout = np.zeros((self._n_readout, self._n_out))  # Initialize Wout with zeros

        # Validation/test runs temporarily overwrite self.input_parameters
        original_input_parameters = self.input_parameters
        try:
            # =================== STEP 2: BAYESIAN HYPERPARAMETER OPTIMIZATION ==============
            self.val_k = 0  # Reset validation counter at the start of training
            # Perform hyperparameter optimization if required
            if self.hyperparameters_to_optimize:
                report = plot_training or logger.isEnabledFor(logging.INFO)
                bo_results = self._optimize_hyperparameters(U_wtv, Y_wtv, validation_strategy,
                                                           print_convergence=report)
            else:
                bo_results = None
            # Expose the BHO output for post-training inspection
            if bo_results is None:
                self.bo_results = None
            else:
                res = bo_results['res']
                self.bo_results = dict(
                    func_vals=np.asarray(res.func_vals), x_iters=list(res.x_iters),
                    x=res.x, fun=res.fun,
                    hp_names=bo_results['hp_names'],
                    n_grid_points=bo_results['n_grid_points'])
            # ====================== STEP 3: RIDGE REGRESSION TRAINING =====================
            # the search, or the keywords, may have changed sigma_in and the normalization, which dr_di caches
            self._invalidate_jacobian_cache()
            # Compute the output weight matrix Wout
            self.Wout = self._solve_ridge_regression(U_wtv, Y_wtv)

            if getattr(self, '_patch', None) is None:
                logger.info('%s', self.training_summary())

            # ========================== STEP 4: TEST AND PLOTTING ======================
            if plot_training:
                self._plot_training_results(U_test, Y_test, bo_results, save_ESN_training, folder)
        finally:
            self.input_parameters = original_input_parameters

    def _train_multi_seed(self, train_data, n_seeds, add_noise, validation_strategy, seed):
        """Backend of ``train(n_seeds > 1)``: train `n_seeds` reservoir realizations
        of this ESN on the same data and settings, adopt the realization with the
        best validation score in place, and store the per-seed scores in
        `seed_scores` (``{seed: score}``) for statistical comparison. Runs one
        process per seed when the ESN and validation strategy pickle (a closure
        strategy doesn't -- falls back to a serial loop with a note). `train` has
        already set its keywords on this network, which the realizations copy. The
        first seed is `seed`; by default, the first seed of the previous call while
        the attribute `seed` (the kept seed) is one of its seeds, else the attribute."""
        previous = getattr(self, 'seed_scores', None) or {}
        if seed is not None:
            base = seed
        elif self.seed in previous:     # trained again: the seeds of the previous call
            base = int(min(previous))
        else:
            base = self.seed
        seeds = [base + i for i in range(n_seeds)]
        try:
            pickle.dumps((self, validation_strategy))
            parallel = True
        except Exception:
            logger.info('n_seeds: the ESN or the validation strategy does not pickle; '
                        'training the seeds serially instead of in parallel.')
            parallel = False
        try:
            if parallel:
                # ponytail: no BLAS-thread throttling in workers; export OMP_NUM_THREADS
                # if n_seeds x BLAS threads oversubscribes the machine.
                with ProcessPoolExecutor(max_workers=min(n_seeds, os.cpu_count() or 1)) as ex:
                    results = list(ex.map(_train_one_seed, [self] * n_seeds, seeds,
                                          [train_data] * n_seeds, [add_noise] * n_seeds,
                                          [validation_strategy] * n_seeds))
            else:
                results = [_train_one_seed(self.copy(), s, train_data, add_noise, validation_strategy)
                           for s in seeds]
        except Exception as error:      # the warnings of the failed seed first: they may explain the error
            for message, category in getattr(error, _SEED_WARNINGS, ()):
                _warn(message, category)
            raise
        # collected by the seeds, issued here, once per call of train (_warnings_once): before the best seed is
        # adopted
        for *_, caught in results:
            for message, category in caught:
                _warn(message, category)
        scores = np.array([score for _, score, *_ in results])
        for s, score, (_, _, log, _) in zip(seeds, scores, results):  # the reports of each seed, in order
            logger.info('seed %s (validation score %.4f)%s', s, score,
                        (':\n' + log.rstrip('\n')) if log.strip() else '')
        best = int(np.nanargmin(scores))
        self.__dict__.clear()
        self.__dict__.update(results[best][0].__dict__)
        self.seed_scores = dict(zip(seeds, scores))
        logger.info('n_seeds=%d: validation score %.4f +/- %.4f (best %.4f @ seed %s, worst %.4f) -> kept seed %s',
                    n_seeds, scores.mean(), scores.std(), scores[best], seeds[best], scores.max(), seeds[best])

    # ---------------------------------------------------------------- training again (train(force=True))
    def _set_options(self, options):
        """Set the keywords of `train` (options of the network, see `_keyword_kind`), with a diagnostic message for
        every changed value (from this network, not from the networks of independent patches, which receive the
        same keywords). After a keyword of the layout (`_LAYOUT_OPTIONS`), check the layout as the constructor does;
        `train` restores the network when the check raises a ValueError."""
        # dt_ESN, W and Win last, whatever the order of the keywords: dt_ESN sets the time step of the data from the
        # new upsample (as the constructor does), and W and Win are checked against the new N_units and inputs
        for key in sorted(options, key=lambda k: _SET_LAST.index(k) + 1 if k in _SET_LAST else 0):
            val = options[key]
            if getattr(self, '_patch', None) is None:
                try:
                    old = getattr(self, key)
                except Exception:               # e.g. a getter that needs another option
                    old = val
                if not _same(old, val):
                    logger.debug('train: %s = %s -> %s', key, old, val)
            setattr(self, key, val)
        if _LAYOUT_OPTIONS.intersection(options):
            self._check_layout()

    def _saved_state(self) -> list:
        """The state of this network and of the networks of its patches, for `_restore_state`: their attributes
        (a shallow copy, enough because `train` assigns new values and new arrays rather than changing them in
        place) and the state of their generator `rng`, which the noise of the inputs advances in place: the state
        of its bit generator, a dict that is cheaper to save than a copy of the generator, or a copy of a generator
        without one."""
        saved = []
        for net in [self, *(self.patches or [])]:
            rng = net.__dict__.get('_rng')
            saved.append((net, dict(net.__dict__),
                          rng.bit_generator.state if hasattr(rng, 'bit_generator') else deepcopy(rng)))
        return saved

    @staticmethod
    def _restore_state(saved):
        """Return every network of `saved` (`_saved_state`) to its saved state."""
        for net, state, rng in saved:
            net.__dict__.clear()
            net.__dict__.update(state)
            if isinstance(rng, dict):           # the generator of the saved attributes, returned to its state
                state['_rng'].bit_generator.state = rng
            elif rng is not None:
                net._rng = rng

    def _inferred_state(self) -> dict:
        """The attributes of `_INFERRED` that this network sets itself (`_UNSET`: the class default), before the
        split of `train` may infer them, for `_record_split`."""
        return {name: self.__dict__.get(name, _UNSET) for name in _INFERRED}

    def _record_split(self, before):
        """Remember which attributes of `_INFERRED` the split of this call inferred from the data (or, for a
        ragged record, wrote back), with the value that each had before (`before`, from `_inferred_state`), for
        `_restore_split`."""
        now = self._inferred_state()
        self._inferred_split = {name: (now[name], value) for name, value in before.items()
                                if not _same(now[name], value)}

    def _restore_split(self, keep=()):
        """Return the attributes that the previous `train` inferred from its data (`_record_split`) to the values
        they had before it, so that this call infers them again from its data: `t_train` and `t_val` (None: inferred
        again), and the names ``param_shift_i``/``param_norm_i`` that it added to `hyperparameters_to_optimize`
        (`_INFERRED`). An attribute set since, or in `keep` (the keywords of this call), is kept. A value set as
        an attribute that equals the inferred value cannot be told from it, and is inferred again: give it to
        `train` to keep it. A network trained before this record (pickled by echostatenetwork 0.2.3 or earlier) has
        it rebuilt from `split_summary` (`_legacy_split_record`)."""
        record = self.__dict__.pop('_inferred_split', None)
        if record is None:
            record = self._legacy_split_record()
        for name, (inferred, before) in record.items():
            if name in keep or not _same(self.__dict__.get(name, _UNSET), inferred):
                continue
            if name in ('t_train', 't_val') and before in (None, _UNSET):
                logger.debug('train: %s is inferred again from the data (the previous call inferred %s).',
                             name, inferred)
            if before is _UNSET:
                self.__dict__.pop(name, None)
            else:
                setattr(self, name, before)

    def _legacy_split_record(self) -> dict:
        """The record of `_record_split` of a network trained before it existed, from `split_summary`. When in doubt,
        a value is taken as given, and kept, as echostatenetwork 0.2.3 kept it. A ragged record writes `t_train`
        back, and its summary says whether `t_val` was inferred. On one series, `t_train` and `t_val` were both
        inferred when the split is the one that `train` infers from the length of the series (the training and
        validation steps fill 80% of it, and `t_val` is 20% of these) and the summary says that `t_val` was
        inferred. A network trained with ``n_seeds > 1`` (it holds `seed_scores`) may have a summary that says
        that `t_val` was given (0.2.3 split the data of the kept seed again to score it without a search): for it,
        the split alone decides, so that a given split equal to the inferred one is inferred again. Otherwise,
        `t_val` was inferred when the summary says so, and `t_train` is taken as given: a `t_train` inferred with a
        given `t_val` cannot be told from a given split that fills 80% of the series. The names that the split
        added to `hyperparameters_to_optimize` are not recorded. With independent patches, the summary of the
        network of the first patch, whose split this network takes."""
        source = self.patches[0] if self._independent and self.patches else self
        summary = source.split_summary
        if not summary or not source.trained:
            return {}
        record = {}
        try:
            if summary.get('ragged'):
                record['t_train'] = (self.t_train, summary.get('t_train_given'))
                val_inferred = summary.get('t_val_inferred')
            else:
                n_wtv = max(self.N_wash + 2, int(round(0.8 * summary['n_steps'])))
                inferred = (self.N_train + self.N_val == n_wtv and self.N_val == max(1, int(round(0.2 * n_wtv)))
                            and bool(summary.get('t_val_inferred') or source.__dict__.get('seed_scores')))
                if inferred:
                    record['t_train'] = (self.t_train, None)
                val_inferred = inferred or summary.get('t_val_inferred')
            if val_inferred:
                record['t_val'] = (self.t_val, None)
        except (KeyError, TypeError, ValueError):       # an incomplete summary: nothing is inferred again
            return {}
        return record

    def _reservoir_options(self) -> dict:
        """The options that define `W` and `Win` (their shape or their pattern of connections): `N_units`,
        `connect`, `Win_type`, the number of inputs and of parameter columns, `param_in_reservoir` (with
        parameters), and the layout of the patches (`patch_size`, and `halo` with patches)."""
        n_param = 0 if self.input_parameters is None else self._n_param(self.input_parameters)
        return dict(N_units=self.N_units, connect=self.connect, Win_type=self.Win_type, inputs=self._n_in,
                    parameters=n_param, param_in_reservoir=bool(self.param_in_reservoir) if n_param else None,
                    patch_size=self.patch_size, halo=None if self.patch_size is None else self.halo)

    def _record_reservoir(self):
        """Give a reservoir without a record of the options that define it the options of the network as its record,
        before `train` sets its keywords, when both matrices fit these options. A reservoir set by hand (`W`, `Win`,
        `from_arrays`, e.g. the charts of a qlroms family that share the reservoir of the first chart) or held by a
        network pickled by echostatenetwork 0.2.3 or earlier then follows the rule of a reservoir drawn by
        `_generate_W_Win` (`_stale_reservoir`): `train` keeps it, or draws a new one when an option of this call
        changes it, so that the networks that share one reservoir draw the same new one. Its seed is unknown (None),
        as for a reservoir drawn from `rng`. An option changed as an attribute before this call is not detected."""
        state = self.__dict__
        if '_reservoir_key' in state or '_W' not in state or '_Win' not in state:
            return
        if state['_W'].shape == (self.N_units, self.N_units) and state['_Win'].shape == (self.N_units, self._n_in + 1):
            self._reservoir_key = dict(self._reservoir_options(), seed=None)

    def _stale_reservoir(self) -> tuple[list, tuple]:
        """Why `W` and `Win` no longer fit the network, and the matrices to draw again (both empty: `train` keeps
        them). A reservoir with a record of the options that define it (`_generate_W_Win`, `_record_reservoir`):
        every option of `_reservoir_options` that changed since, and both matrices. The seed is not an option of the
        record: a trained network keeps its reservoir whatever the seed given to `train`, and the first call keeps
        a reservoir drawn or set before it. Matrices without a record (e.g. after ``esn.N_units = 20; esn.W = W20``,
        which leaves the old `Win`): the matrix whose shape no longer fits, alone."""
        state = self.__dict__
        key = state.get('_reservoir_key')
        if key is None:
            shapes = {'_W': (self.N_units, self.N_units), '_Win': (self.N_units, self._n_in + 1)}
            drop = tuple(attr for attr, shape in shapes.items() if attr in state and state[attr].shape != shape)
            return [f'the shape of {attr[1:]} ({state[attr].shape} -> {shapes[attr]})' for attr in drop], drop
        if '_W' not in state and '_Win' not in state:
            return [], ()
        stale = [f'{name} ({key.get(name)} -> {value})' for name, value in self._reservoir_options().items()
                 if key.get(name) != value]
        return stale, ('_W', '_Win') if stale else ()

    def training_summary(self) -> str:
        """One-line summary of the last `train` call: the data split (from
        `_split_and_format_data`, stored in `split_summary`), the resulting
        train/validation windows, and the selected hyperparameters (with the number
        of Bayesian-optimization evaluations when a search ran, or, with 'tikh' alone,
        the number of values of `tikh_range` that its one evaluation compares)."""
        if self._independent and self.patches is not None:
            return '\n'.join(f'patch {p}: {net.training_summary()}'
                             for p, net in enumerate(self.patches))
        s = self.split_summary
        if s is None:
            raise RuntimeError('no training summary yet: call train() first.')
        if s['ragged']:
            data = (f"{s['segments_train']}/{s['segments_in']} segments -> {s['pairs']} pairs, "
                    f"{s['segments_test']} held out")
            if s['segments_dropped']:
                data += f", {s['segments_dropped']} dropped (< N_wash+2)"
            if s['t_train_given'] is not None:
                data += " (given t_train ignored: a segmented corpus is never capped)"
        else:
            data = f"{self.N_train}+{self.N_val} train+val steps, {s['n_test']} test"
        hp_names = ('rho', 'sigma_in', 'tikh') + \
            (('sigma_param',) if self.sigma_param is not None else ()) + \
            (('leak_rate',) if self.leak_rate != 1.0 else ())
        hps = ', '.join(f'{name}={self._get_hyperparam(name):.3g}' for name in hp_names)
        # realized fold count of the last validation run (strategies may cap or
        # multiply the requested N_folds -- see val_fold_step), so an N_folds
        # override is visible here rather than silently absorbed
        folds = getattr(self, 'n_folds_realized', None)
        if folds is not None:
            hps += f" | {folds} validation fold{'s' if folds != 1 else ''}"
        if self.bo_results is None:
            bo = ''
        elif self.bo_results['hp_names']:
            bo = f" | BHO: {len(self.bo_results['func_vals'])} evals over {self.bo_results['hp_names']}"
        else:       # 'tikh' alone: one evaluation of the validation strategy selects tikh
            n = len(self.tikh_range)
            bo = f" | tikh alone: 1 eval over the {n} value{'s' if n != 1 else ''} of tikh_range, no BHO"
        return (f"trained: {data} | t_train={self.t_train:.3g}, t_val={self.t_val:.3g}"
                + (' (inferred)' if s['t_val_inferred'] else '') + f" | {hps}{bo}")


    def copy(self):
        """Return a deep copy of this `EchoStateNetwork`.

        Returns
        -------
        EchoStateNetwork
            A new, independent instance with the same state and matrices. The copy of a trained network is
            trained: ``train(force=True)`` trains it again, e.g. on other data (see `train`).
        """
        return deepcopy(self)

    def to_arrays(self) -> dict:
        """Deployment state as a flat dict of plain numpy-compatible values, ready
        for ``np.savez`` (no pickled objects). The inverse is `from_arrays`.

        Captures exactly what `step`/`reservoir_to_physical`/`outputs_to_inputs`
        read -- configuration scalars, `bias_in`/`bias_out` (pinned as data, so a
        future class-default change cannot alter a stored model), `norm`/`shift`
        (which carry any BHO-tuned parameter normalization), and the matrices
        (`W`/`Win` as exact CSR triplets, so a rebuilt reservoir is bit-identical).
        Matrices appear only when generated. A ragged (list) `input_parameters` is
        collapsed to ``np.zeros((N_param, 1))``: after training only its column
        count is read at run time. Dropped entirely: BO search state and ranges,
        split/seed summaries, `WCout` and other derived caches. The rng is NOT
        stored -- a reloaded model's stochastic noise stream restarts from `seed`.
        The parallel layout (`patch_size`, `halo`, `periodic`, `shared`) is stored only
        when it is set, so a dict without these keys loads as a single reservoir;
        independent patches store the arrays of each patch network under the key
        prefix ``patch<p>_``.
        """
        out = dict(N_dim=self.N_dim, N_units=self.N_units, N_wash=self.N_wash,
                   upsample=self.upsample, seed=self.seed, dt_ESN=self.dt_ESN,
                   rho=self.rho, sigma_in=self.sigma_in, tikh=self.tikh,
                   **({} if self.sigma_param is None else dict(sigma_param=self.sigma_param)),
                   leak_rate=self.leak_rate, noise=self.noise,
                   Win_type=self.Win_type, norm_method=self.norm_method,
                   noise_type=self.noise_type, readout_input=str(self.readout_input),
                   bias_in=np.asarray(self.bias_in), bias_out=np.asarray(self.bias_out),
                   observed_idx=np.asarray(self.observed_idx),
                   norm=np.asarray(self.norm), shift=np.asarray(self.shift))
        if self.patch_size is not None:
            out.update(patch_size=int(self.patch_size), halo=int(self.halo),
                       periodic=bool(self.periodic), shared=bool(self.shared))
        if self._independent and self.patches is not None:   # one key prefix per patch
            for p, net in enumerate(self.patches):
                out.update({f'patch{p}_{k}': v for k, v in net.to_arrays().items()})
        ip = self.input_parameters
        if ip is not None:
            out['input_parameters'] = (np.zeros((self._n_param(ip), 1))
                                       if isinstance(ip, list) else np.asarray(ip))
        if hasattr(self, '_Wout'):
            out['Wout'] = np.asarray(self._Wout)
        if hasattr(self, '_W'):
            out.update(W_data=self._W.data, W_indices=self._W.indices,
                       W_indptr=self._W.indptr)
        if hasattr(self, '_Win'):
            if issparse(self._Win):
                out.update(Win_data=self._Win.data, Win_indices=self._Win.indices,
                           Win_indptr=self._Win.indptr)
            else:
                out['Win'] = np.asarray(self._Win)
        return out

    @classmethod
    def from_arrays(cls, arrays) -> 'EchoStateNetwork':
        """Rebuild a deployable ESN from `to_arrays`'s dict (values may be 0-d numpy
        scalars from an npz, or memory-mapped arrays). `trained` is True iff the
        matrix keys are present, and ``train(force=True)`` then trains it again. Assignment runs through the
        normal constructor and property setters, so every shape assertion still guards the load."""
        def f(key):
            return float(np.asarray(arrays[key]))

        def s(key):
            return str(np.asarray(arrays[key]))

        n_units = int(np.asarray(arrays['N_units']))
        upsample = int(np.asarray(arrays['upsample']))
        ip = np.asarray(arrays['input_parameters']) if 'input_parameters' in arrays else None
        patches = (dict(patch_size=int(np.asarray(arrays['patch_size'])),
                        halo=int(np.asarray(arrays['halo'])),
                        periodic=bool(np.asarray(arrays['periodic'])),
                        shared=bool(np.asarray(arrays['shared'])) if 'shared' in arrays else True)
                   if 'patch_size' in arrays else {})
        esn = cls(np.zeros((int(np.asarray(arrays['N_dim'])), 1)),
                  **patches,
                  dt=f('dt_ESN') / upsample,
                  N_units=n_units, N_wash=int(np.asarray(arrays['N_wash'])),
                  upsample=upsample, Win_type=s('Win_type'),
                  norm_method=s('norm_method'), noise_type=s('noise_type'),
                  noise=f('noise'), rho=f('rho'), sigma_in=f('sigma_in'),
                  **({'sigma_param': f('sigma_param')} if 'sigma_param' in arrays else {}),
                  tikh=f('tikh'), leak_rate=f('leak_rate'),
                  observed_idx=np.asarray(arrays['observed_idx']),
                  input_parameters=ip,
                  readout_input={'True': True, 'False': False}.get(
                      str(np.asarray(arrays['readout_input'])),
                      str(np.asarray(arrays['readout_input'])))
                  if 'readout_input' in arrays else False)
        esn.seed = int(np.asarray(arrays['seed']))
        esn.bias_in = np.asarray(arrays['bias_in'])
        esn.bias_out = np.asarray(arrays['bias_out'])
        if 'Win' in arrays:
            esn.Win = np.asarray(arrays['Win'])
        elif 'Win_data' in arrays:
            esn.Win = csr_matrix((np.asarray(arrays['Win_data']),
                                  np.asarray(arrays['Win_indices']),
                                  np.asarray(arrays['Win_indptr'])),
                                 shape=(n_units, esn._n_in + 1))
        if 'W_data' in arrays:
            esn.W = csr_matrix((np.asarray(arrays['W_data']),
                                np.asarray(arrays['W_indices']),
                                np.asarray(arrays['W_indptr'])),
                               shape=(n_units, n_units))
        if 'Wout' in arrays:
            esn.Wout = np.asarray(arrays['Wout'])
        esn.norm = np.asarray(arrays['norm'])
        esn.shift = np.asarray(arrays['shift'])
        if esn._independent and 'patch0_N_dim' in arrays:
            esn._init_patches()      # options, metric and closed-loop sites of each patch
            for p in range(esn.N_patches):
                prefix = f'patch{p}_'
                loaded = EchoStateNetwork.from_arrays(
                    {k[len(prefix):]: arrays[k] for k in arrays.keys() if k.startswith(prefix)})
                for name in ('_Win', '_W', '_Wout', '_norm', '_shift', 'rho', 'sigma_in',
                             'tikh', 'leak_rate', 'bias_in', 'bias_out', 'seed'):
                    if hasattr(loaded, name):
                        setattr(esn.patches[p], name, getattr(loaded, name))
        return esn

    # _______________________________________________________________________________________ HELPER METHODS FOR ESN INITIALIZATION & TRAINING



    def _generate_W_Win(self, seed=None):
        """Generate `Win` (random, sparse or dense) and `W` (Erdős–Rényi, rescaled to
        unit spectral radius), if not already set.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, with a new generator. Defaults to the
            generator of the attribute `seed`, `rng`. When it draws both matrices,
            the network records the options that define them (`_reservoir_options`)
            and the seed (None for `rng`), so that `train` keeps them while these
            are unchanged.

        Raises
        ------
        ValueError
            If `Win_type` is not one of `WIN_TYPES`.

        Returns
        -------
        None
            Sets `Win` and `W` in place.
        """
        if self._independent:   # one independent realization per patch
            if self.patches is None:
                self._init_patches(seed)
            for net in self.patches:
                net._generate_W_Win()
            return

        if seed is None:
            rng0 = self.rng
        else:
            rng0 = np.random.default_rng(seed)
        drawn = not hasattr(self, '_Win') and not hasattr(self, '_W')     # both drawn here, none set by hand

        # Input matrix: Sparse random matrix where only one element per row is different from zero.
        # Parameter columns (if any) are the exception: every neuron connects densely to them,
        # since they carry a single global forcing rather than a per-neuron-selected observation.
        if not hasattr(self, '_Win'):
            # input columns: N_dim_in, or the window of one patch with the parallel layout
            n_in = self._n_in
            Win = lil_matrix((self.N_units,
                              n_in + 1))  # +1 accounts for input bias
            if self.Win_type in ('sparse', 'sparse_dense_bias'):
                N_param = self._n_param(self.input_parameters) if self.input_parameters is not None else 0
                N_state = n_in - N_param
                dense_bias = self.Win_type == 'sparse_dense_bias'
                # columns eligible for the single sparse connection: state columns (+ bias, last column, if sparse)
                sparse_cols = np.arange(N_state) if dense_bias else np.append(np.arange(N_state), n_in)
                for j in range(self.N_units):
                    Win[j, rng0.choice(sparse_cols)] = rng0.uniform(low=-1, high=1)
                if dense_bias:
                    Win[:, n_in] = rng0.uniform(low=-1, high=1, size=(self.N_units, 1))
                if N_param > 0 and self.param_in_reservoir:
                    Win[:, N_state:n_in] = rng0.uniform(
                        low=-1, high=1, size=(self.N_units, N_param))
            elif self.Win_type == 'dense':
                for j in range(self.N_units):
                    Win[j, :] = rng0.uniform(low=-1, high=1, size=n_in + 1)
                if not self.param_in_reservoir and self.input_parameters is not None:
                    Win[:, n_in - self._n_param(self.input_parameters):n_in] = 0
            else:
                raise ValueError(f"Win type {self.Win_type} not implemented {WIN_TYPES}")
            # Store
            self.Win = Win

        # Reservoir state matrix: Erdos-Renyi network
        if not hasattr(self, '_W'):
            W = csr_matrix(rng0.uniform(low=-1, high=1, size=(self.N_units, self.N_units)) *
                        (rng0.random(size=(self.N_units, self.N_units)) < (1 - self.sparsity)))
            # scale W by the spectral radius to have unitary spectral radius.
            # ARPACK's default starting vector is unseeded, which makes the
            # radius (and so W, W_out) differ between processes with the same
            # seed; a fixed v0 keeps it deterministic without consuming rng0.
            v0 = np.full(self.N_units, 1.0 / np.sqrt(self.N_units))
            try:
                spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', v0=v0,
                                                     return_eigenvectors=False))[0]
            except ArpackNoConvergence:
                # seed-dependent ARPACK stagnation on small reservoirs; dense
                # eigenvalues are exact and cheap at typical N_units
                spectral_radius = np.abs(np.linalg.eigvals(W.toarray())).max()
            self.W = (1. / spectral_radius) * W
        if drawn:       # what drew W and Win: train() keeps them while it holds (_stale_reservoir)
            # seed None: drawn from the generator rng, after the noise of the inputs, which no given seed reproduces
            self._reservoir_key = dict(self._reservoir_options(), seed=seed)


    def _compute_RR_terms(self, U_wtv, Y_wtv):
        """
        Computes the Ridge Regression (RR) terms, including left-hand side (LHS) and right-hand side (RHS)
        matrices, for training the output weights.

        Parameters
        ----------
        U_wtv : np.ndarray (L x Nt x N_dim)
            Wash-train-validation input data. L segments, each with Nt time steps and N_dim dimensions.
        Y_wtv : np.ndarray (L x Nt x N_dim)
            Corresponding output labels for input data.

        Returns
        -------
            tuple:
                - LHS (np.ndarray): Left-hand side matrix for ridge regression.
                - RHS (np.ndarray): Right-hand side matrix for ridge regression.
                - U_RR (list): List of input states split by L-segments.
                - R_RR (list): List of reservoir states split by L-segments, each
                  ``(Nt, N_r)`` (the patches stacked with the parallel layout).

        With the parallel layout, every time step contributes one regression sample
        per patch, ``[r_p; bias_out] -> y[sites of p]``, so LHS and RHS sum over the
        time steps and the patches of the shared readout.
        """

        LHS = np.zeros((self._n_readout, self._n_readout))
        RHS = np.zeros((self._n_readout, self._n_out))
        R_RR = [None] * len(U_wtv)
        U_RR = [None] * len(U_wtv)


        for ll in range(len(U_wtv)):

            U_wash_l = U_wtv[ll][:self.N_wash]
            # Y_wash_l = Y_wtv[ll][:self.N_wash]
            Uin_l = U_wtv[ll][self.N_wash:]
            Yout_l = Y_wtv[ll][self.N_wash:]

            assert Uin_l.shape[0] == Yout_l.shape[0], \
                f'Inconsistent shapes for training data at segment {ll}: {Uin_l.shape} vs {Yout_l.shape}'

            assert Uin_l.shape[0] > 0, \
                f'Not enough data for training at segment {ll}: {Uin_l.shape}'

            # Washout phase to initialize reservoir state
            N_ens = U_wash_l.shape[-1] if U_wash_l.ndim == 3 else 1
            r_out = np.zeros((self.N_r, N_ens))
            for u_in in U_wash_l:
                _, r_out = self.step(u_in, r_out)

            if Yout_l.ndim == 3:
                assert Yout_l.shape[-1] == 1, f'Yout_l has shape {Yout_l.shape}, only 1 sample at a time is allowed'
                Yout_l = Yout_l[..., 0]

            # Open-loop train phase: one pass over the whole segment, states filled
            r_open = np.zeros((Uin_l.shape[0], self.N_r, N_ens))
            y_open = np.zeros((Uin_l.shape[0], self.N_dim, N_ens))
            for ii, u_in in enumerate(Uin_l):
                u_out, r_out = self.step(u_in, r_out)
                y_open[ii], r_open[ii] = u_out, r_out

            if y_open.ndim > 2:
                y_open, r_open = y_open.squeeze(axis=-1), r_open.squeeze(axis=-1)

            R_RR[ll] = r_open # type: ignore
            U_RR[ll] = y_open # type: ignore

            if self.patch_size is not None:
                # one sample per step and patch, row t * N_patches + p: the patches share Wout
                r_open = r_open.reshape(-1, self.N_units)
                Yout_l = Yout_l.reshape(-1, self.patch_size)

            # Compute matrices for linear regression system
            bias_out = np.ones([r_open.shape[0], 1]) * self.bias_out
            if self.readout_input:   # row ii also reads the (normalised) input that produced r_open[ii]
                U2 = Uin_l[..., 0] if Uin_l.ndim == 3 else Uin_l
                r_aug = np.hstack((r_open, self.normalize_input(U2.T).T[:, self._readout_rows], bias_out))
            else:
                r_aug = np.hstack((r_open, bias_out))

            LHS += np.dot(r_aug.T, r_aug)
            RHS += np.dot(r_aug.T, Yout_l)

        return LHS, RHS, U_RR, R_RR


    def _solve_ridge_regression(self, U_wtv, Y_wtv):
        """
        Solves the ridge regression problem to compute the output weight matrix (Wout).

        Parameters
        ----------
        U_wtv : np.ndarray (L x Nt x N_dim)
            Input data for ridge regression (train/valiladion). L segments, each with Nt time steps and N_dim dimensions.
        Y_wtv : np.ndarray (L x Nt x N_dim)
            Target labels for ridge regression. L segments, each with Nt time steps and N_dim dimensions.

        Returns
        -------
            np.ndarray: Computed output weight matrix (Wout).
        """
        LHS, RHS = self._compute_RR_terms(U_wtv, Y_wtv)[:2]
        return np.linalg.solve(self._regularized(LHS, self.tikh), RHS)

    def _regularized(self, LHS, tikh):
        """A copy of the ridge left-hand side with `tikh` on the diagonal; the bias
        row of the readout (the last one) is left out when `regularize_bias_out` is False."""
        LHS = LHS.copy()
        LHS.ravel()[::LHS.shape[1] + 1] += tikh
        if not self.regularize_bias_out:
            LHS[-1, -1] -= tikh
        return LHS


    def _UY_from_raw_data(self, data, add_noise=True, seed=None):
        """
        Extracts input (U) and output (Y) matrices from raw data.

        Args:
            data (np.ndarray or list[np.ndarray]): Raw time series data, either a
                regular [(L) x Nt x N_dim] array, or a list of L segments of shape
                (Nt_l, N_dim) with possibly different Nt_l (e.g. variable-length
                cluster-dwell chunks) -- see _UY_from_ragged_data.

        Returns:
            tuple: (U, Y). Regular (L, Nt, N_dim_in/N_dim) arrays for ndarray input;
            lists of L arrays of shape (Nt_l, N_dim_in/N_dim) for ragged input.
        """
        if isinstance(data, (list, tuple)):
            return self._UY_from_ragged_data(data, add_noise=add_noise, seed=seed)

        #   APPLY UPSAMPLE AND OBSERVED INDICES ________________________
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        # Set labels always as the full state
        Y = data[:, ::self.upsample].copy()
        # Inputs are the observed components of the state, which can be a subset of the full state
        U = Y[:, :, self.observed_idx].copy()

        assert Y.shape[-1] >= U.shape[-1]

        if self.input_parameters is not None:
            # Broadcast the per-segment parameter (e.g. cluster id) across all Nt steps
            L, Nt = U.shape[0], U.shape[1]
            N_param = self.input_parameters.shape[0]
            assert self.input_parameters.shape == (N_param, L), \
                f'input_parameters must have shape (N_param, L)=({N_param}, {L}), got {self.input_parameters.shape}'
            params_tiled = np.broadcast_to(self.input_parameters.T[:, None, :], (L, Nt, N_param))
            U = np.concatenate([U, params_tiled], axis=-1)

        assert U.shape[-1] == self.N_dim_in

        if add_noise:
            #  ==================== ADD NOISE TO TRAINING INPUT ====================== ##
            # Add noise to the inputs if distinction inputs/labels is not given.
            # Larger noise promotes stability in long term, but hinders time accuracy
            U_std = np.std(U, axis=1, keepdims=True)
            if seed is None:
                rng0 = self.rng
            else:
                rng0 = np.random.default_rng(seed)
            U += rng0.normal(loc=0, scale=self.noise * U_std, size=U.shape)

        return U, Y

    def _UY_from_ragged_data(self, segments, add_noise=True, seed=None):
        """Per-segment version of _UY_from_raw_data for a list of L segments of shape
        (Nt_l, N_dim), Nt_l possibly different per segment (e.g. cluster-dwell chunks
        of different lengths -- no ensemble axis is supported in this path).
        """
        L = len(segments)
        per_step_params = isinstance(self.input_parameters, list)
        if self.input_parameters is not None and not per_step_params:
            N_param = self.input_parameters.shape[0]
            assert self.input_parameters.shape == (N_param, L), \
                f'input_parameters must have shape (N_param, L)=({N_param}, {L}), got {self.input_parameters.shape}'
        elif per_step_params:
            assert len(self.input_parameters) == L, \
                f'input_parameters list must have length L={L}, got {len(self.input_parameters)}'

        rng0 = self.rng if seed is None else np.random.default_rng(seed)
        U, Y = [], []
        for l, seg in enumerate(segments):
            y_l = np.asarray(seg)[::self.upsample]
            u_l = y_l[:, self.observed_idx]
            assert y_l.shape[-1] >= u_l.shape[-1]

            if per_step_params:
                # Explicit per-timestep parameter (e.g. cluster id varying within a
                # segment that spans a transition), already upsample-aligned by the caller.
                params_l = np.asarray(self.input_parameters[l])[::self.upsample]
                assert params_l.shape[0] == y_l.shape[0], \
                    f'input_parameters[{l}] must have {y_l.shape[0]} steps (post-upsample), got {params_l.shape[0]}'
                u_l = np.concatenate([u_l, params_l], axis=-1)
            elif self.input_parameters is not None:
                N_param = self.input_parameters.shape[0]
                params_l = np.broadcast_to(self.input_parameters[:, l], (y_l.shape[0], N_param))
                u_l = np.concatenate([u_l, params_l], axis=-1)
            assert u_l.shape[-1] == self.N_dim_in

            if add_noise:
                u_std = np.std(u_l, axis=0, keepdims=True)
                u_l = u_l + rng0.normal(loc=0, scale=self.noise * u_std, size=u_l.shape)

            U.append(u_l)
            Y.append(y_l)
        return U, Y

    def _split_and_format_data(self, data=None, add_noise=True):
        """Format raw data into washout/train/validation and test sets, adding noise
        and computing/storing `norm`/`shift` (and, for a parametric ESN, tuning the
        parameter normalization -- see the class docstring).

        Parameters
        ----------
        data : np.ndarray
            Input time series data, shape ``(L, Nt, N_dim)`` (or ``(Nt, N_dim)``), or a
            ragged list of L segments of shape ``(Nt_l, N_dim)`` -- see
            `_UY_from_ragged_data` for the segmented-corpus rules.
        add_noise : bool
            Whether to add noise to the training input data. Default True.

        Returns
        -------
        U_wtv : np.ndarray
            Wash-train-validation input data.
        Y_wtv : np.ndarray
            Corresponding labels for train/validation data.
        U_test : np.ndarray
            Test input data.
        Y_test : np.ndarray
            Test labels.

        Raises
        ------
        ValueError
            If `data` is None, or shorter than `N_train + N_val` steps.
        """
        if data is None:
            raise ValueError('No training data provided to format_training_data method.')
        is_ragged = isinstance(data, (list, tuple))
        if not is_ragged and data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        U, Y = self._UY_from_raw_data(data, add_noise=add_noise) # dimensions: L x Nt x N_dim_in/N_dim

        #   SEPARATE INTO WASH/TRAIN/VAL/TEST SETS ______________________
        if is_ragged:
            # Segments may have very different lengths (e.g. cluster-dwell chunks).
            # Two rules decide what enters training:
            #   1. a segment must yield at least one open-loop pair past its own
            #      washout (N_wash + 2 raw points). Validation length does NOT gate
            #      training inclusion -- the probing strategies use whatever tail a
            #      segment has (_SegmentRVC_Noise), so tying the drop rule to N_val
            #      (as before) silently wasted training data whenever t_val was long.
            #   2. t_train does NOT cap a segmented corpus. A caller who cut a
            #      trajectory into segments already chose how much data to train on;
            #      t_train is a *window length* for one long trajectory and has no
            #      sensible reading across many short ones -- read as a total pair
            #      budget it silently shrank every segment to a couple of steps
            #      (a per-dwell-sized t_train over hundreds of dwells -> ~2% of each,
            #      i.e. an ESN trained on almost nothing). So every usable segment is
            #      kept in full and t_train is written back from what was kept.
            # U_test/Y_test just reuse the kept segments -- run_test is a visual
            # diagnostic, not the metric BHO optimizes, so an in-sample check is an
            # acceptable trade for not fragmenting already-scarce data further.
            min_len = self.N_wash + 2
            usable = [(U_l, Y_l) for U_l, Y_l in zip(U, Y) if U_l.shape[0] >= min_len]
            if not usable:
                raise ValueError(f'No segment has >= N_wash+2={min_len} steps; reduce N_wash.')

            t_val_inferred = self.t_val is None
            if self.t_val is None:
                # infer the validation window from the dwell (segment) lengths: the
                # median usable segment's closed-loop tail past its own washout
                median_len = int(np.median([U_l.shape[0] for U_l, _ in usable]))
                n_val = max(1, median_len - self.N_wash - 1)
                self.t_val = n_val * self.dt_ESN
                if n_val < 10:
                    _warn(
                        f'inferred validation window is degenerate: N_val={n_val} '
                        f'step(s), because the median segment ({median_len} steps) '
                        f'barely exceeds the washout (N_wash={self.N_wash}). '
                        f'Closed-loop validation over so few steps is meaningless; '
                        f'provide longer segments or set t_val explicitly.')

            # Use ALL the input data -- the first 80% of the segments (they arrive in
            # time order) train/validate, the last 20% are held out as the run_test
            # diagnostic window (with a single segment, test falls back to in-sample
            # reuse). t_train is written back from what was kept, so N_train is
            # defined afterwards even though nothing was capped by it.
            t_train_given = self.t_train
            # t_test == 0 means NO diagnostic holdout: every segment trains
            # (run_test then reuses the training segments in-sample)
            n_wtv = (len(usable) if self.t_test == 0
                     else max(1, int(round(0.8 * len(usable)))))
            wtv, test = usable[:n_wtv], usable[n_wtv:]
            U_wtv = [U_l[:-1] for U_l, _ in wtv]
            Y_wtv = [Y_l[1:] for _, Y_l in wtv]
            U_test = [U_l[:-1] for U_l, _ in test] or U_wtv
            Y_test = [Y_l[1:] for _, Y_l in test] or Y_wtv
            kept_pairs = sum(u.shape[0] - self.N_wash for u in U_wtv)
            self.t_train = max(kept_pairs - self.N_val, 1) * self.dt_ESN
            # quiet by design: training_summary() makes one compact line from this, which train() logs
            self.split_summary = dict(
                ragged=True, segments_in=len(U), segments_dropped=len(U) - len(usable),
                segments_train=len(wtv), segments_test=len(test), pairs=kept_pairs,
                t_val_inferred=t_val_inferred, t_train_given=t_train_given)
        else:
            # a single unsegmented trajectory has no dwell lengths to infer t_val
            # from; when omitted it defaults to 20% of the train/val window, and an
            # omitted t_train consumes all the data (80% train/val, 20% test).
            t_val_inferred = self.t_val is None
            if self.t_train is None:
                n_wtv = max(self.N_wash + 2, int(round(0.8 * U.shape[1])))
                if self.t_val is None:
                    self.t_val = max(1, int(round(0.2 * n_wtv))) * self.dt_ESN
                self.t_train = max(1, n_wtv - self.N_val) * self.dt_ESN
            elif self.t_val is None:
                self.t_val = max(1, int(round(0.2 * self.N_train))) * self.dt_ESN
            # n_test counts usable test PAIRS: U_test = U[:, N_wtv:-1] drops the
            # final step (no successor), hence the -1
            self.split_summary = dict(
                ragged=False, n_steps=U.shape[1],
                n_test=max(U.shape[1] - self.N_train - self.N_val - 1, 0),
                t_val_inferred=t_val_inferred)
            N_wtv = self.N_train + self.N_val
            if U.shape[1] < N_wtv:
                raise ValueError(f'Increase the length of the training data signal. {U.shape} < {N_wtv}')

            U_wtv = U[:, :N_wtv - 1].copy()
            Y_wtv = Y[:, 1:N_wtv].copy()

            U_test = U[:, N_wtv:-1].copy()
            Y_test = Y[:, N_wtv+1:].copy()

            assert U_wtv.shape[1] == Y_wtv.shape[1], \
                f'Inconsistent shapes for train/validation data: {U_wtv.shape} vs {Y_wtv.shape}'
            assert U_test.shape[1] == Y_test.shape[1], \
                f'Inconsistent shapes for test data: {U_test.shape} vs {Y_test.shape}'

            if Y_wtv.ndim not in [2, 3]:
                raise ValueError(f'Inconsistent ensemble size for train/validation data: {Y_wtv.shape}')

        # compute norm (normalize inputs by component range). Parameter columns (e.g.
        # cluster id) are excluded: they are a raw/categorical added input, not a
        # physical quantity to rescale, so they get identity normalization instead.
        N_obs = len(self.observed_idx)
        if is_ragged:
            # U_wtv is a ragged list (variable Nt_l): pool every segment's samples into
            # one long pseudo-trajectory so _set_norm sees a regular array. _set_norm
            # only squeezes its leading (L) axis when L>1, so flatten explicitly --
            # here L=1 by construction (one pooled trajectory).
            obs_pool = np.concatenate([u[:, :N_obs] for u in U_wtv], axis=0)
            norm_obs, shift_obs = EchoStateNetwork._set_norm(obs_pool[np.newaxis], method=self.norm_method)
            norm_obs, shift_obs = norm_obs.reshape(-1), shift_obs.reshape(-1)
        else:
            norm_obs, shift_obs = EchoStateNetwork._set_norm(U_wtv[..., :N_obs], method=self.norm_method)
        if self.patch_size is not None:
            # the patches share Win, so every site takes the same shift and scale,
            # computed from the samples of all the sites pooled together
            pooled = np.reshape(obs_pool if is_ragged else U_wtv[..., :N_obs], (1, -1, 1))
            norm_obs, shift_obs = (np.full(N_obs, np.ravel(v)[0]) for v in
                                   EchoStateNetwork._set_norm(pooled, method=self.norm_method))
        if self.input_parameters is not None:
            N_param = self._n_param(self.input_parameters)
            # Identity normalization for the parameter columns, unless the Bayesian
            # search already tuned them: re-formatting after train() (e.g. to slice
            # validation data) must not clobber the tuned values and leave a raw
            # parameter swamping the reservoir.
            mode = self.optimize_parameter_normalization
            if isinstance(mode, str) and mode == 'data':
                # the parameter columns normalized like the state (norm_method over the training values), before
                # the search: sigma_in then scales state and parameters alike, and the search keeps its dimensions
                params = (np.concatenate([u[:, N_obs:] for u in U_wtv], axis=0) if is_ragged
                          else np.asarray(U_wtv)[..., N_obs:N_obs + N_param])
                norm_param, shift_param = (np.ravel(v) for v in EchoStateNetwork._set_norm(
                    np.reshape(params, (1, -1, N_param)), method=self.norm_method))
            elif (mode and getattr(self, '_norm', None) is not None
                    and self._norm.size == N_obs + N_param):
                norm_param, shift_param = self._norm[-N_param:], self._shift[-N_param:]
            else:
                norm_param, shift_param = np.ones(N_param), np.zeros(N_param)
            norm_obs = np.concatenate([norm_obs, norm_param])
            shift_obs = np.concatenate([shift_obs, shift_param])
            if mode and not isinstance(mode, str):
                # tune shift/norm per parameter via the same BO loop as rho/sigma_in/tikh,
                # instead of leaving them fixed at identity (see class docstring above).
                # Size param_shift_range/param_norm_range from the real, final
                # input_parameters (not the caller's construction-time placeholder --
                # see __init__), unless the caller already customized them.
                if 'param_shift_range' not in vars(self) or 'param_norm_range' not in vars(self):
                    lo, hi = self._param_range(self.input_parameters)
                    half = max((hi - lo) / 2, 1e-6)
                    if 'param_shift_range' not in vars(self):
                        self.param_shift_range = (lo, hi)
                    if 'param_norm_range' not in vars(self):
                        self.param_norm_range = (half / 3, half * 3)
                # Reassign (not .append) so this becomes an instance attribute, since
                # hyperparameters_to_optimize is otherwise a shared mutable class default.
                param_hp_names = [f'param_shift_{i}' for i in range(N_param)] + \
                                  [f'param_norm_{i}' for i in range(N_param)]
                self.hyperparameters_to_optimize = self.hyperparameters_to_optimize + \
                    [name for name in param_hp_names if name not in self.hyperparameters_to_optimize]
        self.norm, self.shift = norm_obs, shift_obs

        return U_wtv, Y_wtv, U_test, Y_test


    # ___________________________________________________________________________________________ BAYESIAN OPTIMIZATION
    def _reset_hyperparams(self, params, names, tikhonov=None):
        """
        Updates specific hyperparameters with new values.

        Parameters
        ----------
        params : list
            List of hyperparameter values to set.
        names : list
            Names of the hyperparameters to update.
        tikhonov : float, optional
            Value to set for the Tikhonov regularization parameter.

        Outputs:
            None. Updates internal hyperparameter values.
        """
        N_obs = len(self.observed_idx)
        for hp, name in zip(params, names):
            if name in ('sigma_in', 'sigma_param'):
                setattr(self, name, 10 ** hp)
            elif name.startswith('param_shift_'):
                self._shift[N_obs + int(name.removeprefix('param_shift_'))] = hp
            elif name.startswith('param_norm_'):
                self._norm[N_obs + int(name.removeprefix('param_norm_'))] = hp
            else:
                setattr(self, name, hp)
        if tikhonov is not None:
            setattr(self, 'tikh', tikhonov)

    def _get_hyperparam(self, name):
        """
        Reads the current value of a hyperparameter by name, including the virtual
        param_shift_i/param_norm_i names that _reset_hyperparams writes directly into
        the shift/norm arrays for (see _reset_hyperparams -- there's no self.param_shift_i
        attribute, unlike e.g. self.rho).
        """
        N_obs = len(self.observed_idx)
        if name.startswith('param_shift_'):
            return self.shift[N_obs + int(name.removeprefix('param_shift_'))]
        if name.startswith('param_norm_'):
            return self.norm[N_obs + int(name.removeprefix('param_norm_'))]
        return getattr(self, name)

    @staticmethod
    def _hp_range_attr(hyper_param):
        """
        Maps a hyperparameter name to the class attribute holding its (min, max) search
        range. Per-parameter names (param_shift_0, param_norm_1, ...) share one range
        attribute (param_shift_range/param_norm_range) rather than one per index.
        """
        if hyper_param.startswith('param_shift_') or hyper_param.startswith('param_norm_'):
            return hyper_param.rsplit('_', 1)[0] + '_range'
        return hyper_param + '_range'


    def _optimize_hyperparameters(self, U_wtv, Y_wtv, validation_strategy=None, print_convergence=True):
        """
        Performs Bayesian hyperparameter optimization to minimize the validation loss.

        Parameters
        ----------
        U_wtv : np.ndarray
            Wash-train-validation input data.
        Y_wtv : np.ndarray
            Corresponding labels for train-validation data.
        validation_strategy : function, optional
            Validation function for hyperparameter tuning.
            Defaults to `_RVC_Noise`.

        Returns
        -------
            dict: ``res``, the result of the search (an ``OptimizeResult``), ``hp_names``, the names of its
            dimensions, and ``n_grid_points``, the number of points of its initial grid. With 'tikh' alone, the
            search space is empty, and the search is one evaluation of the validation strategy, which selects
            `tikh` from `tikh_range` (``hp_names = []``, ``n_grid_points = 1``, ``res.x = []``). Without 'tikh'
            in `hyperparameters_to_optimize`, every evaluation keeps the fixed `tikh`: the strategies read the
            grid ``tikh_range = [tikh]`` during the search, and `tikh_range` is restored afterwards.
        """
        if 'tikh' in self.hyperparameters_to_optimize:
            return self._search_hyperparameters(U_wtv, Y_wtv, validation_strategy, print_convergence)
        saved = self.__dict__.get('tikh_range', _UNSET)
        self.tikh_range = [self.tikh]
        try:
            return self._search_hyperparameters(U_wtv, Y_wtv, validation_strategy, print_convergence)
        finally:
            if saved is _UNSET:
                del self.tikh_range                 # the class default
            else:
                self.tikh_range = saved

    def _search_hyperparameters(self, U_wtv, Y_wtv, validation_strategy, print_convergence):
        """Body of `_optimize_hyperparameters`, with the grid of `tikh` that the strategies read."""
        # Prepare search grid, space, and hyperparameter names
        search_grid, search_space, hp_names = self._hyperparameter_search(print_convergence=print_convergence)
        # Track optimal Tikhonov regularization (one evaluation with 'tikh' alone, whatever N_func_evals)
        tikh_opt = np.zeros(max(self.N_func_evals, 1))

        # Use default or provided validation strategy
        if validation_strategy is None:
            validation_strategy = self._RVC_Noise

        # Prepare the validation function
        val_func = partial(validation_strategy,
                           case=self,
                           U_wtv=U_wtv.copy(),
                           Y_wtv=Y_wtv.copy(),
                           tikh_opt=tikh_opt,
                           hp_names=hp_names,
                           print_convergence=print_convergence
                           )

        if not search_space:
            # 'tikh' alone (issue #1): every evaluation of the strategy selects tikh from tikh_range, so the search
            # is one evaluation at the fixed rho, sigma_in and leak_rate; gp_minimize cannot fit a Gaussian process
            # on zero dimensions. The result has the fields of a skopt result that train() and the plots read.
            # Imported here: scipy.optimize slows the import of the package.
            from scipy.optimize import OptimizeResult

            fun = float(val_func([]))
            result = OptimizeResult(x=[], fun=fun, func_vals=np.array([fun]), x_iters=[[]], models=[])
        else:
            from skopt import gp_minimize
            from skopt.learning import GaussianProcessRegressor as GPR
            from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern

            # Configure ARD 5/2 Matern Kernel for Gaussian Process
            kernel_ = (ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0)) *
                       Matern(length_scale=[0.2] * len(search_space), nu=2.5, length_scale_bounds=(1e-2, 1e1)))

            # Gaussian Process reconstruction
            gp_estimator = GPR(kernel=kernel_,
                               normalize_y=True,
                               n_restarts_optimizer=3,
                               noise=self.bho_noise,
                               random_state=10)

            # Perform Bayesian Optimization
            result = gp_minimize(val_func,  # function to minimize
                                 search_space,  # bounds
                                 base_estimator=gp_estimator,  # GP kernel
                                 acq_func="gp_hedge",  # acquisition function
                                 n_calls=self.N_func_evals,  # number of evaluations
                                 x0=search_grid,  # Initial grid points
                                 n_initial_points=self.N_initial_rand,  # random initial points
                                 n_restarts_optimizer=3,  # tries per acquisition
                                 random_state=10)
        assert result is not None, 'gp_minimize retuned a None instance'
        # Process results
        f_iters = np.array(result.func_vals)
        best_idx = np.argmin(f_iters)

        # Update hyperparameters with the best result
        self._reset_hyperparams(result.x, hp_names, tikhonov=tikh_opt[best_idx])

        logger.info('%soptimum: %s, validation score %.4f',
                    '' if getattr(self, '_patch', None) is None else f'patch {self._patch}: ',
                    ', '.join([f'{name}={self._get_hyperparam(name):.3g}' for name in hp_names]
                              + [f'tikh={self.tikh:.0e}']), result.fun)  # type: ignore

        return dict(res=result,
                    hp_names=hp_names,
                    n_grid_points=len(search_grid))

    def _hyperparameter_search(self, print_convergence=True):
        """
        Prepares the search grid and search space for Bayesian hyperparameter optimization.
        TODO: add noise to the optional input_parameters to optimize.

        Returns
        -------
            tuple:
                - search_grid (list): List of initial grid points for optimization.
                - search_space (list): Search space objects for each hyperparameter.
                - hp_names (list): Names of the hyperparameters being optimized. 'tikh' is never one of them: with
                  'tikh' in `hyperparameters_to_optimize`, every evaluation of the validation strategy selects it
                  from `tikh_range`; without it, every evaluation keeps the fixed `tikh`
                  (`_optimize_hyperparameters`). With 'tikh' alone, the three lists are ``[[]]``, ``[]`` and ``[]``.
        """
        parameters = [hp for hp in self.hyperparameters_to_optimize if hp != 'tikh']
        if parameters:          # a search over 'tikh' alone needs no scikit-optimize
            from skopt.space import Real

        # Grid points per axis, shrunk (never grown) so the full factorial grid fits
        # within N_func_evals. A fixed N_grid blows up combinatorially once many
        # hyperparameters are in play (e.g. optimize_parameter_normalization adds 2
        # dims per parameter) -- N_grid=3 with 8 dims would be a 6561-point grid.
        # ponytail: coarser-per-axis is a blunt way to keep this fast; a real
        # high-dimensional design (e.g. Latin hypercube) would use the eval budget
        # better if this ever needs finer resolution with many parameters.
        n_grid_eff = self.N_grid if not parameters else \
            max(1, min(self.N_grid, int(self.N_func_evals ** (1.0 / len(parameters)))))

        param_grid, search_space = [], []
        for hyper_param in parameters:
            range_ = getattr(self, self._hp_range_attr(hyper_param))  # type: tuple[float,float]
            param_grid.append(np.linspace(*range_, 2 * n_grid_eff + 1)[1::2] if self.grid_midpoints
                              else np.linspace(*range_, n_grid_eff))
            search_space.append(Real(*range_, name=hyper_param))

        # The first n_grid_eff^len(parameters) points are from grid search
        search_grid = product(*param_grid, repeat=1)
        search_grid = [list(sg) for sg in search_grid]

        # Print optimization header
        if print_convergence:
            patch = '' if getattr(self, '_patch', None) is None else f' (patch {self._patch})'
            if parameters:
                n_bayes = max(0, self.N_func_evals - len(search_grid))
                logger.info('hyperparameter search: a %d^%d grid and %d point%s of Bayesian optimization%s',
                            n_grid_eff, len(parameters), n_bayes, '' if n_bayes == 1 else 's', patch)
            else:
                n = len(self.tikh_range)
                logger.info('hyperparameter search: tikh alone, selected from the %d value%s of tikh_range in one '
                            'evaluation, without Bayesian optimization%s', n, 's' if n != 1 else '', patch)
            logger.info('%s', validation.convergence_header(self))

        return search_grid, search_space, parameters

    # ___________________________________________________________________________________________ NORMALIZATION METHODS

    @staticmethod
    def _set_norm(train_data, method=None):
        """
        Computes the normalization factor for the input data.
        Parameters
        ----------
        train_data : np.ndarray
            Wash-train-validation training input data. (Nens x Nt x Ndim).
        Returns
        -------
            float: Normalization factor based on the range of the input data.
        """
        # assert train_data.ndim in [3, 4], f'U_wtv must be a 3D array, got {train_data.ndim}D: ({train_data.shape})'

        if train_data.ndim == 3:
            L, _, Ndim = train_data.shape
            Nens = 1
        elif train_data.ndim == 4:
            L, _, Ndim, Nens = train_data.shape
        elif train_data.ndim == 2:
            L = 1
            Nens = 1
            Ndim = train_data.shape[1]
        else:
            raise ValueError(f'U_wtv must be a 2D, 3D or 4D array, got {train_data.ndim}D: ({train_data.shape})')

        if method is None:
            return np.ones(Ndim), np.zeros(Ndim)

        shift = np.mean(train_data, axis=1)

        shifted_data  = train_data - shift[:, np.newaxis, :]

        if method == 'std':
            shift = np.mean(train_data, axis=1)
            norm = np.std(shifted_data, axis=1)
        elif method == 'max':
            norm = np.max(shifted_data, axis=1)
        elif method == 'mean':
            norm = np.mean(abs(shifted_data), axis=1)
        elif method == 'range':
            m = np.min(shifted_data, axis=1)
            M = np.max(shifted_data, axis=1)
            norm = M - m
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if L > 1:
            norm = np.mean(norm, axis=0)
            shift = np.mean(shift, axis=0)
        if Nens > 1:
            norm = np.mean(norm, axis=-1)
            shift = np.mean(shift, axis=-1)

        if np.any(abs(norm) < 1e-12):
            norm[abs(norm) < 1e-12] = 1.0  # Prevent division by zero

        return norm, shift

    # ___________________________________________________________________________________________ VALIDATION STRATEGIES

    # Implemented as module-level functions in validation.py; aliased as
    # staticmethods so existing references keep working: self._RVC_Noise is
    # train()'s default. The qlESN segment strategies live in
    # qlroms.data_driven_qlroms.validation (they are dwell-corpus specific).
    _RVC_Noise = staticmethod(validation.RVC_Noise)
    _single_series_validation = staticmethod(validation.single_series_validation)
    _SSV = staticmethod(validation.SSV)
    _WFV = staticmethod(validation.WFV)
    _KFV = staticmethod(validation.KFV)

    def compute_nMAE(self, Y_true, Y_pred, norm=1.0):
        r"""Error metric used throughout training/validation/testing: the normalized
        mean absolute error
        $\mathrm{mean}(|\mathbf{Y}_\mathrm{true} - \mathbf{Y}_\mathrm{pred}|) /
        \mathrm{mean}(|\texttt{norm}|)$.

        Parameters
        ----------
        Y_true : np.ndarray
            Ground-truth values.
        Y_pred : np.ndarray
            Predicted values, same shape as `Y_true`.
        norm : float or np.ndarray
            Normalization factor (e.g. the data range per component). Default 1.0.

        Returns
        -------
        float
            Normalized error.
        """
        return np.mean(np.abs(Y_true - Y_pred)) / np.mean(np.abs(norm))

    def compute_nRMSE(self, Y_true, Y_pred, norm=1.0):
        """Deprecated alias of `compute_nMAE` -- the metric was never an RMSE."""
        _warn('compute_nRMSE computes a normalized MAE; use compute_nMAE', DeprecationWarning)
        return self.compute_nMAE(Y_true, Y_pred, norm)

    # _______________________________________________________________________________________ TEST & PLOTTING FUNCTIONS


    def run_test(self,
                 U_test,
                 Y_test,
                 pdf_file=None,
                 Nt_test=None,
                 max_L_tests=None,
                 nbins=20,
                 max_short_tests=None,
                 long_term=True,
                 short_term=True,
                 ):
        """Evaluate the trained ESN on test data with closed-loop forecasts, logging the
        error metrics (logger 'echostatenetwork', silent by default; see `set_verbose`) and
        building diagnostic figures.

        Parameters
        ----------
        U_test : np.ndarray
            Test input data, shape ``(L, Nt, N_dim)``.
        Y_test : np.ndarray
            Ground-truth outputs for the test data, same leading shape as `U_test`.
        pdf_file : PdfPages, optional
            Currently unused by this method (present for API compatibility).
        Nt_test : int, optional
            Length of each short-term test, in ESN steps. Defaults to `N_val`.
        max_L_tests : int, optional
            Maximum number of segments (of the `L` available) to evaluate, each at most once; with more
            segments, `run_test` draws `max_L_tests` of them with the generator `rng`. Defaults to the
            attribute `max_L_tests` (10).
        nbins : int
            Number of bins for the prediction PDFs (long-term test only). Default 20.
        max_short_tests : int, optional
            Maximum number of short-term test figures to draw per tested segment. Defaults to the attribute
            `max_short_tests` (10). Every short-term test of the test tail is scored, whether it is drawn or not.
        long_term : bool
            If True, run one long-term (full-length) closed-loop forecast per tested
            segment and plot it. Default True.
        short_term : bool
            If True, run short-term (`Nt_test`-long) closed-loop forecasts over the whole test tail of every
            tested segment, and plot up to `max_short_tests` of them per segment. Default True.

        Returns
        -------
        list
            ``[fig_long] + figures_short``: the long-term test figure (None if `long_term` is False or more than
            one segment is tested) followed by the short-term test figures of every tested segment (``[None]`` if
            `short_term` is False). ``[]`` when no segment can be tested.
        """

        if max_L_tests is None:
            max_L_tests = self.max_L_tests
        if max_short_tests is None:
            max_short_tests = self.max_short_tests
        if max_L_tests < 1:
            _warn('run_test: max_L_tests < 1; no test is run.')
            return []
        if Nt_test is None:
            Nt_test = self.N_val

        # U_test/Y_test may be a regular ndarray or a list of L segments of possibly
        # different length (e.g. cluster-dwell chunks) -- normalize to a list either way.
        if isinstance(U_test, np.ndarray):
            if U_test.ndim == 1:
                U_test = U_test[np.newaxis, :, np.newaxis]
            elif U_test.ndim == 2:
                U_test = U_test[np.newaxis, :, :]
            U_test = list(U_test)
        if isinstance(Y_test, np.ndarray):
            if Y_test.ndim == 1:
                Y_test = Y_test[np.newaxis, :, np.newaxis]
            elif Y_test.ndim == 2:
                Y_test = Y_test[np.newaxis, :, :]
            Y_test = list(Y_test)

        rng0 = self.rng

        L, Nq = len(U_test), U_test[0].shape[1]

        if Nq > 10:
            nrows, dims = 10, rng0.choice(Nq, 10, replace=False)
        else:
            nrows, dims = self.N_dim, np.arange(Nq)
            if Nq == 1:
                dims = [dims]

        observed_idx_np = np.array(self.observed_idx)

        # Select test cases (with a maximum of max_L_tests), restricted to segments
        # with enough test-tail length: the long-term washout+forecast in predict_Y
        # needs len(Y_test_l) >= 2*N_wash (segments may differ in length -- e.g.
        # cluster-dwell chunks -- so this isn't guaranteed for every Li).
        min_needed = 2 * self.N_wash if long_term else self.N_wash
        testable = [Li for Li in range(L) if len(Y_test[Li]) >= min_needed]
        if not testable:
            _warn(f'run_test: no segment has >= {min_needed} test steps; no test is run.')
            return []

        if len(testable) > 1:
            if max_L_tests < len(testable):         # each segment is tested at most once
                L_indices = np.sort(rng0.choice(testable, max_L_tests, replace=False))
            else:
                L_indices = np.array(testable)
        else:
            L_indices = [testable[0]]

        N_ens = U_test[0].shape[-1] if U_test[0].ndim == 3 else 1
        # Prediction function
        def predict_Y(_input, _target):

            # Perform washout (open-loop without extra forecast step)
            r_out = np.zeros((self.N_r, N_ens))
            u_out = np.zeros((self.N_dim, N_ens))
            # sized by the washout, not by the target: a short-term test whose window is
            # shorter than N_wash would otherwise overflow this buffer (IndexError)
            u_open = np.zeros((self.N_wash,) + _target.shape[1:])


            for ii, u_in in enumerate(_input[:self.N_wash]):
                u_out, r_out = self.step(u_in, r_out)
                try:
                    u_open[ii] = u_out.squeeze()
                except Exception:
                    u_open[ii] = u_out.copy()


            Y_closed = np.zeros_like(_target)

            # Reroll input_parameters to the segment's own value for this run, then
            # restore the original (N_param, L) array so callers see the full parameter
            # set again afterwards (e.g. a later re-format of the training data).
            original_input_parameters = self.input_parameters
            if self.input_parameters is not None:
                # Parameter (e.g. cluster id) may vary within _input (a segment straddling
                # a transition); read off its own first step and hold it fixed for this run.
                N_param = self._n_param(self.input_parameters)
                self.input_parameters = _input[0, -N_param:].reshape(N_param, N_ens)

            try:
                for i in range(Y_closed.shape[0]):
                    u_input = self.outputs_to_inputs(full_state=u_out)
                    u_out, r_out = self.step(u_input, r_out)
                    try:
                        Y_closed[i] = u_out.squeeze()
                    except Exception:
                        Y_closed[i] = u_out.copy()
            finally:
                self.input_parameters = original_input_parameters

            return Y_closed, u_open

        # Plotting function
        def plot_time(_axs, _time, _pred_closed, _pred_open, _inputs, _target, _err=None):
            if not isinstance(_axs, (list, np.ndarray)):
                _axs = [_axs]

            t_wash_in = _time[:self.N_wash] - self.dt_ESN
            t_wash_out = _time[:self.N_wash]
            t_out = _time[self.N_wash:]

            for dim_i, _ax in zip(range(self.N_dim), _axs):
                _ax.plot(t_out, _target[:, dim_i], 'k', label=f'truth dim {dim_i}')
                # Plot the input if observed
                if dim_i in self.observed_idx:
                    _i = np.argmin(abs(observed_idx_np-dim_i))
                    _ax.plot(t_wash_in, _inputs[:self.N_wash, _i], 'x', c='C4', ms=5, label='Washout')

                _ax.plot(t_wash_out, _pred_open[:, dim_i], '-co', mfc='none', label='ESN open loop')
                _ax.plot(t_out, _pred_closed[:, dim_i], '--r', dashes=[2, .5],
                         label=[f'ESN closed-loop prediction \n error = {_err:.4}' if _err is not None else 'ESN closed-loop prediction'])
                _ax.set(ylabel=f'$u_{dim_i}$')
                _ax.set(ylim=ylims[dim_i])

        test_counter, errors_all, figures_long, figures_short = 0, [], [], []
        hist_args = dict(bins=nbins, density=True, orientation='horizontal', stacked=False)


        for Li in L_indices:

            # Select dataset
            U_test_l, Y_test_l = U_test[Li], Y_test[Li]


            norm_l = np.max(Y_test_l, axis=0) - np.min(Y_test_l, axis=0)

            t_l = (np.arange(U_test_l.shape[0])) * self.dt_ESN
            # set ylims for plotting
            ylims = [[np.min(Y_test_l[:, dim_i])*1.05, np.max(Y_test_l[:, dim_i])*1.05] for dim_i in range(self.N_dim)]

            # plot tests statistics if the test dataset is long or requested
            if long_term:

                # predict over the entire test set
                Y_closed, U_open = predict_Y(U_test_l[:-1], Y_test_l[self.N_wash:])

                err_long = np.log10(self.compute_nMAE(Y_closed, Y_test_l[self.N_wash:], norm=norm_l))

                fig_long, grid = plt.subplots(nrows=self.N_dim, ncols=2, figsize=[10, 2.5 * self.N_dim],
                                         sharex='col', sharey='row', layout='tight', width_ratios=[5, 1])

                if self.N_dim == 1:
                    axs, axs_pdf = [grid[0]], [grid[1]]
                else:
                    axs, axs_pdf = grid[:, 0], grid[:, 1]

                plot_time(_axs=axs,
                          _time=t_l,
                          _pred_closed=Y_closed,
                           _pred_open=U_open,
                          _inputs=U_test_l,
                          _target=Y_test_l[self.N_wash:],)

                # Plot histograms]
                for dim_i, ax_2 in enumerate(axs_pdf):
                    if dim_i in self.observed_idx:
                        _i = np.argmin(abs(observed_idx_np - dim_i))
                        ax_2.hist(U_test_l[:, _i], color='k', lw=2, alpha=0.6, histtype='step', **hist_args)

                    ax_2.hist(Y_test_l[:, dim_i], color='k', lw=.85, histtype='step', **hist_args)
                    ax_2.hist(Y_closed[:, dim_i], color='r', ls='--', histtype='stepfilled', alpha=0.5, **hist_args)
                    ax_2.hist(Y_closed[:, dim_i], color='r', ls='--', histtype='step', **hist_args)

                # axs[0].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0))
                plt.suptitle(f'Li = {Li}, observed idx = {self.observed_idx}, error = {err_long:.4}')
                axs[-1].set(xlabel='$t/T$')
                figures_long.append(fig_long)


            if short_term:
                i0 = 0 # reset time index for each Li
                n_figures = 0                      # the figures of this segment, at most max_short_tests
                short_term_error = 0.
                max_test_time = U_test_l.shape[0]  # per-segment: segments may differ in length

                while i0 + Nt_test < max_test_time:     # every window of the test tail is scored
                    test_counter += 1

                    i1 = i0 + Nt_test + self.N_wash

                    current_input = U_test_l[i0:i1-1].copy()
                    current_target = Y_test_l[i0+self.N_wash:i1].copy()
                    current_time = t_l[i0:i1]

                    # predict
                    Y_closed, U_open = predict_Y(current_input, current_target)

                    current_error = np.log10(self.compute_nMAE(current_target, Y_closed, norm=norm_l))

                    short_term_error += current_error


                    if n_figures < max_short_tests:
                        fig_short, axs_short = plt.subplots(nrows=nrows, ncols=1, figsize=[8, 1.5 * nrows], sharex='all', layout='tight')
                        if nrows == 1:
                            axs_short = [axs_short]

                        plot_time(_axs=axs_short, _time=current_time, _pred_closed=Y_closed, _pred_open=U_open,
                                  _inputs=current_input, _target=current_target, _err=current_error)


                        axs_short[0].legend(title=f'Test {test_counter}: Li = {Li}', loc='upper left',
                                            bbox_to_anchor=(1, 1), fontsize='x-small')
                        axs_short[-1].set(xlabel='$t/T$')

                        figures_short.append(fig_short)
                        n_figures += 1
                    i0 += Nt_test

                errors_all.append(short_term_error / max(1, (i0 // Nt_test)))

        fig_long = figures_long[0] if len(figures_long) == 1 else None
        if not figures_short:
            figures_short = [None]

        # Compute errors over all Lis
        if test_counter > 0:
            errors_all = np.array(errors_all)
            logger.info('run_test: log10 nMAE of the short closed-loop tests (%d tests), mean per segment over %d '
                        'segments: min %.3g, max %.3g, mean %.3g', test_counter, len(errors_all),
                        np.min(errors_all), np.max(errors_all), np.mean(errors_all))

        return [fig_long] + figures_short


    def _plot_training_results(self, U_test, Y_test, results, save_ESN_training, folder):
        """
        Plots training results, including Bayesian optimization convergence and test results.
        `results` is None when no search ran; with 'tikh' alone (one evaluation), there is no convergence to plot.
        """
        all_figs = []

        if results is not None and results['hp_names']:
            from skopt.plots import plot_convergence

            # Plot Bayesian optimization convergence
            fig1 = plt.figure()
            plot_convergence(results['res'])
            all_figs.append(fig1)
            # Plot Gaussian Process reconstruction
            all_figs.extend(self._plot_BO(results))
        # Plot Wout matrix
        all_figs.append(self.plot_Wout())
        # Plot test results if applicable (run_test itself skips gracefully if no
        # segment has enough test-tail length). Capped by max_L_tests/max_short_tests:
        # run_test makes one long-term figure per tested segment plus one per
        # short-term test, so a multi-segment (e.g. parametric) network otherwise
        # emits dozens of figures.
        max_test_len = max(len(u) for u in U_test) if isinstance(U_test, list) else U_test.shape[1]
        if self.perform_test and max_test_len >= self.N_val:
            test_figs = self.run_test(U_test, Y_test,
                                      long_term=True, short_term=True,
                                      max_L_tests=self.max_L_tests,
                                      max_short_tests=self.max_short_tests)
            all_figs = all_figs + test_figs

        if save_ESN_training:
            if folder is None:
                folder = self.figs_folder
            os.makedirs(folder, exist_ok=True)
            save_pdf = plt_pdf.PdfPages(f'{folder}{self.filename}_Training.pdf')
            [add_pdf_page(save_pdf, fig) for fig in all_figs]
            save_pdf.close()



    def _plot_BO(self, results_bayesian_optimization):
        """Visualize the Bayesian-optimization Gaussian-process reconstruction as one
        subplot per consecutive hyperparameter pair, all pairs sharing a single figure
        (parametric ESNs can optimize a dozen hyperparameters -- one figure each is
        unreadable).

        Parameters
        ----------
        results_bayesian_optimization : dict
            Dictionary with ``hp_names`` (labels of the optimized hyperparameters),
            ``res`` (the `skopt` result carrying the GP reconstruction) and
            ``n_grid_points`` (initial grid evaluations, marked differently from the
            BO-acquired points).

        Returns
        -------
        list
            ``[fig]`` with the shared figure, or ``[]`` when fewer than two
            hyperparameters were optimized (nothing to contour).
        """

        hp_names = results_bayesian_optimization['hp_names']
        res = results_bayesian_optimization['res']
        n_grid_points = results_bayesian_optimization['n_grid_points']

        f_iters = np.array(res.func_vals)

        if len(hp_names) < 2:  # nothing to contour
            return []

        gp = res.models[-1]
        res_x = np.array(res.x_iters)
        best_x = res.x  # best-found point in the full (possibly >2D) hyperparameter space
        best_idx = np.argmin(f_iters)
        amin = np.amin([10, np.max(f_iters)])
        n_len = 100  # points to evaluate the GP at

        # All consecutive hyperparameter pairs share one figure (parametric ESNs can
        # optimize a dozen hyperparameters -- one figure each is unreadable).
        n_pairs = len(hp_names) - 1
        ncols = min(3, n_pairs)
        nrows = int(np.ceil(n_pairs / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), layout='constrained')
        axs = np.atleast_1d(axs).ravel()

        for hpi, ax in enumerate(axs[:n_pairs]):
            range_1 = getattr(self, self._hp_range_attr(hp_names[hpi]))  # type: tuple[float, float]
            range_2 = getattr(self, self._hp_range_attr(hp_names[hpi + 1]))  # type: tuple[float, float]

            xx, yy = np.meshgrid(np.linspace(*range_1, n_len), np.linspace(*range_2, n_len))

            # res.space is the full len(hp_names)-D search space, so every evaluated point
            # needs one value per hyperparameter, not just the 2 being plotted here -- hold
            # every other dimension fixed at its best-found value (a partial-dependence slice)
            x_x = np.tile(best_x, (xx.size, 1)).astype(float)
            x_x[:, hpi] = xx.ravel()
            x_x[:, hpi + 1] = yy.ravel()
            x_gp = res.space.transform(x_x.tolist())  # gp prediction needs norm. format

            # Final GP reconstruction for each realization at the evaluation points
            y_pred = np.clip(-gp.predict(x_gp), a_min=-amin, a_max=-np.min(f_iters)).reshape(n_len, n_len)

            cf = ax.contourf(xx, yy, y_pred, levels=20, cmap='Blues')
            ax.contour(xx, yy, y_pred, levels=20, colors='black', linewidths=1, linestyles='solid', alpha=0.3)
            fig.colorbar(cf, ax=ax, label='-$\\log_{10}$(MSE)')
            #   Plot the n_tot search points (columns hpi/hpi+1 -- the pair being
            #   visualized in *this* subplot, not always the first two hyperparameters)
            for rx, mk in zip([res_x[:n_grid_points], res_x[n_grid_points:]], ['v', 's']):
                ax.plot(rx[:, hpi], rx[:, hpi + 1], mk, c='w', alpha=.8, mec='k', ms=8)
            ax.plot(res_x[best_idx, hpi], res_x[best_idx, hpi + 1], '*r', alpha=.8, mec='r', ms=8)
            ax.set(xlabel=hp_names[hpi], ylabel=hp_names[hpi + 1])

        for ax in axs[n_pairs:]:
            ax.axis('off')

        return [fig]


    def plot_Wout(self):
        """Visualize the trained read-out matrix `Wout`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with a single heat-map axis of ``Wout.T``.
        """
        fig, ax = plt.subplots()
        im = ax.matshow(self.Wout.T, cmap="PRGn", aspect=4., vmin=-np.max(self.Wout), vmax=np.max(self.Wout))
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.colorbar(im, orientation='horizontal', extend='both')
        ax.set(ylabel='$N_u$', xlabel='$N_r$', title='$\\mathbf{W}_\\mathrm{out}$')
        return fig




if __name__ == "__main__":

    print(vars(EchoStateNetwork))

    pass
