"""User-facing messages (issue #7). Warnings point at the line that calls echostatenetwork (not inside the package,
nor at scikit-optimize, which calls the validation strategy back) and appear once per call, however many seeds,
patches or evaluations of the search raise them: a keyword of the constructor or of `train` that the network ignores,
fewer validation folds than requested (WFV, KFV) and a degenerate validation window. The reports (the search, the
summary of `train` and the diagnostics) go to the logger 'echostatenetwork', silent by default and shown by
`set_verbose`; `train` trains a trained network again only with `force=True`."""
import inspect
import logging
import re
import warnings

import numpy as np
import pytest

import echostatenetwork
from echostatenetwork import EchoStateNetwork, validation

DATA = np.random.default_rng(0).normal(size=(200, 4)).cumsum(axis=0) * 0.1
QUICK = dict(dt=1, N_units=15, upsample=1, t_train=120, t_val=30, t_test=40, N_wash=5, hyperparameters_to_optimize=[],
             seed=0)
RETRAIN = 'training the trained network again'   # a diagnostic message: train(force=True)
SUMMARY = 'trained:'                             # the start of the training summary, the report of train
FOLDS = 'of the requested N_folds'               # fewer folds fit than WFV or KFV request
DEGENERATE = 'inferred validation window is degenerate'


@pytest.fixture(autouse=True)
def quiet():
    yield
    echostatenetwork.set_verbose(False)


def _next_line():
    """Number of the line after the line of the caller."""
    return inspect.currentframe().f_back.f_lineno + 1


def _found(records, match='are ignored'):
    """The warnings among `records` whose message contains `match` (by default, about ignored keywords)."""
    return [w for w in records if match in str(w.message)]


def _serial(*args, **kwargs):
    """The default validation strategy; wrapped in a lambda, it does not pickle, so the seeds train serially."""
    return validation.RVC_Noise(*args, **kwargs)


def _series_esn(**kwargs):
    """A small network with a search of three evaluations and 100 requested folds, of which 4 fit in `DATA`."""
    options = {**QUICK, 'N_folds': 100, 'N_grid': 2, 'N_func_evals': 3, 'hyperparameters_to_optimize': ['rho'],
               **kwargs}
    if set(options['hyperparameters_to_optimize']) - {'tikh'}:
        pytest.importorskip('skopt')        # a Bayesian search needs scikit-optimize, an optional dependency
    return EchoStateNetwork(DATA.T, **options)


class _Subclass(EchoStateNetwork):
    """A subclass defined outside echostatenetwork that passes its keywords on, as romda's ESN_model does."""

    def __init__(self, y, **kwargs):
        self.line = _next_line()
        super().__init__(y, **kwargs)


class _WithOption(EchoStateNetwork):
    """A subclass with an option of its own, which a plain EchoStateNetwork does not hold."""
    my_option = 0


class _CallableOption(EchoStateNetwork):
    """A subclass that holds an option of EchoStateNetwork as a callable of the class."""
    validation_metric = staticmethod(validation.nMSE)


# ------------------------------------------------------------------------------------------- ignored keywords
def test_a_typo_warns_at_the_caller_and_keeps_the_default():
    match = re.escape("EchoStateNetwork: keyword(s) ['N_unit'] are not attributes of EchoStateNetwork and are ignored")
    with pytest.warns(UserWarning, match=match) as rec:
        line = _next_line()
        esn = EchoStateNetwork(DATA.T, dt=1, N_unit=50)
    (w,) = _found(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert esn.N_units == EchoStateNetwork.N_units and 'N_unit' not in vars(esn)
    # through a subclass, at the line of the subclass
    with pytest.warns(UserWarning, match=re.escape("_Subclass: keyword(s) ['rhoo']")) as rec:
        esn = _Subclass(DATA.T, rhoo=0.5)
    (w,) = _found(rec)
    assert (w.filename, w.lineno) == (__file__, esn.line) and esn.rho == EchoStateNetwork.rho
    # in train
    esn = EchoStateNetwork(DATA.T, **QUICK)
    match = re.escape("EchoStateNetwork.train: keyword(s) ['rho_rnage'] are not attributes of the network and are "
                      "ignored (did you mean 'rho_range' for 'rho_rnage'?)")
    with pytest.warns(UserWarning, match=match) as rec:
        line = _next_line()
        esn.train(DATA, plot_training=False, rho_rnage=(0.5, 0.6))
    (w,) = _found(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert esn.rho_range == EchoStateNetwork.rho_range and 'rho_rnage' not in vars(esn) and esn.trained
    # a method (copy) or a read-only property (trained) is not set
    with pytest.warns(UserWarning) as rec:
        esn = EchoStateNetwork(DATA.T, **QUICK, copy=1, trained=True)
    (w,) = _found(rec)
    assert "['copy'] name methods of EchoStateNetwork" in str(w.message)
    assert "['trained'] are read-only properties" in str(w.message) and callable(esn.copy) and not esn.trained
    # in code run by exec, whose globals have no module name
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        exec('EchoStateNetwork(Y, N_unit=3)', {'EchoStateNetwork': EchoStateNetwork, 'Y': DATA.T})
    (w,) = _found(rec)
    assert w.filename == '<string>'


@pytest.mark.parametrize('kwargs, hint', [
    (dict(N_unit=50), "(did you mean 'N_units' for 'N_unit'?)"),
    (dict(traned=True), None),                                  # not 'trained', a read-only property: options only
    (dict(train=True, n_seed=2), "(did you mean 'n_seeds' for 'n_seed'?)"),       # the parameters of train
    (dict(train=True, dt=0.5), "(did you mean 'dt_ESN' for 'dt'? train sets the time step of the data, dt, with "
                               "dt_ESN = dt * upsample.)"),
])
def test_the_hint_names_the_closest_option(kwargs, hint):
    kwargs = dict(kwargs)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        if kwargs.pop('train', False):
            esn = EchoStateNetwork(DATA.T, **QUICK)
            esn.train(DATA, plot_training=False, **kwargs)
            assert esn.dt_physical == 1
        else:
            EchoStateNetwork(DATA.T, dt=1, **kwargs)
    (w,) = _found(rec)
    assert (hint in str(w.message)) if hint else 'did you mean' not in str(w.message)


def test_known_keywords_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        esn = EchoStateNetwork(DATA.T, **QUICK, rho=0.8, Win_type='sparse_dense_bias', val_k=0)
        esn.train(DATA, plot_training=False, sigma_in=0.01, N_func_evals=5, tikh_range=[1e-6, 1e-8])
        esn.train(DATA, plot_training=False, rho=0.7, Win=esn.Win.toarray(), force=True)  # held once trained
    assert esn.rho == 0.7 and esn.sigma_in == 0.01
    # an option that a subclass holds as a staticmethod is set by train, without a warning
    esn = _CallableOption(DATA.T, **QUICK)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        esn.train(DATA, plot_training=False, validation_metric=validation.log_nMAE)
    assert esn.validation_metric is validation.log_nMAE
    # an option of a subclass stays on the parallel network: the networks of the patches neither receive it nor warn
    esn = _WithOption(DATA.T, **QUICK, patch_size=2, halo=1, shared=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warnings.filterwarnings('ignore', category=DeprecationWarning)          # of scikit-optimize
        esn.train(DATA, plot_training=False, my_option=5)
    assert esn.my_option == 5 and all('my_option' not in vars(net) for net in esn.patches)


@pytest.mark.parametrize('layout, kwargs, message', [
    ({}, dict(Win=np.ones((15, 5))), "['Win'] are attributes that the network does not hold yet and are ignored (set "
                                     "Win on the network before calling train, which keeps them)"),
    (dict(patch_size=2, halo=1, shared=False), dict(Win=np.ones((15, 5))),
     "['Win'] are attributes that the network does not hold yet and are ignored (with independent patches, the "
     "network of every patch holds its own, in patches[p])"),
    ({}, dict(Wout=np.zeros((16, 4))), "['Wout'] are computed by train and are ignored"),
    ({}, dict(step=1), "['step'] name methods of the network"),
    ({}, dict(patch_sites=None, N_train=5), "['N_train', 'patch_sites'] are read-only properties of the network"),
], ids=['not_held_yet', 'held_by_the_patches', 'computed_by_train', 'method', 'read_only'])
def test_a_keyword_that_train_cannot_set_says_why(layout, kwargs, message):
    # a getter that raises (patch_sites without patches) does not prevent the warning
    esn = EchoStateNetwork(DATA.T, **QUICK, **layout)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        line = _next_line()
        esn.train(DATA, plot_training=False, **kwargs)
    (w,) = _found(rec)
    assert message in str(w.message) and (w.filename, w.lineno) == (__file__, line) and esn.trained
    if 'Win' in kwargs and not layout:
        assert not np.array_equal(esn.Win.toarray(), kwargs['Win'])     # drawn from the seed, as without Win
    if 'Wout' in kwargs:
        assert np.abs(esn.Wout).max() > 0


@pytest.mark.parametrize('layout, n_seeds, serial', [
    ({}, 3, True), ({}, 2, False), (dict(patch_size=2, halo=1, shared=False), 1, False),
    (dict(patch_size=2, halo=1, shared=False), 2, True)],
    ids=['serial_seeds', 'parallel_seeds', 'patches', 'patches_and_serial_seeds'])
def test_an_ignored_keyword_warns_once_per_call(layout, n_seeds, serial):
    esn = EchoStateNetwork(DATA.T, **QUICK, **layout)
    strategy = (lambda *a, **k: _serial(*a, **k)) if serial else None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        line = _next_line()
        esn.train(DATA, plot_training=False, n_seeds=n_seeds, typo_key=1, validation_strategy=strategy)
    (w,) = _found(rec)
    assert (w.filename, w.lineno) == (__file__, line) and esn.trained


# --------------------------------------------------------------------- a trained network, and direct calls
def test_a_trained_network_is_not_trained_again_without_force():
    esn = _WithOption(DATA.T, **QUICK)
    esn.train(DATA, plot_training=False)
    Wout = esn.Wout.copy()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        esn.train(DATA, plot_training=False, rho=0.5, typo_key=1)       # skipped: rho and Wout unchanged
    assert len(_found(rec)) == 1                                         # and the keyword is still named
    assert _found(rec, '_WithOption.train: the network is already trained and is not trained again; pass force=True')
    assert esn.rho == 0.9 and np.array_equal(esn.Wout, Wout)
    esn.train(DATA, plot_training=False, rho=0.5, force=True)
    assert esn.rho == 0.5 and not np.array_equal(esn.Wout, Wout)


def test_warnings_of_direct_calls_point_at_the_caller():
    assert not hasattr(EchoStateNetwork, 'verbose')
    with pytest.warns(FutureWarning, match="'verbose' is removed") as rec:
        lines = [_next_line()]
        esn = EchoStateNetwork(DATA.T, **QUICK, verbose=True)
    with pytest.warns(FutureWarning, match=r"EchoStateNetwork\.train: the keyword 'verbose' is removed") as train:
        lines.append(_next_line())
        esn.train(DATA, plot_training=False, verbose=False)              # not the warning of an unknown keyword
    assert esn.trained and 'verbose' not in vars(esn)
    with pytest.warns(UserWarning, match='not trained again') as again:
        lines.append(_next_line())
        esn.train(DATA, plot_training=False)
    with pytest.warns(DeprecationWarning, match='use compute_nMAE') as alias:
        lines.append(_next_line())
        assert esn.compute_nRMSE(DATA[:5], DATA[5:10], norm=2.0) == esn.compute_nMAE(DATA[:5], DATA[5:10], norm=2.0)
    short = DATA[:3]
    with pytest.warns(UserWarning, match='no test is run') as test:
        lines.append(_next_line())
        assert esn.run_test(short[None, :-1], short[None, 1:]) == []
    records = [*rec, *train, *again, *alias, *test]
    assert [(w.filename, w.lineno) for w in records] == [(__file__, line) for line in lines]


def _legacy_metric(case, Y_true, Y_pred, norm=1.0):
    """A validation metric of the user that calls a deprecated function of the package."""
    error = case.compute_nRMSE(Y_true, Y_pred, norm)
    return float(np.log10(error)) if np.isfinite(error) and error > 0 else 10.0


def test_a_warning_from_a_callback_of_the_user_points_at_the_callback_once_per_call():
    # the first frame outside the package, skipping scikit-optimize: the line of the callback that calls the package,
    # not the line that calls train, once however many evaluations of the search call the callback
    esn = _series_esn(N_folds=4, validation_metric=_legacy_metric)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        esn.train(DATA, plot_training=False)
    (w,) = _found(rec, 'use compute_nMAE')
    assert (w.filename, w.lineno, w.category) == (__file__, _legacy_metric.__code__.co_firstlineno + 2,
                                                  DeprecationWarning)


# ------------------------------------------------------------------------- warnings of the training (issue #7, item 5)
@pytest.mark.parametrize('options, n_seeds, strategy', [
    ({}, 1, validation.WFV),                                                   # called back by scikit-optimize
    ({}, 3, validation.KFV),                                                   # in three processes
    (dict(hyperparameters_to_optimize=[]), 3, validation.WFV),                  # the score of every seed
    (dict(patch_size=2, halo=1, shared=False), 1, validation.WFV),             # the search of every patch
    (dict(patch_size=2, halo=1, shared=False), 2, validation.WFV),             # every patch and seed
    ({}, 2, lambda *a, **k: validation.WFV(*a, **k)),                          # serial seeds: it does not pickle
], ids=['search', 'seeds', 'seeds_without_search', 'patches', 'patches_and_seeds', 'serial_seeds'])
def test_fold_count_warning_once_at_the_caller(options, n_seeds, strategy):
    esn = _series_esn(**options)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        line = _next_line()
        esn.train(DATA, plot_training=False, validation_strategy=strategy, n_seeds=n_seeds)
    (w,) = _found(rec, FOLDS)                       # not once per seed, nor per evaluation of the search
    assert (w.filename, w.lineno, w.category) == (__file__, line, UserWarning)
    assert re.match(r'(WFV|KFV): only 4 of the requested N_folds=100', str(w.message)) and esn.trained


def test_warning_of_the_seeds_follows_the_filters_of_the_caller():
    esn = _series_esn()
    with pytest.warns(UserWarning, match=FOLDS):    # issued by this process, not by the processes of the seeds
        esn.train(DATA, plot_training=False, validation_strategy=validation.KFV, n_seeds=2)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warnings.filterwarnings('ignore', category=DeprecationWarning)          # of scikit-optimize
        warnings.filterwarnings('ignore', message='.*' + FOLDS)
        esn.train(DATA, plot_training=False, validation_strategy=validation.KFV, n_seeds=2, force=True)
    with warnings.catch_warnings(), pytest.raises(UserWarning, match=FOLDS):
        warnings.simplefilter('error', UserWarning)
        esn.train(DATA, plot_training=False, validation_strategy=validation.KFV, n_seeds=2, force=True)


def test_every_call_of_train_warns_at_its_own_line():
    # once per call, not once per network or per process: the networks of the patches forget it between calls
    esn = _series_esn(patch_size=2, halo=1, shared=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('default')            # once per location, as Python shows warnings by default
        lines = [_next_line()]
        esn.train(DATA, plot_training=False, validation_strategy=validation.WFV)
        lines.append(_next_line())
        esn.train(DATA, plot_training=False, validation_strategy=validation.WFV, force=True)
    assert [(w.filename, w.lineno) for w in _found(rec, FOLDS)] == [(__file__, line) for line in lines]


def test_degenerate_validation_window_warning_once_at_the_caller():
    rng = np.random.default_rng(1)
    segments = [rng.normal(size=(n, 2)).cumsum(axis=0) * 0.1 for n in (12, 13, 14, 12, 13)]
    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=10, upsample=1, N_wash=5, hyperparameters_to_optimize=[],
                           seed=0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        line = _next_line()
        esn.train(segments, plot_training=False, n_seeds=3)
    (w,) = _found(rec, DEGENERATE)                  # not once per seed: the split and the score of every seed
    assert (w.filename, w.lineno) == (__file__, line)


def _fails_after_WFV(*args, **kwargs):
    """A validation strategy that fails after the fold-count warning of WFV."""
    validation.WFV(*args, **kwargs)
    raise RuntimeError('the strategy fails after WFV')


def test_the_warnings_of_a_failed_seed_are_issued_before_its_error():
    # the warnings that a failed seed collected are issued, at the caller, before its error
    esn = _series_esn()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        with pytest.raises(RuntimeError, match='fails after WFV'):
            line = _next_line()
            esn.train(DATA, plot_training=False, validation_strategy=_fails_after_WFV, n_seeds=2)
    (w,) = _found(rec, FOLDS)
    assert (w.filename, w.lineno) == (__file__, line)
    assert not esn.trained                                              # and the network is as it was


# ------------------------------------------------------------------------------------- the reports (verbosity)
def _train_twice(esn):
    esn.train(DATA, plot_training=False)
    esn.train(DATA, plot_training=False, rho=0.8, force=True)   # trained again, with an option changed


def test_set_verbose_on_and_off(capsys, caplog):
    _train_twice(EchoStateNetwork(DATA.T, **QUICK))             # silent by default: no summary, no diagnostics
    out = capsys.readouterr()
    assert out.out == '' and out.err == ''
    assert RETRAIN not in caplog.text and SUMMARY not in caplog.text

    echostatenetwork.set_verbose(True)
    echostatenetwork.set_verbose(True)                                 # idempotent: one handler
    logger = logging.getLogger('echostatenetwork')
    assert sum(h.get_name() == 'echostatenetwork.verbose' for h in logger.handlers) == 1
    esn = EchoStateNetwork(DATA.T, **QUICK)
    _train_twice(esn)
    assert RETRAIN in caplog.text and 'rho = 0.9 -> 0.8' in caplog.text
    assert esn.training_summary() in caplog.text
    out = capsys.readouterr()
    assert out.out == '' and '[echostatenetwork] train(force=True): ' + RETRAIN in out.err
    assert '[echostatenetwork] ' + SUMMARY in out.err

    echostatenetwork.set_verbose(False)
    assert not any(h.get_name() == 'echostatenetwork.verbose' for h in logger.handlers)
    assert logger.level == logging.NOTSET                               # the default, not a forced WARNING
    caplog.clear()
    _train_twice(EchoStateNetwork(DATA.T, **QUICK))
    assert RETRAIN not in caplog.text and SUMMARY not in caplog.text
    assert capsys.readouterr().err == ''


def test_seeds_report_through_the_parent(capsys, caplog):
    echostatenetwork.set_verbose(True)
    esn = EchoStateNetwork(DATA.T, **QUICK)
    esn.train(DATA, plot_training=False, n_seeds=2)
    assert caplog.text.count(SUMMARY) == 2                              # each seed's summary, logged by the parent
    assert 'n_seeds=2' in caplog.text and 'kept seed' in caplog.text
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('hyperparameters, n_func_evals, header', [
    (['rho'], 4, 'a 2^1 grid and 2 points of Bayesian optimization'),
    (['tikh'], 4, 'tikh alone, selected from the 4 values of tikh_range in one evaluation'),       # issue #1
])
def test_search_table_header_and_every_line_prefixed(capsys, hyperparameters, n_func_evals, header):
    if hyperparameters != ['tikh']:
        pytest.importorskip('skopt')                    # a Bayesian search
    echostatenetwork.set_verbose(True)
    esn = EchoStateNetwork(DATA.T, **QUICK)
    esn.train(DATA, plot_training=False, hyperparameters_to_optimize=hyperparameters, N_func_evals=n_func_evals,
              N_grid=2)
    err = capsys.readouterr().err
    assert header in err and 'hyperparameter search' in err and 'log_nMAE' in err
    assert all(line.startswith('[echostatenetwork] ') for line in err.splitlines() if line.strip())
    if hyperparameters == ['tikh']:          # the optimum lists tikh only
        assert 'Bayesian optimization' not in err.replace('without Bayesian optimization', '')
        assert 'optimum: tikh=' in err and 'optimum: ,' not in err
        assert 'tikh alone: 1 eval over the 4 values of tikh_range, no BHO' in err


def test_run_test_scores_every_window_and_caps_the_figures_per_segment(caplog):
    # the number of scored short-term tests is set by the test tail, not by the figures: max_short_tests caps the
    # figures of each segment, every segment is scored the same way, and each segment is tested at most once
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    esn = EchoStateNetwork(DATA.T, **QUICK)
    esn.train(DATA, plot_training=False)
    segments = np.stack([DATA[:80], DATA[80:160]])
    n_windows = (80 - 1 - esn.N_val - 1) // esn.N_val + 1          # windows of N_val steps that fit in a segment
    try:
        for max_short_tests, n_figures in ((1, 2), (10, 2 * n_windows)):
            caplog.clear()
            with caplog.at_level(logging.INFO, logger='echostatenetwork'):
                figures = esn.run_test(segments[:, :-1], segments[:, 1:], long_term=False, max_L_tests=10,
                                       max_short_tests=max_short_tests)
            assert figures[0] is None and len(figures) == 1 + n_figures
            assert f'({2 * n_windows} tests), mean per segment over 2 segments' in caplog.text
        figures = esn.run_test(segments[:1, :-1], segments[:1, 1:], short_term=False)
        assert figures[0] is not None and figures[1:] == [None]    # one segment: its long-term figure
        assert esn.run_test(segments[:, :-1], segments[:, 1:], short_term=False)[0] is None
        with pytest.warns(UserWarning, match='max_L_tests < 1'):
            assert esn.run_test(segments[:, :-1], segments[:, 1:], max_L_tests=0) == []
    finally:
        plt.close('all')
