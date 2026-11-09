"""User-facing warnings (issue #7): each warning points at the line that calls echostatenetwork (not inside the
package, nor at scikit-optimize, which calls the validation strategy back) and appears once per call, however many
seeds, patches or evaluations of the search raise it. A keyword of the constructor or of `train` that the network
ignores always warns; so do fewer validation folds than requested (WFV, KFV) and a degenerate validation window."""
import inspect
import re
import warnings

import numpy as np
import pytest

from echostatenetwork import EchoStateNetwork, validation

DATA = np.random.default_rng(0).normal(size=(200, 4)).cumsum(axis=0) * 0.1
QUICK = dict(dt=1, N_units=15, upsample=1, t_train=120, t_val=30, t_test=40, N_wash=5, hyperparameters_to_optimize=[],
             seed=0)


def _next_line():
    """Number of the line after the line of the caller."""
    return inspect.currentframe().f_back.f_lineno + 1


def _ignored(records):
    """The warnings about ignored keywords among `records`."""
    return [w for w in records if 'are ignored' in str(w.message)]


def _serial(*args, **kwargs):
    """The default validation strategy; wrapped in a lambda, it does not pickle, so the seeds train serially."""
    return validation.RVC_Noise(*args, **kwargs)


class _Subclass(EchoStateNetwork):
    """A subclass defined outside echostatenetwork that passes its keywords on, as romda's ESN_model does."""

    def __init__(self, y, **kwargs):
        self.line = _next_line()
        super().__init__(y, **kwargs)


def test_constructor_typo_warns_at_the_caller():
    match = re.escape("EchoStateNetwork: keyword(s) ['N_unit'] are not attributes of EchoStateNetwork and are ignored")
    with pytest.warns(UserWarning, match=match) as rec:
        line = _next_line()
        esn = EchoStateNetwork(DATA.T, dt=1, N_unit=50)
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert "did you mean 'N_units' for 'N_unit'?" in str(w.message)
    assert esn.N_units == EchoStateNetwork.N_units and 'N_unit' not in vars(esn)


def test_constructor_typo_through_a_subclass_warns_at_the_subclass():
    with pytest.warns(UserWarning, match=re.escape("_Subclass: keyword(s) ['rhoo']")) as rec:
        esn = _Subclass(DATA.T, rhoo=0.5)
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, esn.line)
    assert esn.rho == EchoStateNetwork.rho


def test_train_typo_warns_at_the_caller_and_keeps_the_default():
    esn = EchoStateNetwork(DATA.T, **QUICK)
    match = re.escape("EchoStateNetwork.train: keyword(s) ['rho_rnage'] are not attributes of the network and are "
                      "ignored (did you mean 'rho_range' for 'rho_rnage'?)")
    with pytest.warns(UserWarning, match=match) as rec:
        line = _next_line()
        esn.train(DATA, plot_training=False, rho_rnage=(0.5, 0.6))
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert esn.rho_range == EchoStateNetwork.rho_range and 'rho_rnage' not in vars(esn) and esn.trained


def test_no_hint_without_a_close_attribute():
    esn = EchoStateNetwork(DATA.T, **QUICK)
    with pytest.warns(UserWarning, match=re.escape("['plot', 'tikhonov'] are not attributes")) as rec:
        esn.train(DATA, plot_training=False, tikhonov=1e-6, plot=True)
    (w,) = _ignored(rec)
    assert 'did you mean' not in str(w.message)


def test_known_keywords_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        esn = EchoStateNetwork(DATA.T, **QUICK, rho=0.8, Win_type='sparse_dense_bias', val_k=0)
        esn.train(DATA, plot_training=False, sigma_in=0.01, N_func_evals=5, tikh_range=[1e-6, 1e-8])
        esn.train(DATA, plot_training=False, rho=0.7, Win=esn.Win.toarray(), force=True)  # held once trained
    assert esn.rho == 0.7 and esn.sigma_in == 0.01


def test_attribute_that_the_network_does_not_hold_yet_is_ignored():
    esn = EchoStateNetwork(DATA.T, **QUICK)
    Win = np.ones((QUICK['N_units'], DATA.shape[1] + 1))
    match = re.escape("keyword(s) ['Win'] are attributes that the network does not hold yet and are ignored "
                      "(set Win on the network before calling train, which keeps them)")
    with pytest.warns(UserWarning, match=match) as rec:
        line = _next_line()
        esn.train(DATA, plot_training=False, Win=Win)
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert 'not attributes' not in str(w.message)
    assert esn.trained and not np.array_equal(esn.Win.toarray(), Win)     # drawn from the seed, as without Win


def test_readout_is_computed_by_train():
    # train computes Wout, so the keyword has no effect and warns, on a trained network as well
    esn = EchoStateNetwork(DATA.T, **QUICK)
    Wout = np.zeros((QUICK['N_units'] + 1, DATA.shape[1]))
    with pytest.warns(UserWarning, match=re.escape("['Wout'] are computed by train and are ignored")) as rec:
        esn.train(DATA, plot_training=False, Wout=Wout)
    (w,) = _ignored(rec)
    assert str(w.message).endswith('and are ignored') and esn.trained
    with pytest.warns(UserWarning, match=re.escape("['Wout'] are computed by train and are ignored")):
        esn.train(DATA, plot_training=False, Wout=Wout, force=True)
    assert np.abs(esn.Wout).max() > 0


def test_serial_seeds_warn_once():
    esn = EchoStateNetwork(DATA.T, **QUICK)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        line = _next_line()
        esn.train(DATA, plot_training=False, n_seeds=3, typo_key=1,
                  validation_strategy=lambda *a, **k: _serial(*a, **k))
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert len(esn.seed_scores) == 3


def test_parallel_seeds_warn_once_in_the_parent():
    esn = EchoStateNetwork(DATA.T, **QUICK)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        line = _next_line()
        esn.train(DATA, plot_training=False, n_seeds=2, typo_key=1)
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert len(esn.seed_scores) == 2


def test_independent_patches_warn_once():
    esn = EchoStateNetwork(DATA.T, **QUICK, patch_size=2, halo=1, shared=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        line = _next_line()
        esn.train(DATA, plot_training=False, typo_key=1)
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert esn.trained


def test_independent_patches_with_serial_seeds_warn_once():
    esn = EchoStateNetwork(DATA.T, **QUICK, patch_size=2, halo=1, shared=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        esn.train(DATA, plot_training=False, n_seeds=2, typo_key=1,
                  validation_strategy=lambda *a, **k: _serial(*a, **k))
    assert len(_ignored(rec)) == 1 and esn.trained


def test_a_trained_network_that_is_not_trained_again_still_names_the_keyword():
    esn = EchoStateNetwork(DATA.T, **QUICK)
    esn.train(DATA, plot_training=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        esn.train(DATA, plot_training=False, typo_key=1)
    assert len(_ignored(rec)) == 1 and any('not trained again' in str(w.message) for w in rec)


def test_a_subclass_that_is_not_trained_again_is_named_in_the_warning():
    esn = _WithOption(DATA.T, **QUICK)
    esn.train(DATA, plot_training=False)
    with pytest.warns(UserWarning, match=re.escape('_WithOption.train: the network is already trained and is not '
                                                   'trained again; pass force=True')):
        esn.train(DATA, plot_training=False)


def test_one_warning_per_line_of_the_caller():
    # Python shows a warning once per location by default: the location is the line of the caller, so two lines
    # give two warnings, and a loop on one line gives one
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('default')
        EchoStateNetwork(DATA.T, typo_key=1)
        EchoStateNetwork(DATA.T, typo_key=1)
        for _ in range(3):
            EchoStateNetwork(DATA.T, typo_key=1)
    assert len(_ignored(rec)) == 3


# ------------------------------------------------------------------------- warnings of the training (issue #7, item 5)
FOLDS = 'of the requested N_folds'                  # fewer folds fit than WFV or KFV request
DEGENERATE = 'inferred validation window is degenerate'


def _series_esn(**kwargs):
    """A small network with a search of three evaluations and 100 requested folds, of which 4 fit in `DATA`."""
    options = {**QUICK, 'N_folds': 100, 'N_grid': 2, 'N_func_evals': 3, 'hyperparameters_to_optimize': ['rho'],
               **kwargs}
    if set(options['hyperparameters_to_optimize']) - {'tikh'}:
        pytest.importorskip('skopt')        # a Bayesian search needs scikit-optimize, an optional dependency
    return EchoStateNetwork(DATA.T, **options)


def _found(records, match):
    """The warnings among `records` whose message contains `match`."""
    return [w for w in records if match in str(w.message)]


@pytest.mark.parametrize('n_seeds', [1, 3])
@pytest.mark.parametrize('strategy', [validation.WFV, validation.KFV], ids=['WFV', 'KFV'])
def test_fold_count_warning_once_at_the_caller(strategy, n_seeds):
    # n_seeds=1: the strategy runs in the search, called back by scikit-optimize; n_seeds=3: in three processes
    esn = _series_esn()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        line = _next_line()
        esn.train(DATA, plot_training=False, validation_strategy=strategy, n_seeds=n_seeds)
    (w,) = _found(rec, FOLDS)                       # not once per seed, nor per evaluation of the search
    assert (w.filename, w.lineno, w.category) == (__file__, line, UserWarning)
    assert str(w.message).startswith(f'{strategy.__name__}: only 4 of the requested N_folds=100')


@pytest.mark.parametrize('options, n_seeds, strategy', [
    (dict(hyperparameters_to_optimize=[]), 3, validation.WFV),                  # the score of every seed
    (dict(patch_size=2, halo=1, shared=False), 1, validation.WFV),             # the search of every patch
    (dict(patch_size=2, halo=1, shared=False), 2, validation.WFV),             # every patch and seed
    ({}, 2, lambda *a, **k: validation.WFV(*a, **k)),                          # serial seeds: it does not pickle
], ids=['seeds_without_search', 'patches', 'patches_and_seeds', 'serial_seeds'])
def test_fold_count_warning_once_without_search_with_patches_and_serial_seeds(options, n_seeds, strategy):
    esn = _series_esn(**options)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        line = _next_line()
        esn.train(DATA, plot_training=False, validation_strategy=strategy, n_seeds=n_seeds)
    (w,) = _found(rec, FOLDS)
    assert (w.filename, w.lineno) == (__file__, line)
    assert esn.trained


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


def test_warnings_of_direct_calls_point_at_the_caller():
    with pytest.warns(FutureWarning, match="'verbose' is removed") as rec:
        lines = [_next_line()]
        esn = EchoStateNetwork(DATA.T, **QUICK, verbose=True)
    esn.train(DATA, plot_training=False)
    with pytest.warns(UserWarning, match='not trained again') as again:
        lines.append(_next_line())
        esn.train(DATA, plot_training=False)
    with pytest.warns(DeprecationWarning, match='use compute_nMAE') as alias:
        lines.append(_next_line())
        esn.compute_nRMSE(DATA[:5], DATA[5:10])
    short = DATA[:3]
    with pytest.warns(UserWarning, match='no test is run') as test:
        lines.append(_next_line())
        esn.run_test(short[None, :-1], short[None, 1:])
    assert [(w.filename, w.lineno) for w in [*rec, *again, *alias, *test]] == [(__file__, line) for line in lines]


# ---------------------------------------- methods, read-only properties and the options of subclasses (issue #7)
class _WithOption(EchoStateNetwork):
    """A subclass with an option of its own, which a plain EchoStateNetwork does not hold."""
    my_option = 0


def test_keywords_that_name_a_method_or_a_read_only_property_are_ignored():
    # a keyword that names a method (copy) or a read-only property (trained) is not set, with one warning
    with pytest.warns(UserWarning) as rec:
        line = _next_line()
        esn = EchoStateNetwork(DATA.T, **QUICK, copy=1, trained=True)
    (w,) = _ignored(rec)
    assert (w.filename, w.lineno) == (__file__, line)
    assert "['copy'] name methods of EchoStateNetwork" in str(w.message)
    assert "['trained'] are read-only properties" in str(w.message)
    assert callable(esn.copy) and not esn.trained
    esn.train(DATA, plot_training=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        # a getter that raises (patch_sites without patches) does not prevent the warning
        assert esn.train(DATA, plot_training=False, patch_sites=None, N_train=5) is None
    (w,) = _ignored(rec)
    assert "['N_train', 'patch_sites'] are read-only properties of the network" in str(w.message)
    assert any('not trained again' in str(r.message) for r in rec)


def test_the_hint_names_the_parameters_of_train():
    esn = EchoStateNetwork(DATA.T, **QUICK)
    with pytest.warns(UserWarning, match=re.escape("(did you mean 'n_seeds' for 'n_seed'?)")):
        esn.train(DATA, plot_training=False, n_seed=2)


def test_the_hint_suggests_dt_ESN_for_dt():
    # dt is a parameter of the constructor only: train sets the time step of the data with dt_ESN = dt * upsample
    esn = EchoStateNetwork(DATA.T, **QUICK)
    esn.train(DATA, plot_training=False)
    match = re.escape("keyword(s) ['dt'] are not attributes of the network and are ignored (did you mean 'dt_ESN' "
                      "for 'dt'? train sets the time step of the data, dt, with dt_ESN = dt * upsample.)")
    with pytest.warns(UserWarning, match=match):
        esn.train(DATA, plot_training=False, dt=0.5, force=True)
    assert esn.dt_physical == 1
    esn.train(DATA, plot_training=False, dt_ESN=0.5 * esn.upsample, t_train=60, t_val=15, force=True)
    assert esn.dt_physical == 0.5 and esn.N_train == 120


def test_the_hint_suggests_only_options():
    # the hint names options only: not 'trained', a read-only property, for 'traned'
    with pytest.warns(UserWarning) as rec:
        EchoStateNetwork(DATA.T, dt=1, traned=True)
    (w,) = _ignored(rec)
    assert 'did you mean' not in str(w.message)
    with pytest.warns(UserWarning, match=re.escape("(did you mean 'Win_type' for 'Win_typ'?)")):
        EchoStateNetwork(DATA.T, dt=1, Win_typ='dense')


class _CallableOption(EchoStateNetwork):
    """A subclass that holds an option of EchoStateNetwork as a callable of the class."""
    validation_metric = staticmethod(validation.nMSE)


def test_an_option_that_a_subclass_holds_as_a_callable_is_set_by_train():
    # an option that the subclass holds as a staticmethod is set by train, without a warning
    esn = _CallableOption(DATA.T, **QUICK)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        esn.train(DATA, plot_training=False, validation_metric=validation.log_nMAE)
    assert esn.validation_metric is validation.log_nMAE
    with pytest.warns(UserWarning, match=re.escape("['step'] name methods of the network")):
        esn.train(DATA, plot_training=False, step=1, force=True)


def test_an_option_of_a_subclass_stays_on_the_network_with_independent_patches():
    # an option of the subclass stays on the parallel network: the networks of the patches neither receive it nor
    # warn
    esn = _WithOption(DATA.T, **QUICK, patch_size=2, halo=1, shared=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warnings.filterwarnings('ignore', category=DeprecationWarning)          # of scikit-optimize
        esn.train(DATA, plot_training=False, my_option=5)
    assert esn.my_option == 5 and all('my_option' not in vars(net) for net in esn.patches)


def test_the_matrices_of_independent_patches_are_held_by_the_patches():
    esn = EchoStateNetwork(DATA.T, **QUICK, patch_size=2, halo=1, shared=False)
    match = re.escape("['Win'] are attributes that the network does not hold yet and are ignored (with independent "
                      "patches, the network of every patch holds its own, in patches[p])")
    with pytest.warns(UserWarning, match=match):
        esn.train(DATA, plot_training=False, Win=np.ones((QUICK['N_units'], 5)))


def test_code_run_by_exec_without_a_module_name_still_warns():
    # code run by exec, whose globals have no module name, still warns
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        exec('EchoStateNetwork(Y, N_unit=3)', {'EchoStateNetwork': EchoStateNetwork, 'Y': DATA.T})
    (w,) = _ignored(rec)
    assert w.filename == '<string>'


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
