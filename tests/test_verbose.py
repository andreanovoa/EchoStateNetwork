"""Verbosity: the package is silent by default, `set_verbose` shows its reports (the search and the summary of `train`,
and the diagnostic messages), and the removed keyword `verbose` warns; `train` trains a trained network again only with
`force=True`."""
import inspect
import logging

import numpy as np
import pytest

import echostatenetwork
from echostatenetwork import EchoStateNetwork

RETRAIN = 'training the trained network again'   # a diagnostic message: train(force=True)
SUMMARY = 'trained:'                             # the start of the training summary, the report of train


def small_esn(**kwargs):
    """A small network on a random walk, without a hyperparameter search, and its data."""
    data = np.random.default_rng(0).normal(size=(200, 3)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=15, upsample=1, t_train=120, t_val=30, t_test=40, N_wash=5,
                           hyperparameters_to_optimize=[], seed=0, **kwargs)
    return esn, data


def train_twice(esn, data):
    esn.train(data, plot_training=False)
    esn.train(data, plot_training=False, rho=0.8, force=True)   # trained again, with an option changed


@pytest.fixture(autouse=True)
def quiet():
    yield
    echostatenetwork.set_verbose(False)


def test_silent_by_default(capsys, caplog):
    train_twice(*small_esn())
    out = capsys.readouterr()
    assert out.out == '' and out.err == ''                              # neither the summary nor the diagnostics
    assert RETRAIN not in caplog.text and SUMMARY not in caplog.text


def test_set_verbose_on_and_off(capsys, caplog):
    echostatenetwork.set_verbose(True)
    echostatenetwork.set_verbose(True)                                 # idempotent: one handler
    logger = logging.getLogger('echostatenetwork')
    assert sum(h.get_name() == 'echostatenetwork.verbose' for h in logger.handlers) == 1
    esn, data = small_esn()
    train_twice(esn, data)
    assert RETRAIN in caplog.text and 'rho = 0.9 -> 0.8' in caplog.text
    assert esn.training_summary() in caplog.text
    out = capsys.readouterr()
    assert out.out == '' and '[echostatenetwork] train(force=True): ' + RETRAIN in out.err
    assert '[echostatenetwork] ' + SUMMARY in out.err

    echostatenetwork.set_verbose(False)
    assert not any(h.get_name() == 'echostatenetwork.verbose' for h in logger.handlers)
    assert logger.level == logging.NOTSET                               # the default, not a forced WARNING
    caplog.clear()
    train_twice(*small_esn())
    assert RETRAIN not in caplog.text and SUMMARY not in caplog.text
    assert capsys.readouterr().err == ''


def test_verbose_keyword_is_removed():
    assert not hasattr(EchoStateNetwork, 'verbose')
    with pytest.warns(FutureWarning, match="'verbose' is removed"):
        esn, _ = small_esn(verbose=False)
    assert 'verbose' not in vars(esn)


def test_verbose_keyword_of_train_is_removed():
    # train(verbose=...) gives the FutureWarning of the constructor, at the line of the caller, and not the warning
    # of an unknown keyword (issue #7)
    esn, data = small_esn()
    with pytest.warns(FutureWarning, match=r"EchoStateNetwork\.train: the keyword 'verbose' is removed") as rec:
        line = inspect.currentframe().f_lineno + 1
        esn.train(data, plot_training=False, verbose=False)
    assert [(w.filename, w.lineno) for w in rec] == [(__file__, line)]
    assert esn.trained and 'verbose' not in vars(esn)


def test_seeds_report_through_the_parent(capsys, caplog):
    echostatenetwork.set_verbose(True)
    esn, data = small_esn()
    esn.train(data, plot_training=False, n_seeds=2)
    assert caplog.text.count(SUMMARY) == 2                              # each seed's summary, logged by the parent
    assert 'n_seeds=2' in caplog.text and 'kept seed' in caplog.text
    assert capsys.readouterr().out == ''


def test_run_test_without_testable_segment_warns():
    esn, data = small_esn()
    esn.train(data, plot_training=False)
    short = data[:3]
    with pytest.warns(UserWarning, match='no test is run'):
        assert esn.run_test(short[None, :-1], short[None, 1:]) == []


def test_run_test_tests_each_segment_at_most_once(caplog):
    # max_L_tests larger than the number of segments: each of the 3 segments is tested once, none twice
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    esn, data = small_esn()
    esn.train(data, plot_training=False)
    segments = np.stack([data[:60], data[60:120], data[120:180]])
    try:
        with caplog.at_level(logging.INFO, logger='echostatenetwork'):
            esn.run_test(segments[:, :-1], segments[:, 1:], max_L_tests=10, long_term=False, max_short_tests=1)
    finally:
        plt.close('all')
    assert 'mean per segment over 3 segments' in caplog.text


def test_trained_network_is_not_trained_again_without_force():
    esn, data = small_esn()
    esn.train(data, plot_training=False)
    Wout = esn.Wout.copy()
    with pytest.warns(UserWarning, match='force=True'):
        esn.train(data, plot_training=False, rho=0.5)                   # skipped: rho and Wout unchanged
    assert esn.rho == 0.9 and np.array_equal(esn.Wout, Wout)
    esn.train(data, plot_training=False, rho=0.5, force=True)
    assert esn.rho == 0.5 and not np.array_equal(esn.Wout, Wout)


def test_search_table_with_verbose_and_every_line_prefixed(capsys):
    pytest.importorskip('skopt')                    # a Bayesian search
    echostatenetwork.set_verbose(True)
    esn, data = small_esn()
    esn.train(data, plot_training=False, hyperparameters_to_optimize=['rho'], N_func_evals=4, N_grid=2,
              force=True)
    err = capsys.readouterr().err.splitlines()
    assert any('hyperparameter search' in line for line in err) and any('log_nMAE' in line for line in err)
    assert all(line.startswith('[echostatenetwork] ') for line in err if line.strip())


@pytest.mark.parametrize('n_func_evals, points', [(3, '1 point'), (4, '2 points')])
def test_search_header_counts_the_points_of_bayesian_optimization(caplog, n_func_evals, points):
    pytest.importorskip('skopt')                    # a Bayesian search
    echostatenetwork.set_verbose(True)
    esn, data = small_esn()
    esn.train(data, plot_training=False, hyperparameters_to_optimize=['rho'], N_func_evals=n_func_evals, N_grid=2,
              force=True)
    assert f'a 2^1 grid and {points} of Bayesian optimization' in caplog.text


def test_search_table_with_tikh_alone(capsys):
    # issue #1: the search over tikh alone is one evaluation, which the header says, and the optimum lists tikh only
    echostatenetwork.set_verbose(True)
    esn, data = small_esn()
    esn.train(data, plot_training=False, hyperparameters_to_optimize=['tikh'], force=True)
    err = capsys.readouterr().err
    assert 'tikh alone, selected from the 4 values of tikh_range in one evaluation' in err
    assert 'Bayesian optimization' not in err.replace('without Bayesian optimization', '')
    assert 'optimum: tikh=' in err and 'optimum: ,' not in err
    assert 'tikh alone: 1 eval over the 4 values of tikh_range, no BHO' in err


def test_training_again_says_why_it_draws_a_new_reservoir_and_infers_the_split_again(caplog):
    # train(force=True): a changed option of the reservoir draws a new one, and a split that the previous call
    # inferred is inferred again from the new data (issue #7)
    echostatenetwork.set_verbose(True)
    data = np.random.default_rng(0).normal(size=(200, 3)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=15, upsample=1, t_test=0, N_wash=5, hyperparameters_to_optimize=[],
                           seed=0)
    esn.train(data, plot_training=False)
    caplog.clear()
    esn.train(data[:150], plot_training=False, N_units=20, force=True)
    assert 'train: new W and Win (seed 0), because N_units (15 -> 20) changed.' in caplog.text
    assert 'train: t_train is inferred again from the data' in caplog.text
    assert 'train: t_val is inferred again from the data' in caplog.text
    caplog.clear()
    esn.train(data[:150], plot_training=False, rho=0.5, force=True)
    assert RETRAIN in caplog.text and 'new W and Win' not in caplog.text
