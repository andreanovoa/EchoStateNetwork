"""A singular ridge regression (a saturated reservoir with a vanishing tikh) falls back on the least-squares solution
and warns through `warnings` (once per call of train, as the other diagnostics of the package), so that one bad
point scores badly instead of stopping the search; a non-singular one gives the result of `np.linalg.solve`,
unchanged."""
import numpy as np
import pytest

from echostatenetwork import EchoStateNetwork, validation

DATA = np.random.default_rng(0).normal(size=(300, 3)).cumsum(axis=0) * 10
SINGULAR = 'ridge matrix is singular'


def saturated_esn(**kwargs):
    """A reservoir saturated by a large input scaling and input bias: its states are +-1, so the columns of the ridge
    left-hand side are collinear and, without regularisation, it is singular."""
    options = dict(dt=1, N_units=50, upsample=1, t_train=200, t_val=40, t_test=40, N_wash=5, rho=0.5, sigma_in=10.,
                   bias_in=np.array([10.]), tikh=0., tikh_range=[0.], seed=0)
    options.update(kwargs)
    return EchoStateNetwork(DATA.T, **options)


def test_non_singular_solve_is_unchanged():
    esn = saturated_esn()
    rng = np.random.default_rng(1)
    M = rng.normal(size=(6, 6))
    LHS, RHS = M @ M.T, rng.normal(size=(6, 2))
    np.testing.assert_array_equal(esn._ridge_solve(LHS, RHS, 1e-3), np.linalg.solve(esn._regularized(LHS, 1e-3), RHS))


def test_collinear_columns_fall_back_on_least_squares():
    esn = saturated_esn()
    X = np.random.default_rng(2).normal(size=(40, 3))
    X = np.hstack((X, X[:, :1]))                              # two collinear columns: a singular X^T X
    LHS, RHS = X.T @ X, X.T @ np.ones((40, 2))
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(LHS, RHS)
    with pytest.warns(UserWarning, match=SINGULAR):
        Wout = esn._ridge_solve(LHS, RHS, 0.)
    assert np.isfinite(Wout).all()
    np.testing.assert_allclose(Wout, np.linalg.lstsq(LHS, RHS, rcond=None)[0])


@pytest.mark.parametrize('strategy', [validation.RVC_Noise, validation.SSV])
def test_search_on_a_saturated_reservoir_completes(strategy):
    esn = saturated_esn(hyperparameters_to_optimize=['tikh'])
    with pytest.warns(UserWarning, match=SINGULAR) as record:
        esn.train(DATA, plot_training=False, validation_strategy=strategy)
    assert np.isfinite(esn.Wout).all()
    assert len([w for w in record if SINGULAR in str(w.message)]) == 1   # once per call of train


def test_training_without_search_on_a_saturated_reservoir_completes():
    esn = saturated_esn(hyperparameters_to_optimize=[])
    with pytest.warns(UserWarning, match=SINGULAR):
        esn.train(DATA, plot_training=False)
    assert np.isfinite(esn.Wout).all()
