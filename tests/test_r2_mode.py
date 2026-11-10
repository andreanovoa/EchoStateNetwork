"""The readout on squared reservoir units (`r2_mode`): every second unit (indices 1, 3, 5, ...) enters the readout
squared, as `r2_mode` of MagriLab/Adjoint-ESN (Pathak et al. 2018)."""
import numpy as np
import pytest
from test_esn_parallel import _random_parallel, _set_norm

from echostatenetwork import EchoStateNetwork, validation


def _small(r2_mode=True, **kw):
    rng = np.random.default_rng(13)
    data = rng.normal(size=(120, 3)).cumsum(axis=0) * 0.1
    kw = {'hyperparameters_to_optimize': [], **kw}
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=60, t_val=20, t_test=20, N_wash=5,
                           seed=0, r2_mode=r2_mode, **kw)
    return esn, data


def _adjoint_esn_before_readout_r2(x, b_out):
    """`before_readout_r2` of MagriLab/Adjoint-ESN (adjoint_esn/esn.py), for one state x of shape (N_units,)."""
    x2 = x.copy()
    x2[1::2] = x2[1::2] ** 2
    return np.hstack((x2, b_out))


def test_the_readout_squares_every_second_unit_as_adjoint_esn():
    esn, data = _small()
    esn.train(data, plot_training=False)
    r = np.random.default_rng(1).normal(size=(esn.N_units, 1))
    features = _adjoint_esn_before_readout_r2(r[:, 0], esn.bias_out)
    np.testing.assert_allclose(esn.reservoir_to_physical(r)[:, 0], features @ esn.Wout, rtol=1e-13)
    assert esn.Wout.shape == (esn.N_units + 1, esn.N_dim)          # the readout keeps its size
    off, _ = _small(r2_mode=False)
    off.train(data, plot_training=False)
    assert not np.allclose(off.Wout, esn.Wout)


def test_the_ridge_regression_reads_the_squared_units():
    esn, data = _small()
    esn.train(data, plot_training=False)
    U, Y = data[None, :-1], data[None, 1:]
    LHS, RHS, _, R_RR = esn._compute_RR_terms(U, Y)
    R = R_RR[0].copy()
    R[:, 1::2] **= 2
    rows = np.hstack([R, np.ones((R.shape[0], 1)) * esn.bias_out])
    np.testing.assert_allclose(LHS, rows.T @ rows, rtol=1e-12)
    np.testing.assert_allclose(RHS, rows.T @ Y[0, esn.N_wash:], rtol=1e-12, atol=1e-12)


def test_the_fold_readouts_of_the_single_series_strategies_read_the_squared_units():
    # one fold that trains on every row: the per-fold readout of the engine behind SSV/WFV/KFV equals the ridge
    # solution of _compute_RR_terms, which reads the squared units
    esn = _random_parallel(N_dim=8, patch_size=2, halo=1, N_wash=5, t_val=10, tikh_range=[1e-6], r2_mode=True)
    rng = np.random.default_rng(7)
    U = rng.normal(size=(1, 60, 8))
    Y = np.roll(U, -1, axis=1)

    def all_rows(case, n_post):
        return [([(0, n_post)], n_post - case.N_val, case.N_val)]
    validation.single_series_validation([], esn, U, Y, np.zeros(1), [], False, all_rows, 'all_rows')
    LHS, RHS = esn._compute_RR_terms(U, Y)[:2]
    LHS[np.diag_indices_from(LHS)] += 1e-6
    np.testing.assert_allclose(esn.Wout, np.linalg.solve(LHS, RHS), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('leak_rate', [1.0, 0.6])
def test_jacobian_matches_finite_differences(leak_rate):
    esn, data = _small(leak_rate=leak_rate)
    esn.train(data, plot_training=False)
    rng = np.random.default_rng(2)
    u, r = data[5].reshape(-1, 1).copy(), 0.3 * rng.normal(size=(esn.N_units, 1))
    J, eps = esn.Jacobian(u, r), 1e-5
    J_fd = np.zeros_like(J)
    for j in range(u.shape[0]):
        up, um = u.copy(), u.copy()
        up[j] += eps
        um[j] -= eps
        J_fd[:, j] = ((esn.step(up, r)[0] - esn.step(um, r)[0]) / (2 * eps))[:, 0]
    np.testing.assert_allclose(J, J_fd, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('shared', [True, False])
def test_patch_jacobian_matches_finite_differences(shared):
    esn = _random_parallel(N_dim=8, patch_size=2, halo=1, periodic=True, shared=shared, r2_mode=True)
    rng = np.random.default_rng(4)
    _set_norm(esn, rng)
    if not shared:
        assert all(net.r2_mode for net in esn.patches)               # the patches inherit the option
    u, r = rng.normal(size=(8, 2)), 0.3 * rng.normal(size=(esn.N_r, 2))
    J, eps = esn.Jacobian(u, r), 1e-5
    J_fd = np.zeros_like(J)
    for j in range(8):
        up, um = u.copy(), u.copy()
        up[j] += eps
        um[j] -= eps
        J_fd[:, j] = (esn.step(up, r)[0] - esn.step(um, r)[0]) / (2 * eps)
    np.testing.assert_allclose(J, J_fd, rtol=1e-6, atol=1e-9)


def test_r2_mode_survives_to_arrays_and_older_arrays_load_without_it():
    esn, data = _small()
    esn.train(data, plot_training=False)
    arrays = esn.to_arrays()
    back = EchoStateNetwork.from_arrays(arrays)
    u, r = data[7].reshape(-1, 1), np.random.default_rng(3).normal(size=(esn.N_units, 1))
    assert back.r2_mode is True
    np.testing.assert_allclose(back.step(u, r)[0], esn.step(u, r)[0], rtol=1e-13)
    older = {k: v for k, v in arrays.items() if k != 'r2_mode'}       # saved before r2_mode existed
    assert EchoStateNetwork.from_arrays(older).r2_mode is False


@pytest.mark.parametrize('strategy', ['_RVC_Noise', '_SSV', '_WFV', '_KFV'])
def test_every_validation_strategy_trains_with_r2_mode(strategy):
    esn, data = _small(rho_range=(0.5, 1.0), N_grid=2, N_func_evals=4, hyperparameters_to_optimize=['rho'],
                       N_folds=2)
    esn.train(data, plot_training=False, validation_strategy=getattr(EchoStateNetwork, strategy))
    assert esn.trained and np.isfinite(esn.bo_results['fun']) and np.isfinite(esn.Wout).all()
