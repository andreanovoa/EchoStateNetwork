"""The readout on squared reservoir units (`r2_mode`): every second unit (indices 1, 3, 5, ...) enters the readout
squared, as `r2_mode` of MagriLab/Adjoint-ESN (Pathak et al. 2018). The Jacobian and the folds of the single-series
strategies with `r2_mode` are tested with the others, in test_esn_core and test_esn_parallel."""
import numpy as np

from echostatenetwork import EchoStateNetwork


def _trained(r2_mode=True):
    data = np.random.default_rng(13).normal(size=(120, 3)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=60, t_val=20, t_test=20, N_wash=5, seed=0,
                           r2_mode=r2_mode, hyperparameters_to_optimize=[])
    esn.train(data, plot_training=False)
    return esn, data


def test_the_readout_and_the_ridge_regression_square_every_second_unit_as_adjoint_esn():
    esn, data = _trained()
    r = np.random.default_rng(1).normal(size=(esn.N_units, 1))
    x2 = r[:, 0].copy()                                   # before_readout_r2 of MagriLab/Adjoint-ESN (adjoint_esn/esn.py)
    x2[1::2] = x2[1::2] ** 2
    np.testing.assert_allclose(esn.reservoir_to_physical(r)[:, 0], x2 @ esn.Wout + esn.intercept, rtol=1e-13)
    assert esn.Wout.shape == (esn.N_units, esn.N_dim)              # the readout keeps its size
    assert not np.allclose(_trained(r2_mode=False)[0].Wout, esn.Wout)
    U, Y = data[None, :-1], data[None, 1:]
    LHS, RHS, _, R_RR = esn._compute_RR_terms(U, Y)
    R = R_RR[0].copy()
    R[:, 1::2] **= 2
    rows = np.hstack([R, np.ones((R.shape[0], 1))])
    np.testing.assert_allclose(LHS, rows.T @ rows, rtol=1e-12)
    np.testing.assert_allclose(RHS, rows.T @ Y[0, esn.N_wash:], rtol=1e-12, atol=1e-12)


def test_r2_mode_survives_to_arrays_and_older_arrays_load_without_it():
    esn, data = _trained()
    arrays = esn.to_arrays()
    back = EchoStateNetwork.from_arrays(arrays)
    u, r = data[7].reshape(-1, 1), np.random.default_rng(3).normal(size=(esn.N_units, 1))
    assert back.r2_mode is True
    np.testing.assert_allclose(back.step(u, r)[0], esn.step(u, r)[0], rtol=1e-13)
    older = {k: v for k, v in arrays.items() if k != 'r2_mode'}       # saved before r2_mode existed
    assert EchoStateNetwork.from_arrays(older).r2_mode is False
