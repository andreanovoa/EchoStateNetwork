"""to_arrays / from_arrays: the pickle-free deployment-state round trip."""
import numpy as np
from conftest import assert_same_closed_loop, npz_round_trip
from scipy.sparse import csr_matrix

from echostatenetwork import EchoStateNetwork


def _trained(seed=0, **kw):
    rng = np.random.default_rng(seed)
    data = np.cumsum(rng.normal(size=(2, 120, 3)), axis=1)
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=20, upsample=1, N_wash=5,
                           t_train=60, t_val=20, t_test=20, seed=seed,
                           hyperparameters_to_optimize=[], **kw)
    esn.train(data, plot_training=False)
    return esn


def test_npz_round_trip_is_bitwise_in_closed_loop(tmp_path):
    esn = _trained()
    back = npz_round_trip(esn, tmp_path)                 # allow_pickle=False: no object arrays may leak in
    assert back.trained and isinstance(back.W, csr_matrix)
    assert np.array_equal(back.Wout, esn.Wout)
    assert np.array_equal(back.W.toarray(), esn.W.toarray())
    assert np.array_equal(np.asarray(back.Win.todense()), np.asarray(esn.Win.todense()))
    assert back.Win_type == esn.Win_type and back.seed == esn.seed
    assert_same_closed_loop(esn, back, np.full((esn.N_dim, 1), 0.3), n=20)


def test_parametric_norm_shift_round_trip():
    esn = _trained(input_parameters=np.array([[0.0, 1.0]]))
    # tuned parametric normalization lives ONLY in _norm/_shift: pin custom values
    n = esn.norm.copy()
    n[-1] = 3.7
    esn.norm = n
    sh = esn.shift.copy()
    sh[-1] = -0.4
    esn.shift = sh
    back = EchoStateNetwork.from_arrays(esn.to_arrays())
    assert np.array_equal(back.norm, esn.norm) and np.array_equal(back.shift, esn.shift)
    assert back.N_dim_in == esn.N_dim_in

    # ragged (per-timestep) parameter lists collapse to zeros((N_param, 1))
    esn.input_parameters = [np.zeros((10, 1)), np.ones((7, 1))]
    d = esn.to_arrays()
    assert d['input_parameters'].shape == (1, 1)
    assert EchoStateNetwork.from_arrays(d).N_dim_in == esn.N_dim_in


def test_untrained_esn_round_trip():
    esn = EchoStateNetwork(np.zeros((3, 1)), dt=1, N_units=15, upsample=1, N_wash=5)
    d = esn.to_arrays()
    assert not any(k in d for k in ('Wout', 'W_data', 'Win_data', 'Win'))
    assert not EchoStateNetwork.from_arrays(d).trained


def test_arrays_saved_before_the_intercept_left_the_readout_load_unchanged():
    # before 0.7, the last row of Wout multiplied the output bias, bias_out, and the input bias was inside the input
    # scaling without an entry scale_bias_in: such a store gives the same network, with the intercept beside Wout
    data = np.random.default_rng(3).normal(size=(200, 3)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=20, upsample=1, t_train=120, t_val=30, t_test=0, N_wash=5,
                           hyperparameters_to_optimize=[], scale_bias_in=True, sigma_in=0.3, seed=0)
    esn.train(data, plot_training=False)
    arrays = esn.to_arrays()
    assert arrays['Wout'].shape == (20, 3) and arrays['intercept'].shape == (3,) and 'bias_out' not in arrays
    old = {k: v for k, v in arrays.items() if k not in ('intercept', 'scale_bias_in')}
    old.update(Wout=np.vstack([esn.Wout, esn.intercept / 2.0]), bias_out=np.array([2.0]))
    u, r = data[7].reshape(-1, 1), np.random.default_rng(4).normal(size=(esn.N_units, 1))
    for stored in (arrays, old):
        back = EchoStateNetwork.from_arrays(stored)
        assert back.scale_bias_in is True and back.Wout.shape == (20, 3)
        np.testing.assert_allclose(back.intercept, esn.intercept, rtol=1e-13)
        for a, b in zip(back.step(u, r), esn.step(u, r)):
            np.testing.assert_allclose(a, b, rtol=1e-13)
    assert EchoStateNetwork(data.T).scale_bias_in is False            # the default: the bias outside sigma_in


def test_a_network_pickled_before_the_intercept_left_the_readout_unpickles_unchanged():
    data = np.random.default_rng(3).normal(size=(200, 3)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=20, upsample=1, t_train=120, t_val=30, t_test=0, N_wash=5,
                           hyperparameters_to_optimize=[], scale_bias_in=True, sigma_in=0.3, seed=0)
    esn.train(data, plot_training=False)
    u, r = data[7].reshape(-1, 1), np.random.default_rng(4).normal(size=(esn.N_units, 1))
    old = EchoStateNetwork.__new__(EchoStateNetwork)       # the attributes that a version before 0.7 pickled
    state = {k: v for k, v in esn.__getstate__().items() if k not in ('_intercept', '_intercept_apart', 'scale_bias_in')}
    state.update(_Wout=np.vstack([esn.Wout, esn.intercept / 2.0]), bias_out=np.array([2.0]))
    old.__setstate__(state)
    assert old.scale_bias_in is True and old.Wout.shape == (20, 3)
    for a, b in zip(old.step(u, r), esn.step(u, r)):
        np.testing.assert_allclose(a, b, rtol=1e-13)
    new = EchoStateNetwork(data.T, dt=1, N_units=20, upsample=1, t_train=120, t_val=30, t_test=0, N_wash=5,
                           hyperparameters_to_optimize=[], seed=0)
    new.train(data, plot_training=False)
    back = new.copy()                                       # copy goes through the pickling methods
    assert back.scale_bias_in is False
    np.testing.assert_array_equal(back.step(u, r)[0], new.step(u, r)[0])
