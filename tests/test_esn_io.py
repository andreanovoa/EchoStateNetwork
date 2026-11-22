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


def test_a_network_without_normalization_survives_npz(tmp_path):
    # norm_method=None was stored as an object array, which np.load(allow_pickle=False) refuses
    data = np.random.default_rng(5).normal(size=(200, 3)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=10, upsample=1, t_test=0, N_wash=5, hyperparameters_to_optimize=[],
                           norm_method=None)
    esn.train(data, plot_training=False)
    assert esn.to_arrays()['norm_method'] == 'None'
    back = npz_round_trip(esn, tmp_path)
    assert back.norm_method is None
    u, r = data[7].reshape(-1, 1), np.random.default_rng(6).normal(size=(esn.N_units, 1))
    np.testing.assert_array_equal(back.step(u, r)[0], esn.step(u, r)[0])
