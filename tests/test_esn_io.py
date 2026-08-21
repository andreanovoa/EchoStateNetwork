"""to_arrays / from_arrays: the pickle-free deployment-state round trip."""
import numpy as np
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


def test_to_from_arrays_bitwise_closed_loop():
    esn = _trained()
    back = EchoStateNetwork.from_arrays(esn.to_arrays())
    assert back.trained
    assert np.array_equal(back.Wout, esn.Wout)
    assert np.array_equal(back.W.toarray(), esn.W.toarray())
    assert np.array_equal(np.asarray(back.Win.todense()), np.asarray(esn.Win.todense()))
    u = np.full((esn.N_dim, 1), 0.3)
    r1 = r2 = np.zeros((esn.N_units, 1))
    for _ in range(20):
        u1, r1 = esn.step(esn.outputs_to_inputs(full_state=u), r1)
        u2, r2 = back.step(back.outputs_to_inputs(full_state=u), r2)
        assert np.array_equal(u1, u2) and np.array_equal(r1, r2)
        u = u1


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


def test_arrays_survive_npz(tmp_path):
    esn = _trained()
    p = tmp_path / 'esn.npz'
    np.savez_compressed(p, **esn.to_arrays())
    z = np.load(p, allow_pickle=False)          # no object arrays may leak in
    back = EchoStateNetwork.from_arrays(z)
    assert isinstance(back.W, csr_matrix)
    assert np.array_equal(back.Wout, esn.Wout)
    assert back.Win_type == esn.Win_type and back.seed == esn.seed
