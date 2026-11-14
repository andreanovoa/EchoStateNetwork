"""open_loop, closed_loop_step and forecast: the public loops of a trained network, and state_positions of a single
reservoir."""
import numpy as np
import pytest

from echostatenetwork import EchoStateNetwork


def _trained(parametric=False, **kw):
    """A small network trained on a random walk (two segments at two parameters when parametric)."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 120, 3)).cumsum(axis=1) / 5
    opts = dict(dt=1, N_units=25, upsample=1, t_train=60, t_val=20, t_test=10, N_wash=5, sigma_in=0.1,
                hyperparameters_to_optimize=[], seed=0, **kw)
    if parametric:
        esn = EchoStateNetwork(data[0].T, input_parameters=np.array([[0.5, 1.5]]), **opts)
        esn.train(data, plot_training=False)
    else:
        esn = EchoStateNetwork(data[0].T, **opts)
        esn.train(data[0], plot_training=False)
    return esn, data[0]


def _parallel():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(150, 8)).cumsum(axis=0) / 5
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=80, t_val=30, t_test=10, N_wash=5,
                           hyperparameters_to_optimize=[], patch_size=2, halo=1, seed=0)
    esn.train(data, plot_training=False)
    return esn, data


def _manual_open_loop(esn, U, r):
    Y, R = [], []
    for u_in in U:
        u, r = esn.step(esn.outputs_to_inputs(u_in.reshape(esn.N_dim, -1)), r)
        Y.append(u)
        R.append(r)
    return np.array(Y), np.array(R)


@pytest.mark.parametrize('make', [_trained, _parallel])
def test_open_loop_is_a_manual_step_loop(make):
    esn, data = make()
    Y, R = esn.open_loop(data[:12])
    Y0, R0 = _manual_open_loop(esn, data[:12], np.zeros((esn.N_r, 1)))
    np.testing.assert_array_equal(Y, Y0[..., 0])
    np.testing.assert_array_equal(R, R0[..., 0])
    assert Y.shape == (12, esn.N_dim) and R.shape == (12, esn.N_r)
    # from a given state, and an ensemble (n, N_dim, m)
    U = np.stack([data[:6], data[3:9]], axis=-1)
    r0 = np.random.default_rng(2).normal(size=(esn.N_r, 2))
    Y, R = esn.open_loop(U, r0=r0)
    Y0, R0 = _manual_open_loop(esn, U, r0)
    np.testing.assert_array_equal(Y, Y0)
    np.testing.assert_array_equal(R, R0)


@pytest.mark.parametrize('make', [_trained, _parallel])
def test_forecast_is_a_manual_single_step_loop(make):
    esn, data = make()
    Y_open, R_open = esn.open_loop(data[:10])
    u, r = Y_open[-1][:, None], R_open[-1][:, None]
    Y, u_end, r_end = esn.forecast(u, r, 15)
    for i in range(15):
        u, r = esn._single_step(u, r)
        np.testing.assert_array_equal(Y[i], u)
    np.testing.assert_array_equal(u_end, u)
    np.testing.assert_array_equal(r_end, r)
    assert Y.shape == (15, esn.N_dim, 1)
    # the forecast continues from its last state
    Y2 = esn.forecast(u_end, r_end, 5)[0]
    np.testing.assert_array_equal(np.concatenate([Y, Y2]), esn.forecast(Y_open[-1], R_open[-1], 20)[0])


def test_alignment_of_the_washout_and_the_forecast():
    # a network trained on a periodic signal: the last washout output predicts row len(wash), and forecast Y[i]
    # row len(wash) + 1 + i. A forecast one step off is farther from the data by about one step of the signal.
    t = 0.3 * np.arange(400)
    data = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    esn = EchoStateNetwork(data.T, dt=1, N_units=60, upsample=1, t_train=250, t_val=60, t_test=10, N_wash=20,
                           sigma_in=0.3, rho=0.8, tikh=1e-10, hyperparameters_to_optimize=[], seed=0)
    esn.train(data, plot_training=False)
    n_wash, n = 40, 30
    Y_open, R_open = esn.open_loop(data[:n_wash])
    Y = esn.forecast(Y_open[-1], R_open[-1], n)[0][..., 0]
    err = {s: np.abs(Y - data[n_wash + 1 + s:n_wash + 1 + s + n]).max() for s in (-1, 0, 1)}
    assert np.abs(Y_open[-1] - data[n_wash]).max() < 1e-2
    assert err[0] < 1e-2 < min(err[-1], err[1])


def test_ensemble_is_per_column():
    esn, data = _trained(parametric=True)
    F = np.array([0.6, 1.4, 1.0])
    U = np.stack([data[:8], data[2:10], data[4:12]], axis=-1)
    Y, R = esn.open_loop(U, input_parameters=F)
    Yf, *_ = esn.forecast(Y[-1], R[-1], 10, input_parameters=F)
    for j, f in enumerate(F):
        Yj, Rj = esn.open_loop(U[..., j], input_parameters=f)
        np.testing.assert_allclose(Y[..., j], Yj, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(R[..., j], Rj, rtol=1e-12, atol=1e-13)
        Yfj = esn.forecast(Y[-1][:, j:j + 1], R[-1][:, j:j + 1], 10, input_parameters=f)[0]
        np.testing.assert_allclose(Yf[..., j], Yfj[..., 0], rtol=1e-10, atol=1e-12)


def test_input_parameters_are_restored():
    esn, data = _trained(parametric=True)
    own = esn.input_parameters
    r0 = np.zeros((esn.N_r, 1))
    # a scalar is the same as one column per member; the parameter changes the output
    a = esn.closed_loop_step(data[0][:, None], r0, input_parameters=0.6)[0]
    b = esn.closed_loop_step(data[0][:, None], r0, input_parameters=np.array([[0.6]]))[0]
    c = esn.closed_loop_step(data[0][:, None], r0, input_parameters=1.4)[0]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert esn.input_parameters is own
    # restored after an exception inside the loop
    with pytest.raises(Exception):
        esn.forecast(np.full((esn.N_dim, 2), np.nan), np.zeros((esn.N_r + 1, 2)), 3, input_parameters=[0.6, 1.4])
    assert esn.input_parameters is own
    with pytest.raises(Exception):
        esn.open_loop(data[:4, :2], input_parameters=0.6)
    assert esn.input_parameters is own
    # a network without parameters refuses them
    plain, _ = _trained()
    with pytest.raises(ValueError, match='without input_parameters'):
        plain.closed_loop_step(data[0], np.zeros(plain.N_r), input_parameters=1.0)


def test_from_arrays_has_the_loops():
    esn, data = _trained(parametric=True)
    back = EchoStateNetwork.from_arrays(esn.to_arrays())
    for net in (esn, back):
        Y, R = net.open_loop(data[:10], input_parameters=0.8)
        out = net.forecast(Y[-1], R[-1], 12, input_parameters=0.8)
        if net is esn:
            ref = (Y, R) + out
        else:
            for x, y in zip((Y, R) + out, ref):
                np.testing.assert_array_equal(x, y)


def test_state_positions_of_a_single_reservoir():
    for win in ('sparse', 'sparse_dense_bias'):
        esn, _ = _trained(Win_type=win)
        W = esn.Win[:, :esn.N_dim]
        W = np.abs(W.toarray() if hasattr(W, 'toarray') else W)
        expected = np.concatenate([np.arange(esn.N_dim), np.where(W.max(1) > 0, W.argmax(1), np.nan)])
        np.testing.assert_array_equal(esn.state_positions, expected)
        assert esn.state_positions.shape == (esn.N_dim + esn.N_r,)
    # a unit that reads several components has no position
    dense, _ = _trained(Win_type='dense')
    with pytest.raises(ValueError, match='no position'):
        dense.state_positions
    with pytest.raises(ValueError, match='no position'):
        EchoStateNetwork(np.zeros((8, 1))).state_positions
