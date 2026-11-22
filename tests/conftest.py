"""Helpers shared by the test modules (imported as ``from conftest import ...``)."""
import numpy as np

from echostatenetwork import EchoStateNetwork


def assert_jacobian(esn, u, r, eps=1e-5, rtol=1e-6, atol=1e-8):
    """Compare `esn.Jacobian(u, r)` with central finite differences of `esn.step`, one input column at a time; return
    the analytical Jacobian."""
    J = esn.Jacobian(u, r)
    columns = []
    for j in range(u.shape[0]):
        up, um = u.copy(), u.copy()
        up[j] += eps
        um[j] -= eps
        columns.append((esn.step(up, r)[0] - esn.step(um, r)[0]) / (2 * eps))
    np.testing.assert_allclose(J, np.stack(columns, axis=1).reshape(J.shape), rtol=rtol, atol=atol)
    return J


def npz_round_trip(esn, tmp_path):
    """`esn` saved with `to_arrays` to a compressed npz file, and loaded back without pickle."""
    path = tmp_path / 'esn.npz'
    np.savez_compressed(path, **esn.to_arrays())
    return EchoStateNetwork.from_arrays(np.load(path, allow_pickle=False))


def assert_same_closed_loop(a, b, y, n=10):
    """The closed loops of `a` and `b` from the output `y` are bit-identical for `n` steps."""
    ya, yb = y, y
    ra = rb = np.zeros((a.N_r, y.shape[1]))
    for _ in range(n):
        ya, ra = a.step(a.outputs_to_inputs(ya), ra)
        yb, rb = b.step(b.outputs_to_inputs(yb), rb)
        assert np.array_equal(ya, yb) and np.array_equal(ra, rb)
    return a.outputs_to_inputs(ya), ra


def random_parallel(N_dim=12, N_units=10, seed=0, n_param=0, readout_input=False, **kw):
    """A parallel network with generated Win/W, `n_param` input parameters, a random readout of the rows that
    `readout_input` reads and, with parameters or a readout of the input, a random normalization; untrained (with
    shared=False, one realization and one readout of its window per patch)."""
    rng = np.random.default_rng(seed)
    if n_param:
        kw['input_parameters'] = rng.normal(size=(n_param, 1))
    esn = EchoStateNetwork(np.zeros((N_dim, 1)), dt=1, N_units=N_units, upsample=1, seed=seed, sigma_in=0.5,
                           readout_input=readout_input, **kw)
    esn._generate_W_Win(seed=seed)
    for net in esn.patches if esn._independent else [esn]:
        net.Wout = rng.normal(size=(net._n_readout, net._n_out))
        if n_param or readout_input:
            net.norm, net.shift = rng.uniform(0.5, 2.0, net.N_dim_in), rng.normal(size=net.N_dim_in)
    return esn


def set_norm(esn, rng):
    """Random per-site input scaling: on the parallel network, or per patch window."""
    for net in [esn] if esn.shared else esn.patches:
        net.norm, net.shift = rng.uniform(0.5, 2.0, net.N_dim), rng.normal(size=net.N_dim)


def lorenz96(n, dt, nx=40, forcing=8.0, seed=0):
    """Lorenz-96 record of n steps on a ring of nx sites (RK4, after a transient)."""
    def rhs(x):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing

    x = forcing + 0.1 * np.random.default_rng(seed).standard_normal(nx)
    X = np.empty((n + 1000, nx))
    for k in range(X.shape[0]):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        X[k] = x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X[1000:]
