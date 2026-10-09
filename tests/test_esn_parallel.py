"""Parallel layout of EchoStateNetwork: one reservoir per patch of sites
(patch_size, halo, periodic, shared), Pathak et al. (2018)."""
import numpy as np
import pytest

from echostatenetwork import EchoStateNetwork


def _random_parallel(N_dim=12, N_units=10, seed=0, **kw):
    """Parallel ESN with generated Win/W and a random readout, without training
    (with shared=False, one realization and one readout of its window per patch)."""
    esn = EchoStateNetwork(np.zeros((N_dim, 1)), dt=1, N_units=N_units, upsample=1,
                           seed=seed, sigma_in=0.5, **kw)
    esn._generate_W_Win(seed=seed)
    rng = np.random.default_rng(seed)
    if esn.shared:
        esn.Wout = rng.normal(size=(N_units + 1, esn.patch_size))
    else:
        for net in esn.patches:
            net.Wout = rng.normal(size=(N_units + 1, net.N_dim))
    return esn


def _set_norm(esn, rng):
    """Random per-site input scaling: on the parallel network, or per patch window."""
    if esn.shared:
        esn.norm, esn.shift = rng.uniform(0.5, 2.0, esn.N_dim), rng.normal(size=esn.N_dim)
    else:
        for net in esn.patches:
            net.norm, net.shift = rng.uniform(0.5, 2.0, net.N_dim), rng.normal(size=net.N_dim)


def _lorenz96(n, dt, nx=40, forcing=8.0, seed=0):
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


@pytest.mark.parametrize('leak_rate', [1.0, 0.4])
def test_one_patch_without_halo_reproduces_the_standard_step(leak_rate):
    # patch_size = N_dim, halo = 0: one reservoir that reads and predicts every site
    rng = np.random.default_rng(0)
    data = rng.normal(size=(80, 6)).cumsum(axis=0)
    kw = dict(dt=1, N_units=15, upsample=1, t_train=40, t_val=15, t_test=10, N_wash=5,
              leak_rate=leak_rate, hyperparameters_to_optimize=[], seed=0)
    std = EchoStateNetwork(data.T, **kw)
    std.train(data, plot_training=False)
    par = EchoStateNetwork(data.T, patch_size=6, halo=0, **kw)
    par.Win, par.W, par.Wout = std.Win, std.W, std.Wout
    par.norm, par.shift = std.norm, std.shift
    assert par.N_patches == 1 and par.N_r == std.N_r == 15

    u, r = data[10:14].T, rng.normal(size=(15, 4))     # an ensemble of 4 members
    for out_par, out_std in zip(par.step(u, r), std.step(u, r)):
        np.testing.assert_array_equal(out_par, out_std)
    np.testing.assert_array_equal(par.reservoir_to_physical(r), std.reservoir_to_physical(r))
    np.testing.assert_allclose(par.Jacobian(u, r), std.Jacobian(u, r), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('shared', [True, False])
def test_a_patch_ignores_the_sites_outside_its_window(periodic, shared):
    esn = _random_parallel(N_dim=12, patch_size=2, halo=1, periodic=periodic, Win_type='dense',
                           shared=shared)
    window_0 = {11, 0, 1, 2} if periodic else {0, 1, 2}    # -1: outside the domain, reads zero
    assert set(esn.patch_sites[0]) - {-1} == window_0

    rng = np.random.default_rng(1)
    u, r = rng.normal(size=(12, 3)), rng.normal(size=(esn.N_r, 3))
    u_out, r_out = esn.step(u, r)
    for site in range(12):
        u_pert = u.copy()
        u_pert[site] += 1.0
        u_out_p, r_out_p = esn.step(u_pert, r)
        for p in range(esn.N_patches):
            sites, units = slice(2 * p, 2 * p + 2), slice(p * esn.N_units, (p + 1) * esn.N_units)
            unchanged = (np.array_equal(u_out_p[sites], u_out[sites])
                         and np.array_equal(r_out_p[units], r_out[units]))
            assert unchanged == (site not in esn.patch_sites[p])


def test_shared_matrices_on_a_ring_are_shift_equivariant():
    esn = _random_parallel(N_dim=12, patch_size=3, halo=2)
    rng = np.random.default_rng(2)
    u, r = rng.normal(size=(12, 2)), rng.normal(size=(esn.N_r, 2))
    u_out, r_out = esn.step(u, r)
    # shift the input by one patch: the output and the reservoirs shift by one patch
    u_s, r_s = esn.step(np.roll(u, 3, axis=0), np.roll(r, esn.N_units, axis=0))
    np.testing.assert_allclose(u_s, np.roll(u_out, 3, axis=0), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(r_s, np.roll(r_out, esn.N_units, axis=0), rtol=1e-13, atol=1e-13)
    # shared matrices, but each patch reads its own window and predicts its own values
    assert not np.allclose(u_out[:3], u_out[3:6])


@pytest.mark.parametrize('shared', [True, False])
def test_ensemble_members_advance_independently(shared):
    esn = _random_parallel(N_dim=12, patch_size=2, halo=2, periodic=False, shared=shared)
    rng = np.random.default_rng(3)
    U, R = rng.normal(size=(12, 5)), rng.normal(size=(esn.N_r, 5))
    u_out, r_out = esn.step(U, R)
    assert u_out.shape == (12, 5) and r_out.shape == (esn.N_r, 5)
    for j in range(5):
        u_j, r_j = esn.step(U[:, j], R[:, j])
        np.testing.assert_allclose(u_j[:, 0], u_out[:, j], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(r_j[:, 0], r_out[:, j], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('periodic, leak_rate', [(True, 1.0), (False, 0.5)])
@pytest.mark.parametrize('shared', [True, False])
def test_patch_jacobian_matches_finite_differences(periodic, leak_rate, shared):
    esn = _random_parallel(N_dim=8, patch_size=2, halo=1, periodic=periodic, leak_rate=leak_rate,
                           shared=shared)
    rng = np.random.default_rng(4)
    _set_norm(esn, rng)                                 # per site, to test the scaling
    u, r = rng.normal(size=(8, 2)), 0.1 * rng.normal(size=(esn.N_r, 2))
    J = esn.Jacobian(u, r)
    assert J.shape == (8, 8, 2)
    eps, J_fd = 1e-5, np.zeros_like(J)
    for j in range(8):
        up, um = u.copy(), u.copy()
        up[j] += eps
        um[j] -= eps
        J_fd[:, j] = (esn.step(up, r)[0] - esn.step(um, r)[0]) / (2 * eps)
    np.testing.assert_allclose(J, J_fd, rtol=1e-6, atol=1e-8)
    # the rows of patch 0 depend only on the columns of its window
    outside = np.setdiff1d(np.arange(8), esn.patch_sites[0])
    assert not J[:2][:, outside].any()


def test_shared_readout_pools_every_patch_in_the_ridge_regression():
    esn = _random_parallel(N_dim=8, patch_size=2, halo=1, N_wash=5)
    rng = np.random.default_rng(5)
    U = rng.normal(size=(1, 60, 8))
    Y = np.roll(U, -1, axis=1)
    LHS, RHS, _, R_RR = esn._compute_RR_terms(U, Y)
    R, n = R_RR[0], esn.N_units
    assert R.shape == (60 - esn.N_wash, esn.N_r)

    def aug(r_p):
        return np.hstack([r_p, np.ones((r_p.shape[0], 1)) * esn.bias_out])

    Y_t = Y[0, esn.N_wash:]
    LHS_sum = sum(aug(R[:, p * n:(p + 1) * n]).T @ aug(R[:, p * n:(p + 1) * n]) for p in range(4))
    RHS_sum = sum(aug(R[:, p * n:(p + 1) * n]).T @ Y_t[:, 2 * p:2 * p + 2] for p in range(4))
    np.testing.assert_allclose(LHS, LHS_sum, rtol=1e-12)
    np.testing.assert_allclose(RHS, RHS_sum, rtol=1e-12, atol=1e-12)


def test_single_series_engine_pools_the_patches_like_the_ridge_regression():
    # one fold that trains on every row: the per-fold readout of the engine behind
    # SSV/WFV/KFV must equal the ridge solution on the pooled patches
    from echostatenetwork import validation
    esn = _random_parallel(N_dim=8, patch_size=2, halo=1, N_wash=5, t_val=10, tikh_range=[1e-6])
    rng = np.random.default_rng(7)
    U = rng.normal(size=(1, 60, 8))
    Y = np.roll(U, -1, axis=1)

    def all_rows(case, n_post):
        return [([(0, n_post)], n_post - case.N_val, case.N_val)]

    validation.single_series_validation([], esn, U, Y, np.zeros(1), [], False, all_rows, 'all_rows')
    LHS, RHS = esn._compute_RR_terms(U, Y)[:2]
    LHS[np.diag_indices_from(LHS)] += 1e-6
    np.testing.assert_allclose(esn.Wout, np.linalg.solve(LHS, RHS), rtol=1e-6, atol=1e-9)


def test_lorenz96_parallel_esn_forecasts_in_closed_loop():
    dt, n_train = 0.02, 3000
    steps_lyap = int(round(1 / 1.67 / dt))          # one Lyapunov time of Lorenz-96 (F = 8)
    X = _lorenz96(n_train + 1500, dt)
    esn = EchoStateNetwork(X[:1].T, dt=dt, upsample=1, N_units=100, patch_size=2, halo=2,
                           N_wash=50, t_train=0.8 * n_train * dt, t_val=0.2 * n_train * dt,
                           t_test=0., rho=0.6, sigma_in=1.0, tikh=1e-6, noise=1e-3, seed=0,
                           hyperparameters_to_optimize=[], verbose=False)
    esn.train(X[:n_train], plot_training=False)
    assert (esn.N_patches, esn.N_r) == (20, 2000)
    assert esn.Win.shape == (100, 7) and esn.Wout.shape == (101, 2)
    assert np.ptp(esn.norm) == 0 and np.ptp(esn.shift) == 0     # pooled over the sites

    positions = esn.state_positions
    assert positions.shape == (40 + 2000,)
    np.testing.assert_array_equal(positions[:40], np.arange(40))
    assert np.all(positions[40:140] == 0.5) and positions[-1] == 38.5

    # washout and closed-loop forecast from 5 held-out starts, advanced as an ensemble
    starts = n_train + 100 + np.arange(5) * 3 * steps_lyap
    r = np.zeros((esn.N_r, 5))
    for k in range(esn.N_wash, 0, -1):
        u, r = esn.step(X[starts - k].T, r)          # u forecasts X[starts]
    scale = np.sqrt(np.mean(np.sum(X ** 2, axis=1)))
    n_steps = 3 * steps_lyap
    valid = np.full(5, n_steps)
    for k in range(n_steps):
        err = np.linalg.norm(u - X[starts + k].T, axis=0) / scale
        valid = np.where((err > 0.5) & (valid == n_steps), k, valid)
        u, r = esn.step(u, r)
    assert np.median(valid) / steps_lyap > 0.2


@pytest.mark.parametrize('strategy', [None, EchoStateNetwork._SSV, EchoStateNetwork._WFV,
                                      EchoStateNetwork._KFV])
def test_validation_strategies_with_patches(strategy):
    rng = np.random.default_rng(6)
    data = rng.normal(size=(200, 8)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, patch_size=2, halo=1,
                           t_train=120, t_val=30, t_test=40, N_wash=5, N_folds=3, N_grid=2,
                           N_func_evals=2, hyperparameters_to_optimize=['rho'], seed=0,
                           verbose=False)
    esn.train(data, plot_training=False, validation_strategy=strategy)
    assert esn.trained and esn.Wout.shape == (13, 2) and np.isfinite(esn.Wout).all()
    assert len(esn.bo_results['func_vals']) == 2


def test_parallel_layout_survives_npz(tmp_path):
    esn = _random_parallel(N_dim=8, patch_size=2, halo=1, periodic=False)
    esn.norm, esn.shift = np.full(8, 2.0), np.full(8, 0.3)
    p = tmp_path / 'esn.npz'
    np.savez_compressed(p, **esn.to_arrays())
    back = EchoStateNetwork.from_arrays(np.load(p, allow_pickle=False))
    assert (back.patch_size, back.halo, back.periodic) == (2, 1, False)
    u, r1, r2 = np.full((8, 1), 0.3), np.zeros((esn.N_r, 1)), np.zeros((esn.N_r, 1))
    u1 = u2 = u
    for _ in range(10):
        u1, r1 = esn.step(u1, r1)
        u2, r2 = back.step(u2, r2)
        assert np.array_equal(u1, u2) and np.array_equal(r1, r2)


# ---------------------------------------------------------------------- independent patches

@pytest.mark.parametrize('periodic', [True, False])
def test_independent_patches_with_the_shared_matrices_reproduce_the_shared_network(periodic):
    shared = _random_parallel(N_dim=12, patch_size=3, halo=2, periodic=periodic, leak_rate=0.7)
    shared.norm, shared.shift = np.full(12, 1.7), np.full(12, -0.4)
    local = EchoStateNetwork(np.zeros((12, 1)), dt=1, N_units=10, upsample=1, sigma_in=0.5,
                             patch_size=3, halo=2, periodic=periodic, leak_rate=0.7, shared=False)
    local._generate_W_Win()
    assert len(local.patches) == 4 and local.patches[0].N_dim == 3 + 2 * 2
    for net in local.patches:           # the matrices and hyperparameters of the shared network
        net.Win, net.W = shared.Win, shared.W
        Wout = np.zeros((11, 7))
        Wout[:, 2:5] = shared.Wout      # the columns of the patch sites within the window
        net.Wout = Wout
        net.norm, net.shift = np.full(7, 1.7), np.full(7, -0.4)
        assert (net.rho, net.sigma_in, net.leak_rate) == (shared.rho, shared.sigma_in, 0.7)

    rng = np.random.default_rng(8)
    u, r = rng.normal(size=(12, 3)), rng.normal(size=(shared.N_r, 3))
    for out_local, out_shared in zip(local.step(u, r), shared.step(u, r)):
        np.testing.assert_array_equal(out_local, out_shared)
    np.testing.assert_array_equal(local.reservoir_to_physical(r), shared.reservoir_to_physical(r))
    # same blocks, summed in a different order
    np.testing.assert_allclose(local.Jacobian(u, r), shared.Jacobian(u, r), rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(local.state_positions, shared.state_positions)


def test_independent_patches_fit_one_ridge_regression_per_patch():
    rng = np.random.default_rng(9)
    data = rng.normal(size=(200, 8)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, patch_size=2, halo=1,
                           shared=False, t_train=150, t_val=30, t_test=0, N_wash=5, tikh=1e-6,
                           hyperparameters_to_optimize=[], seed=0, verbose=False)
    esn.train(data, plot_training=False, add_noise=False)
    assert esn.trained and len(esn.patches) == 4
    # independent random realizations, one per patch
    assert not np.array_equal(esn.patches[0].W.toarray(), esn.patches[1].W.toarray())
    for p, net in enumerate(esn.patches):
        # direct ridge solve on the samples of patch p alone: open loop on its window
        window = esn._patch_record(data, p)[:esn.N_train + esn.N_val]
        r, R = np.zeros((12, 1)), []
        for u_in in window[:-1]:
            _, r = net.step(u_in, r)
            R.append(r[:, 0])
        R = np.hstack([np.array(R)[esn.N_wash:], np.ones((len(R) - esn.N_wash, 1)) * net.bias_out])
        Y = window[esn.N_wash + 1:, 1:3]                  # the sites of patch p
        Wout_p = np.linalg.solve(R.T @ R + 1e-6 * np.eye(13), R.T @ Y)
        np.testing.assert_allclose(net.Wout[:, 1:3], Wout_p, rtol=1e-6, atol=1e-9)
        # each patch normalizes its own window
        assert net.norm.shape == (4,) and np.ptp(net.norm) > 0


def test_independent_patches_select_their_hyperparameters_per_patch():
    rng = np.random.default_rng(10)
    data = rng.normal(size=(200, 8)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, patch_size=2, halo=1,
                           shared=False, t_train=120, t_val=30, t_test=40, N_wash=5, N_folds=2,
                           N_grid=2, N_func_evals=3, hyperparameters_to_optimize=['rho', 'sigma_in'],
                           seed=0, verbose=False)
    esn.train(data, plot_training=False)
    for net in esn.patches:            # one search of its own per patch
        assert net.bo_results['hp_names'] == ['rho', 'sigma_in']
        assert len(net.bo_results['func_vals']) == 3
    assert len(esn.training_summary().splitlines()) == 4
    # validation probes: the patch sites come from the forecast, the halo from the data
    net = esn.patches[0]
    u_in = net._closed_loop_input(np.ones((4, 1)), np.zeros(4))
    np.testing.assert_array_equal(u_in[:, 0], [0, 1, 1, 0])


def test_independent_patches_survive_npz(tmp_path):
    esn = _random_parallel(N_dim=8, patch_size=2, halo=1, periodic=False, shared=False)
    for p, net in enumerate(esn.patches):        # hyperparameters and scaling of their own
        net.rho, net.sigma_in, net.tikh = 0.5 + 0.1 * p, 0.2 * (p + 1), 10.0 ** -p
        net.norm, net.shift = np.full(4, 1.0 + p), np.full(4, 0.1 * p)
    path = tmp_path / 'esn.npz'
    np.savez_compressed(path, **esn.to_arrays())
    back = EchoStateNetwork.from_arrays(np.load(path, allow_pickle=False))
    assert (back.patch_size, back.halo, back.periodic, back.shared) == (2, 1, False, False)
    assert back.trained
    for net, net_back in zip(esn.patches, back.patches):
        assert (net_back.rho, net_back.sigma_in, net_back.tikh) == (net.rho, net.sigma_in, net.tikh)
        assert net_back._forecast_sites == net._forecast_sites
    u1 = u2 = np.full((8, 1), 0.3)
    r1, r2 = np.zeros((esn.N_r, 1)), np.zeros((esn.N_r, 1))
    for _ in range(10):
        u1, r1 = esn.step(u1, r1)
        u2, r2 = back.step(u2, r2)
        assert np.array_equal(u1, u2) and np.array_equal(r1, r2)


def test_errors_of_independent_patches():
    with pytest.raises(ValueError, match='shared=False requires'):
        EchoStateNetwork(np.zeros((8, 1)), shared=False)
    esn = _random_parallel(N_dim=8, patch_size=2, halo=1, shared=False)
    for name in ('Win', 'W', 'Wout'):
        with pytest.raises(AttributeError, match=rf'patches\[p\]\.{name}'):
            getattr(esn, name)
    with pytest.raises(AttributeError, match='patches'):
        esn.Wout = np.zeros((11, 2))
    with pytest.raises(ValueError, match='N_r'):
        esn.step(np.zeros((8, 1)), np.zeros((esn.N_units, 1)))


@pytest.mark.parametrize('kwargs, match', [
    (dict(patch_size=3), 'divides'),                       # 8 sites
    (dict(patch_size=0), 'divides'),
    (dict(patch_size=2, halo=-1), 'halo'),
    (dict(patch_size=2, halo=4), 'periodic domain'),       # window of 10 sites on a ring of 8
    (dict(patch_size=2, input_parameters=np.zeros((1, 1))), 'input_parameters'),
    (dict(patch_size=2, readout_input=True), 'readout_input'),
    (dict(patch_size=2, observed_idx=np.arange(0, 8, 2)), 'observed_idx'),
])
def test_incompatible_layouts_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EchoStateNetwork(np.zeros((8, 1)), **kwargs)


def test_errors_of_the_parallel_api():
    # a non-periodic domain pads the windows, so the halo may exceed the domain
    padded = EchoStateNetwork(np.zeros((8, 1)), patch_size=2, halo=4, periodic=False)
    assert (padded.patch_sites == -1).any()
    # a single reservoir has no positions and no patches
    single = EchoStateNetwork(np.zeros((8, 1)))
    assert (single.N_patches, single.N_r) == (1, single.N_units)
    with pytest.raises(ValueError, match='no position'):
        single.state_positions
    with pytest.raises(ValueError, match='patch_size'):
        single.patch_sites
    # the reservoir state stacks the patches
    esn = _random_parallel(N_dim=8, patch_size=2, halo=1)
    with pytest.raises(ValueError, match='N_r'):
        esn.step(np.zeros((8, 1)), np.zeros((esn.N_units, 1)))
