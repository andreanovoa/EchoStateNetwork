"""Parallel layout of EchoStateNetwork: one reservoir per patch of sites
(patch_size, halo, periodic, shared), Pathak et al. (2018)."""
import numpy as np
import pytest
from conftest import assert_jacobian, assert_same_closed_loop, lorenz96, npz_round_trip, random_parallel, set_norm

from echostatenetwork import EchoStateNetwork, validation


@pytest.mark.parametrize('leak_rate, parametric', [(1.0, False), (0.4, False), (1.0, True)])
def test_one_patch_without_halo_reproduces_the_standard_step(leak_rate, parametric):
    # patch_size = N_dim, halo = 0: one reservoir that reads and predicts every site; the parameter columns follow
    # the window, as the standard network's
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 80, 6)).cumsum(axis=1)
    kw = dict(dt=1, N_units=15, upsample=1, t_train=40, t_val=15, t_test=10, N_wash=5,
              leak_rate=leak_rate, hyperparameters_to_optimize=[], seed=0)
    if parametric:
        kw['input_parameters'] = np.array([[0.0, 1.0]])
    std = EchoStateNetwork(data[0].T, **kw)
    std.train(data if parametric else data[0], plot_training=False)
    par = EchoStateNetwork(data[0].T, patch_size=6, halo=0, **kw)
    par.Win, par.W, par.Wout = std.Win, std.W, std.Wout
    par.norm, par.shift = std.norm, std.shift
    assert par.N_patches == 1 and par.N_r == std.N_r == 15
    if parametric:
        std.input_parameters = par.input_parameters = np.array([[0.3, 0.7, 0.3, 0.7]])
    u, r = std.outputs_to_inputs(data[0, 10:14].T), rng.normal(size=(15, 4))     # an ensemble of 4 members
    for out_par, out_std in zip(par.step(u, r), std.step(u, r)):
        np.testing.assert_allclose(out_par, out_std, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(par.reservoir_to_physical(r), std.reservoir_to_physical(r), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(par.Jacobian(u, r), std.Jacobian(u, r), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('shared', [True, False])
def test_a_patch_ignores_the_sites_outside_its_window(periodic, shared):
    esn = random_parallel(N_dim=12, patch_size=2, halo=1, periodic=periodic, Win_type='dense', shared=shared)
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
    # the members of an ensemble advance independently
    for j in range(3):
        u_j, r_j = esn.step(u[:, j], r[:, j])
        np.testing.assert_allclose(u_j[:, 0], u_out[:, j], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(r_j[:, 0], r_out[:, j], rtol=1e-13, atol=1e-13)


def test_shared_matrices_on_a_ring_are_shift_equivariant():
    esn = random_parallel(N_dim=12, patch_size=3, halo=2)
    rng = np.random.default_rng(2)
    u, r = rng.normal(size=(12, 2)), rng.normal(size=(esn.N_r, 2))
    u_out, r_out = esn.step(u, r)
    # shift the input by one patch: the output and the reservoirs shift by one patch
    u_s, r_s = esn.step(np.roll(u, 3, axis=0), np.roll(r, esn.N_units, axis=0))
    np.testing.assert_allclose(u_s, np.roll(u_out, 3, axis=0), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(r_s, np.roll(r_out, esn.N_units, axis=0), rtol=1e-13, atol=1e-13)
    # shared matrices, but each patch reads its own window and predicts its own values
    assert not np.allclose(u_out[:3], u_out[3:6])


@pytest.mark.parametrize('periodic, leak_rate', [(True, 1.0), (False, 0.5)])
@pytest.mark.parametrize('shared', [True, False])
@pytest.mark.parametrize('options', [
    dict(),
    dict(r2_mode=True),
    dict(n_param=2),
    dict(n_param=2, readout_input='params', param_in_reservoir=False, Win_type='dense'),
    dict(n_param=2, readout_input='params', Win_type='dense'),
    dict(n_param=2, readout_input=True, Win_type='dense'),
], ids=['plain', 'r2_mode', 'params', 'readout_params_only', 'readout_params', 'readout_input'])
def test_patch_jacobian_matches_finite_differences(options, shared, periodic, leak_rate):
    esn = random_parallel(N_dim=8, patch_size=2, halo=1, periodic=periodic, leak_rate=leak_rate, shared=shared,
                          **options)
    rng = np.random.default_rng(4)
    if not options.get('n_param'):
        set_norm(esn, rng)                                 # per site, to test the scaling
    if options.get('r2_mode') and not shared:
        assert all(net.r2_mode for net in esn.patches)     # the patches inherit the option
    n_in = esn.N_dim_in
    u, r = rng.normal(size=(n_in, 2)), 0.1 * rng.normal(size=(esn.N_r, 2))
    J = assert_jacobian(esn, u, r)
    assert J.shape == (8, n_in, 2)
    # the rows of patch 0 depend only on the columns of its window (and on the parameters)
    outside = np.setdiff1d(np.arange(8), esn.patch_sites[0])
    assert not J[:2][:, outside].any()


@pytest.mark.parametrize('readout_input', [False, True])
def test_shared_readout_pools_every_patch_in_the_ridge_regression(readout_input):
    # the sample of patch p reads [r_p; 1], or [r_p; window_p; p; 1] with readout_input=True, from the input that
    # produced r_p
    n_param = 2 if readout_input else 0
    esn = random_parallel(N_dim=8, patch_size=2, halo=1, periodic=False, N_wash=5, n_param=n_param,
                          readout_input=readout_input)
    rng = np.random.default_rng(15)
    U = rng.normal(size=(1, 60, 8))
    if n_param:
        U = np.concatenate([U, np.broadcast_to(esn.input_parameters.T, (1, 60, n_param))], axis=-1)
    Y = np.roll(U[..., :8], -1, axis=1)
    LHS, RHS, _, R_RR = esn._compute_RR_terms(U, Y)
    R, n, U_t, Y_t = R_RR[0], esn.N_units, U[0, esn.N_wash:], Y[0, esn.N_wash:]
    assert R.shape == (60 - esn.N_wash, esn.N_r)

    def aug(p):
        inputs = []
        if readout_input:
            U_norm = esn.normalize_input(U_t.T).T
            padded = np.hstack([U_norm[:, :8], np.zeros((U_t.shape[0], 1))])
            inputs = [padded[:, esn.patch_sites[p]], U_norm[:, 8:]]
        return np.hstack([R[:, p * n:(p + 1) * n], *inputs, np.ones((R.shape[0], 1)) * esn.bias_out])

    assert aug(0).shape[1] == esn._n_readout
    np.testing.assert_allclose(LHS, sum(aug(p).T @ aug(p) for p in range(4)), rtol=1e-12)
    np.testing.assert_allclose(RHS, sum(aug(p).T @ Y_t[:, 2 * p:2 * p + 2] for p in range(4)),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('options', [dict(), dict(readout_input=True, n_param=2), dict(r2_mode=True)],
                         ids=['plain', 'readout_input', 'r2_mode'])
def test_single_series_engine_pools_the_patches_like_the_ridge_regression(options):
    # one fold that trains on every row: the per-fold readout of the engine behind SSV/WFV/KFV must equal the ridge
    # solution on the pooled patches
    esn = random_parallel(N_dim=8, patch_size=2, halo=1, N_wash=5, t_val=10, tikh_range=[1e-6], **options)
    rng = np.random.default_rng(7)
    U = rng.normal(size=(1, 60, 8))
    if options.get('n_param'):
        U = np.concatenate([U, np.broadcast_to(esn.input_parameters.T, (1, 60, 2))], axis=-1)
    Y = np.roll(U[..., :8], -1, axis=1)

    def all_rows(case, n_post):
        return [([(0, n_post)], n_post - case.N_val, case.N_val)]

    validation.single_series_validation([], esn, U, Y, np.zeros(1), [], False, all_rows, 'all_rows')
    LHS, RHS = esn._compute_RR_terms(U, Y)[:2]
    LHS[np.diag_indices_from(LHS)] += 1e-6
    np.testing.assert_allclose(esn.Wout, np.linalg.solve(LHS, RHS), rtol=1e-6, atol=1e-9)


def test_lorenz96_parallel_esn_forecasts_in_closed_loop():
    dt, n_train = 0.02, 3000
    steps_lyap = int(round(1 / 1.67 / dt))          # one Lyapunov time of Lorenz-96 (F = 8)
    X = lorenz96(n_train + 1500, dt)
    esn = EchoStateNetwork(X[:1].T, dt=dt, upsample=1, N_units=100, patch_size=2, halo=2,
                           N_wash=50, t_train=0.8 * n_train * dt, t_val=0.2 * n_train * dt,
                           t_test=0., rho=0.6, sigma_in=1.0, tikh=1e-6, noise=1e-3, seed=0,
                           hyperparameters_to_optimize=[])
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
                           N_func_evals=2, hyperparameters_to_optimize=['rho'], seed=0)
    esn.train(data, plot_training=False, validation_strategy=strategy)
    assert esn.trained and esn.Wout.shape == (13, 2) and np.isfinite(esn.Wout).all()
    assert len(esn.bo_results['func_vals']) == 2


def test_every_patch_reads_the_input_parameters():
    esn = random_parallel(N_dim=8, patch_size=2, halo=1, n_param=2)
    assert esn.N_dim_in == 10 and esn.Win.shape == (10, 4 + 2 + 1)
    assert np.all(esn.Win.toarray()[:, 4:6] != 0)          # dense parameter columns
    rng = np.random.default_rng(5)
    u, r = rng.normal(size=(10, 3)), 0.1 * rng.normal(size=(esn.N_r, 3))
    du = u.copy()
    du[8] += 1.0                                            # every reservoir moves with the parameters
    changed = np.abs(esn.step(du, r)[1] - esn.step(u, r)[1]).reshape(4, esn.N_units, 3).max(axis=(1, 2))
    assert np.all(changed > 0)


# ---------------------------------------------------------------------- independent patches

@pytest.mark.parametrize('periodic', [True, False])
def test_independent_patches_with_the_shared_matrices_reproduce_the_shared_network(periodic):
    shared = random_parallel(N_dim=12, patch_size=3, halo=2, periodic=periodic, leak_rate=0.7)
    # a normalization that differs from site to site, so that the networks of the patches are not interchangeable:
    # a patch stepped with the window or the shift of another patch changes the outputs
    scale = np.random.default_rng(9)
    norm, shift = scale.uniform(0.5, 2.0, 12), scale.normal(size=12)
    shared.norm, shared.shift = norm, shift
    local = EchoStateNetwork(np.zeros((12, 1)), dt=1, N_units=10, upsample=1, sigma_in=0.5,
                             patch_size=3, halo=2, periodic=periodic, leak_rate=0.7, shared=False)
    local._generate_W_Win()
    assert len(local.patches) == 4 and local.patches[0].N_dim == 3 + 2 * 2
    for net, sites in zip(local.patches, local.patch_sites):     # the matrices and hyperparameters of the shared
        net.Win, net.W = shared.Win, shared.W                       # network, and its normalization of the window
        Wout = np.zeros((11, 7))
        Wout[:, 2:5] = shared.Wout      # the columns of the patch sites within the window
        net.Wout = Wout
        # outside a non-periodic domain, any value: the position reads zero once normalized
        net.norm, net.shift = np.where(sites >= 0, norm[sites], 3.0), np.where(sites >= 0, shift[sites], 5.0)
        assert (net.rho, net.sigma_in, net.leak_rate) == (shared.rho, shared.sigma_in, 0.7)

    rng = np.random.default_rng(8)
    u, r = rng.normal(size=(12, 3)), rng.normal(size=(shared.N_r, 3))
    # equal up to rounding, not bit for bit: the reservoir update is the same, but the readout multiplies
    # Wout by N_ens columns in each patch network and by N_patches * N_ens columns in the shared network,
    # and a BLAS library may sum a column in a different order for a different shape of the product
    # (Accelerate does for these shapes), so the outputs can differ by a few units in the last place
    tol = dict(rtol=1e-12, atol=1e-14)
    for out_local, out_shared in zip(local.step(u, r), shared.step(u, r)):
        np.testing.assert_allclose(out_local, out_shared, **tol)
    np.testing.assert_allclose(local.reservoir_to_physical(r), shared.reservoir_to_physical(r), **tol)
    # same blocks, summed in a different order
    np.testing.assert_allclose(local.Jacobian(u, r), shared.Jacobian(u, r), **tol)
    np.testing.assert_array_equal(local.state_positions, shared.state_positions)


def test_independent_patches_fit_one_ridge_regression_per_patch():
    rng = np.random.default_rng(9)
    data = rng.normal(size=(200, 8)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, patch_size=2, halo=1,
                           shared=False, t_train=150, t_val=30, t_test=0, N_wash=5, tikh=1e-6,
                           hyperparameters_to_optimize=[], seed=0)
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
                           seed=0)
    esn.train(data, plot_training=False)
    for net in esn.patches:            # one search of its own per patch
        assert net.bo_results['hp_names'] == ['rho', 'sigma_in']
        assert len(net.bo_results['func_vals']) == 3
    assert len(esn.training_summary().splitlines()) == 4
    # validation probes: the patch sites come from the forecast, the halo from the data
    net = esn.patches[0]
    u_in = net._closed_loop_input(np.ones((4, 1)), np.zeros(4))
    np.testing.assert_array_equal(u_in[:, 0], [0, 1, 1, 0])


@pytest.mark.parametrize('shared', [True, False])
def test_patches_search_tikh_alone(shared):
    # issue #1 with the parallel layout: the shared network, or the network of every patch, selects tikh from
    # tikh_range in one evaluation of the validation strategy
    data = np.random.default_rng(10).normal(size=(200, 8)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, patch_size=2, halo=1, shared=shared,
                           t_train=120, t_val=30, t_test=40, N_wash=5, N_folds=2,
                           hyperparameters_to_optimize=['tikh'], seed=0)
    esn.train(data, plot_training=False)
    nets = [esn] if shared else esn.patches
    for net in nets:
        assert net.bo_results['hp_names'] == [] and len(net.bo_results['func_vals']) == 1
        assert net.tikh in net.tikh_range and np.isfinite(net.Wout).all()
    assert esn.training_summary().count('tikh alone') == len(nets)


def test_errors_of_the_parallel_api():
    with pytest.raises(ValueError, match='shared=False requires'):
        EchoStateNetwork(np.zeros((8, 1)), shared=False)
    esn = random_parallel(N_dim=8, patch_size=2, halo=1, shared=False)
    for name in ('Win', 'W', 'Wout'):
        with pytest.raises(AttributeError, match=rf'patches\[p\]\.{name}'):
            getattr(esn, name)
    with pytest.raises(AttributeError, match='patches'):
        esn.Wout = np.zeros((11, 2))
    for net in (esn, random_parallel(N_dim=8, patch_size=2, halo=1)):     # the reservoir state stacks the patches
        with pytest.raises(ValueError, match='N_r'):
            net.step(np.zeros((8, 1)), np.zeros((net.N_units, 1)))
    # a non-periodic domain pads the windows, so the halo may exceed the domain
    assert (EchoStateNetwork(np.zeros((8, 1)), patch_size=2, halo=4, periodic=False).patch_sites == -1).any()
    # a single reservoir has no patches (and no positions before Win is drawn)
    single = EchoStateNetwork(np.zeros((8, 1)))
    assert (single.N_patches, single.N_r) == (1, single.N_units)
    with pytest.raises(ValueError, match='patch_size'):
        single.patch_sites


@pytest.mark.parametrize('kwargs, match', [
    (dict(patch_size=3), 'divides'),                       # 8 sites
    (dict(patch_size=0), 'divides'),
    (dict(patch_size=2, halo=-1), 'halo'),
    (dict(patch_size=2, halo=4), 'periodic domain'),       # window of 10 sites on a ring of 8
    (dict(patch_size=2, observed_idx=np.arange(0, 8, 2)), 'observed_idx'),
])
def test_incompatible_layouts_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EchoStateNetwork(np.zeros((8, 1)), **kwargs)


@pytest.mark.parametrize('patch', [None, 2])
def test_sigma_param_scales_the_parameter_columns(patch, tmp_path):
    """sigma_param equal to sigma_in is the network without it; another value changes only the parameter input, and
    the Jacobian (parameter columns included) matches finite differences; the value survives npz."""
    layout = dict(patch_size=patch, halo=1) if patch else {}
    esn = random_parallel(N_dim=8, input_parameters=np.array([[0.5]]), **layout)
    rng = np.random.default_rng(3)
    esn.norm, esn.shift = rng.uniform(0.5, 2.0, 9), rng.normal(size=9)
    u, r = rng.normal(size=(9, 2)), 0.1 * rng.normal(size=(esn.N_r, 2))
    base = esn.step(u, r)
    esn.sigma_param = esn.sigma_in
    for a, b in zip(esn.step(u, r), base):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
    esn.sigma_param = 2.0
    assert not np.allclose(esn.step(u, r)[1], base[1])
    assert_jacobian(esn, u, r)
    assert npz_round_trip(esn, tmp_path).sigma_param == 2.0


def test_sigma_param_is_searched():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(2, 120, 4)).cumsum(axis=1) * 0.1
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=20, upsample=1, t_train=80, t_val=20, t_test=20, N_wash=5,
                           N_folds=2, N_grid=2, N_func_evals=8, seed=0, input_parameters=np.array([[0.0, 1.0]]),
                           optimize_parameter_normalization='data', sigma_param_range=(-2, 0),
                           hyperparameters_to_optimize=['rho', 'sigma_in', 'sigma_param', 'tikh'])
    esn.train(data, plot_training=False)
    assert esn.bo_results['hp_names'] == ['rho', 'sigma_in', 'sigma_param']
    assert 10 ** -2 <= esn.sigma_param <= 1.0


# ---------------------------------------------------------------------- readout_input with the parallel layout

def _lorenz96_forcings(forcings=(6.0, 8.0), n=900, dt=0.02):
    """Two Lorenz-96 records of 10 sites, one per forcing, and the forcings as (1, L) parameters."""
    data = np.stack([lorenz96(n, dt, nx=10, forcing=F, seed=i) for i, F in enumerate(forcings)])
    return data, np.array([forcings])


def _l96_readout_params(data, params, dt=0.02, **kw):
    """The parallel L96-10 network of the washout-free ESN-DA paper, small: two patches of five sites, halo 2,
    the forcing in the readout only."""
    n = data.shape[1]
    options = dict(dt=dt, upsample=1, N_units=40, patch_size=5, halo=2, N_wash=20, t_train=0.6 * n * dt,
                   t_val=0.2 * n * dt, t_test=0.1 * n * dt, rho=0.6, sigma_in=0.5, tikh=1e-6, noise=1e-3, seed=0,
                   hyperparameters_to_optimize=[], input_parameters=params, readout_input='params',
                   param_in_reservoir=False, optimize_parameter_normalization='data')
    return EchoStateNetwork(data[0, :1].T, **{**options, **kw})


@pytest.mark.parametrize('param_in_reservoir', [False, True])
@pytest.mark.parametrize('readout_input', ['params', True])
def test_one_patch_without_halo_reproduces_the_single_readout_input_network(readout_input, param_in_reservoir):
    # patch_size = N_dim, halo = 0: the training (ridge regression included) and the step of the single reservoir;
    # norm_method=None, because the shared patches pool the normalization of the sites
    rng = np.random.default_rng(11)
    data = rng.normal(size=(2, 120, 6)).cumsum(axis=1) * 0.1
    kw = dict(dt=1, N_units=15, upsample=1, t_train=80, t_val=20, t_test=10, N_wash=5, seed=0,
              hyperparameters_to_optimize=[], input_parameters=np.array([[0.0, 1.0]]), norm_method=None,
              optimize_parameter_normalization='data', readout_input=readout_input,
              param_in_reservoir=param_in_reservoir)
    std = EchoStateNetwork(data[0].T, **kw)
    std.train(data, plot_training=False, add_noise=False)
    par = EchoStateNetwork(data[0].T, patch_size=6, halo=0, **kw)
    par.train(data, plot_training=False, add_noise=False)
    assert par.Wout.shape == std.Wout.shape == (std._n_readout, 6)
    np.testing.assert_allclose(par.Wout, std.Wout, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(par.shift, std.shift)     # the parameter is normalized as for one reservoir
    np.testing.assert_array_equal(par.norm, std.norm)
    u, r = np.vstack([data[0, 10:14].T, [[0.2, 0.4, 0.6, 0.8]]]), 0.1 * rng.normal(size=(15, 4))
    for out_par, out_std in zip(par.step(u, r), std.step(u, r)):
        np.testing.assert_allclose(out_par, out_std, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(par.reservoir_to_physical(r, u=u), std.reservoir_to_physical(r, u=u),
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(par.Jacobian(u, r), std.Jacobian(u, r), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('readout_input', ['params', True])
def test_one_independent_patch_without_halo_reproduces_the_single_readout_input_network(readout_input):
    # with shared=False, the network of the patch is a single reservoir on the whole record, with its own seed
    rng = np.random.default_rng(12)
    data = rng.normal(size=(2, 120, 6)).cumsum(axis=1) * 0.1
    kw = dict(dt=1, N_units=15, upsample=1, t_train=80, t_val=20, t_test=10, N_wash=5,
              hyperparameters_to_optimize=[], input_parameters=np.array([[0.0, 1.0]]),
              optimize_parameter_normalization='data', readout_input=readout_input, param_in_reservoir=False)
    local = EchoStateNetwork(data[0].T, patch_size=6, halo=0, shared=False, seed=0, **kw)
    local.train(data, plot_training=False, add_noise=False)
    std = EchoStateNetwork(data[0].T, seed=local.patches[0].seed, **kw)
    std.train(data, plot_training=False, add_noise=False)
    np.testing.assert_allclose(local.patches[0].Wout, std.Wout, rtol=1e-10, atol=1e-12)
    u, r = np.vstack([data[0, 10:13].T, [[0.2, 0.5, 0.9]]]), 0.1 * rng.normal(size=(15, 3))
    for out_local, out_std in zip(local.step(u, r), std.step(u, r)):
        np.testing.assert_allclose(out_local, out_std, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(local.reservoir_to_physical(r, u=u), std.reservoir_to_physical(r, u=u),
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('shared', [True, False])
def test_readout_params_reads_the_parameter_only_in_the_readout(shared, periodic):
    # param_in_reservoir=False: the reservoir update does not depend on the parameter, the readout does,
    # through the direct term Wout_p^T (p - shift_p) / norm_p of every patch
    esn = random_parallel(N_dim=8, patch_size=2, halo=1, shared=shared, periodic=periodic, n_param=2,
                          readout_input='params', param_in_reservoir=False)
    rng = np.random.default_rng(13)
    u, r = rng.normal(size=(10, 3)), 0.1 * rng.normal(size=(esn.N_r, 3))
    u_out, r_out = esn.step(u, r)
    u2 = u.copy()
    u2[8:] += rng.normal(size=(2, 3))
    u2_out, r2_out = esn.step(u2, r)
    np.testing.assert_array_equal(r2_out, r_out)
    assert not np.allclose(u2_out, u_out)
    for p in range(esn.N_patches):
        net = esn.patches[p] if not shared else esn
        W_p, norm_p = net.Wout[esn.N_units:esn.N_units + 2], net.norm[-2:]
        if not shared:
            W_p = W_p[:, 1:3]                            # the sites of the patch within its window
        np.testing.assert_allclose(u2_out[2 * p:2 * p + 2] - u_out[2 * p:2 * p + 2],
                                   W_p.T @ ((u2[8:] - u[8:]) / norm_p[:, None]), rtol=1e-10, atol=1e-12)
    # the readout reads [r_p; p; 1]: reservoir_to_physical needs the input, and reproduces the step
    with pytest.raises(ValueError, match='pass u='):
        esn.reservoir_to_physical(r_out)
    np.testing.assert_allclose(esn.reservoir_to_physical(r_out, u=u), u_out, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('shared', [True, False])
def test_lorenz96_readout_params_parallel_esn_trains_and_forecasts(shared, periodic):
    data, forcings = _lorenz96_forcings()
    esn = _l96_readout_params(data, forcings, shared=shared, periodic=periodic)
    esn.train(data, plot_training=False)
    assert esn.trained
    for net in esn.patches if not shared else [esn]:
        assert net.Wout.shape == (40 + 1 + 1, net._n_out)                # [r_p; F; 1]
        assert not np.asarray(net.Win[:, net._n_in - 1].toarray() if hasattr(net.Win, 'toarray')
                              else net.Win[:, net._n_in - 1]).any()      # F not in the reservoir
        np.testing.assert_allclose(net.shift[-1], 7.0)                  # F normalized as for one reservoir
    # closed loop at each forcing, from the same washout: finite, and the forcing changes the forecast
    forecasts = []
    for F in (6.0, 8.0):
        esn.input_parameters = np.array([[F]])
        r = np.zeros((esn.N_r, 1))
        for u_in in data[1, :20]:
            u, r = esn.step(esn.outputs_to_inputs(u_in[:, None]), r)
        forecast = []
        for _ in range(50):
            forecast.append(u[:, 0])
            u, r = esn.step(esn.outputs_to_inputs(u), r)
        forecasts.append(np.array(forecast))
    assert all(np.isfinite(f).all() for f in forecasts)
    assert not np.allclose(*forecasts)
    # the one-step open-loop forecast at the forcing of the record tracks the record
    esn.input_parameters = np.array([[8.0]])
    r, err = np.zeros((esn.N_r, 1)), []
    for k in range(200):
        u, r = esn.step(esn.outputs_to_inputs(data[1, k][:, None]), r)
        if k >= 20:
            err.append(np.abs(u[:, 0] - data[1, k + 1]).mean())
    assert np.mean(err) < 0.1 * np.abs(data[1]).mean()


@pytest.mark.parametrize('strategy', [None, EchoStateNetwork._SSV, EchoStateNetwork._WFV, EchoStateNetwork._KFV])
@pytest.mark.parametrize('shared', [True, False])
def test_readout_params_through_the_validation_strategies(strategy, shared):
    # None: the default search over the normalization of the parameters, with RVC_Noise; the single-series
    # strategies: tikh alone on one record
    if strategy is None:
        data, forcings = _lorenz96_forcings(n=500)
        options = dict(optimize_parameter_normalization=True, hyperparameters_to_optimize=['rho', 'tikh'],
                       N_func_evals=4, N_grid=2, N_folds=2)
    else:
        data, forcings = _lorenz96_forcings(forcings=(8.0,), n=500)
        options = dict(hyperparameters_to_optimize=['tikh'], t_val=0.1 * 500 * 0.02)
    esn = _l96_readout_params(data, forcings, shared=shared, N_units=20, **options)
    esn.train(data, plot_training=False, validation_strategy=strategy)
    assert esn.trained
    for net in esn.patches if not shared else [esn]:
        assert net.Wout.shape == (22, net._n_out) and np.isfinite(net.Wout).all()
        if strategy is None:
            assert {'param_shift_0', 'param_norm_0'} <= set(net.bo_results['hp_names'])


@pytest.mark.parametrize('shared', [True, False])
def test_parallel_network_survives_npz(shared, tmp_path):
    # the layout, the readout of the parameters, the options read by a later training (param_in_reservoir, connect,
    # optimize_parameter_normalization) and, with independent patches, the hyperparameters of every patch; the closed
    # loop and the Jacobian of the reloaded network are bit-identical
    esn = random_parallel(N_dim=8, patch_size=2, halo=1, shared=shared, periodic=False, n_param=2,
                          readout_input='params', param_in_reservoir=False)
    for p, net in enumerate(esn.patches or []):
        net.rho, net.sigma_in, net.tikh = 0.5 + 0.1 * p, 0.2 * (p + 1), 10.0 ** -p
    esn.connect, esn.optimize_parameter_normalization = 4, 'data'
    back = npz_round_trip(esn, tmp_path)
    assert (back.patch_size, back.halo, back.periodic, back.shared) == (2, 1, False, shared)
    assert back.trained and back.connect == 4 and back.optimize_parameter_normalization == 'data'
    for net, net_back in zip(esn.patches or [esn], back.patches or [back]):
        assert net_back.readout_input == 'params' and net_back.param_in_reservoir is False
        assert (net_back.rho, net_back.sigma_in, net_back.tikh) == (net.rho, net.sigma_in, net.tikh)
        if not shared:
            assert net_back._forecast_sites == net._forecast_sites
    u, r = assert_same_closed_loop(esn, back, np.full((8, 1), 0.3))
    np.testing.assert_array_equal(back.Jacobian(u, r), esn.Jacobian(u, r))
    # a store saved before these training options existed loads with the defaults
    arrays = {k: v for k, v in esn.to_arrays().items()
              if k not in ('param_in_reservoir', 'connect', 'optimize_parameter_normalization')}
    old = EchoStateNetwork.from_arrays(arrays)
    assert old.param_in_reservoir is True and old.connect == EchoStateNetwork.connect


@pytest.mark.parametrize('shared', [True, False])
def test_train_again_with_another_readout_input(shared):
    # readout_input changes Wout only: the reservoirs are kept, the readout gains (or loses) the parameter rows
    data, forcings = _lorenz96_forcings(n=400)
    esn = _l96_readout_params(data, forcings, shared=shared, N_units=20, readout_input=False,
                              param_in_reservoir=True)
    esn.train(data, plot_training=False)
    nets = esn.patches if not shared else [esn]
    Win = [net.Win.copy() for net in nets]
    assert all(net.Wout.shape[0] == 21 for net in nets)
    esn.train(data, plot_training=False, readout_input='params', force=True)
    nets = esn.patches if not shared else [esn]
    assert all(net.readout_input == 'params' and net.Wout.shape[0] == 22 for net in nets)
    for before, net in zip(Win, nets):
        assert (abs(before - net.Win)).max() == 0
    esn.input_parameters = np.array([[7.0]])
    u, r = esn.outputs_to_inputs(data[0, 10][:, None]), np.zeros((esn.N_r, 1))
    assert np.isfinite(esn.step(u, r)[0]).all()
