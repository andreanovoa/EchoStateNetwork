"""The single network: segments and input parameters, the data split, the reservoir equation and its Jacobian, the
readout options, the validation strategies and their folds, the stability gate, the ridge regression and the search
over tikh alone."""
import re
import sys
import warnings

import numpy as np
import pytest
from conftest import assert_jacobian

from echostatenetwork import EchoStateNetwork, validation


def _segments(lengths, N_dim, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(nt, N_dim)) for nt in lengths]


@pytest.mark.parametrize('per_step', [False, True])
def test_parametric_network_trains_on_ragged_segments(per_step):
    # Segments (e.g. cluster-dwell chunks) may differ in length, and the parameter may be one value per segment,
    # (N_param, L), or vary within a segment, a list of (Nt_l, N_param) arrays. An unnormalized parameter column would
    # swamp the reservoir through Win, so a parametric network tunes its normalization unless the caller opts out.
    lengths = [60, 90, 45, 120, 75]
    segments = _segments(lengths, 4, seed=2)
    if per_step:
        params = [np.full((nt, 1), float(i)) for i, nt in enumerate(lengths)]
        params[1][lengths[1] // 2:] = 5.0                                 # flips mid-segment
    else:
        params = np.array([[0.0, 1.0, 2.0, 0.0, 1.0]])
    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=20, upsample=1, t_train=30, t_val=10, N_wash=5,
                           hyperparameters_to_optimize=[], input_parameters=params)
    esn.train(segments, plot_training=False)
    assert esn.N_dim_in == 5 and esn.Wout.shape == (esn.N_units + 1, 4)
    assert esn.optimize_parameter_normalization is True
    assert esn.param_norm_range[0] <= esn.norm[-1] <= esn.param_norm_range[1]


def test_search_with_per_step_parameters_and_leak_rate():
    # _RVC_Noise and run_test slice off the parameter columns with input_parameters.shape[0], which must work for a
    # per-timestep list as well; leak_rate is searched as rho is
    lengths = [60, 90, 45]
    per_step = [np.full((nt, 1), float(i)) for i, nt in enumerate(lengths)]
    esn = EchoStateNetwork(np.zeros((2, 1)), dt=1, N_units=15, upsample=1, t_train=30, t_val=10, N_wash=5,
                           N_folds=2, N_grid=2, N_func_evals=5, hyperparameters_to_optimize=['rho', 'leak_rate'],
                           input_parameters=per_step, optimize_parameter_normalization=False)
    esn.train(_segments(lengths, 2, seed=4), plot_training=False)
    assert esn.trained and esn.bo_results['hp_names'] == ['rho', 'leak_rate']
    assert esn.leak_rate_range[0] <= esn.leak_rate <= esn.leak_rate_range[1]


def test_bo_results_exposed_after_training():
    # Downstream code (qlroms' per-chart ESNs) plots esn.bo_results['func_vals'], so train() keeps a SLIMMED copy of
    # the search: the raw skopt OptimizeResult holds the corpus and the GP models, which bloat (or, with a closure as
    # validation strategy, break) every pickle of a trained network.
    import pickle
    data = np.random.default_rng(9).normal(size=(2, 80, 3))
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=15, upsample=1, t_train=40, t_val=10, t_test=10, N_wash=5,
                           N_folds=2, N_grid=2, N_func_evals=3, hyperparameters_to_optimize=['rho'])
    assert esn.bo_results is None
    esn.train(data, plot_training=False, validation_strategy=lambda *a, **k: EchoStateNetwork._RVC_Noise(*a, **k))
    assert len(esn.bo_results['func_vals']) == 3 and esn.bo_results['hp_names'] == ['rho']
    pickle.dumps(esn)


def test_ragged_segments_ignore_t_train_budget():
    # A segment survives if it yields one trainable pair past its own washout (t_val does not gate it), and t_train
    # does not cap a segmented corpus (read as a total budget of pairs, it shrank every segment to a couple of
    # steps): every usable segment is kept in full, and t_train is written back from what was kept.
    lengths = [15, 20, 500, 30, 18]
    segments = _segments(lengths, 2, seed=5)
    splits = []
    for t_train in (100, 1000):                       # far below, and above, the ~550 trainable pairs
        esn = EchoStateNetwork(segments[0].T, dt=1, N_units=15, upsample=1, t_train=t_train, t_val=10, N_wash=3,
                               hyperparameters_to_optimize=[], input_parameters=np.array([np.arange(5.0)]))
        U_wtv, Y_wtv, U_test, _ = esn._split_and_format_data(segments)
        splits.append([u.shape[0] for u in U_wtv])
        assert len(U_test) == 1 and splits[-1] == [nt - 1 for nt in lengths[:4]]    # 80/20, none dropped
        assert esn.N_train == max(sum(u.shape[0] - esn.N_wash for u in U_wtv) - esn.N_val, 1)
        assert all(u.shape[0] == y.shape[0] for u, y in zip(U_wtv, Y_wtv))
    esn.train(segments, plot_training=False)
    assert esn.trained


def test_rvc_fold_placement_adapts_to_each_segments_own_length():
    # _RVC_Noise placed its folds from the nominal N_train, as if every segment were that long: on short segments the
    # later folds sliced out of bounds, gave NaN in compute_nMAE, and a NaN-catch reset the accumulated score to a
    # constant, so every candidate looked identical. Fold placement is computed per segment.
    rng = np.random.default_rng(6)
    lengths = list(rng.integers(50, 200, size=20)) + [900, 950]          # most far shorter than N_train=500
    segments = [rng.normal(size=(nt, 2)).cumsum(axis=0) * 0.1 for nt in lengths]
    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=15, upsample=1, t_train=500, t_val=50, N_wash=5, N_folds=4,
                           hyperparameters_to_optimize=['rho', 'sigma_in'], N_grid=2, N_func_evals=4,
                           input_parameters=np.array([np.arange(22) % 2.0]), seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # "Mean of empty slice" -> fail
        esn.train(segments, plot_training=False)
    assert esn.trained


def test_omitted_t_train_and_t_val_are_inferred():
    """No t_train: ALL the data, split 80/20 into train/val and test (whole segments, or one series); no t_val: the
    post-washout length of the median segment, or 20% of the train/val window of one series. The inferred values are
    written back; t_test=0 trains on every segment."""
    lengths = [40, 45, 50, 55, 60, 42, 48, 52, 58, 44]
    segments = _segments(lengths, 2, seed=7)
    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=15, upsample=1, N_wash=5, hyperparameters_to_optimize=[],
                           input_parameters=np.array([np.arange(10.0)]))
    assert esn.t_train is None and esn.t_val is None
    U_wtv, _, U_test, _ = esn._split_and_format_data(segments)
    assert len(U_wtv) == 8 and len(U_test) == 2
    assert [u.shape[0] for u in U_wtv] == [nt - 1 for nt in lengths[:8]]
    assert esn.t_train is not None and esn.N_val == int(np.median(lengths)) - esn.N_wash - 1
    esn.train(segments, plot_training=False)
    assert esn.trained

    data = np.random.default_rng(8).normal(size=(200, 2)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=15, upsample=1, N_wash=5, hyperparameters_to_optimize=[])
    U_wtv, _, U_test, _ = esn._split_and_format_data(data)
    n_wtv = esn.N_train + esn.N_val
    assert n_wtv == 160 and esn.N_val == round(0.2 * 160)
    assert U_wtv.shape[1] == n_wtv - 1 and U_test.shape[1] == data.shape[0] - n_wtv - 1

    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=10, upsample=1, N_wash=5, t_test=0.0,
                           hyperparameters_to_optimize=[])
    esn.train(segments, plot_training=False)
    assert (esn.split_summary['segments_train'], esn.split_summary['segments_test']) == (10, 0)


def test_rr_terms_match_returned_states():
    """LHS/RHS must be EXACTLY the Gram products of the returned open-loop states:
    one unchunked pass per segment, full-segment state arrays assigned in place
    (the chunked N_split path and its per-chunk appends were removed)."""
    rng = np.random.default_rng(3)
    Nt, N_dim = 400, 3
    y = np.stack([np.sin(0.07 * np.arange(Nt) + 2 * k) for k in range(N_dim)], axis=1)
    y += 0.01 * rng.standard_normal(y.shape)

    esn = EchoStateNetwork(y.T, dt=1, N_units=40, upsample=1, N_wash=20,
                           t_train=300, t_val=50, seed=0,
                           hyperparameters_to_optimize=[])
    esn._generate_W_Win(seed=0)
    esn.Wout = np.zeros((esn.N_units + 1, esn.N_dim))
    esn.norm, esn.shift = np.ones(esn.N_dim), np.zeros(esn.N_dim)
    U_wtv, Y_wtv = esn._UY_from_raw_data(y[np.newaxis], add_noise=False)[:2]
    LHS, RHS, U_RR, R_RR = esn._compute_RR_terms(U_wtv, Y_wtv)

    assert R_RR[0].shape == (U_wtv[0].shape[0] - esn.N_wash, esn.N_units)
    assert U_RR[0].shape == (U_wtv[0].shape[0] - esn.N_wash, esn.N_dim)
    r_aug = np.hstack([R_RR[0], np.ones((R_RR[0].shape[0], 1)) * esn.bias_out])
    assert np.array_equal(LHS, r_aug.T @ r_aug)
    assert np.array_equal(RHS, r_aug.T @ Y_wtv[0][esn.N_wash:])


def test_spectral_radius_arpack_fallback():
    # N_units=150, seed=0 stagnates ARPACK (0/1 eigenvectors at 1501 iterations);
    # the dense fallback must still deliver a unit-spectral-radius reservoir
    esn = EchoStateNetwork(np.zeros((3, 1)), dt=0.1, N_units=150, N_wash=10, seed=0)
    esn._generate_W_Win(seed=esn.seed)
    assert np.isclose(np.abs(np.linalg.eigvals(esn.W.toarray())).max(), 1.0, atol=1e-8)


@pytest.mark.parametrize('leak_rate, scale_bias_in', [(1.0, True), (0.35, True), (1.0, False)])
def test_step_is_the_reservoir_equation(leak_rate, scale_bias_in):
    # r' = (1 - a) r + a tanh(sigma_in Win_u u + s b_in Win_b + rho W r), with s = sigma_in, or s = 1 when
    # scale_bias_in=False; leak_rate defaults to 1 (the plain tanh map), and scale_bias_in survives to_arrays
    rng = np.random.default_rng(12)
    data = rng.normal(size=(60, 3))
    leak = {} if leak_rate == 1.0 else dict(leak_rate=leak_rate)
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=30, t_val=10, t_test=10, N_wash=5,
                           sigma_in=0.01, bias_in=np.array([0.7]), scale_bias_in=scale_bias_in,
                           hyperparameters_to_optimize=[], **leak)
    esn.train(data, plot_training=False)
    assert esn.leak_rate == leak_rate
    u, r = np.repeat(data[:1].T, 2, axis=1), rng.normal(size=(esn.N_units, 2))
    Win = esn.Win.toarray()
    s = esn.sigma_in if scale_bias_in else 1.0
    x = np.tanh(esn.sigma_in * Win[:, :-1] @ esn.normalize_input(u) + s * 0.7 * Win[:, -1:] + esn.rho * esn.W.dot(r))
    expected = (1 - leak_rate) * r + leak_rate * x
    np.testing.assert_allclose(esn.step(u, r)[1], expected)
    loaded = EchoStateNetwork.from_arrays(esn.to_arrays())
    assert loaded.scale_bias_in is scale_bias_in
    np.testing.assert_allclose(loaded.step(u, r)[1], expected)


@pytest.mark.parametrize('r2_mode', [False, True])
@pytest.mark.parametrize('leak_rate', [1.0, 0.6])
def test_jacobian_matches_finite_differences(leak_rate, r2_mode):
    data = np.random.default_rng(13).normal(size=(120, 3)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=60, t_val=20, t_test=20, N_wash=5, seed=0,
                           leak_rate=leak_rate, r2_mode=r2_mode, hyperparameters_to_optimize=[])
    esn.train(data, plot_training=False)
    r = 0.3 * np.random.default_rng(2).normal(size=(esn.N_units, 1))
    assert_jacobian(esn, data[5].reshape(-1, 1).copy(), r, atol=1e-9 if r2_mode else 1e-8)


def _train_single_series(strategy, seed=20, **esn_kwargs):
    """Small single-series training run for the Racca & Magri (2021)
    single-series validation strategies (_SSV/_WFV/_KFV); returns the ESN.
    `esn_kwargs` override the constructor defaults (e.g. val_fold_step)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(200, 3)).cumsum(axis=0) * 0.1
    kwargs = dict(dt=1, N_units=15, upsample=1,
                  t_train=120, t_val=30, t_test=40, N_wash=5,
                  N_folds=3, N_grid=2, N_func_evals=3,
                  hyperparameters_to_optimize=['rho'], seed=0)
    kwargs.update(esn_kwargs)
    esn = EchoStateNetwork(data.T, **kwargs)
    esn.train(data, plot_training=False, validation_strategy=strategy)
    return esn


def test_single_series_strategies_retrain_per_fold():
    # SSV, WFV and KFV train and keep the search trace. WFV and KFV retrain Wout per fold on different windows, so on
    # the same data, reservoir and seed the three objectives differ: identical traces would mean that the folds
    # collapsed into one. A ragged corpus is refused, with a pointer to the segment strategies.
    traces = []
    for strategy in (EchoStateNetwork._SSV, EchoStateNetwork._WFV, EchoStateNetwork._KFV):
        esn = _train_single_series(strategy)
        assert esn.trained and esn.Wout.shape == (esn.N_units + 1, 3)
        traces.append(esn.bo_results['func_vals'])
        assert len(traces[-1]) == 3
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not np.allclose(traces[a], traces[b])
    segments = _segments((60, 90, 45), 2, seed=23)
    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=10, upsample=1, t_train=30, t_val=10, N_wash=5,
                           N_grid=2, N_func_evals=2, hyperparameters_to_optimize=['rho'])
    with pytest.raises(ValueError, match='segment strategies'):
        esn.train(segments, plot_training=False, validation_strategy=EchoStateNetwork._WFV)


def test_realized_fold_count():
    # A val_fold_step smaller than N_val overlaps the validation intervals (the chaotic fold variants of Racca &
    # Magri 2021), so more folds fit in the same data. An unreachable N_folds warns of the count used, and the summary
    # reports the realized count, not the request (also for the default strategy).
    for strategy in (EchoStateNetwork._WFV, EchoStateNetwork._KFV):
        counts = []
        for step in (None, 10):                                          # 10 < N_val = 30: overlap
            with pytest.warns(UserWarning, match=r'using \d+ folds') as record:
                esn = _train_single_series(strategy, N_folds=100, N_func_evals=2, val_fold_step=step)
            note = next(str(w.message) for w in record if 'using' in str(w.message))
            counts.append(int(re.search(r'using (\d+) folds', note).group(1)))
            assert esn.n_folds_realized == counts[-1]
            assert f'{counts[-1]} validation folds' in esn.training_summary()
        assert counts[1] > counts[0]
    data = np.random.default_rng(21).normal(size=(660, 3))
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=450, t_val=50, t_test=100, N_wash=5,
                           N_folds=6, N_grid=2, N_func_evals=2, hyperparameters_to_optimize=['rho'])
    esn.train(data, plot_training=False)
    assert esn.n_folds_realized == 6 and '6 validation folds' in esn.training_summary()


def test_val_fold_step_none_matches_explicit_n_val_bit_identically():
    # The default val_fold_step=None is the regular fold geometry: with the advance spelled out as N_val, the trace of
    # the search is identical. The SAME instance is trained again (W goes through ARPACK's start vector) without input
    # noise (drawn from the advancing instance rng), so everything else is deterministic.
    data = np.random.default_rng(20).normal(size=(200, 3)).cumsum(axis=0) * 0.1
    for strategy in (EchoStateNetwork._WFV, EchoStateNetwork._KFV):
        esn = EchoStateNetwork(data.T, dt=1, N_units=15, upsample=1, t_train=120, t_val=30, t_test=40, N_wash=5,
                               N_folds=3, N_grid=2, N_func_evals=3, hyperparameters_to_optimize=['rho'], seed=0)
        traces = []
        for step in (None, 30):
            esn.val_fold_step = step
            esn.train(data, plot_training=False, add_noise=False, validation_strategy=strategy, force=True)
            traces.append(esn.bo_results['func_vals'].copy())
        assert np.array_equal(*traces)


def test_validation_metric_is_shared_across_strategies():
    # A custom probe metric set on the ESN is used by RVC and the single-series
    # engine alike (strategy geometry and probe scoring are orthogonal).
    calls = []

    def counting_metric(case, Y_true, Y_pred, norm):
        calls.append(Y_true.shape)
        return float(np.mean((Y_true - Y_pred) ** 2))

    rng = np.random.default_rng(11)
    data = rng.normal(size=(2, 80, 3))
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=12, upsample=1,
                            t_train=40, t_val=10, t_test=0., N_wash=5, N_folds=2,
                            hyperparameters_to_optimize=[],
                            validation_metric=counting_metric)
    esn.train(data, plot_training=False)
    U_wtv, Y_wtv = esn._split_and_format_data(data)[:2]
    out = validation.RVC_Noise([], esn.copy(), U_wtv, Y_wtv, np.zeros(1), [],
                               print_convergence=False)
    assert calls and np.isfinite(out)


@pytest.mark.parametrize('readout_input, in_reservoir', [(True, True), ('params', True), ('params', False)])
def test_readout_input_reads_the_input_directly(readout_input, in_reservoir):
    # readout_input=True: Wout^T [r; u_norm; bias], the "input skip" readout; 'params': Wout^T [r; p_norm; bias], and
    # param_in_reservoir=False zeroes Win's parameter columns, so p reaches the output only through the readout. The
    # Jacobian carries the direct term, reservoir_to_physical needs u, and to_arrays keeps both options.
    rng = np.random.default_rng(22)
    data = rng.normal(size=(2, 80, 3))
    params = dict(input_parameters=np.array([[0.0, 1.0]])) if readout_input == 'params' else {}
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=12, upsample=1, t_train=40, t_val=10, t_test=10, N_wash=5,
                           hyperparameters_to_optimize=[], readout_input=readout_input, param_in_reservoir=in_reservoir,
                           **params)
    esn.train(data, plot_training=False)
    rows = slice(3, 4) if params else slice(0, 3)
    assert esn.Wout.shape == (esn.N_units + 1 + rows.stop - rows.start, 3)
    if params:
        assert (abs(esn.Win[:, 3]).sum() > 0) == in_reservoir
    u = np.vstack([data[0, 5].reshape(-1, 1), [[0.5]]])[:esn.N_dim_in]
    r = 0.1 * rng.normal(size=(esn.N_units, 1))
    u_out, r_out = esn.step(u, r)
    manual = esn.Wout.T @ np.vstack([r_out, esn.normalize_input(u)[rows], esn.bias_out * np.ones((1, 1))])
    np.testing.assert_allclose(u_out, manual)
    np.testing.assert_allclose(esn.reservoir_to_physical(r_out, u=u), u_out)
    with pytest.raises(ValueError, match='readout_input'):
        esn.reservoir_to_physical(r_out)
    assert_jacobian(esn, u, r, eps=1e-4, rtol=1e-5)
    back = EchoStateNetwork.from_arrays(esn.to_arrays())
    assert back.readout_input == readout_input and back.param_in_reservoir == in_reservoir
    assert np.array_equal(back.step(u, r)[0], u_out)


@pytest.mark.parametrize("mode", ["data", False])   # True: one value of the parameter, no range to search
@pytest.mark.parametrize("one_segment", [True, False])
def test_parametric_network_trains_on_one_series(mode, one_segment):
    # one series, (Nt, N_dim) or (1, Nt, N_dim): _set_norm keeps the axis of the segments, which the parameter
    # columns of the normalization must not trip on
    data = np.random.default_rng(23).normal(size=(1, 200, 3))
    esn = EchoStateNetwork(data[0, :1].T, dt=1, N_units=10, upsample=1, t_train=120, t_val=30, t_test=10,
                           N_wash=5, hyperparameters_to_optimize=[], input_parameters=np.array([[8.0]]),
                           optimize_parameter_normalization=mode)
    esn.train(data[0] if one_segment else data, plot_training=False)
    assert esn.trained and esn.norm.shape == esn.shift.shape == (4,)
    if mode == 'data':
        assert esn.shift[-1] == 8.0


_SEEDED_TRAIN_SCRIPT = """
import sys
import numpy as np
from echostatenetwork import EchoStateNetwork
data = np.random.default_rng(0).normal(size=(1, 300, 3))
esn = EchoStateNetwork(data[0].T, dt=1, N_units=200, upsample=1, t_train=200, t_val=40,
                       t_test=20, N_wash=10, hyperparameters_to_optimize=[], seed=3)
esn.train(data, plot_training=False)
np.savez(sys.argv[1], W=esn.W.toarray(), Wout=esn.Wout)
"""


def test_seeded_training_is_bit_identical_across_processes(tmp_path):
    # ARPACK's default start vector is unseeded, so the spectral-radius scaling of W
    # must pass a fixed v0 or W (and Wout) drift in the last digits between processes
    import subprocess
    import sys

    outs = []
    for k in range(2):
        out = tmp_path / f'run{k}.npz'
        subprocess.run([sys.executable, '-c', _SEEDED_TRAIN_SCRIPT, str(out)],
                       check=True, capture_output=True)
        outs.append(np.load(out))
    assert np.array_equal(outs[0]['W'], outs[1]['W'])
    assert np.array_equal(outs[0]['Wout'], outs[1]['Wout'])


def test_dense_bias_column_and_unregularized_bias_row():
    # 'sparse_dense_bias': one state column per neuron and the bias column in every neuron;
    # regularize_bias_out=False leaves the Tikhonov factor off the bias row only.
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1, 200, 3))
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=30, upsample=1, t_train=150, t_val=20, t_test=10,
                           N_wash=5, hyperparameters_to_optimize=[], Win_type='sparse_dense_bias', seed=0)
    esn.train(data, plot_training=False)
    Win = esn.Win.toarray()
    assert np.all(Win[:, -1] != 0) and np.all((Win[:, :-1] != 0).sum(1) == 1)
    assert esn.to_arrays()['Win_type'] == 'sparse_dense_bias'
    LHS = np.eye(esn._n_readout)
    esn.regularize_bias_out = False
    assert np.allclose(np.diag(esn._regularized(LHS, 0.5)), [1.5] * (esn._n_readout - 1) + [1.0])
    assert np.allclose(LHS, np.eye(esn._n_readout))                     # a copy: the input is untouched


def test_bho_midpoint_grid_noise_and_stability_gate():
    # grid_midpoints: no initial point on the boundary; bho_noise reaches the GP; the stability gate
    # penalizes a candidate whose long closed loop blows up, and not one that stays bounded
    rng = np.random.default_rng(0)
    t = np.arange(600) * 0.05
    data = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)[None] + 0.01 * rng.normal(size=(1, 600, 3))
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=40, upsample=1, t_train=400, t_val=50, t_test=50, N_wash=10,
                           hyperparameters_to_optimize=['rho', 'sigma_in', 'tikh'], N_grid=2, N_func_evals=6,
                           grid_midpoints=True, bho_noise='gaussian', stability_steps=200, seed=0)
    esn.train(data, plot_training=False)
    init = np.array(esn.bo_results['x_iters'][:4])                    # the 2 x 2 initial grid (rho, sigma_in)
    for k, (lo, hi) in enumerate((esn.rho_range, esn.sigma_in_range)):
        assert np.all((init[:, k] > lo) & (init[:, k] < hi))
    U, Y, *_ = esn._split_and_format_data(data)
    W = esn.Wout
    flags = validation.unstable(esn, U, Y, np.stack([W, 1e3 * W]))
    assert flags[1] == 1.0


def test_stability_gate_from_several_starts():
    """The washouts of the gate spread over every segment, in turn; one start is the start of the first segment
    (the former gate); several starts flag the unstable readout and keep the parameters of each record. Fewer starts
    than segments spread over the list of segments, not the first ones only."""
    rng = np.random.default_rng(1)
    t = np.arange(300) * 0.05
    data = np.stack([np.stack([np.sin(t + a), np.cos(t + a)], axis=1) for a in (0.0, 1.0)]) + 0.01 * rng.normal(size=(2, 300, 2))
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=30, upsample=1, t_train=200, t_val=40, t_test=60, N_wash=10,
                           hyperparameters_to_optimize=[], seed=0, input_parameters=np.array([[0.0, 1.0]]),
                           optimize_parameter_normalization=False)
    esn.train(data, plot_training=False)
    U, Y, *_ = esn._split_and_format_data(data)
    esn.stability_starts = 1
    assert validation._stability_starts(esn, U) == [(0, 0)]
    esn.stability_starts = 4
    starts = validation._stability_starts(esn, U)
    assert [l for l, _ in starts] == [0, 1, 0, 1] and starts[0][1] == starts[1][1] == 0 and starts[2][1] > 0
    before = esn.input_parameters.copy()
    esn.stability_steps = 100
    W = esn.Wout
    flags = validation.unstable(esn, U, Y, np.stack([W, 1e3 * W]))
    assert flags[1] == 1.0
    np.testing.assert_array_equal(esn.input_parameters, before)

    class Case:
        N_wash, stability_starts = 5, 3
    assert [l for l, _ in validation._stability_starts(Case, [np.zeros((50, 2))] * 7)] == [0, 2, 4]


def test_stability_gate_compares_each_loop_with_its_own_record():
    """Two records of different amplitude: the loops of a readout that reproduces each record pass the gate. A band
    around the pooled records would reject the loops of the small one."""
    t = np.arange(400) * 0.05
    data = np.stack([a * np.stack([np.sin(t), np.cos(t)], axis=1) for a in (1.0, 4.0)])
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=60, upsample=1, t_train=300, t_val=50, t_test=50, N_wash=20,
                           hyperparameters_to_optimize=[], seed=0, input_parameters=np.array([[0.0, 1.0]]),
                           optimize_parameter_normalization=False, rho=0.5, sigma_in=0.5, tikh=1e-8, noise=0.0)
    esn.train(data, plot_training=False)
    U, Y, *_ = esn._split_and_format_data(data, add_noise=False)
    esn.stability_steps, esn.stability_starts = 200, 2
    assert [l for l, _ in validation._stability_starts(esn, U)] == [0, 1]
    assert validation.unstable(esn, U, Y, esn.Wout[None])[0] == 0.0


def test_stability_penalty_rejects_a_failing_candidate(monkeypatch):
    """The default penalty is a hard rejection: a Tikhonov candidate that fails the gate loses to a candidate that
    passes it, whatever its short-fold score. At the former default of 1.0 (one decade of log_nMAE), a failing
    candidate still won with a short-fold score better by more than one decade (L96-40s: -2.27 + 1 < -1.2)."""
    data = np.random.default_rng(0).normal(size=(300, 3)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, upsample=1, N_units=20, t_train=200, t_val=40, t_test=0, N_wash=10,
                           hyperparameters_to_optimize=[], tikh_range=[1e-12, 1e-6], stability_steps=1)
    esn.train(data, plot_training=False)
    U_wtv, Y_wtv = esn._split_and_format_data(data, add_noise=False)[:2]
    scores = {}                                   # short-fold score of each candidate, keyed by its readout:

    def metric(case, Y_true, Y_pred, norm):       # the first candidate scores two decades better than the second
        key = case.Wout.ctypes.data
        if key not in scores:
            scores[key] = (-3.0, -1.0)[len(scores)]
        return scores[key]

    monkeypatch.setattr(validation, 'unstable', lambda case, U, Y, Wout_tik: np.array([1.0, 0.0]))   # and fails

    def evaluation(penalty):
        probe = esn.copy()
        probe.val_k, probe.stability_penalty, probe.validation_metric = 0, penalty, metric
        scores.clear()
        score = validation.RVC_Noise([], probe, U_wtv, Y_wtv, np.zeros(1), [], print_convergence=False)
        assert len(scores) == 2
        return score, probe.tikh

    assert evaluation(EchoStateNetwork.stability_penalty) == (-1.0, 1e-6)   # the candidate that passes the gate
    assert evaluation(1.0) == (-2.0, 1e-12)                                  # the former default: the failing one


def test_stability_gate_checks_the_tail_of_the_loop(monkeypatch):
    """A loop that follows the record for three quarters of its length and then collapses to the mean keeps the
    standard deviation of the whole loop inside the band (sqrt(0.75) of the record's); the band on the last half of
    the loop (stability_tail) rejects it, and stability_tail=0 restores the test on the whole loop only."""
    t = np.arange(400) * 0.05
    data = np.stack([np.sin(t), np.cos(t)], axis=1)[None]
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=20, upsample=1, t_train=300, t_val=50, t_test=50, N_wash=10,
                           hyperparameters_to_optimize=[], seed=0, noise=0.0, stability_steps=300)
    esn.train(data, plot_training=False)
    U, Y, *_ = esn._split_and_format_data(data, add_noise=False)
    calls = []

    def step(u, r):                                   # the loop reproduces the record, then collapses to its mean
        k = len(calls) - esn.N_wash                   # steps into the loop (negative during the washout)
        calls.append(k)
        y = data[0, k % len(t)] if k < 0.75 * esn.stability_steps else np.zeros(2)
        return y.reshape(2, 1), r

    monkeypatch.setattr(esn, 'step', step)
    assert esn.stability_tail == 0.5
    assert validation.unstable(esn, U, Y, esn.Wout[None])[0] == 1.0
    calls.clear()
    esn.stability_tail = 0
    assert validation.unstable(esn, U, Y, esn.Wout[None])[0] == 0.0


def test_parameter_columns_normalized_like_the_state():
    """optimize_parameter_normalization='data': the parameter columns take the shift and scale of norm_method over
    the training values, and the search keeps its own hyperparameters only."""
    rng = np.random.default_rng(2)
    data = rng.normal(size=(3, 80, 2))
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=20, upsample=1, t_train=50, t_val=10, t_test=10, N_wash=5,
                           hyperparameters_to_optimize=['rho'], N_grid=2, N_func_evals=2, N_folds=2, seed=0,
                           input_parameters=np.array([[2.0, 4.0, 8.0]]), optimize_parameter_normalization='data',
                           norm_method='range')
    esn.train(data, plot_training=False)
    np.testing.assert_allclose(esn.norm[-1], 6.0)                 # range of the training values, 8 - 2
    np.testing.assert_allclose(esn.shift[-1], 14.0 / 3)           # their mean (equal segments)
    assert esn.hyperparameters_to_optimize == ['rho']


# ---------------------------------------------------------------------- a singular ridge regression
SINGULAR = 'ridge matrix is singular'


def _saturated_esn(**kwargs):
    """A reservoir saturated by a large input scaling and input bias: its states are +-1, so the columns of the ridge
    left-hand side are collinear and, without regularisation, it is singular."""
    data = np.random.default_rng(0).normal(size=(300, 3)).cumsum(axis=0) * 10
    options = dict(dt=1, N_units=50, upsample=1, t_train=200, t_val=40, t_test=40, N_wash=5, rho=0.5, sigma_in=10.,
                   bias_in=np.array([10.]), tikh=0., tikh_range=[0.], seed=0)
    return EchoStateNetwork(data.T, **{**options, **kwargs}), data


def test_ridge_solve_falls_back_on_least_squares_only_when_singular():
    # a non-singular solve is np.linalg.solve, unchanged; a singular one (two collinear columns) warns and returns the
    # least-squares solution, so that one bad point of a search scores badly instead of stopping it
    esn, _ = _saturated_esn()
    rng = np.random.default_rng(1)
    M = rng.normal(size=(6, 6))
    LHS, RHS = M @ M.T, rng.normal(size=(6, 2))
    np.testing.assert_array_equal(esn._ridge_solve(LHS, RHS, 1e-3), np.linalg.solve(esn._regularized(LHS, 1e-3), RHS))
    X = rng.normal(size=(40, 3))
    X = np.hstack((X, X[:, :1]))
    LHS, RHS = X.T @ X, X.T @ np.ones((40, 2))
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(LHS, RHS)
    with pytest.warns(UserWarning, match=SINGULAR):
        Wout = esn._ridge_solve(LHS, RHS, 0.)
    np.testing.assert_allclose(Wout, np.linalg.lstsq(LHS, RHS, rcond=None)[0])


@pytest.mark.parametrize('strategy', [None, validation.RVC_Noise, validation.SSV])
def test_training_on_a_saturated_reservoir_completes(strategy):
    # without a search (None), and with a search over tikh: the singular ridge warns once per call of train
    esn, data = _saturated_esn(hyperparameters_to_optimize=[] if strategy is None else ['tikh'])
    with pytest.warns(UserWarning, match=SINGULAR) as record:
        esn.train(data, plot_training=False, validation_strategy=strategy)
    assert np.isfinite(esn.Wout).all()
    assert len([w for w in record if SINGULAR in str(w.message)]) == 1


# ---------------------------------------------------------------------- 'tikh' alone in the search (issue #1)
def _issue1_network(**kwargs):
    """The network and record of issue #1: 'tikh' is the only hyperparameter to optimize."""
    data = np.random.default_rng(0).normal(size=(300, 3)).cumsum(axis=0) * 0.1
    options = dict(dt=1, upsample=1, N_units=20, t_train=200, t_val=40, t_test=0, N_wash=10,
                   hyperparameters_to_optimize=['tikh'])
    return EchoStateNetwork(data.T, **{**options, **kwargs}), data


def _one_evaluation(esn, data, strategy, tikh):
    """The validation score of `strategy` on a copy of `esn` whose Tikhonov factor is fixed at `tikh`."""
    probe = esn.copy()
    probe.val_k, probe.tikh_range = 0, [tikh]
    U_wtv, Y_wtv = probe._split_and_format_data(data, add_noise=False)[:2]
    return strategy([], probe, U_wtv, Y_wtv, np.zeros(1), [], print_convergence=False)


@pytest.mark.parametrize('strategy', ['RVC_Noise', 'SSV', 'WFV', 'KFV'])
def test_tikh_alone_is_one_evaluation_of_the_strategy(strategy, monkeypatch):
    # Every evaluation of the strategy selects tikh from tikh_range, so the search over tikh alone is one
    # evaluation at the fixed rho, sigma_in and leak_rate: no Bayesian optimization, and no scikit-optimize
    # (echostatenetwork 0.2.3 gave the empty space to gp_minimize, which raised "need at least one array to
    # concatenate").
    monkeypatch.setitem(sys.modules, 'skopt', None)            # any import of scikit-optimize fails
    strategy = getattr(validation, strategy)
    esn, data = _issue1_network()
    tikh_range = list(esn.tikh_range)
    esn.train(data, plot_training=False, add_noise=False, validation_strategy=strategy)

    bo = esn.bo_results
    assert bo['hp_names'] == [] and bo['x'] == [] and bo['x_iters'] == [[]] and bo['n_grid_points'] == 1
    assert len(bo['func_vals']) == 1 and bo['fun'] == bo['func_vals'][0]
    assert (esn.rho, esn.sigma_in, esn.leak_rate) == (0.9, 1e-3, 1.0) and list(esn.tikh_range) == tikh_range
    # tikh is the value of tikh_range with the lowest score, and fun is that score
    scores = [_one_evaluation(esn, data, strategy, tikh) for tikh in tikh_range]
    assert esn.tikh == tikh_range[int(np.argmin(scores))]
    assert bo['fun'] == pytest.approx(min(scores), rel=1e-9)
    assert esn.training_summary().endswith(f'tikh alone: 1 eval over the {len(tikh_range)} values of tikh_range, '
                                           'no BHO')


def test_tikh_alone_needs_no_search_budget():
    # other lists without a dimension: 'tikh' twice, and no budget of evaluations (N_func_evals=0)
    esn, data = _issue1_network(hyperparameters_to_optimize=['tikh', 'tikh'], N_func_evals=0)
    esn.train(data, plot_training=False)
    assert esn.bo_results['hp_names'] == [] and len(esn.bo_results['func_vals']) == 1
    assert esn.tikh in esn.tikh_range and np.isfinite(esn.Wout).all()


def test_seed_scores_with_tikh_alone_and_without_a_search():
    # n_seeds > 1 keeps the best seed, scored with bo_results['fun'], here the score of the one evaluation of tikh
    # alone. Without a search, a seed is scored at its own fixed tikh, the one of the network that is kept, and not at
    # the best value of tikh_range, which the network does not use.
    esn, data = _issue1_network()
    esn.train(data, plot_training=False, n_seeds=2)
    scores = esn.seed_scores
    assert esn.bo_results['hp_names'] == [] and len(esn.bo_results['func_vals']) == 1
    assert esn.seed == min(scores, key=scores.get) and scores[esn.seed] == esn.bo_results['fun']

    esn, data = _issue1_network(hyperparameters_to_optimize=[], tikh=1e-16)
    esn.train(data, plot_training=False, add_noise=False, n_seeds=2)
    assert esn.tikh == 1e-16 and esn.bo_results is None
    fixed = _one_evaluation(esn, data, validation.RVC_Noise, 1e-16)
    best = min(_one_evaluation(esn, data, validation.RVC_Noise, tikh) for tikh in esn.tikh_range)
    assert fixed != pytest.approx(best)                          # the two scores differ on this record
    assert esn.seed_scores[esn.seed] == pytest.approx(fixed, rel=1e-9)


@pytest.mark.parametrize('searched', [[], ['tikh']])
def test_plot_training_without_a_bayesian_optimization(searched):
    # No convergence trace and no landscape without a Bayesian optimization: plot_training=True draws Wout only
    # (perform_test=False), without a search and with 'tikh' alone.
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    plt.close('all')
    esn, data = _issue1_network(hyperparameters_to_optimize=searched, perform_test=False)
    try:
        esn.train(data, plot_training=True)
        assert esn.trained and len(plt.get_fignums()) == 1
    finally:
        plt.close('all')


def test_tikh_alone_with_parametric_inputs():
    # optimize_parameter_normalization (the default with input_parameters) adds the shift and the norm of the
    # parameter to the search, which is then a Bayesian optimization; without it, tikh alone is one evaluation.
    pytest.importorskip('skopt')                                         # a Bayesian search
    rng = np.random.default_rng(1)
    data, params = rng.normal(size=(2, 80, 3)), np.array([[0.0, 1.0]])
    options = dict(dt=1, N_units=15, upsample=1, t_train=40, t_val=10, t_test=10, N_wash=5, N_folds=2, N_grid=2,
                   N_func_evals=3, input_parameters=params, hyperparameters_to_optimize=['tikh'])
    esn = EchoStateNetwork(data[0].T, **options)
    esn.train(data, plot_training=False)
    assert esn.bo_results['hp_names'] == ['param_shift_0', 'param_norm_0']
    assert len(esn.bo_results['func_vals']) == 3
    esn = EchoStateNetwork(data[0].T, optimize_parameter_normalization=False, **options)
    esn.train(data, plot_training=False)
    assert esn.bo_results['hp_names'] == [] and len(esn.bo_results['func_vals']) == 1
    assert esn.tikh in esn.tikh_range


@pytest.mark.parametrize("layout", [{}, dict(patch_size=2, halo=1)])
def test_batched_ridge_sums_equal_one_pass(layout):
    """_compute_RR_terms accumulates the ridge sums `batch` steps at a time: a batch of 7 gives the sums and states
    of one batch, keep_states=False returns no states, and train() gives the same readout whatever the batch."""
    rng = np.random.default_rng(5)
    data = np.cumsum(rng.normal(size=(2, 60, 4)), axis=1) * 0.1
    kw = dict(dt=1, N_units=12, upsample=1, N_wash=5, t_train=40, t_val=10, t_test=0, noise=0.0,
              hyperparameters_to_optimize=[], seed=0, verbose=False, **layout)
    esn = EchoStateNetwork(data[0].T, **kw)
    esn.train(data, plot_training=False)
    U = [d[:-1] for d in data]
    Y = [d[1:] for d in data]
    LHS, RHS, U_RR, R_RR = esn._compute_RR_terms(U, Y, batch=10 ** 9)
    LHS7, RHS7, U_RR7, R_RR7 = esn._compute_RR_terms(U, Y, batch=7)
    assert np.allclose(LHS7, LHS, rtol=1e-12, atol=1e-12) and np.allclose(RHS7, RHS, rtol=1e-12, atol=1e-12)
    assert all(np.array_equal(a, b) for a, b in zip(R_RR7, R_RR)) and all(
        np.array_equal(a, b) for a, b in zip(U_RR7, U_RR))
    lhs, rhs, u_none, r_none = esn._compute_RR_terms(U, Y, keep_states=False, batch=7)
    assert np.array_equal(lhs, LHS7) and u_none == r_none == [None, None]
    small = EchoStateNetwork(data[0].T, readout_batch=3, **kw)
    small.train(data, plot_training=False)
    assert np.allclose(small.Wout, esn.Wout, rtol=1e-9, atol=1e-11)
