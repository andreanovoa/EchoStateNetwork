"""Tests for the parametric EchoStateNetwork (src/esn_core.py)."""
import re
import warnings

import numpy as np
import pytest

from echostatenetwork import EchoStateNetwork


def test_parametric_esn_trains_and_shapes():
    rng = np.random.default_rng(0)
    N_dim, Nt, L = 4, 80, 2
    data = rng.normal(size=(L, Nt, N_dim))
    params = np.array([[0.0, 1.0]])  # (N_param=1, L=2), e.g. cluster id per segment

    esn = EchoStateNetwork(data[0].T, dt=1, N_units=20, upsample=1,
                            t_train=40, t_val=10, t_test=10, N_wash=5,
                            hyperparameters_to_optimize=[],
                            input_parameters=params)
    esn.train(data, plot_training=False)

    assert esn.N_dim_in == N_dim + 1
    assert esn.Wout.shape == (esn.N_units + 1, N_dim)
    # An un-normalized parameter column would swamp the reservoir through Win's dense
    # parameter connections, so the constructor auto-enables optimize_parameter_normalization
    # for any parametric ESN unless the caller opts out (see esn_core.py's class docstring) --
    # the parameter column is tuned to O(1) via the same BHO loop as rho/sigma_in/tikh, not
    # left at the identity (norm=1, shift=0).
    assert esn.optimize_parameter_normalization is True
    assert esn.param_norm_range[0] <= esn.norm[-1] <= esn.param_norm_range[1]


def test_parametric_esn_closed_loop_hyperparameter_search():
    # Exercises _RVC_Noise's closed-loop run (outputs_to_inputs) with input_parameters set.
    rng = np.random.default_rng(1)
    N_dim, Nt, L = 3, 80, 2
    data = rng.normal(size=(L, Nt, N_dim))
    params = np.array([[0.0, 1.0]])

    esn = EchoStateNetwork(data[0].T, dt=1, N_units=15, upsample=1,
                            t_train=40, t_val=10, t_test=10, N_wash=5,
                            N_folds=2, N_grid=2, N_func_evals=2,
                            hyperparameters_to_optimize=['rho'],
                            input_parameters=params)
    esn.train(data, plot_training=False)

    assert esn.trained


def test_bo_results_exposed_after_training():
    # Downstream code (e.g. qlrom's per-chart ESNs) plots BHO convergence traces from
    # esn.bo_results['func_vals'] -- the per-evaluation validation-loss trace -- so
    # train() must keep the optimization output instead of discarding it. It is a
    # SLIMMED copy: the raw skopt OptimizeResult retains the training corpus and GP
    # models, which would bloat (or, with closure validation strategies, break)
    # every pickle of a trained ESN.
    import pickle

    rng = np.random.default_rng(9)
    data = rng.normal(size=(2, 80, 3))
    n_func_evals = 3

    esn = EchoStateNetwork(data[0].T, dt=1, N_units=15, upsample=1,
                            t_train=40, t_val=10, t_test=10, N_wash=5,
                            N_folds=2, N_grid=2, N_func_evals=n_func_evals,
                            hyperparameters_to_optimize=['rho'])
    assert esn.bo_results is None  # nothing to inspect before training

    def local_strategy(*args, **kwargs):  # unpicklable closure, the regression case
        return EchoStateNetwork._RVC_Noise(*args, **kwargs)

    esn.train(data, plot_training=False, validation_strategy=local_strategy)

    assert isinstance(esn.bo_results, dict)
    assert len(esn.bo_results['func_vals']) == n_func_evals
    assert esn.bo_results['hp_names'] == ['rho']
    pickle.dumps(esn)  # must not choke on (or retain) the BHO internals

    esn_no_bho = EchoStateNetwork(data[0].T, dt=1, N_units=15, upsample=1,
                                   t_train=40, t_val=10, t_test=10, N_wash=5,
                                   hyperparameters_to_optimize=[])
    esn_no_bho.train(data, plot_training=False)
    assert esn_no_bho.bo_results is None


def test_parametric_esn_ragged_segments():
    # Segments (e.g. cluster-dwell chunks) may have different lengths; train_data can
    # be a list of (Nt_l, N_dim) arrays instead of a regular (L, Nt, N_dim) array.
    rng = np.random.default_rng(2)
    N_dim = 4
    lengths = [60, 90, 45, 120, 75]
    segments = [rng.normal(size=(nt, N_dim)) for nt in lengths]
    params = np.array([[0.0, 1.0, 2.0, 0.0, 1.0]])  # (1, L=5) cluster id per segment

    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=20, upsample=1,
                            t_train=30, t_val=10, N_wash=5,
                            hyperparameters_to_optimize=[],
                            input_parameters=params)
    esn.train(segments, plot_training=False)

    assert esn.N_dim_in == N_dim + 1
    assert esn.Wout.shape == (esn.N_units + 1, N_dim)
    assert esn.optimize_parameter_normalization is True
    assert esn.param_norm_range[0] <= esn.norm[-1] <= esn.param_norm_range[1]


def test_parametric_esn_per_timestep_varying_parameter():
    # input_parameters can be a list of (Nt_l, N_param) arrays instead of a constant
    # (N_param, L) array, for segments where the parameter itself varies within the
    # segment (e.g. a window straddling a cluster transition).
    rng = np.random.default_rng(3)
    N_dim = 2
    lengths = [60, 90, 45]
    segments = [rng.normal(size=(nt, N_dim)) for nt in lengths]

    per_step = []
    for i, nt in enumerate(lengths):
        label = np.full((nt, 1), float(i))
        if i == 1:
            label[nt // 2:] = 5.0  # flips mid-segment
        per_step.append(label)

    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=20, upsample=1,
                            t_train=30, t_val=10, N_wash=5,
                            hyperparameters_to_optimize=[],
                            input_parameters=per_step)
    esn.train(segments, plot_training=False)

    assert esn.N_dim_in == N_dim + 1
    assert esn.optimize_parameter_normalization is True
    assert esn.param_norm_range[0] <= esn.norm[-1] <= esn.param_norm_range[1]


def test_parametric_esn_per_timestep_varying_parameter_with_bho():
    # _RVC_Noise and run_test's predict_Y both read input_parameters.shape[0] to slice
    # off the parameter columns for their own closed-loop runs; this must also work
    # when input_parameters is a per-timestep list rather than a constant ndarray.
    rng = np.random.default_rng(4)
    N_dim = 2
    lengths = [60, 90, 45]
    segments = [rng.normal(size=(nt, N_dim)) for nt in lengths]
    per_step = [np.full((nt, 1), float(i)) for i, nt in enumerate(lengths)]

    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=15, upsample=1,
                            t_train=30, t_val=10, N_wash=5,
                            N_folds=2, N_grid=2, N_func_evals=2,
                            hyperparameters_to_optimize=['rho'],
                            input_parameters=per_step)
    esn.train(segments, plot_training=False)

    assert esn.trained


def test_ragged_segments_ignore_t_train_budget():
    # Short segments must survive as long as they yield one trainable pair past
    # their own washout (N_wash + 2 raw points -- t_val must NOT gate training
    # inclusion). t_train must NOT cap a segmented corpus: read as a total pair
    # budget it shrank every segment to a couple of steps. Every usable segment is
    # kept in full whether or not t_train is given, and t_train is written back
    # from what was kept.
    rng = np.random.default_rng(5)
    N_dim = 2
    lengths = [15, 20, 500, 30, 18]
    segments = [rng.normal(size=(nt, N_dim)) for nt in lengths]
    params = np.array([[float(i) for i in range(len(lengths))]])

    # t_train=100 is far below the corpus (~550 trainable pairs): ignored, not a cap
    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=15, upsample=1,
                            t_train=100, t_val=10, N_wash=3,
                            hyperparameters_to_optimize=[],
                            input_parameters=params)
    U_wtv, Y_wtv, U_test, _ = esn._split_and_format_data(segments)

    assert len(U_wtv) == 4 and len(U_test) == 1     # 80/20 segment split, none dropped
    assert [u.shape[0] for u in U_wtv] == [nt - 1 for nt in lengths[:4]]  # kept in full
    kept_pairs = sum(u.shape[0] - esn.N_wash for u in U_wtv)
    assert esn.N_train == max(kept_pairs - esn.N_val, 1)   # written back from the data
    for u, y in zip(U_wtv, Y_wtv):
        assert u.shape[0] == y.shape[0]  # washout+train windows stay aligned

    # a generous t_train gives exactly the same split -- it is informational here
    esn_all = EchoStateNetwork(segments[0].T, dt=1, N_units=15, upsample=1,
                                t_train=1000, t_val=10, N_wash=3,
                                hyperparameters_to_optimize=[],
                                input_parameters=params)
    U_full, _, _, _ = esn_all._split_and_format_data(segments)
    assert [u.shape[0] for u in U_full] == [u.shape[0] for u in U_wtv]

    esn.train(segments, plot_training=False)
    assert esn.trained


def test_rvc_fold_placement_adapts_to_each_segments_own_length():
    # _RVC_Noise's fold spacing/count used to be derived once from the nominal
    # N_train, as if every segment were that long. With segments now genuinely
    # variable-length (some far shorter than N_train), that silently sliced out of
    # bounds for later folds on short segments -- an empty U_wash/Y_val, NaN in
    # compute_nRMSE, and a NaN-catch that *reset* the whole accumulated score to a
    # constant, making every hyperparameter combination look identical. Fold
    # placement must instead be computed per segment.
    rng = np.random.default_rng(6)
    N_dim = 2
    # most segments far shorter than N_train=500 (below), a couple long enough for
    # all N_folds folds at the old, nominal spacing
    lengths = list(rng.integers(50, 200, size=20)) + [900, 950]
    segments = [rng.normal(size=(nt, N_dim)).cumsum(axis=0) * 0.1 for nt in lengths]
    params = np.array([[float(i % 2) for i in range(len(lengths))]])

    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=15, upsample=1,
                            t_train=500, t_val=50, N_wash=5, N_folds=4,
                            hyperparameters_to_optimize=['rho', 'sigma_in'],
                            N_grid=2, N_func_evals=4,
                            input_parameters=params, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # "Mean of empty slice" -> fail
        esn.train(segments, plot_training=False)

    assert esn.trained


def test_omitted_t_train_uses_all_data_and_infers_t_val():
    """No t_train -> ALL the data is used with an 80/20 train-val/test split of the
    segments; no t_val -> inferred from the median segment (dwell) length. The
    inferred values are written back so N_train/N_val are defined afterwards."""
    rng = np.random.default_rng(7)
    N_dim = 2
    lengths = [40, 45, 50, 55, 60, 42, 48, 52, 58, 44]
    segments = [rng.normal(size=(nt, N_dim)) for nt in lengths]
    params = np.array([[float(i) for i in range(len(lengths))]])

    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=15, upsample=1,
                            N_wash=5, hyperparameters_to_optimize=[],
                            input_parameters=params)   # t_train/t_val omitted
    assert esn.t_train is None and esn.t_val is None

    U_wtv, Y_wtv, U_test, Y_test = esn._split_and_format_data(segments)

    # 80/20 segment split: 8 train/val segments kept whole, 2 held out for test
    assert len(U_wtv) == 8
    assert len(U_test) == 2
    assert [u.shape[0] for u in U_wtv] == [nt - 1 for nt in lengths[:8]]

    # t_val inferred from the median segment length's post-washout tail
    assert esn.t_val is not None and esn.t_train is not None
    assert esn.N_val == int(np.median(lengths)) - esn.N_wash - 1

    esn.train(segments, plot_training=False)
    assert esn.trained


def test_omitted_t_train_single_trajectory_80_20():
    """Single unsegmented trajectory, nothing provided: 80% train/val, 20% test,
    t_val = 20% of the train/val window."""
    rng = np.random.default_rng(8)
    data = rng.normal(size=(200, 2)).cumsum(axis=0) * 0.1

    esn = EchoStateNetwork(data.T, dt=1, N_units=15, upsample=1, N_wash=5,
                            hyperparameters_to_optimize=[])
    U_wtv, Y_wtv, U_test, Y_test = esn._split_and_format_data(data)

    n_wtv = esn.N_train + esn.N_val
    assert n_wtv == 160                      # 80% of 200
    assert esn.N_val == round(0.2 * 160)     # 20% of the train/val window
    assert U_wtv.shape[1] == n_wtv - 1
    assert U_test.shape[1] == data.shape[0] - n_wtv - 1


def test_rr_terms_invariant_to_n_split():
    """With reservoir continuity across contiguous chunks, the ridge system must be
    EXACTLY the same whatever N_split is -- it is pure computation batching."""
    rng = np.random.default_rng(3)
    Nt, N_dim = 400, 3
    y = np.stack([np.sin(0.07 * np.arange(Nt) + 2 * k) for k in range(N_dim)], axis=1)
    y += 0.01 * rng.standard_normal(y.shape)

    def rr(n_split):
        esn = EchoStateNetwork(y.T, dt=1, N_units=40, upsample=1, N_wash=20,
                               t_train=300, t_val=50, seed=0,
                               hyperparameters_to_optimize=[])
        esn.N_split = n_split
        esn._generate_W_Win(seed=0)
        esn.Wout = np.zeros((esn.N_units + 1, esn.N_dim))
        esn.norm, esn.shift = np.ones(esn.N_dim), np.zeros(esn.N_dim)
        U_wtv, Y_wtv = esn._UY_from_raw_data(y[np.newaxis], add_noise=False)[:2]
        return esn._compute_RR_terms(U_wtv, Y_wtv)[:2]

    LHS1, RHS1 = rr(1)
    LHS4, RHS4 = rr(4)
    assert np.allclose(LHS1, LHS4, atol=1e-12)
    assert np.allclose(RHS1, RHS4, atol=1e-12)


def test_spectral_radius_arpack_fallback():
    # N_units=150, seed=0 stagnates ARPACK (0/1 eigenvectors at 1501 iterations);
    # the dense fallback must still deliver a unit-spectral-radius reservoir
    import numpy as np

    from echostatenetwork import EchoStateNetwork

    y = np.random.default_rng(0).standard_normal((200, 3))
    esn = EchoStateNetwork(y, dt=0.1, N_units=150, N_wash=10, seed=0)
    esn._generate_W_Win(seed=esn.seed)
    W = esn.W.toarray() if hasattr(esn.W, "toarray") else np.asarray(esn.W)
    rho = np.abs(np.linalg.eigvals(W)).max()
    assert np.isclose(rho, 1.0, atol=1e-8)


def test_leak_rate_default_is_plain_tanh_step():
    # leak_rate defaults to 1.0 (no leak): the update IS the plain tanh map
    rng = np.random.default_rng(11)
    data = rng.normal(size=(60, 3))
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=30,
                           t_val=10, t_test=10, N_wash=5,
                           hyperparameters_to_optimize=[])
    esn.train(data, plot_training=False)
    assert esn.leak_rate == 1.0
    u, r = data[:1].T, rng.normal(size=(esn.N_units, 1))
    u_aug = np.concatenate((esn.normalize_input(u), esn.bias_in * np.ones((1, 1))))
    expected = np.tanh(esn.sigma_in * esn.Win.dot(u_aug) + esn.rho * esn.W.dot(r))
    np.testing.assert_allclose(esn.step(u, r)[1], expected)


def test_leaky_step_matches_leaky_integrator_formula():
    rng = np.random.default_rng(12)
    data = rng.normal(size=(60, 3))
    alpha = 0.35
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=30,
                           t_val=10, t_test=10, N_wash=5, leak_rate=alpha,
                           hyperparameters_to_optimize=[])
    esn.train(data, plot_training=False)
    u, r = data[:1].T, rng.normal(size=(esn.N_units, 1))
    u_aug = np.concatenate((esn.normalize_input(u), esn.bias_in * np.ones((1, 1))))
    x_tanh = np.tanh(esn.sigma_in * esn.Win.dot(u_aug) + esn.rho * esn.W.dot(r))
    np.testing.assert_allclose(esn.step(u, r)[1], (1 - alpha) * r + alpha * x_tanh)


def test_jacobian_matches_finite_differences_with_and_without_leak():
    rng = np.random.default_rng(13)
    data = rng.normal(size=(80, 3))
    for alpha in (1.0, 0.6):
        esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=40,
                               t_val=15, t_test=10, N_wash=5, leak_rate=alpha,
                               hyperparameters_to_optimize=[])
        esn.train(data, plot_training=False)
        u = data[5].reshape(-1, 1).copy()
        r = 0.1 * rng.normal(size=(esn.N_units, 1))
        J = esn.Jacobian(u, r)
        eps = 1e-4   # smaller eps only adds FD roundoff (verified quadratic convergence)
        J_fd = np.zeros_like(J)
        for j in range(u.shape[0]):
            up, um = u.copy(), u.copy()
            up[j] += eps
            um[j] -= eps
            J_fd[:, j] = ((esn.step(up, r)[0] - esn.step(um, r)[0]) / (2 * eps))[:, 0]
        np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-8)


def test_leak_rate_optimized_in_bayesian_search():
    rng = np.random.default_rng(14)
    data = rng.normal(size=(2, 80, 3))
    esn = EchoStateNetwork(data[0].T, dt=1, N_units=15, upsample=1, t_train=40,
                           t_val=10, t_test=10, N_wash=5, N_folds=2, N_grid=2,
                           N_func_evals=4,
                           hyperparameters_to_optimize=['leak_rate'])
    esn.train(data, plot_training=False)
    lo, hi = esn.leak_rate_range
    assert lo <= esn.leak_rate <= hi
    assert 'leak_rate' in esn.bo_results['hp_names']
    assert 'leak_rate' in esn.training_summary() or esn.leak_rate == 1.0


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


def test_ssv_single_shot_validation_trains():
    esn = _train_single_series(EchoStateNetwork._SSV)
    assert esn.trained
    assert esn.Wout.shape == (esn.N_units + 1, 3)
    assert isinstance(esn.bo_results, dict)
    assert len(esn.bo_results['func_vals']) == 3


def test_wfv_walk_forward_validation_trains():
    esn = _train_single_series(EchoStateNetwork._WFV)
    assert esn.trained
    assert esn.Wout.shape == (esn.N_units + 1, 3)
    assert isinstance(esn.bo_results, dict)
    assert len(esn.bo_results['func_vals']) == 3


def test_kfv_k_fold_validation_trains():
    esn = _train_single_series(EchoStateNetwork._KFV)
    assert esn.trained
    assert esn.Wout.shape == (esn.N_units + 1, 3)
    assert isinstance(esn.bo_results, dict)
    assert len(esn.bo_results['func_vals']) == 3


def test_wfv_kfv_retrain_per_fold_giving_distinct_objectives():
    # _WFV/_KFV retrain Wout per fold on different training windows, so on the
    # same data/reservoir/seed their objective landscapes must differ from
    # _SSV's (single fold, single training prefix) and from each other. Every
    # run is deterministic (fixed data seed, reservoir seed, and gp_minimize
    # random_state), so identical traces would mean the fold machinery collapsed
    # into a single shared objective.
    traces = {}
    for name, strategy in [('SSV', EchoStateNetwork._SSV),
                           ('WFV', EchoStateNetwork._WFV),
                           ('KFV', EchoStateNetwork._KFV)]:
        traces[name] = _train_single_series(strategy).bo_results['func_vals']
    assert not np.allclose(traces['SSV'], traces['WFV'])
    assert not np.allclose(traces['SSV'], traces['KFV'])
    assert not np.allclose(traces['WFV'], traces['KFV'])


def test_wfv_rejects_ragged_segments():
    # The single-series strategies are defined for one contiguous series; a
    # ragged (list-of-segments) corpus must raise and point to the segment-aware
    # strategy instead.
    rng = np.random.default_rng(23)
    segments = [rng.normal(size=(nt, 2)) for nt in (60, 90, 45)]
    esn = EchoStateNetwork(segments[0].T, dt=1, N_units=10, upsample=1,
                            t_train=30, t_val=10, N_wash=5,
                            N_grid=2, N_func_evals=2,
                            hyperparameters_to_optimize=['rho'])
    with pytest.raises(ValueError, match='_SegmentRVC_Noise'):
        esn.train(segments, plot_training=False,
                  validation_strategy=EchoStateNetwork._WFV)


def test_val_fold_step_overlapping_intervals_multiply_folds(capsys):
    # Chaotic fold variants of Racca & Magri (2021): a val_fold_step smaller
    # than N_val advances consecutive validation intervals by less than their
    # length, so they overlap and more folds fit in the same data. Request an
    # unreachable N_folds so both runs print the auto-reduction note, and read
    # the achieved fold count from it.
    def achieved_folds(strategy, step):
        _train_single_series(strategy, N_folds=100, N_func_evals=2,
                             val_fold_step=step)
        out = capsys.readouterr().out
        return int(re.search(r'using (\d+) folds', out).group(1))

    for strategy in (EchoStateNetwork._WFV, EchoStateNetwork._KFV):
        k_regular = achieved_folds(strategy, None)
        k_chaotic = achieved_folds(strategy, 10)  # < N_val = 30 -> overlap
        assert k_chaotic > k_regular


def test_val_fold_step_none_matches_explicit_n_val_bit_identically():
    # The default val_fold_step=None must reproduce the regular non-overlapping
    # fold geometry bit-identically: with the fold advance spelled out
    # explicitly as N_val (t_val=30, dt_ESN=1), the BHO trace is identical.
    # Retrain the SAME instance (regenerating W is not bit-reproducible --
    # the spectral-radius rescaling goes through ARPACK's random start
    # vector) and disable the input noise (drawn from the advancing
    # instance rng); everything left is deterministic.
    rng = np.random.default_rng(20)
    data = rng.normal(size=(200, 3)).cumsum(axis=0) * 0.1
    for strategy in (EchoStateNetwork._WFV, EchoStateNetwork._KFV):
        esn = EchoStateNetwork(data.T, dt=1, N_units=15, upsample=1,
                               t_train=120, t_val=30, t_test=40, N_wash=5,
                               N_folds=3, N_grid=2, N_func_evals=3,
                               hyperparameters_to_optimize=['rho'], seed=0)
        traces = {}
        for step in (None, 30):
            esn.val_fold_step = step
            esn.train(data, plot_training=False, add_noise=False,
                      validation_strategy=strategy)
            traces[step] = esn.bo_results['func_vals'].copy()
        assert np.array_equal(traces[None], traces[30])


def test_training_summary_reports_realized_fold_count():
    # An N_folds override is otherwise invisible (strategies may cap or multiply
    # it -- see val_fold_step); the realized count must surface in the summary.
    rng = np.random.default_rng(21)
    data = rng.normal(size=(660, 3))
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=450,
                           t_val=50, t_test=100, N_wash=5, N_folds=6,
                           N_grid=2, N_func_evals=2,
                           hyperparameters_to_optimize=['rho'])
    esn.train(data, plot_training=False)
    assert esn.n_folds_realized == 6
    assert '6 validation folds' in esn.training_summary()

    esn_kfv = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=450,
                               t_val=50, t_test=100, N_wash=5, N_folds=100,
                               N_grid=2, N_func_evals=2,
                               hyperparameters_to_optimize=['rho'])
    esn_kfv.train(data, plot_training=False, validation_strategy=EchoStateNetwork._KFV)
    # capped to what fits: the summary shows the REALIZED count, not the request
    assert esn_kfv.n_folds_realized < 100
    assert f'{esn_kfv.n_folds_realized} validation folds' in esn_kfv.training_summary()
