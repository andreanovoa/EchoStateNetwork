"""Validation strategies for EchoStateNetwork's Bayesian hyperparameter search.

Plain functions with the shared signature
``(x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence)``: ``case`` is the
EchoStateNetwork being validated, ``x`` the hyperparameter values under evaluation.
Pass one as ``train(validation_strategy=...)``; the class aliases
(``EchoStateNetwork._RVC_Noise`` etc.) keep the old spelling working.

- RVC_Noise: chaotic recycle validation, within-segment folds (the default).
- SegmentRVC_Noise: one washed probe per segment, for ragged short-segment corpora.
- RecycledSegmentRVC_Noise: SegmentRVC_Noise with probe seeds recycled from the
  ridge pass (exact and free) instead of re-washed from zero.
- SSV / WFV / KFV: the single-series strategies of Racca & Magri (2021), sharing
  the single_series_validation engine (one teacher-forced pass, prefix-sum ridge).
"""

import numpy as np


def RVC_Noise(x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence=True):
    """
    Implements Chaotic Recycle Validation for hyperparameter optimization.

    Parameters
    ----------
    x : list
        Hyperparameter values to evaluate.
    case : EchoStateNetwork
        Instance of the ESN being validated.
    U_wtv : np.ndarray
        Wash-train-validation input data.
    Y_wtv : np.ndarray
        Corresponding labels for train/validation data.
    tikh_opt : np.ndarray
        Array to store optimal Tikhonov regularization values.
    hp_names : list
        Names of the hyperparameters being optimized.

    Returns
    -------
        float: Mean (over folds) log10 closed-loop normalized MAE of the best Tikhonov candidate.

    See Also
    --------
    SSV, WFV, KFV : the single-series validation strategies of
        Racca & Magri (2021) (single shot, walk forward, K-fold),
        available for comparison on a single contiguous training series.
    SegmentRVC_Noise : recycle validation for a ragged corpus of segments.
    """
    # Re-set hyperparams as the optimization goes on
    if hp_names:
        case._reset_hyperparams(x, hp_names)

    N_tikh = len(case.tikh_range)
    nMAE = np.zeros(N_tikh)

    # Train using tv: Wout_tik is passed with all the combinations of tikh_ and target noise
    # This must result in L-Xa timeseries
    LHS, RHS, _, _ = case._compute_RR_terms(U_wtv, Y_wtv)
    Wout_tik = np.empty((N_tikh, case.N_units + 1, case.N_dim))

    # print(f'Computing Wout for tikhonov values: {case.tikh_range}')
    # print(f'LHS shape: {LHS.shape}, RHS shape: {RHS.shape}')

    for tik_j in range(N_tikh):
        LHS_reg = LHS.copy()
        LHS_reg.ravel()[::LHS.shape[1] + 1] += case.tikh_range[tik_j]
        Wout_tik[tik_j] = np.linalg.solve(LHS_reg, RHS)

    # print(U_wtv.shape, Y_wtv.shape, 'U_wtv, Y_wtv shapes in RVC noise')
    # Perform Validation in different folds
    n_looop = 0  # count the number of validation tests performed
    for U_l, Y_l in zip(U_wtv, Y_wtv):  # Each set of training data
        norm_l = np.max(Y_l, axis=0) - np.min(Y_l, axis=0)

        if case.input_parameters is not None:
            # Parameter (e.g. cluster id) may vary within U_l (a segment straddling
            # a transition); read off U_l's own first step and hold it fixed for
            # this validation run (a short window, usually within one regime).
            N_param = case._n_param(case.input_parameters)
            case.input_parameters = U_l[0, -N_param:].reshape(N_param, 1)

        # Segments may have very different lengths (e.g. cluster-dwell chunks), so
        # fold placement/spacing and count are computed per segment: fitting
        # case.N_folds evenly-spaced folds derived from the nominal case.N_train
        # (as if every segment were that long) silently slices out of bounds for
        # any shorter segment -- an empty U_wash/Y_val and a NaN in
        # compute_nMAE instead of a genuine hyperparameter signal.
        Nt_l = U_l.shape[0]
        usable_span = Nt_l - case.N_wash - case.N_val
        if usable_span < 0:
            continue  # segment too short to hold even one validation fold
        n_folds_l = min(case.N_folds, usable_span + 1)
        N_fw_l = usable_span // (n_folds_l - 1) if n_folds_l > 1 else 0

        for fold in range(n_folds_l):
            n_looop += 1
            # p is the washout window's own start within the segment (not an
            # offset past it); p + N_wash + N_val <= Nt_l always holds by
            # construction of N_fw_l, so this never slices past the segment
            # end into an empty Y_val (the "case.N_wash + ..." variant did,
            # whenever fold spacing was wide enough for the last fold to run
            # past Nt_l -- silently returning an empty slice and a NaN nMAE).
            p = fold * N_fw_l

            # Select washout and validation data
            U_wash = U_l[p:p + case.N_wash]
            Y_val = Y_l[p + case.N_wash:p + case.N_wash + case.N_val]

            for tik_j in range(N_tikh):  # cloop for each tikh_-noise combination

                case.Wout = Wout_tik[tik_j]

                # Washout inside the tikhonov loop (open-loop, no extra forecast
                # step): the closed loop below mutates u_out/r_out, so each
                # candidate Wout must start from its own washout -- otherwise
                # every tikh_ but the first begins from the previous candidate's
                # final state, and the first from a washout with a stale Wout.
                r_out = np.zeros((case.N_units, 1))
                u_out = np.zeros((case.N_dim, 1))
                for u_in in U_wash:
                    u_out, r_out = case.step(u_in, r_out)

                # Y_close = case.closedLoop(case.N_val)[0][1:].squeeze()
                Y_closed = np.zeros_like(Y_val)

                for i in range(Y_closed.shape[0]):
                    u_input = case.outputs_to_inputs(full_state=u_out)
                    u_out, r_out = case.step(u_input, r_out)
                    Y_closed[i] = u_out[:, 0].copy()

                # Compute normalized MSE; a diverged run contributes a fixed
                # per-fold penalty (log10 nMAE of 10) instead of resetting the
                # whole accumulated sum for this tikhonov value, which would
                # erase every previous fold's genuine signal.
                err = np.log10(case.compute_nMAE(Y_val, Y_closed, norm=norm_l))
                nMAE[tik_j] += err if np.isfinite(err) else 10.0

    if n_looop == 0:
        raise ValueError('No segment is long enough to hold a single validation '
                          'fold (need >= N_wash+N_val steps); reduce t_val or N_wash.')
    case.n_folds_realized = n_looop   # surfaced by training_summary()

    # select and save the optimal tikhonov and noise level in the targets
    a = nMAE.argmin()
    tikh_opt[case.val_k] = case.tikh_range[a]
    case.tikh = case.tikh_range[a]
    normalized_best_MAE = nMAE[a] / n_looop

    case.val_k += 1
    if print_convergence:
        print(case.val_k, end="")
        for hp in case.hyperparameters_to_optimize:
            print(f'\t {case._get_hyperparam(hp):.3e}', end="")
        print(f'\t {normalized_best_MAE:.4f}')

    return normalized_best_MAE

def SegmentRVC_Noise(x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence=True):
    """Recycle validation for a corpus of many, possibly short, roughly-independent
    segments (e.g. qlESN's per-cluster dwell chunks). Expects U_wtv/Y_wtv as a
    *list* of segments (the ragged path in _split_and_format_data), not a
    regular ndarray -- for a single long trajectory, use RVC_Noise instead.

    RVC_Noise carves several within-segment folds out of each segment, spaced
    by N_train/N_folds; for a 40-step dwell segment with N_wash=15, that leaves
    only a handful of steps to place folds in, so they end up almost identical
    (barely independent) or -- if N_folds is set for a long-trajectory workload
    -- silently degenerate (a fold's window running past the segment's own end,
    an empty validation slice, a NaN caught and replaced by a constant, see
    RVC_Noise's own comment). Neither is a good hyperparameter signal.

    Here every segment instead contributes exactly *one* washout+closed-loop
    probe, using whatever tail is available (up to N_val steps, or less for a
    short segment) -- fold-placement arithmetic disappears entirely, and with
    typically hundreds of segments there's already plenty of independent signal
    without needing several folds out of any single one. case.N_folds caps how
    many segments are probed per call (a fixed, evenly spaced subsample), for
    cost control on large corpora, rather than folds within one segment.
    """
    if hp_names:
        case._reset_hyperparams(x, hp_names)

    N_tikh = len(case.tikh_range)
    nMAE = np.zeros(N_tikh)

    LHS, RHS, _, _ = case._compute_RR_terms(U_wtv, Y_wtv)
    Wout_tik = np.empty((N_tikh, case.N_units + 1, case.N_dim))
    for tik_j in range(N_tikh):
        LHS_reg = LHS.copy()
        LHS_reg.ravel()[::LHS.shape[1] + 1] += case.tikh_range[tik_j]
        Wout_tik[tik_j] = np.linalg.solve(LHS_reg, RHS)

    # Segments long enough to hold a washout window at all; cap how many get
    # probed (random subsample) rather than probing every one of possibly
    # hundreds on every BHO evaluation.
    probeable = [l for l in range(len(U_wtv)) if U_wtv[l].shape[0] > case.N_wash]
    if not probeable:
        raise ValueError('No segment is long enough to hold a washout window '
                          f'(need > N_wash={case.N_wash} steps); reduce N_wash.')
    if len(probeable) > case.N_folds:
        # deterministic, evenly spaced subset -- NOT a fresh random draw per
        # call: the BHO objective must be comparable across hyperparameter
        # evaluations, and resampling the probe set every evaluation injects
        # noise the GP mistakes for hyperparameter signal.
        pick = np.unique(np.linspace(0, len(probeable) - 1, case.N_folds).astype(int))
        probeable = [probeable[i] for i in pick]

    n_looop = 0
    for l in probeable:
        U_l, Y_l = U_wtv[l], Y_wtv[l]
        Nt_l = U_l.shape[0]
        norm_l = np.max(Y_l, axis=0) - np.min(Y_l, axis=0)

        if case.input_parameters is not None:
            # Parameter (e.g. cluster id) may vary within U_l (a segment straddling
            # a transition); read off U_l's own first step and hold it fixed for
            # this validation run (a short window, usually within one regime).
            N_param = case._n_param(case.input_parameters)
            case.input_parameters = U_l[0, -N_param:].reshape(N_param, 1)

        n_looop += 1
        n_val_l = min(case.N_val, Nt_l - case.N_wash)  # whatever tail this segment has

        U_wash = U_l[:case.N_wash]
        Y_val = Y_l[case.N_wash: case.N_wash + n_val_l]

        for tik_j in range(N_tikh):
            case.Wout = Wout_tik[tik_j]

            # Washout per tikhonov candidate: the reservoir path is
            # Wout-independent, but the closed-loop seed u_out is the readout of
            # the washed reservoir and must reflect THIS candidate's Wout (a
            # shared post-washout copy would seed every candidate with the
            # previous/stale readout).
            r_out = np.zeros((case.N_units, 1))
            u_out = np.zeros((case.N_dim, 1))
            for u_in in U_wash:
                u_out, r_out = case.step(u_in, r_out)
            Y_closed = np.zeros_like(Y_val)
            for i in range(Y_closed.shape[0]):
                u_input = case.outputs_to_inputs(full_state=u_out)
                u_out, r_out = case.step(u_input, r_out)
                Y_closed[i] = u_out[:, 0].copy()

            # per-probe penalty for a diverged run, as in RVC_Noise (never
            # reset the accumulated sum -- that erases the other probes' signal)
            err = np.log10(case.compute_nMAE(Y_val, Y_closed, norm=norm_l))
            nMAE[tik_j] += err if np.isfinite(err) else 10.0

    if n_looop == 0:
        raise ValueError('No probeable segment available for validation after filtering; '
                         'reduce N_wash or provide longer segments.')
    case.n_folds_realized = n_looop   # surfaced by training_summary()

    # select and save the optimal tikhonov and noise level in the targets
    a = nMAE.argmin()
    tikh_opt[case.val_k] = case.tikh_range[a]
    case.tikh = case.tikh_range[a]
    normalized_best_MAE = nMAE[a] / n_looop

    case.val_k += 1
    if print_convergence:
        print(case.val_k, end="")
        for hp in case.hyperparameters_to_optimize:
            print(f'\t {case._get_hyperparam(hp):.3e}', end="")
        print(f'\t {normalized_best_MAE:.4f}')

    return normalized_best_MAE

def RecycledSegmentRVC_Noise(x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence=True):
    """`SegmentRVC_Noise` with the probe seeds RECYCLED from the ridge pass instead
    of re-washed from zero: `_compute_RR_terms` has already driven the reservoir
    open-loop over every segment and returns those trajectories (`R_RR`), so the
    state at post-washout index 0 is bit-identical to a dedicated washout of the
    same length -- and free. The candidate-specific closed-loop seed is one matmul
    (`reservoir_to_physical` of the shared state with this candidate's readout),
    the same pattern `single_series_validation` uses for `SSV`/`WFV`/`KFV`.

    Why it exists (measured on the rotating-cylinder corpus): with short segments
    the washed probe's seed sits one step before the synchronisation cliff, so its
    closed-loop error largely measures the washout transient, not the
    hyperparameters -- washed losses are inflated and select conservative
    hyperparameters through a seed artefact. Recycled seeding makes the validation
    loss interpretable as forecast skill over the probe horizon.

    Caveat, from the same study: the washed transient acted as an accidental
    stability regulariser, so recycled selection is more aggressive (higher rho,
    lower tikh) and can test worse on horizons far beyond the probe length.
    Stability preference must be encoded deliberately -- keep a rho cap and a tikh
    floor in the search ranges, and prefer a longer `t_val` where segments allow.
    """
    if hp_names:
        case._reset_hyperparams(x, hp_names)

    N_tikh = len(case.tikh_range)
    nMAE = np.zeros(N_tikh)

    # One ridge pass serves training AND every probe seed.
    LHS, RHS, _, R_open = case._compute_RR_terms(U_wtv, Y_wtv)
    Wout_tik = np.empty((N_tikh, case.N_units + 1, case.N_dim))
    for tik_j in range(N_tikh):
        LHS_reg = LHS.copy()
        LHS_reg.ravel()[::LHS.shape[1] + 1] += case.tikh_range[tik_j]
        Wout_tik[tik_j] = np.linalg.solve(LHS_reg, RHS)

    # R_open[l][i] is the state after consuming U_l[N_wash + i], so a probe seeded
    # at index 0 forecasts Y_l[N_wash + 1:] -- needs at least one target row.
    probeable = [l for l in range(len(U_wtv)) if U_wtv[l].shape[0] >= case.N_wash + 2]
    if not probeable:
        raise ValueError('No segment is long enough to hold a recycled probe '
                         f'(need >= N_wash+2={case.N_wash + 2} steps); reduce N_wash.')
    if len(probeable) > case.N_folds:
        # deterministic, evenly spaced subset, as SegmentRVC_Noise (a fresh draw per
        # call would inject noise the BHO's GP mistakes for hyperparameter signal)
        pick = np.unique(np.linspace(0, len(probeable) - 1, case.N_folds).astype(int))
        probeable = [probeable[i] for i in pick]

    n_looop = 0
    with np.errstate(over='ignore', invalid='ignore'):   # diverged probes are penalised, not warned
        for l in probeable:
            U_l, Y_l = U_wtv[l], Y_wtv[l]
            norm_l = np.max(Y_l, axis=0) - np.min(Y_l, axis=0)
            if case.input_parameters is not None:
                N_param = case._n_param(case.input_parameters)
                case.input_parameters = U_l[0, -N_param:].reshape(N_param, 1)
            n_looop += 1
            n_val_l = min(case.N_val, U_l.shape[0] - case.N_wash - 1)
            Y_val = Y_l[case.N_wash + 1: case.N_wash + 1 + n_val_l]
            r_seed = R_open[l][0][:, np.newaxis]

            for tik_j in range(N_tikh):
                case.Wout = Wout_tik[tik_j]
                u_out = case.reservoir_to_physical(r_seed)   # this candidate's own readout
                r_out = r_seed.copy()
                Y_closed = np.zeros_like(Y_val)
                for i in range(Y_closed.shape[0]):
                    u_input = case.outputs_to_inputs(full_state=u_out)
                    u_out, r_out = case.step(u_input, r_out)
                    Y_closed[i] = u_out[:, 0].copy()
                err = np.log10(case.compute_nMAE(Y_val, Y_closed, norm=norm_l))
                nMAE[tik_j] += err if np.isfinite(err) else 10.0   # per-probe divergence penalty

    case.n_folds_realized = n_looop   # surfaced by training_summary()

    a = nMAE.argmin()
    tikh_opt[case.val_k] = case.tikh_range[a]
    case.tikh = case.tikh_range[a]
    normalized_best_MAE = nMAE[a] / n_looop

    case.val_k += 1
    if print_convergence:
        print(case.val_k, end="")
        for hp in case.hyperparameters_to_optimize:
            print(f'\t {case._get_hyperparam(hp):.3e}', end="")
        print(f'\t {normalized_best_MAE:.4f}')

    return normalized_best_MAE

def single_series_validation(x, case, U_wtv, Y_wtv, tikh_opt, hp_names,
                              print_convergence, folds_of, strategy_name):
    """Shared engine for the single-series validation strategies of
    Racca & Magri (2021): `SSV`, `WFV` and `KFV` differ only in fold
    geometry, which each supplies via `folds_of`; everything else -- the
    teacher-forced open-loop pass, the per-fold ridge solves, the closed-loop
    probes, the Tikhonov grid and the BHO bookkeeping -- lives here.

    Noise handling is identical to `RVC_Noise`: U_wtv is already the noisy
    copy built by `_split_and_format_data` (inputs only; targets are clean),
    and no extra noise is added here.

    The open-loop reservoir trajectory does not depend on `Wout`, so ONE
    teacher-forced pass over the series per objective call serves every fold
    and every Tikhonov candidate. Each fold's ridge system is then assembled
    from prefix sums of per-interval Gram terms -- per-fold retraining is
    pure arithmetic, with no reservoir recomputation.

    Parameters
    ----------
    x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence
        As in `RVC_Noise`, except U_wtv/Y_wtv must be regular ndarrays with
        a single segment (``L == 1``); a segmented or ragged corpus raises a
        ValueError.
    folds_of : callable
        ``folds_of(case, n_post) -> list of (train_ranges, i0, n_val_k)``
        with ``n_post`` the number of post-washout rows; ``train_ranges`` a
        list of half-open row ranges ``(r0, r1)`` the fold trains on, ``i0``
        the first validation row, ``n_val_k`` the validation length.
    strategy_name : str
        Name used in error messages ('SSV', 'WFV', 'KFV').

    Returns
    -------
    float
        Mean (over folds) log10 closed-loop normalized error of the best
        Tikhonov candidate -- the scalar the Bayesian optimization minimizes.
    """
    # Re-set hyperparams as the optimization goes on
    if hp_names:
        case._reset_hyperparams(x, hp_names)

    if not isinstance(U_wtv, np.ndarray) or U_wtv.ndim != 3 or U_wtv.shape[0] != 1:
        raise ValueError(
            f'{strategy_name} requires a single contiguous training series '
            '(regular ndarray with L==1); got a segmented/ragged corpus. '
            'Use RVC_Noise or SegmentRVC_Noise instead.')

    U_l, Y_l = U_wtv[0], Y_wtv[0]
    norm_l = np.max(Y_l, axis=0) - np.min(Y_l, axis=0)

    if case.input_parameters is not None:
        # Hold the series' own leading parameter vector fixed for the
        # closed-loop probes (same slice trick as RVC_Noise; train()
        # restores the original input_parameters afterwards).
        N_param = case._n_param(case.input_parameters)
        case.input_parameters = U_l[0, -N_param:].reshape(N_param, 1)

    # ONE teacher-forced open-loop pass (Wout-independent). The returned
    # open-loop readouts (U_RR) are computed with the scratch Wout train()
    # zeroed before BHO -- stale, so they are discarded; the total LHS/RHS
    # are rebuilt per fold from interval sums below.
    _, _, _, R_RR = case._compute_RR_terms(U_wtv, Y_wtv)
    R = R_RR[0]  # (Nt - N_wash, N_units)
    r_aug = np.hstack([R, np.ones((R.shape[0], 1)) * case.bias_out])
    Y_t = Y_l[case.N_wash:]  # row i <-> input U_l[N_wash + i]
    n_post = r_aug.shape[0]

    if n_post < case.N_val + 1:
        raise ValueError(
            f'{strategy_name}: the series holds {n_post} post-washout steps, but '
            f'one validation interval (N_val={case.N_val}) plus at least one '
            'training step is required; reduce t_val or N_wash.')

    folds = folds_of(case, n_post)
    case.n_folds_realized = len(folds)   # surfaced by training_summary()

    # Prefix Gram sums at every training-range boundary: each fold's ridge
    # system is a difference of prefixes, so every teacher-forced row enters
    # exactly one Gram product regardless of the number of folds.
    bounds = sorted({0, *(b for tr, _, _ in folds for rng_ in tr for b in rng_)})
    A = np.zeros((case.N_units + 1, case.N_units + 1))
    B = np.zeros((case.N_units + 1, case.N_dim))
    prefix, prev = {}, 0
    for b in bounds:
        if b > prev:
            block = r_aug[prev:b]
            A = A + block.T @ block  # new arrays: stored references stay valid
            B = B + block.T @ Y_t[prev:b]
            prev = b
        prefix[b] = (A, B)

    N_tikh = len(case.tikh_range)
    nMAE = np.zeros(N_tikh)
    r_washed = None  # lazily-computed genuine washout state (i0 == 0 folds)

    for train_ranges, i0, n_val_k in folds:
        LHS = np.zeros_like(A)
        RHS = np.zeros_like(B)
        for r0, r1 in train_ranges:
            LHS += prefix[r1][0] - prefix[r0][0]
            RHS += prefix[r1][1] - prefix[r0][1]

        Y_val = Y_t[i0:i0 + n_val_k]

        # Reservoir seed for the closed-loop probe, free from the
        # teacher-forced pass: R[i0-1] is the open-loop state after
        # consuming the ENTIRE history up to input U_l[N_wash+i0-1]
        # (equivalent to, and longer than, a washout on the N_wash steps
        # preceding the interval). For i0 == 0 (interval flush with the training
        # washout) step open-loop through the genuine washout window
        # U_l[:N_wash] once; the reservoir path is Wout-independent, so one
        # pass serves every fold and Tikhonov candidate.
        if i0 >= 1:
            r_seed = R[i0 - 1][:, np.newaxis]
        else:
            if r_washed is None:
                r_washed = np.zeros((case.N_units, 1))
                for u_in in U_l[:case.N_wash]:
                    _, r_washed = case.step(u_in, r_washed)
            r_seed = r_washed

        for tik_j in range(N_tikh):
            LHS_reg = LHS.copy()
            LHS_reg.ravel()[::LHS_reg.shape[1] + 1] += case.tikh_range[tik_j]
            case.Wout = np.linalg.solve(LHS_reg, RHS)

            # Closed-loop seed: THIS candidate's readout of the shared
            # washed state -- equivalent to RVC_Noise's washout inside the
            # Tikhonov loop, since the open-loop reservoir path is
            # Wout-independent and only the seed readout depends on Wout.
            r_out = r_seed.copy()
            u_out = case.reservoir_to_physical(r_out)

            Y_closed = np.zeros_like(Y_val)
            for i in range(Y_closed.shape[0]):
                u_input = case.outputs_to_inputs(full_state=u_out)
                u_out, r_out = case.step(u_input, r_out)
                Y_closed[i] = u_out[:, 0].copy()

            # Per-fold penalty for a diverged closed loop, as in RVC_Noise
            # (never reset the accumulated sum -- that erases the other
            # folds' signal).
            err = np.log10(case.compute_nMAE(Y_val, Y_closed, norm=norm_l))
            nMAE[tik_j] += err if np.isfinite(err) else 10.0

    # select and save the optimal tikhonov (same bookkeeping as RVC_Noise:
    # _optimize_hyperparameters reads tikh_opt[best_idx] after BHO)
    a = nMAE.argmin()
    tikh_opt[case.val_k] = case.tikh_range[a]
    case.tikh = case.tikh_range[a]
    normalized_best_MAE = nMAE[a] / len(folds)

    case.val_k += 1
    if print_convergence:
        print(case.val_k, end="")
        for hp in case.hyperparameters_to_optimize:
            print(f'\t {case._get_hyperparam(hp):.3e}', end="")
        print(f'\t {normalized_best_MAE:.4f}')

    return normalized_best_MAE

def SSV(x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence=True):
    """Single shot validation (SSV) of Racca & Magri (2021).

    The series is split once: Wout is trained (per Tikhonov candidate) on
    everything before the last validation interval, and the closed-loop
    error is computed on that single interval of ``N_val`` steps at the end
    of the series, the reservoir washed out open-loop on the data
    immediately preceding it. Racca & Magri (2021) show SSV must not be
    relied on for chaotic time series (a single validation interval
    correlates weakly with test error); it is provided for comparison with
    the multi-interval strategies (`WFV`, `KFV`, `RVC_Noise`).

    Fold geometry: with ``n`` post-washout steps, one fold -- training rows
    ``[0, n - N_val)``, validation rows ``[n - N_val, n)``.

    Parameters
    ----------
    x : list
        Hyperparameter values to evaluate (aligned with `hp_names`).
    case : EchoStateNetwork
        Instance of the ESN being validated.
    U_wtv : np.ndarray
        Wash-train-validation input data, shape ``(1, Nt, N_dim_in)`` -- a
        single contiguous series. A segmented/ragged corpus raises a
        ValueError (use `RVC_Noise` or `SegmentRVC_Noise` for those).
    Y_wtv : np.ndarray
        Corresponding labels, shape ``(1, Nt, N_dim)``.
    tikh_opt : np.ndarray
        Array to store optimal Tikhonov regularization values.
    hp_names : list
        Names of the hyperparameters being optimized.
    print_convergence : bool
        Print one convergence row per evaluation.

    Returns
    -------
    float
        log10 closed-loop normalized error on the single validation
        interval (best Tikhonov candidate).

    References
    ----------
    Racca & Magri (2021). Robust optimization and validation of echo state
    networks for learning chaotic dynamics. Neural Networks, 142, 252-268
    (arXiv:2103.03174).
    """
    def folds_of(case_, n_post):
        i0 = n_post - case_.N_val
        return [([(0, i0)], i0, case_.N_val)]

    return single_series_validation(
        x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence,
        folds_of, 'SSV')

def WFV(x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence=True):
    """Walk forward validation (WFV) of Racca & Magri (2021).

    A fixed-length training window slides forward by ``step`` per fold;
    each fold retrains Wout on its own window (pure arithmetic on
    per-interval ridge sums -- the teacher-forced reservoir pass is shared)
    and validates closed-loop on the ``N_val`` steps immediately after it,
    the reservoir washed out open-loop on the data immediately preceding
    the interval. Hyperparameters minimize the mean closed-loop error over
    the folds, which is far more robust for chaotic series than `SSV`.

    Fold geometry: the advance between consecutive folds is
    ``step = val_fold_step or N_val``; the default ``val_fold_step = None``
    gives the regular WFV whose validation intervals tile the series
    without overlap, while ``val_fold_step`` of ~one Lyapunov time in ESN
    steps (< ``N_val``) gives the paper's chaotic version with overlapping
    intervals and correspondingly more folds. With ``n`` post-washout
    steps and ``K = min(N_folds, 1 + (n - 1 - N_val) // step)`` folds
    (reduced with a printed note when the requested ``N_folds`` do not
    fit), the training-window length is
    ``m = n - (K - 1) * step - N_val``; fold ``k`` (``k = 0..K-1``) trains
    on rows ``[k * step, k * step + m)`` and validates on
    ``[k * step + m, k * step + m + N_val)`` -- the last validation
    interval always ends at row ``n``.

    Parameters
    ----------
    x : list
        Hyperparameter values to evaluate (aligned with `hp_names`).
    case : EchoStateNetwork
        Instance of the ESN being validated.
    U_wtv : np.ndarray
        Wash-train-validation input data, shape ``(1, Nt, N_dim_in)`` -- a
        single contiguous series. A segmented/ragged corpus raises a
        ValueError (use `RVC_Noise` or `SegmentRVC_Noise` for those).
    Y_wtv : np.ndarray
        Corresponding labels, shape ``(1, Nt, N_dim)``.
    tikh_opt : np.ndarray
        Array to store optimal Tikhonov regularization values.
    hp_names : list
        Names of the hyperparameters being optimized.
    print_convergence : bool
        Print one convergence row per evaluation.

    Returns
    -------
    float
        Mean (over folds) log10 closed-loop normalized error of the best
        Tikhonov candidate.

    References
    ----------
    Racca & Magri (2021). Robust optimization and validation of echo state
    networks for learning chaotic dynamics. Neural Networks, 142, 252-268
    (arXiv:2103.03174). The chaotic versions (subscript c) advance the
    folds by one Lyapunov time instead of the validation-interval length,
    so consecutive validation intervals overlap; see `val_fold_step`.
    """
    def folds_of(case_, n_post):
        step = case_.val_fold_step or case_.N_val
        K = min(case_.N_folds, 1 + (n_post - 1 - case_.N_val) // step)
        if K < case_.N_folds and case_.val_k == 0:
            print(f'WFV: only {K} of the requested N_folds={case_.N_folds} '
                  f'walk-forward folds fit in {n_post} post-washout steps '
                  f'(N_val={case_.N_val}, step={step}); using {K} folds.')
        m = n_post - (K - 1) * step - case_.N_val  # fixed training-window length
        folds = []
        for k in range(K):
            s = k * step
            folds.append(([(s, s + m)], s + m, case_.N_val))
        return folds

    return single_series_validation(
        x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence,
        folds_of, 'WFV')

def KFV(x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence=True):
    """K-fold validation (KFV) of Racca & Magri (2021).

    Leave-one-interval-out: ``N_folds`` validation intervals of length
    ``N_val`` cover the post-washout data (after an initial offset
    absorbing the remainder, cf. the ``b*v`` offset in the paper); each
    fold retrains Wout on ALL rows outside its own interval (pure
    arithmetic on per-interval ridge sums -- the teacher-forced reservoir
    pass is shared) and validates closed-loop on it, the reservoir washed
    out open-loop on the data immediately preceding the interval.

    Note: the shared teacher-forced pass drives the reservoir open-loop
    through the held-out interval too -- the same recycling of training
    data that `RVC_Noise` embraces for its washout windows. `RVC_Noise`
    (recycle validation) matches KFV's accuracy at lower cost by also
    training Wout once on all the data.

    Fold geometry: the advance between consecutive validation intervals
    is ``step = val_fold_step or N_val``; the default
    ``val_fold_step = None`` gives the regular KFV whose intervals tile
    the data without overlap, while ``val_fold_step`` of ~one Lyapunov
    time in ESN steps (< ``N_val``) gives the paper's chaotic version with
    overlapping intervals and correspondingly more folds. With ``n``
    post-washout steps and ``K = min(N_folds, 1 + (n - N_val) // step)``
    intervals (reduced with a printed note when fewer fit), the initial
    offset is ``n - (K - 1) * step - N_val``; fold ``k`` (``k = 0..K-1``)
    validates on rows ``[offset + k * step, offset + k * step + N_val)``
    and trains on all rows outside its own interval.

    Parameters
    ----------
    x : list
        Hyperparameter values to evaluate (aligned with `hp_names`).
    case : EchoStateNetwork
        Instance of the ESN being validated.
    U_wtv : np.ndarray
        Wash-train-validation input data, shape ``(1, Nt, N_dim_in)`` -- a
        single contiguous series. A segmented/ragged corpus raises a
        ValueError (use `RVC_Noise` or `SegmentRVC_Noise` for those).
    Y_wtv : np.ndarray
        Corresponding labels, shape ``(1, Nt, N_dim)``.
    tikh_opt : np.ndarray
        Array to store optimal Tikhonov regularization values.
    hp_names : list
        Names of the hyperparameters being optimized.
    print_convergence : bool
        Print one convergence row per evaluation.

    Returns
    -------
    float
        Mean (over folds) log10 closed-loop normalized error of the best
        Tikhonov candidate.

    References
    ----------
    Racca & Magri (2021). Robust optimization and validation of echo state
    networks for learning chaotic dynamics. Neural Networks, 142, 252-268
    (arXiv:2103.03174). The chaotic versions (subscript c) advance the
    folds by one Lyapunov time instead of the validation-interval length,
    so consecutive validation intervals overlap; see `val_fold_step`.
    """
    def folds_of(case_, n_post):
        step = case_.val_fold_step or case_.N_val
        K = min(case_.N_folds, 1 + (n_post - case_.N_val) // step)
        if K < case_.N_folds and case_.val_k == 0:
            print(f'KFV: only {K} of the requested N_folds={case_.N_folds} '
                  f'validation intervals fit in {n_post} post-washout steps '
                  f'(N_val={case_.N_val}, step={step}); using {K} folds.')
        offset = n_post - (K - 1) * step - case_.N_val
        folds = []
        for k in range(K):
            i0 = offset + k * step
            train_ranges = [(r0, r1) for r0, r1 in
                            [(0, i0), (i0 + case_.N_val, n_post)] if r1 > r0]
            folds.append((train_ranges, i0, case_.N_val))
        return folds

    return single_series_validation(
        x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence,
        folds_of, 'KFV')

