"""Training a trained network again, `train(force=True)` (issue #7): the reservoir (`W`, `Win`) is kept while the
options that define it are unchanged, whatever the seed, and drawn again otherwise; a given split is kept and an
inferred one is inferred again from the new data; `dt_ESN` follows `upsample`. The single network, the seeds of
``n_seeds > 1``, and the parallel layout with shared and with independent patches."""
import pickle
import warnings
from functools import partial

import numpy as np
import pytest

from echostatenetwork import EchoStateNetwork, validation
from echostatenetwork.esn import _patch_metric

DATA = np.random.default_rng(0).normal(size=(400, 3)).cumsum(axis=0) * 0.1
SITES = np.random.default_rng(1).normal(size=(300, 8)).cumsum(axis=0) * 0.1
QUICK = dict(dt=1, N_units=15, upsample=1, t_test=0, N_wash=5, hyperparameters_to_optimize=[], seed=0)
PATCHES = dict(QUICK, N_units=12, patch_size=2, halo=1)
GIVEN = dict(t_train=120, t_val=30)


def trained(data=DATA, **options):
    """A network trained once on `data`; `t_train` and `t_val` are inferred unless given."""
    esn = EchoStateNetwork(DATA.T, **{**QUICK, **options})
    esn.train(data, plot_training=False)
    return esn


def same(a, b):
    """Whether two matrices, sparse or dense, are equal up to rounding: the spectral radius that scales `W` comes
    from ARPACK, which may differ in the last place between processes (e.g. 1.6e-15 after a Bayesian search)."""
    dense = [m.toarray() if hasattr(m, 'toarray') else np.asarray(m) for m in (a, b)]
    return dense[0].shape == dense[1].shape and np.allclose(*dense, rtol=1e-12, atol=1e-14)


def split(esn):
    return esn.t_train, esn.t_val


def reservoirs(esn):
    return [net.W for net in esn.patches] if esn.patches else [esn.W]


# ---------------------------------------------------------------------------------------------- the reservoir
def test_the_reservoir_is_kept_while_the_options_that_define_it_are_unchanged():
    esn = trained(**GIVEN)
    W, Win, Wout = esn.W, esn.Win, esn.Wout.copy()
    esn.train(DATA, plot_training=False, rho=0.5, tikh=1e-6, noise=1e-3, readout_input=True, force=True)
    assert esn.W is W and esn.Win is Win and esn.Wout.shape == (18, 3)        # readout_input changes Wout only
    esn.train(DATA, plot_training=False, rho=0.9, readout_input=False, force=True)
    assert esn.W is W and esn.Win is Win and not np.array_equal(esn.Wout, Wout)


@pytest.mark.parametrize('layout', [{}, dict(patch_size=2, halo=1), dict(patch_size=2, halo=1, shared=False)],
                         ids=['single', 'shared', 'independent'])
def test_a_seed_alone_keeps_the_reservoir_in_every_layout(layout):
    # the seed is not an option of the reservoir: a reservoir drawn before the first train, or by it, is kept by a
    # later call with another seed alone, with a warning, and the attribute seed is not changed
    esn = EchoStateNetwork(SITES.T, **{**PATCHES, **layout})
    esn._generate_W_Win(seed=5)
    kept = reservoirs(esn)
    with pytest.warns(UserWarning, match='seed=6 is ignored'):
        esn.train(SITES, plot_training=False, seed=6)
    with pytest.warns(UserWarning, match='seed=3 is ignored'):
        esn.train(SITES, plot_training=False, seed=3, force=True)
    assert all(a is b for a, b in zip(reservoirs(esn), kept)) and esn.seed == 0
    if not layout:
        with pytest.warns(UserWarning, match=r'seed=6 is ignored: the network keeps its reservoir \(W, Win\), drawn '
                                             r'from seed 5'):
            esn.train(SITES, plot_training=False, seed=6, force=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            esn.train(SITES, plot_training=False, seed=5, force=True)  # the seed of the reservoir: no warning
        new = EchoStateNetwork(SITES.T, **PATCHES)
        new.train(SITES, plot_training=False, seed=4)                  # a new network draws from its seed
        assert not same(esn.W, new.W)


@pytest.mark.parametrize('option', [dict(N_units=30), dict(connect=6), dict(Win_type='dense'),
                                    dict(Win_type='sparse'), dict(observed_idx=np.array([0, 2]))])
def test_a_changed_option_draws_the_reservoir_that_a_new_network_draws(option):
    # issue #7, item 1: a changed option draws the reservoir, and fits the readout, that a new network with this
    # option draws from the same seed
    esn = trained(**GIVEN)
    esn.train(DATA, plot_training=False, force=True, **option)
    new = EchoStateNetwork(DATA.T, **{**QUICK, **GIVEN, **option})
    new.train(DATA, plot_training=False, seed=0)
    assert same(esn.W, new.W) and same(esn.Win, new.Win)
    # the readouts agree to the digits that the conditioning of the ridge system leaves (tikh = 1e-12)
    assert np.allclose(esn.Wout, new.Wout, rtol=1e-4, atol=1e-5 * np.abs(new.Wout).max())


def test_a_changed_option_draws_from_the_given_seed_which_the_reservoir_keeps():
    esn = trained(**GIVEN)
    esn.train(DATA, plot_training=False, seed=4, N_units=16, force=True)   # drawn from seed 4
    new = EchoStateNetwork(DATA.T, **{**QUICK, 'N_units': 16}, **GIVEN)
    new.train(DATA, plot_training=False, seed=4)
    assert same(esn.W, new.W) and same(esn.Win, new.Win)
    esn.train(DATA, plot_training=False, N_units=15, force=True)          # the reservoir keeps seed 4
    new = EchoStateNetwork(DATA.T, **QUICK, **GIVEN)
    new.train(DATA, plot_training=False, seed=4)
    assert same(esn.W, new.W) and same(esn.Win, new.Win) and esn.seed == 0
    W = esn.W
    esn.train(DATA, plot_training=False, force=True)
    assert esn.W is W


def test_parameters_added_or_removed_by_train():
    # parameters added by train draw a reservoir with their columns; the names param_shift_0 and param_norm_0 leave
    # hyperparameters_to_optimize with the parameters, or with optimize_parameter_normalization=False
    pytest.importorskip('skopt')                                         # a Bayesian search
    two = np.stack([DATA, DATA[::-1]])
    esn = EchoStateNetwork(DATA.T, **QUICK, **GIVEN)
    esn.train(two, plot_training=False)
    esn.train(two, plot_training=False, input_parameters=np.array([[0.0, 1.0]]), force=True)
    assert esn.Win.shape == (15, 5) and esn.Wout.shape == (15, 3) and np.isfinite(esn.Wout).all()
    assert esn.optimize_parameter_normalization and 'param_shift_0' in esn.hyperparameters_to_optimize
    esn = EchoStateNetwork(DATA.T, **QUICK, **GIVEN, input_parameters=np.array([[0.0, 1.0]]), N_grid=1,
                           N_func_evals=2)
    esn.train(two, plot_training=False)
    assert esn.hyperparameters_to_optimize == ['param_shift_0', 'param_norm_0']
    esn.train(two, plot_training=False, force=True)                    # the same parameters: the same search
    assert esn.hyperparameters_to_optimize == ['param_shift_0', 'param_norm_0']
    esn.train(two, plot_training=False, optimize_parameter_normalization=False, force=True)
    assert esn.hyperparameters_to_optimize == [] and esn.bo_results is None
    esn.train(two, plot_training=False, optimize_parameter_normalization=True, force=True)
    esn.train(two, plot_training=False, input_parameters=None, force=True)
    assert esn.hyperparameters_to_optimize == [] and esn.Win.shape == (15, 4) and np.isfinite(esn.Wout).all()


def test_matrices_set_by_hand_are_kept_while_the_options_of_the_network_are_unchanged():
    esn, source = EchoStateNetwork(DATA.T, **QUICK), EchoStateNetwork(DATA.T, **QUICK)
    source._generate_W_Win(seed=7)
    esn.Win, esn.W = source.Win, source.W                                # e.g. a reservoir shared by networks
    with pytest.warns(UserWarning, match='seed=3 is ignored'):          # no record of their seed: kept
        esn.train(DATA, plot_training=False, seed=3)
    assert esn.W is source.W and esn.Win is source.Win
    esn.train(DATA, plot_training=False, rho=0.5, force=True)
    assert esn.W is source.W
    esn.train(DATA, plot_training=False, N_units=20, force=True)
    assert esn.W.shape == (20, 20) and esn.Win.shape == (20, 4)


@pytest.mark.parametrize('option', [dict(connect=6), dict(Win_type='dense'), dict(N_units=20),
                                    dict(observed_idx=np.array([0, 2]))])
def test_a_reservoir_without_a_record_follows_the_rule_of_a_drawn_one(option):
    # set by hand, by from_arrays or by echostatenetwork 0.2.3: the options of the network at the start of the call
    # are those of the reservoir, and an option changed by the call, connect and Win_type included, draws a new one
    # from the attribute seed, with the type of Win of the new Win_type
    esn = trained(**GIVEN)
    del esn.__dict__['_reservoir_key']
    W = esn.W
    esn.train(DATA, plot_training=False, rho=0.5, force=True)
    assert esn.W is W
    esn.train(DATA, plot_training=False, force=True, **option)
    new = EchoStateNetwork(DATA.T, **{**QUICK, **GIVEN, 'rho': 0.5, **option})
    new.train(DATA, plot_training=False, seed=0)
    assert same(esn.W, new.W) and same(esn.Win, new.Win) and type(esn.Win) is type(new.Win)


@pytest.mark.parametrize('option', [dict(connect=6), dict(N_units=20), dict(seed=5), dict(seed=5, connect=6)])
def test_networks_that_share_one_reservoir_keep_or_draw_the_same_one(option):
    # a qlroms family with share_reservoir=True: the first network draws the reservoir, the others hold its matrices,
    # set by hand. A seed alone keeps the one shared reservoir; a changed option draws the same new reservoir for
    # every network of the family
    first = EchoStateNetwork(DATA.T, **QUICK, **GIVEN)
    first._generate_W_Win()
    family = [first] + [EchoStateNetwork(DATA.T, **QUICK, **GIVEN) for _ in range(2)]
    for esn in family[1:]:
        esn.Win, esn.W = first.Win, first.W
    for esn in family:
        esn.train(DATA, plot_training=False, force=True)
    for esn in family:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                             # a seed alone is ignored with a warning
            esn.train(DATA[:300], plot_training=False, force=True, **option)
    assert all(same(esn.W, first.W) and same(esn.Win, first.Win) for esn in family[1:])
    assert (first.W is family[1].W) == ('seed' in option and len(option) == 1)


@pytest.mark.parametrize('matrix', ['W', 'Win', 'W given to train'])
def test_a_matrix_set_for_a_new_N_units_is_kept_and_the_other_one_is_drawn(matrix):
    # a matrix set by hand for the new N_units, or given to train with it (and set after N_units, so that its shape
    # is checked against the new value), is kept; only the other one, whose shape no longer fits, is drawn again
    esn = trained(**GIVEN)
    if matrix == 'W given to train':
        esn.train(DATA, plot_training=False, W=np.eye(20), N_units=20, force=True)
        np.testing.assert_array_equal(esn.W.toarray(), np.eye(20))
    else:
        source = EchoStateNetwork(DATA.T, **{**QUICK, 'N_units': 20})
        source._generate_W_Win(seed=3)
        esn.N_units = 20
        setattr(esn, matrix, getattr(source, matrix))
        esn.train(DATA, plot_training=False, force=True)
        assert getattr(esn, matrix) is getattr(source, matrix)
    assert esn.W.shape == (20, 20) and esn.Win.shape == (20, 4) and np.isfinite(esn.Wout).all()


def test_the_jacobian_follows_a_network_trained_again_with_its_reservoir():
    esn = trained(**GIVEN)
    u, r = DATA[0][:, None], np.zeros((15, 1))
    esn.Jacobian(u, r)                                                  # caches dr_di
    esn.train(DATA[:300] * 3, plot_training=False, sigma_in=0.5, force=True)    # Win kept, sigma_in and norm new
    cached = esn.Jacobian(u, r)
    esn._invalidate_jacobian_cache()
    np.testing.assert_array_equal(cached, esn.Jacobian(u, r))


# ------------------------------------------------------------------------------------ the split and the time step
def test_an_inferred_split_is_inferred_again_and_a_given_one_is_kept():
    esn = trained()                                                     # t_train and t_val inferred
    esn.train(DATA[:200], plot_training=False, force=True)              # issue #7, item 3: failed on shorter data
    assert split(esn) == split(trained(DATA[:200]))
    esn.train(DATA, plot_training=False, force=True)                    # the same data give the same split
    assert split(esn) == split(trained())

    esn = trained(t_train=200)                                          # given t_train, inferred t_val
    esn.train(DATA[:300], plot_training=False, t_val=10, force=True)    # a t_val given to train
    assert split(esn) == (200, 10)
    esn.train(DATA, plot_training=False, force=True)
    assert split(esn) == (200, 10)
    esn.t_val = 20                                                      # set as an attribute
    esn.train(DATA, plot_training=False, force=True)
    assert split(esn) == (200, 20)

    segments = [DATA[a:b] for a, b in ((0, 80), (80, 170), (170, 270), (270, 380))]
    shorter = [segment[:40] for segment in segments]
    esn = trained(segments)
    esn.train(shorter, plot_training=False, force=True)                 # t_val inferred from the shorter segments
    assert split(esn) == split(trained(shorter))


def test_dt_ESN_follows_upsample():
    esn = trained(dt=0.5, t_train=60, t_val=15)
    assert (esn.dt_ESN, esn.N_train) == (0.5, 120)
    esn.train(DATA, plot_training=False, upsample=2, force=True)        # issue #7, item 2: dt_ESN stayed 0.5
    assert (esn.dt_ESN, esn.dt_physical, esn.N_train, esn.N_val) == (1.0, 0.5, 60, 15)
    assert esn.training_summary().startswith('trained: 60+15 train+val steps')
    esn.upsample = 4
    assert esn.dt_ESN == 2.0
    esn.dt_ESN = 1.0                                                    # sets the time step of the data
    assert esn.dt_physical == 0.25
    # dt_ESN and upsample given together set the time step of the data, dt_ESN / upsample, whatever their order
    for options in (dict(dt_ESN=2.0, upsample=2), dict(upsample=2, dt_ESN=2.0)):
        esn = trained(**GIVEN)
        esn.train(DATA, plot_training=False, force=True, **options)
        assert (esn.dt_ESN, esn.dt_physical, esn.N_train) == (2.0, 1.0, 60)
    # the constructor takes dt_ESN, with a warning when a given dt disagrees
    esn = EchoStateNetwork(DATA.T, dt_ESN=0.5, upsample=5)
    assert np.isclose(esn.dt_ESN, 0.5) and np.isclose(esn.dt_physical, 0.1)
    with pytest.warns(UserWarning, match=r'dt_ESN=0.5 and dt=1 disagree'):
        esn = EchoStateNetwork(DATA.T, dt=1, dt_ESN=0.5, upsample=1)
    assert esn.dt_ESN == 0.5
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert EchoStateNetwork(DATA.T, dt=0.1, dt_ESN=0.5).dt_ESN == 0.5


# --------------------------------------------------------------------------------------------------- n_seeds
def test_seeds_trained_again_are_the_same_seeds_and_the_kept_one_is_a_single_network():
    # the kept seed keeps the norm that its Wout was fitted with, and its summary marks t_val as inferred, as a
    # single network trained with that seed; trained again, the seeds are the same, not counted from the kept seed
    options = {**QUICK, 'noise': 5e-2}
    esn = EchoStateNetwork(DATA.T, **options)
    esn.train(DATA, plot_training=False, n_seeds=3, seed=10)
    seeds, kept = sorted(esn.seed_scores), esn.seed
    alone = EchoStateNetwork(DATA.T, **{**options, 'seed': kept})
    alone.train(DATA, plot_training=False, seed=kept)
    np.testing.assert_allclose(esn.norm, alone.norm, rtol=1e-12)
    np.testing.assert_allclose(esn.Wout, alone.Wout, rtol=1e-9, atol=1e-12)
    assert esn.training_summary() == alone.training_summary() and '(inferred)' in esn.training_summary()
    esn.train(DATA[:200], plot_training=False, n_seeds=3, force=True)   # the split inferred again in the workers
    assert sorted(esn.seed_scores) == seeds and split(esn) == split(trained(DATA[:200]))
    esn.train(DATA, plot_training=False, n_seeds=3, force=True)
    assert esn.seed == kept and sorted(esn.seed_scores) == seeds


def test_a_call_that_raises_leaves_the_network_as_it_was():
    esn = trained(**GIVEN)
    W, Wout, state = esn.W, esn.Wout, esn.rng.bit_generator.state
    with pytest.raises(ValueError, match='Increase the length'):
        esn.train(DATA[:100], plot_training=False, N_units=20, force=True)
    assert esn.N_units == 15 and esn.W is W and esn.Wout is Wout and split(esn) == (120, 30)
    assert esn.rng.bit_generator.state == state and np.isfinite(esn.step(DATA[0], np.zeros(15))[0]).all()
    # a call with seeds that raises restores the keywords and the inferred split
    esn = trained()
    summary = esn.training_summary()
    with warnings.catch_warnings(), pytest.raises(UserWarning, match='WFV: only'):
        warnings.simplefilter('error', UserWarning)
        esn.train(DATA, plot_training=False, n_seeds=2, N_folds=100, validation_strategy=validation.WFV,
                  force=True)
    assert esn.N_folds == EchoStateNetwork.N_folds and split(esn) == split(trained())
    assert esn.training_summary() == summary


def test_a_search_without_tikh_leaves_tikh_range():
    pytest.importorskip('skopt')                                         # a Bayesian search
    # a search without 'tikh' keeps the fixed tikh and leaves tikh_range, which a later search over 'tikh' scans
    esn = trained(**GIVEN, hyperparameters_to_optimize=['rho'], N_grid=2, N_func_evals=2,
                  tikh_range=[1e-6, 1e-9], tikh=1e-3)
    assert esn.tikh == 1e-3 and esn.tikh_range == [1e-6, 1e-9]
    esn.train(DATA, plot_training=False, hyperparameters_to_optimize=['tikh'], force=True)
    assert esn.tikh in (1e-6, 1e-9) and 'over the 2 values of tikh_range' in esn.training_summary()
    default = trained(**GIVEN, hyperparameters_to_optimize=['rho'], N_grid=2, N_func_evals=2)
    assert 'tikh_range' not in vars(default)                            # the class default, unchanged


# ------------------------------------------------------------------------------------------ parallel layout
def test_shared_patches_draw_their_reservoir_again_for_a_new_halo_or_N_units():
    esn = EchoStateNetwork(SITES.T, **PATCHES)
    esn.train(SITES, plot_training=False)
    esn.train(SITES, plot_training=False, halo=2, N_units=10, force=True)
    assert esn.W.shape == (10, 10) and esn.Win.shape == (10, 7) and esn.N_r == 40
    assert np.isfinite(esn.step(SITES[0], np.zeros(esn.N_r))[0]).all()
    with pytest.raises(ValueError, match='patch_size=3'):                # checked as by the constructor
        esn.train(SITES, plot_training=False, patch_size=3, halo=0, force=True)
    assert (esn.patch_size, esn.halo, esn.t_train) == (2, 2, 192)       # and left as it was
    # the layout switches between shared and independent patches
    esn.train(SITES, plot_training=False, shared=False, force=True)
    assert esn.trained and len(esn.patches) == 4 and not hasattr(esn, '_W')
    esn.train(SITES, plot_training=False, shared=True, force=True)
    assert esn.trained and esn.patches is None and esn.W.shape == (10, 10)
    assert np.isfinite(esn.step(SITES[0], np.zeros(esn.N_r))[0]).all()


def test_independent_patches_keep_their_networks_unless_the_layout_changes():
    esn = EchoStateNetwork(SITES.T, shared=False, **PATCHES)
    esn.train(SITES, plot_training=False)
    nets, Ws = list(esn.patches), [net.W for net in esn.patches]
    esn.train(SITES[:150], plot_training=False, rho=0.5, force=True)
    assert all(a is b and a.W is W and a.rho == 0.5 for a, b, W in zip(esn.patches, nets, Ws))
    short = EchoStateNetwork(SITES.T, shared=False, **PATCHES)
    short.train(SITES[:150], plot_training=False)
    assert split(esn) == split(short) == split(esn.patches[-1])        # the split inferred again
    esn.train(SITES, plot_training=False, N_units=10, force=True)       # each patch draws its reservoir again
    assert all(net.W.shape == (10, 10) for net in esn.patches) and esn.patches[0] is nets[0]
    single = EchoStateNetwork(np.zeros((4, 1)), **{**QUICK, 'N_units': 10, 'seed': esn.patches[0].seed})
    single._generate_W_Win(seed=esn.patches[0].seed)                    # from the seed of the patch
    assert same(esn.patches[0].W, single.W) and same(esn.patches[0].Win, single.Win)
    esn.train(SITES, plot_training=False, halo=2, force=True)           # new networks, with a wider window
    assert all(net.N_dim == 6 and net.Win.shape == (10, 7) for net in esn.patches)
    nets = list(esn.patches)
    with pytest.warns(UserWarning, match=r'seed=3 is ignored: the networks of the patches keep their reservoirs and '
                                         r'their seeds \(spawned from seed 0\); train spawns new seeds only when'):
        esn.train(SITES, plot_training=False, seed=3, force=True)
    assert all(a is b for a, b in zip(esn.patches, nets))
    # an option that redraws the reservoirs: the networks keep their seeds, not their reservoirs
    seeds = [net.seed for net in esn.patches]
    with pytest.warns(UserWarning, match=r'seed=3 is ignored: the networks of the patches keep their seeds \(spawned '
                                         r'from seed 0\), from which they draw their new reservoirs;') as rec:
        esn.train(SITES, plot_training=False, seed=3, N_units=8, force=True)
    assert not any('keep their reservoirs' in str(w.message) for w in rec)
    assert [net.seed for net in esn.patches] == seeds and all(net.W.shape == (8, 8) for net in esn.patches)
    esn.train(SITES, plot_training=False, seed=3, halo=1, force=True)   # new networks, from seed 3
    assert all(a is not b for a, b in zip(esn.patches, nets))
    assert esn._patches_key['seed'] == 3 and np.isfinite(esn.step(SITES[0], np.zeros(esn.N_r))[0]).all()


@pytest.mark.parametrize('name, value', [('N_units', 10), ('connect', 6), ('Win_type', 'dense')])
def test_independent_patches_take_an_option_set_as_an_attribute(name, value):
    # an option set as an attribute of the parallel network reaches the networks of the patches, which draw new
    # reservoirs that fit it
    esn = EchoStateNetwork(SITES.T, shared=False, **PATCHES)
    esn.train(SITES, plot_training=False)
    Ws = [net.W for net in esn.patches]
    setattr(esn, name, value)
    esn.train(SITES, plot_training=False, force=True)
    assert all(getattr(net, name) == value and net.W is not W for net, W in zip(esn.patches, Ws))
    assert np.isfinite(esn.step(SITES[0], np.zeros(esn.N_r))[0]).all()


def test_independent_patches_take_the_split_and_the_metric_of_the_parallel_network():
    esn = EchoStateNetwork(SITES.T, shared=False, **PATCHES)
    esn.train(SITES, plot_training=False)
    esn.t_val = 20                                                      # set as an attribute of the parallel network
    metric = partial(validation.log_nMAE)
    esn.train(SITES, plot_training=False, validation_metric=metric, force=True)
    assert esn.t_val == 20 and all(net.t_val == 20 for net in esn.patches)
    for net in esn.patches:                                             # wrapped: it scores the sites of the patch
        assert net.validation_metric.func is _patch_metric and net.validation_metric.keywords['metric'] is metric
    esn.validation_metric = validation.nMSE                             # set as an attribute
    esn.train(SITES, plot_training=False, force=True)
    assert all(net.validation_metric.keywords['metric'] is validation.nMSE for net in esn.patches)


def test_independent_patches_keep_their_own_search_unless_the_option_changes_on_the_parallel_network():
    pytest.importorskip('skopt')                                         # a Bayesian search
    esn = EchoStateNetwork(SITES.T, shared=False, **{**PATCHES, 'hyperparameters_to_optimize': ['rho']},
                           N_grid=2, N_func_evals=2, N_folds=2)
    esn.train(SITES, plot_training=False)
    rhos = [net.rho for net in esn.patches]                             # selected by the search of every patch
    esn.train(SITES, plot_training=False, hyperparameters_to_optimize=[], force=True)
    assert [net.rho for net in esn.patches] == rhos
    esn.rho = 0.5                                                       # set on the parallel network
    esn.train(SITES, plot_training=False, force=True)
    assert all(net.rho == 0.5 for net in esn.patches)


# ------------------------------------- networks pickled by echostatenetwork 0.2.3, seeds and the time step (issue #7)
def legacy(esn, cls=EchoStateNetwork):
    """`esn` as echostatenetwork 0.2.3 pickled it: `dt_ESN` stored, and no record of the split or of the reservoir."""
    state = dict(vars(esn))
    state['dt_ESN'] = state.pop('_dt_data') * esn.upsample
    del state['_inferred_split'], state['_reservoir_key']
    state['scale_bias_in'] = esn.scale_bias_in      # explicit: a pickle without it has the bias inside sigma_in
    old = cls.__new__(cls)
    old.__setstate__(state)
    return old


class _OwnSetstate(EchoStateNetwork):
    """A subclass whose `__setstate__` does not call the one of EchoStateNetwork, as romda's LatentROM."""

    def __setstate__(self, state):
        self.__dict__.update(state)


def test_a_pickle_stores_dt_ESN_for_older_versions_and_an_older_pickle_keeps_it():
    esn = trained(dt=0.01, upsample=3, t_train=1.2, t_val=0.3)
    state = esn.__getstate__()
    assert np.isclose(state['dt_ESN'], 0.03) and '_dt_data' in state  # 0.2.3 reads dt_ESN
    back = pickle.loads(pickle.dumps(esn))
    assert 'dt_ESN' not in vars(back) and back.dt_ESN == esn.dt_ESN
    old = legacy(esn)
    back = pickle.loads(pickle.dumps(old))
    assert np.isclose(back.dt_ESN, 0.03) and np.isclose(back.dt_physical, 0.01) and 'dt_ESN' not in vars(back)
    np.testing.assert_array_equal(back.step(DATA[0], np.zeros(back.N_r))[0], esn.step(DATA[0], np.zeros(15))[0])


@pytest.mark.parametrize('cls', [EchoStateNetwork, _OwnSetstate])
def test_a_network_pickled_before_this_version_keeps_its_dt_ESN_and_infers_its_split_again(cls):
    esn = cls(DATA.T, **{**QUICK, 'dt': 0.01, 'upsample': 3})
    esn.train(DATA, plot_training=False)                                # t_train and t_val inferred
    old = legacy(esn, cls)
    assert np.isclose(old.dt_ESN, 0.03) and np.isclose(old.dt_physical, 0.01) and old.N_train == esn.N_train
    old.train(DATA[:200], plot_training=False, force=True)             # the inferred split, inferred again
    new = cls(DATA.T, **{**QUICK, 'dt': 0.01, 'upsample': 3})
    new.train(DATA[:200], plot_training=False)
    assert np.allclose(split(old), split(new))


@pytest.mark.parametrize('given, seeds', [((280, 40), False), ((256, 64), False), (None, True)],
                         ids=['fills_80_percent', 'equal_to_the_inferred_one', 'inferred_with_seeds'])
def test_the_split_of_a_network_pickled_before_this_version(given, seeds):
    # 280 + 40 steps fill 80% of the 400 steps, as an inferred t_train does, but 40 is not 20% of them; 256 + 64 is
    # the split that train infers, but split_summary says that t_val was given: a given split is kept on other data,
    # as 0.2.3 kept it. With seeds, 0.2.3 wrote t_val_inferred=False into split_summary when it scored the kept seed
    # without a search: with seed_scores, the inferred split is recognized, and inferred again from shorter data.
    assert split(trained()) == (256, 64)
    if given:
        old = legacy(trained(t_train=given[0], t_val=given[1]))
        old.train(np.concatenate([DATA, DATA]), plot_training=False, force=True)
        assert split(old) == given
    else:
        old = legacy(trained())
        old.split_summary = {**old.split_summary, 't_val_inferred': False}
        old.seed_scores = {0: -1.0, 1: -0.5}                            # trained with n_seeds=2
        old.train(DATA[:200], plot_training=False, force=True)
        assert split(old) == split(trained(DATA[:200]))


def test_the_seed_warning_of_independent_patches_pickled_before_this_version():
    # echostatenetwork 0.2.3 kept no record of the seed that the seeds of the patches were spawned from
    esn = EchoStateNetwork(SITES.T, shared=False, **PATCHES)
    esn.train(SITES, plot_training=False)
    state = dict(vars(esn))
    for name in ('_patches_key', '_patch_options_seen', '_inferred_split'):
        del state[name]
    for net in state['patches']:
        del net.__dict__['_reservoir_key'], net.__dict__['_inferred_split']
    old = EchoStateNetwork.__new__(EchoStateNetwork)
    old.__setstate__(state)
    Ws = [net.W for net in old.patches]
    with pytest.warns(UserWarning, match=r'the networks of the patches keep their reservoirs and their seeds '
                                         r'\(patches\[p\]\.seed\);') as rec:
        old.train(SITES, plot_training=False, seed=3, force=True)
    assert not any('seed None' in str(w.message) for w in rec)
    assert all(net.W is W for net, W in zip(old.patches, Ws))
