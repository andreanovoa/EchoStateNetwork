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


# ---------------------------------------------------------------------------------------------- the reservoir
def test_the_reservoir_is_kept_while_the_options_that_define_it_are_unchanged():
    esn = trained(t_train=120, t_val=30)
    W, Win, Wout = esn.W, esn.Win, esn.Wout.copy()
    esn.train(DATA, plot_training=False, rho=0.5, tikh=1e-6, noise=1e-3, readout_input=True, force=True)
    assert esn.W is W and esn.Win is Win and esn.Wout.shape == (19, 3)        # readout_input changes Wout only
    esn.train(DATA, plot_training=False, rho=0.9, readout_input=False, force=True)
    assert esn.W is W and esn.Win is Win and not np.array_equal(esn.Wout, Wout)


def test_a_seed_alone_keeps_the_reservoir_and_warns():
    # the user decided that train(force=True) keeps the reservoir, and the seed is not an option of the reservoir.
    # Before round 2 of the review, a new seed drew a new reservoir for a network whose reservoir was drawn, but not
    # for a network that holds the same matrices set by hand, which split the shared reservoir of a qlroms family
    esn = trained(t_train=120, t_val=30)
    W, Win = esn.W, esn.Win
    with pytest.warns(UserWarning, match=r'seed=4 is ignored: the network keeps its reservoir \(W, Win\), which'):
        esn.train(DATA, plot_training=False, seed=4, force=True)
    assert esn.W is W and esn.Win is Win and esn.seed == 0              # the attribute seed is not changed
    new = EchoStateNetwork(DATA.T, **QUICK, t_train=120, t_val=30)
    new.train(DATA, plot_training=False, seed=4)
    assert not same(esn.W, new.W)


def test_the_first_train_keeps_a_reservoir_drawn_before_it_whatever_the_seed():
    esn = EchoStateNetwork(DATA.T, **QUICK, t_train=120, t_val=30)
    esn._generate_W_Win(seed=5)
    W, Win = esn.W, esn.Win
    with pytest.warns(UserWarning, match=r'seed=6 is ignored: the network keeps its reservoir \(W, Win\), drawn '
                                         r'from seed 5'):
        esn.train(DATA, plot_training=False, seed=6)
    assert esn.W is W and esn.Win is Win
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        esn.train(DATA, plot_training=False, seed=5, force=True)       # the seed of the reservoir: no warning
    assert esn.W is W


@pytest.mark.parametrize('option', [dict(N_units=30), dict(connect=6), dict(Win_type='dense'),
                                    dict(Win_type='sparse_dense_bias'), dict(observed_idx=np.array([0, 2]))])
def test_a_changed_option_draws_the_reservoir_that_a_new_network_draws(option):
    # issue #7, item 1: before, N_units and observed_idx failed with "matmul: dimension mismatch", and connect and
    # Win_type were ignored (the old W and Win were kept)
    esn = trained(t_train=120, t_val=30)
    esn.train(DATA, plot_training=False, force=True, **option)
    new = EchoStateNetwork(DATA.T, **{**QUICK, 't_train': 120, 't_val': 30, **option})
    new.train(DATA, plot_training=False, seed=0)
    assert same(esn.W, new.W) and same(esn.Win, new.Win) and np.allclose(esn.Wout, new.Wout)


def test_parameters_added_by_train_draw_a_reservoir_with_their_columns():
    two = np.stack([DATA, DATA[::-1]])
    esn = EchoStateNetwork(DATA.T, **QUICK, t_train=120, t_val=30)
    esn.train(two, plot_training=False)
    esn.train(two, plot_training=False, input_parameters=np.array([[0.0, 1.0]]), force=True)
    assert esn.Win.shape == (15, 5) and esn.Wout.shape == (16, 3) and np.isfinite(esn.Wout).all()


def test_a_changed_option_draws_from_the_given_seed_which_the_reservoir_keeps():
    esn = trained(t_train=120, t_val=30)
    esn.train(DATA, plot_training=False, seed=4, N_units=16, force=True)   # drawn from seed 4
    new = EchoStateNetwork(DATA.T, **{**QUICK, 'N_units': 16}, t_train=120, t_val=30)
    new.train(DATA, plot_training=False, seed=4)
    assert same(esn.W, new.W) and same(esn.Win, new.Win)
    esn.train(DATA, plot_training=False, N_units=15, force=True)          # the reservoir keeps seed 4
    new = EchoStateNetwork(DATA.T, **QUICK, t_train=120, t_val=30)
    new.train(DATA, plot_training=False, seed=4)
    assert same(esn.W, new.W) and same(esn.Win, new.Win) and esn.seed == 0
    W = esn.W
    esn.train(DATA, plot_training=False, force=True)
    assert esn.W is W


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
    # are those of the reservoir, and an option changed by the call draws a new one from the attribute seed. Before,
    # connect and Win_type were ignored for such a reservoir (Win_type='dense' kept a sparse Win)
    esn = trained(t_train=120, t_val=30)
    del esn.__dict__['_reservoir_key']
    W = esn.W
    esn.train(DATA, plot_training=False, rho=0.5, force=True)
    assert esn.W is W
    esn.train(DATA, plot_training=False, force=True, **option)
    new = EchoStateNetwork(DATA.T, **{**QUICK, 't_train': 120, 't_val': 30, 'rho': 0.5, **option})
    new.train(DATA, plot_training=False, seed=0)
    assert same(esn.W, new.W) and same(esn.Win, new.Win) and type(esn.Win) is type(new.Win)


@pytest.mark.parametrize('option', [dict(connect=6), dict(Win_type='dense'), dict(N_units=20), dict(seed=5),
                                    dict(seed=5, connect=6)])
def test_networks_that_share_one_reservoir_keep_or_draw_the_same_one(option):
    # a qlroms family with share_reservoir=True: the first network draws the reservoir, the others hold its matrices,
    # set by hand. Before round 2 of the review, connect, Win_type or a seed drew a new reservoir for the first
    # network only; qlroms then gave every network that reservoir after their Wout were fitted on the old one
    first = EchoStateNetwork(DATA.T, **QUICK, t_train=120, t_val=30)
    first._generate_W_Win()
    family = [first] + [EchoStateNetwork(DATA.T, **QUICK, t_train=120, t_val=30) for _ in range(2)]
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


@pytest.mark.parametrize('matrix', ['W', 'Win'])
def test_a_matrix_set_by_hand_after_a_new_N_units_is_kept_and_the_other_one_is_drawn(matrix):
    # before, the rows of W alone decided: a new Win set by hand was dropped with the old W, and a new W set by hand
    # with the old Win failed in the ridge regression
    esn = trained(t_train=120, t_val=30)
    source = EchoStateNetwork(DATA.T, **{**QUICK, 'N_units': 20})
    source._generate_W_Win(seed=3)
    esn.N_units = 20
    setattr(esn, matrix, getattr(source, matrix))
    esn.train(DATA, plot_training=False, force=True)
    assert getattr(esn, matrix) is getattr(source, matrix)
    assert esn.W.shape == (20, 20) and esn.Win.shape == (20, 4) and np.isfinite(esn.Wout).all()


# ------------------------------------------------------------------------------------------------- the split
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


def test_the_split_of_ragged_segments_is_inferred_again():
    segments = [DATA[a:b] for a, b in ((0, 80), (80, 170), (170, 270), (270, 380))]
    shorter = [segment[:40] for segment in segments]
    esn = trained(segments)
    esn.train(shorter, plot_training=False, force=True)                 # before, t_val stayed that of the first call
    assert split(esn) == split(trained(shorter))


# --------------------------------------------------------------------------------------------- the time step
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


def test_a_network_pickled_with_a_stored_dt_ESN_keeps_it():
    esn = trained(dt=0.01, upsample=3, t_train=1.2, t_val=0.3)
    state = dict(vars(esn))
    state['dt_ESN'] = state.pop('_dt_data') * 3                         # the state before dt_ESN followed upsample
    old = EchoStateNetwork.__new__(EchoStateNetwork)
    old.__setstate__(state)
    back = pickle.loads(pickle.dumps(old))
    assert np.isclose(back.dt_ESN, 0.03) and np.isclose(back.dt_physical, 0.01) and 'dt_ESN' not in vars(back)
    np.testing.assert_array_equal(back.step(DATA[0], np.zeros(back.N_r))[0], esn.step(DATA[0], np.zeros(15))[0])


# --------------------------------------------------------------------------------------------------- n_seeds
def test_seeds_trained_again_are_the_same_seeds_on_new_data():
    esn = EchoStateNetwork(DATA.T, **QUICK)
    esn.train(DATA, plot_training=False, n_seeds=3, seed=10)
    seeds, kept = sorted(esn.seed_scores), esn.seed
    esn.train(DATA[:200], plot_training=False, n_seeds=3, force=True)   # the split inferred again in the workers
    assert sorted(esn.seed_scores) == seeds                             # before, 11, 12, 13: from the kept seed
    assert split(esn) == split(trained(DATA[:200]))
    esn.train(DATA, plot_training=False, n_seeds=3, force=True)
    assert esn.seed == kept and sorted(esn.seed_scores) == seeds


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
    with pytest.warns(UserWarning, match='seed=3 is ignored: the networks of the patches keep their reservoirs'):
        esn.train(SITES, plot_training=False, seed=3, force=True)
    assert all(a is b for a, b in zip(esn.patches, nets))
    esn.train(SITES, plot_training=False, seed=3, halo=1, force=True)   # new networks, from seed 3
    assert all(a is not b for a, b in zip(esn.patches, nets))
    assert esn._patches_key['seed'] == 3 and np.isfinite(esn.step(SITES[0], np.zeros(esn.N_r))[0]).all()


@pytest.mark.parametrize('layout', [{}, dict(patch_size=2, halo=1), dict(patch_size=2, halo=1, shared=False)])
def test_a_seed_alone_keeps_the_reservoir_in_every_layout(layout):
    # before, train(seed=0, force=True) drew a new reservoir for a single network and for shared patches, and kept
    # the networks of independent patches, whose seeds were spawned from the attribute seed 0
    esn = EchoStateNetwork(SITES.T, **{**PATCHES, **layout})
    esn.train(SITES, plot_training=False)
    reservoirs = [net.W for net in esn.patches] if esn.patches else [esn.W]
    with pytest.warns(UserWarning, match='seed=3 is ignored'):
        esn.train(SITES, plot_training=False, seed=3, force=True)
    assert all(a is b for a, b in zip([net.W for net in esn.patches] if esn.patches else [esn.W], reservoirs))


def test_the_first_train_keeps_the_networks_of_patches_built_before_it_whatever_the_seed():
    # before, independent patches were built again for a seed other than the one they were spawned from
    esn = EchoStateNetwork(SITES.T, shared=False, **PATCHES)
    esn._generate_W_Win()
    Ws = [net.W for net in esn.patches]
    with pytest.warns(UserWarning, match='seed=3 is ignored'):
        esn.train(SITES, plot_training=False, seed=3)
    assert all(net.W is W for net, W in zip(esn.patches, Ws))


def test_independent_patches_take_the_split_and_the_metric_of_the_parallel_network():
    esn = EchoStateNetwork(SITES.T, shared=False, **PATCHES)
    esn.train(SITES, plot_training=False)
    esn.t_val = 20                                                      # set as an attribute of the parallel network
    metric = partial(validation.log_nMAE)
    esn.train(SITES, plot_training=False, validation_metric=metric, force=True)
    assert esn.t_val == 20 and all(net.t_val == 20 for net in esn.patches)
    for net in esn.patches:                                             # wrapped: it scores the sites of the patch
        assert net.validation_metric.func is _patch_metric and net.validation_metric.keywords['metric'] is metric


def test_the_layout_switches_between_shared_and_independent_patches():
    esn = EchoStateNetwork(SITES.T, **PATCHES)
    esn.train(SITES, plot_training=False)
    esn.train(SITES, plot_training=False, shared=False, force=True)     # before, an AttributeError in Wout
    assert esn.trained and len(esn.patches) == 4 and not hasattr(esn, '_W')
    esn.train(SITES, plot_training=False, shared=True, force=True)
    assert esn.trained and esn.patches is None and esn.W.shape == (12, 12)
    assert np.isfinite(esn.step(SITES[0], np.zeros(esn.N_r))[0]).all()


# ------------------------------------------------------------------------------- round 1 of the review (issue #7)
@pytest.mark.parametrize('name, value', [('N_units', 10), ('connect', 6), ('Win_type', 'dense')])
def test_independent_patches_take_an_option_set_as_an_attribute(name, value):
    # before, the networks of the patches kept N_units 12 and their reservoirs: step failed with "matmul: dimension
    # mismatch", and connect and Win_type were ignored
    esn = EchoStateNetwork(SITES.T, shared=False, **PATCHES)
    esn.train(SITES, plot_training=False)
    Ws = [net.W for net in esn.patches]
    setattr(esn, name, value)
    esn.train(SITES, plot_training=False, force=True)
    assert all(getattr(net, name) == value and net.W is not W for net, W in zip(esn.patches, Ws))
    assert np.isfinite(esn.step(SITES[0], np.zeros(esn.N_r))[0]).all()


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


def test_a_search_without_tikh_leaves_tikh_range():
    pytest.importorskip('skopt')                                         # a Bayesian search
    # before, the search wrote tikh_range = [tikh], so that a later search over 'tikh' compared one value
    esn = trained(t_train=120, t_val=30, hyperparameters_to_optimize=['rho'], N_grid=2, N_func_evals=2,
                  tikh_range=[1e-6, 1e-9], tikh=1e-3)
    assert esn.tikh == 1e-3 and esn.tikh_range == [1e-6, 1e-9]
    esn.train(DATA, plot_training=False, hyperparameters_to_optimize=['tikh'], force=True)
    assert esn.tikh in (1e-6, 1e-9) and 'over the 2 values of tikh_range' in esn.training_summary()
    default = trained(t_train=120, t_val=30, hyperparameters_to_optimize=['rho'], N_grid=2, N_func_evals=2)
    assert 'tikh_range' not in vars(default)                            # the class default, unchanged


def test_parameters_removed_by_train_leave_the_search_of_their_normalization():
    pytest.importorskip('skopt')                                         # a Bayesian search
    # before, the names param_shift_0 and param_norm_0 stayed in hyperparameters_to_optimize: IndexError
    two = np.stack([DATA, DATA[::-1]])
    esn = EchoStateNetwork(DATA.T, **QUICK, t_train=120, t_val=30, input_parameters=np.array([[0.0, 1.0]]),
                           N_grid=1, N_func_evals=2)
    esn.train(two, plot_training=False)
    assert esn.hyperparameters_to_optimize == ['param_shift_0', 'param_norm_0']
    esn.train(two, plot_training=False, force=True)                    # the same parameters: the same search
    assert esn.hyperparameters_to_optimize == ['param_shift_0', 'param_norm_0']
    esn.train(two, plot_training=False, optimize_parameter_normalization=False, force=True)
    assert esn.hyperparameters_to_optimize == [] and esn.bo_results is None
    esn.train(two, plot_training=False, optimize_parameter_normalization=True, force=True)
    esn.train(two, plot_training=False, input_parameters=None, force=True)
    assert esn.hyperparameters_to_optimize == [] and esn.Win.shape == (15, 4) and np.isfinite(esn.Wout).all()


def test_the_jacobian_follows_a_network_trained_again_with_its_reservoir():
    esn = trained(t_train=120, t_val=30)
    u, r = DATA[0][:, None], np.zeros((15, 1))
    esn.Jacobian(u, r)                                                  # caches dr_di
    esn.train(DATA[:300] * 3, plot_training=False, sigma_in=0.5, force=True)    # Win kept, sigma_in and norm new
    cached = esn.Jacobian(u, r)
    esn._invalidate_jacobian_cache()
    np.testing.assert_array_equal(cached, esn.Jacobian(u, r))


def test_a_call_that_raises_leaves_the_network_as_it_was():
    esn = trained(t_train=120, t_val=30)
    W, Wout, state = esn.W, esn.Wout, esn.rng.bit_generator.state
    with pytest.raises(ValueError, match='Increase the length'):
        esn.train(DATA[:100], plot_training=False, N_units=20, force=True)
    assert esn.N_units == 15 and esn.W is W and esn.Wout is Wout and split(esn) == (120, 30)
    assert esn.rng.bit_generator.state == state and np.isfinite(esn.step(DATA[0], np.zeros(15))[0]).all()


def test_a_call_with_seeds_that_raises_leaves_the_inferred_split():
    # before, the keywords and the split reset of the parent stayed: t_train and t_val None, training_summary raised
    esn = trained()
    summary = esn.training_summary()
    with warnings.catch_warnings(), pytest.raises(UserWarning, match='WFV: only'):
        warnings.simplefilter('error', UserWarning)
        esn.train(DATA, plot_training=False, n_seeds=2, N_folds=100, validation_strategy=validation.WFV,
                  force=True)
    assert esn.N_folds == EchoStateNetwork.N_folds and split(esn) == split(trained())
    assert esn.training_summary() == summary


def legacy(esn, cls=EchoStateNetwork):
    """`esn` as echostatenetwork 0.2.3 pickled it: `dt_ESN` stored, and no record of the split or of the reservoir."""
    state = dict(vars(esn))
    state['dt_ESN'] = state.pop('_dt_data') * esn.upsample
    del state['_inferred_split'], state['_reservoir_key']
    old = cls.__new__(cls)
    old.__setstate__(state)
    return old


class _OwnSetstate(EchoStateNetwork):
    """A subclass whose `__setstate__` does not call the one of EchoStateNetwork, as romda's LatentROM."""

    def __setstate__(self, state):
        self.__dict__.update(state)


@pytest.mark.parametrize('cls', [EchoStateNetwork, _OwnSetstate])
def test_a_network_pickled_before_this_version_keeps_its_dt_ESN_and_infers_its_split_again(cls):
    esn = cls(DATA.T, **{**QUICK, 'dt': 0.01, 'upsample': 3})
    esn.train(DATA, plot_training=False)                                # t_train and t_val inferred
    old = legacy(esn, cls)
    assert np.isclose(old.dt_ESN, 0.03) and np.isclose(old.dt_physical, 0.01) and old.N_train == esn.N_train
    old.train(DATA[:200], plot_training=False, force=True)             # before, "Increase the length" for old pickles
    new = cls(DATA.T, **{**QUICK, 'dt': 0.01, 'upsample': 3})
    new.train(DATA[:200], plot_training=False)
    assert np.allclose(split(old), split(new))


def test_a_network_pickled_by_this_version_stores_dt_ESN_for_older_versions():
    esn = trained(dt=0.01, upsample=3, t_train=1.2, t_val=0.3)
    state = esn.__getstate__()
    assert np.isclose(state['dt_ESN'], 0.03) and '_dt_data' in state  # 0.2.3 reads dt_ESN
    back = pickle.loads(pickle.dumps(esn))
    assert 'dt_ESN' not in vars(back) and back.dt_ESN == esn.dt_ESN


def test_the_constructor_takes_dt_ESN():
    # before, dt_ESN was overwritten by dt * upsample without a warning
    esn = EchoStateNetwork(DATA.T, dt_ESN=0.5, upsample=5)
    assert np.isclose(esn.dt_ESN, 0.5) and np.isclose(esn.dt_physical, 0.1)
    with pytest.warns(UserWarning, match=r'dt_ESN=0.5 and dt=1 disagree'):
        esn = EchoStateNetwork(DATA.T, dt=1, dt_ESN=0.5, upsample=1)
    assert esn.dt_ESN == 0.5
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert EchoStateNetwork(DATA.T, dt=0.1, dt_ESN=0.5).dt_ESN == 0.5


def test_independent_patches_take_a_metric_set_as_an_attribute():
    esn = EchoStateNetwork(SITES.T, shared=False, **PATCHES)
    esn.train(SITES, plot_training=False)
    esn.validation_metric = validation.nMSE                             # set on the parallel network
    esn.train(SITES, plot_training=False, force=True)
    assert all(net.validation_metric.keywords['metric'] is validation.nMSE for net in esn.patches)


# ------------------------------------------------------------------------------- round 2 of the review (issue #7)
def test_a_network_pickled_before_this_version_keeps_a_given_split_that_fills_80_percent_of_the_series():
    # 280 + 40 steps fill 80% of the 400 steps, as an inferred t_train does. Before, the given t_train was taken as
    # inferred and replaced on other data; 0.2.3 kept it
    old = legacy(trained(t_train=280, t_val=40))
    old.train(np.concatenate([DATA, DATA]), plot_training=False, force=True)
    assert split(old) == (280, 40)


def test_a_network_pickled_before_this_version_with_seeds_infers_its_split_again():
    # 0.2.3 split the data of the kept seed again, which wrote t_val_inferred=False into split_summary: before,
    # "Increase the length" on shorter data
    old = legacy(trained())
    old.split_summary = {**old.split_summary, 't_val_inferred': False}
    old.train(DATA[:200], plot_training=False, force=True)
    assert split(old) == split(trained(DATA[:200]))


def test_the_kept_seed_keeps_its_normalization_and_its_inferred_split():
    # before, the score of a seed without a search split the data of the kept network again: a new noise gave it
    # another norm than the one that fitted its Wout, and its summary lost '(inferred)'
    options = {**QUICK, 'noise': 5e-2}
    esn = EchoStateNetwork(DATA.T, **options)
    esn.train(DATA, plot_training=False, n_seeds=2)
    alone = EchoStateNetwork(DATA.T, **{**options, 'seed': esn.seed})
    alone.train(DATA, plot_training=False, seed=esn.seed)
    np.testing.assert_allclose(esn.norm, alone.norm, rtol=1e-12)
    np.testing.assert_allclose(esn.Wout, alone.Wout, rtol=1e-9, atol=1e-12)
    assert esn.training_summary() == alone.training_summary() and '(inferred)' in esn.training_summary()


@pytest.mark.parametrize('options', [dict(dt_ESN=2.0, upsample=2), dict(upsample=2, dt_ESN=2.0)])
def test_dt_ESN_and_upsample_given_to_train_do_not_depend_on_their_order(options):
    # before, dt_ESN given first was divided by the old upsample: dt_ESN 4.0
    esn = trained(t_train=120, t_val=30)
    esn.train(DATA, plot_training=False, force=True, **options)
    assert (esn.dt_ESN, esn.dt_physical, esn.N_train) == (2.0, 1.0, 60)


def test_a_reservoir_given_to_train_with_a_new_N_units_is_set_after_it():
    # before, W given before N_units failed the check of its shape against the old N_units
    esn = trained(t_train=120, t_val=30)
    esn.train(DATA, plot_training=False, W=np.eye(20), N_units=20, force=True)
    np.testing.assert_array_equal(esn.W.toarray(), np.eye(20))
    assert esn.Win.shape == (20, 4) and np.isfinite(esn.Wout).all()
