"""EchoStateNetwork: a pure-numpy echo state network / reservoir computer.

Shared ESN implementation for qlrom, romda and related projects.
"""

import io
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy

# Validation methods
from functools import cached_property, partial
from itertools import product

import matplotlib.backends.backend_pdf as plt_pdf
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, issparse, lil_matrix
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import eigs as sparse_eigs

from . import validation


def add_pdf_page(pdf, fig):
    """Save one matplotlib figure as a page in an open PdfPages, then close it."""
    pdf.savefig(fig)
    plt.close(fig)


def _train_one_seed(case, seed, train_data, add_noise, validation_strategy, kwargs):
    """Worker for `EchoStateNetwork.train(n_seeds=...)` (module-level so it pickles
    into worker processes): regenerate the reservoir for this seed, train fully,
    and return ``(case, score)`` with one validation score comparable across seeds
    -- the BHO minimum when a search ran, otherwise one evaluation of the
    validation strategy on a sacrificial copy (the strategy mutates `Wout`/`tikh`
    while scanning `tikh_range`). Stdout is captured and returned instead of
    printed: worker prints interleave across processes, and a forked Jupyter
    kernel's ZMQ stream is not fork-safe -- the parent prints the logs in order."""
    log = io.StringIO()
    with redirect_stdout(log):
        for attr in ('_W', '_Win', '_Wout'):
            if hasattr(case, attr):
                delattr(case, attr)
        case.seed = seed
        case.train(train_data, add_noise=add_noise, plot_training=False,
                   validation_strategy=validation_strategy, seed=seed, **kwargs)
        if case.bo_results is not None:
            score = case.bo_results['fun']
        else:
            strategy = validation_strategy or validation.RVC_Noise
            U_wtv, Y_wtv = case._split_and_format_data(train_data, add_noise=add_noise)[:2]
            probe = case.copy()
            probe.val_k = 0
            score = strategy([], probe, U_wtv, Y_wtv, np.zeros(1), [],
                             print_convergence=False)
    return case, float(score), log.getvalue()

# XDG_RUNTIME_DIR = 'tmp/'


class EchoStateNetwork:
    """
    The EchoStateNetwork class implements a reservoir computing model for time series prediction. It is based on
    the Echo State Network (ESN) approach, which uses a randomly connected reservoir of neurons to map inputs
    into high-dimensional space. This allows the model to capture complex dynamics with efficient training.

    Attributes:
        - Reservoir and network hyperparameters (e.g., N_units, rho, sigma_in, tikh,
          leak_rate -- the leaky-integrator rate, 1.0 = no leak, etc.)
        - Training, validation, and test configuration (e.g., t_train, t_val, t_test, N_wash, etc.)
        - Optimization settings for Bayesian hyperparameter search (e.g., hyperparameters_to_optimize, rho_range, etc.)
        - Bayesian optimization output of the last train() call (bo_results; None when no BHO ran)
        - Input and output weight matrices (Win, Wout) and reservoir state matrix (W)

    References:
        Based on https://github.com/alberacca/Echo-State-Networks, which implements
        Racca & Magri (2021). Robust optimization and validation of echo state
        networks for learning chaotic dynamics. Neural Networks, 142, 252-268
        (arXiv:2103.03174).
    """

    bias_in = np.array([0.1])
    bias_out = np.array([1.0])  # symmetry breaking

    # Slim copy of the last skopt result; None before training or when BHO is off.
    bo_results: dict | None = None

    # Split summary from the last train() call; source for training_summary().
    split_summary: dict | None = None

    connect = 3  # neuron connectivity
    figs_folder = './figs_ESN/'
    filename = 'my_ESN'

    # Parametric input: (N_param, L) for constant per-segment parameters,
    # or list of (Nt_l, N_param) arrays when the parameter varies within a segment.
    input_parameters: np.ndarray | None = None

    N_folds = 4
    val_fold_step = None
    N_func_evals = 20
    N_grid = 4
    N_initial_rand = 0
    N_units = 100
    N_wash = 50

    max_L_tests = 10
    max_short_tests = 10
    perform_test = True

    observed_idx = None

    # t_train/t_val can be inferred from the data inside train().
    t_val = None
    t_train = None
    t_test = 0.5
    upsample = 5
    Win_type = 'sparse'
    norm_method = 'range'

    # Default hyperparameters and optimization ranges.
    noise = 1e-10
    noise_type = 'gauss'
    hyperparameters_to_optimize = ['rho', 'sigma_in', 'tikh']
    rho = 0.9
    rho_range = (.8, 1.05)
    sigma_in = 10 ** -3
    sigma_in_range = (-5, -1)
    tikh = 1e-12
    tikh_range = [1e-8, 1e-10, 1e-12, 1e-16]
    leak_rate = 1.0
    leak_rate_range = (0.1, 1.0)

    # Dense parameter columns can swamp the reservoir if the raw values are not rescaled.
    optimize_parameter_normalization = False
    param_shift_range = (-1.0, 1.0)
    param_norm_range = (0.1, 10.0)

    def __init__(self, y, dt=1., **kwargs):
        """Initialize the reservoir dimensions and time step; matrices are built at
        training time (see `train` / `_generate_W_Win`), not here.

        Parameters
        ----------
        y : np.ndarray
            Sample physical state used only to infer `N_dim`, shape ``(N_dim, N_samples)``
            (or 1D, treated as ``N_samples=1``).
        dt : float
            Time step of the input data, such that ``dt_ESN = dt * upsample``.
        **kwargs
            Any `EchoStateNetwork` class attribute to override (e.g. ``N_units``,
            ``rho``, ``observed_idx``, ``input_parameters``, ``optimize_parameter_normalization``).

        Raises
        ------
        AssertionError
            If `y` has more than two dimensions.
        """

        if y.ndim == 1:
            y = y[:, np.newaxis]
        elif y.ndim > 2:
            raise AssertionError(f'y.shape={y.shape}. The input y must have 2 dimension')


        #   Initialise state dimensions and reservoir state to zeros ------------ #
        self.N_dim = y.shape[0] # Dimension of the physical system i.e., the output dimension
        self.observed_idx = kwargs.pop('observed_idx', np.arange(self.N_dim)) # Default to full observability

        # Set provided input parameters ------------------------- #
        keys = list(kwargs.keys())
        [setattr(self, key, kwargs.pop(key)) for key in keys if hasattr(EchoStateNetwork, key)]

        # Default to rescaling parametric inputs unless the user explicitly disables it.
        # Dense parameter columns can otherwise dominate the reservoir dynamics.
        if self.input_parameters is not None and 'optimize_parameter_normalization' not in vars(self):
            self.optimize_parameter_normalization = True

        # Define time steps and windows.
        self.dt_ESN = dt * self.upsample

        # Initialize ESN matrices.
        self.val_k = kwargs.get('val_k', 0)
        self.initialised = False

    @property
    def trained(self):
        """Flag to check if the model has been trained"""
        return hasattr(self, '_Win') and hasattr(self, '_Wout') and hasattr(self, '_W')

    @property
    def W(self) -> csr_matrix:
        """The reservoir (recurrent) connectivity matrix, shape ``(N_units, N_units)``,
        stored in CSR format. Rescaled to unit spectral radius when generated (see
        `_generate_W_Win`), so `rho` is the *effective* spectral radius used in `step`.
        """
        return self._W


    @property
    def rng(self):
        if not hasattr(self, '_rng'):
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    @property
    def seed(self):
        if not hasattr(self, '_seed'):
            self._seed = 0
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value
        if hasattr(self, '_rng'):
            del self._rng

    @W.setter
    def W(self, value):
        """
        Setter for the reservoir state matrix (W). Converts the input to CSR format.
        """
        if not isinstance(value, csr_matrix):
            value = csr_matrix(value)

        # Ensure the matrix is square and has the correct dimensions
        assert value.shape == (self.N_units, self.N_units), \
            f'W must be a square matrix of shape ({self.N_units}, {self.N_units}), but got {value.shape}'

        # Set the reservoir state matrix
        self._W = value

    @property
    def Win(self) -> np.ndarray | csr_matrix:
        """The input matrix, shape ``(N_units, N_dim_in + 1)`` (the last column
        multiplies the input bias `bias_in`). Sparse (``Win_type='sparse'``, one
        random connection per neuron to a state or bias column, but densely
        connected to any `input_parameters` columns) or dense
        (``Win_type='dense'``); see `_generate_W_Win`.
        """
        return self._Win

    @Win.setter
    def Win(self, value):
        """
        Setter for the input matrix (Win). Converts the input to CSR format if sparse.
        """

        assert self.Win_type in ['sparse', 'dense'], \
                f"Win type {self.Win_type} not implemented ['sparse', 'dense']"

        if self.Win_type == 'sparse' and not isinstance(value, csr_matrix):
            value = csr_matrix(value)
        elif self.Win_type == 'dense' and hasattr(value, 'toarray'):
            value = value.toarray()


        # Ensure the matrix has the correct dimensions
        assert value.shape ==  (self.N_units, self.N_dim_in+1), \
            f'Win must be a square matrix of shape ({self.N_units}, {self.N_dim_in + 1}), but got {value.shape}'

        # Set the input matrix
        self._Win = value
        self._invalidate_jacobian_cache()


    def _invalidate_jacobian_cache(self):
        self.__dict__.pop('dr_di', None)


    @property
    def Wout(self) -> np.ndarray:
        """The trained (ridge-regression) read-out matrix, shape
        ``(N_units + 1, N_dim)`` -- the last row multiplies the output bias
        `bias_out`. Used in `reservoir_to_physical`.
        """
        return self._Wout

    @Wout.setter
    def Wout(self, value: np.ndarray):
        """
        Setter for the reservoir state matrix (W).
        """
        # Ensure the matrix has the correct dimensions
        assert value.shape == (self.N_units + 1, self.N_dim), \
            f'Wout must be a matrix of shape ({self.N_units + 1}, {self.N_dim}), but got {value.shape}'
        # Set the output matrix
        self._Wout = value

    @property
    def val_k(self):
        """int: Number of Bayesian-optimization validation evaluations performed so
        far (reset to 0 at the start of each `train` call). Defaults to 0.
        """
        if not hasattr(self, '_val_k'):
            return 0
        return self._val_k

    @val_k.setter
    def val_k(self, value):
        """
        Setter for the validation counter.
        """
        if not isinstance(value, int):
            raise TypeError('val_k must be an integer')
        self._val_k = value

    @property
    def dt_physical(self):
        """float: Time step of the underlying physical data, ``dt_ESN / upsample``."""
        return self.dt_ESN / self.upsample

    @property
    def N_train(self):
        """int: Number of training steps, ``round(t_train / dt_ESN)``.

        Raises ValueError if `t_train` is not yet set (when omitted it is
        inferred from the data at `train` time -- see `_split_and_format_data`).
        """
        if self.t_train is None:
            raise ValueError("t_train is not set; it is inferred from the data at train() "
                             "time when omitted (see _split_and_format_data).")
        return int(round(self.t_train / self.dt_ESN))

    @property
    def N_val(self):
        """int: Number of validation steps, ``round(t_val / dt_ESN)``.

        Raises ValueError if `t_val` is not yet set (when omitted it is
        inferred from the data at `train` time -- see `_split_and_format_data`).
        """
        if self.t_val is None:
            raise ValueError("t_val is not set; it is inferred from the data at train() "
                             "time when omitted (see _split_and_format_data).")
        return int(round(self.t_val / self.dt_ESN))

    @property
    def N_test(self):
        """int: Number of test steps, ``round(t_test / dt_ESN)``."""
        return int(round(self.t_test / self.dt_ESN))

    @property
    def WCout(self):
        r"""Least-squares fit of `W` from `Wout`'s state block,
        $\arg\min_{\mathbf{X}} \lVert \mathbf{W}_\mathrm{out}[:N_\mathrm{units}]\,\mathbf{X}
        - \mathbf{W} \rVert_F$, shape ``(N_dim, N_units)``. Sketched (commented-out) in
        `Jacobian` as a building block for a closed-loop Jacobian; not currently used
        anywhere, since the closed-loop Jacobian is unimplemented.
        """
        # if not hasattr(self, '_WCout'):
        #     return None
        return self._WCout

    @WCout.setter
    def WCout(self, value=None):
        """
        Setter for the closed-loop reservoir weight matrix (W Cout).
        """
        assert self.trained, 'ESN must be trained with washout before calling step method. Call ESN.train() first.'
        if value is None:
            self._WCout = np.linalg.lstsq(self.Wout[:-1], self.W.toarray(), rcond=None)[0]
        else:
            self._WCout = value


    @property
    def sparsity(self):
        r"""float: Fraction of possible reservoir connections that are zero,
        $1 - \texttt{connect}/(N_\mathrm{units}-1)$ (each neuron connects, on
        average, to `connect` others out of the $N_\mathrm{units}-1$ possible).
        """
        return 1. - self.connect / (self.N_units - 1)


    @staticmethod
    def _n_param(input_parameters):
        """Number of parameter columns, whether input_parameters is a constant-per-segment
        ndarray (N_param, L) or a list of L per-timestep (Nt_l, N_param) arrays."""
        return input_parameters[0].shape[-1] if isinstance(input_parameters, list) else input_parameters.shape[0]

    @staticmethod
    def _param_range(input_parameters):
        """(min, max) of input_parameters, whether ndarray or list-of-arrays (see _n_param)."""
        if isinstance(input_parameters, list):
            return (min(np.asarray(p).min() for p in input_parameters),
                    max(np.asarray(p).max() for p in input_parameters))
        return float(input_parameters.min()), float(input_parameters.max())

    @property
    def N_dim_in(self):
        """int: Number of ESN input dimensions -- the observed state components
        (``len(observed_idx)``) plus, for a parametric ESN, the parameter count of `input_parameters`
        (rows of the ``(N_param, L)`` array, or columns of each per-timestep
        ``(Nt_l, N_param)`` segment).
        """
        if self.input_parameters is None:
            return len(self.observed_idx)
        return len(self.observed_idx) + self._n_param(self.input_parameters)


    @property
    def norm(self):
        """np.ndarray: Per-component scale factor used by `normalize_input`, shape
        ``(N_dim_in,)``. Defaults to ones (no scaling) until set by
        `_split_and_format_data`/`_set_norm`.
        """
        if not hasattr(self, '_norm'):
            return np.ones((self.N_dim_in,))
        return self._norm

    @norm.setter
    def norm(self, value):
        """
        Setter for the normalization factor. Ensures it is N_dim_in.
        """
        if hasattr(self, 'N_dim_in'):
            assert value.size == self.N_dim_in, \
                f'Normalization factor must be dimension Ndim={self.N_dim_in}, got {value.shape}'
        self._norm = value.flatten()


    @cached_property
    def dr_di(self) -> csr_matrix | np.ndarray:
        r"""Linear (pre-activation) part of the input-to-reservoir Jacobian,
        $\sigma_\mathrm{in}\,\mathbf{W}_\mathrm{in,1}\,\mathrm{diag}(1/\texttt{norm})$,
        shape ``(N_units, N_dim_in)``, where $\mathbf{W}_\mathrm{in,1}$ is `Win` with
        the bias column dropped. This is *not* the full $\partial\mathbf{r}/\partial\mathbf{u}$:
        `Jacobian` additionally applies the $\mathrm{diag}(1-\mathbf{r}^2)$ factor from
        differentiating $\tanh$. Cached via `functools.cached_property` and
        invalidated whenever `Win` is reassigned.
        """
        norm = self.norm.copy()

        Win_1 = self.Win[:, :self.N_dim_in]  # type: Union[csr_matrix, np.ndarray]
        g = self.sigma_in * 1.0 / norm

        if issparse(Win_1):
            # .multiply returns a COO matrix: convert back to CSR for efficient products
            return csr_matrix(Win_1.multiply(g[np.newaxis, :]))
        else:
            return Win_1 * g[np.newaxis, :]


    @property
    def shift(self):
        """np.ndarray: Per-component offset used by `normalize_input`, shape
        ``(N_dim_in,)``. Defaults to zeros (no shift) until set by
        `_split_and_format_data`/`_set_norm`.
        """
        if not hasattr(self, '_shift'):
            return np.zeros((self.N_dim_in,))
        return self._shift


    @shift.setter
    def shift(self, value):
        """
        Setter for the shift factor. Ensures it is N_dim_in (nb. after initialization).
        """

        if hasattr(self, 'N_dim_in'):
            assert value.size == self.N_dim_in, \
                f'Shift factor must be dimension Ndim={self.N_dim_in}, got {value.shape}'
        self._shift = value.flatten()

    # _______________________________________________________________________________________________________ STEP & JACOBIAN
    def step(self, u, r):
        r"""Advance the reservoir by one open-loop time step,
        $\mathbf{r}_{n+1} = (1-\alpha)\,\mathbf{r}_n +
        \alpha\tanh(\sigma_\mathrm{in}\mathbf{W}_\mathrm{in}[\mathbf{u}_n; b_\mathrm{in}]
        + \rho\mathbf{W}\mathbf{r}_n)$ with $\alpha$ = `leak_rate` (the default
        $\alpha=1$ is the plain tanh update, no leak), and read out the corresponding
        physical state (see the class docstring for the full formulation).

        Parameters
        ----------
        u : np.ndarray
            Input state at the current time step, shape ``(N_dim_in, N_ens)``.
        r : np.ndarray
            Reservoir state at the current time step, shape ``(N_units, N_ens)``.

        Returns
        -------
        u_out : np.ndarray
            Physical output at the next time step, shape ``(N_dim, N_ens)``.
        r_out : np.ndarray
            Updated reservoir state, shape ``(N_units, N_ens)``.
        """
        # Normalise input data and augment with input bias (ESN symmetry parameter)

        # assert self.trained, 'ESN must be trained with washout before calling step method. Call ESN.train() first.'

        if u.ndim == 1:
            u = np.expand_dims(u, axis=-1)
        elif u.ndim == 3:
            assert u.shape[0] == 1, f'Input u has shape {u.shape}, only 1 sample at a time is allowed'
            u = u[0]
        if r.ndim == 1:
            r = np.expand_dims(r, axis=-1)
        elif r.ndim == 3:
            assert r.shape[0] == 1, f'Input r has shape {r.shape}, only 1 sample at a time is allowed'
            r = r[0]

        # Normalize input
        u_norm = self.normalize_input(u)

        # Augment input with bias
        bias_in = self.bias_in * np.ones((1, u.shape[-1]))
        u_aug = np.concatenate((u_norm, bias_in))

        # Forecast the reservoir state (leaky-integrator; leak_rate=1 -> plain tanh)
        x_tanh = np.tanh(self.sigma_in * self.Win.dot(u_aug) + self.rho * self.W.dot(r))
        r_out = x_tanh if self.leak_rate == 1.0 else \
            (1.0 - self.leak_rate) * r + self.leak_rate * x_tanh

        # compute output from ESN if not during training
        u_out = self.reservoir_to_physical(r_out)
        return u_out, r_out



    def reservoir_to_physical(self, r):
        """Convert the reservoir state to the physical state via the output matrix.

        Parameters
        ----------
        r : np.ndarray
            Reservoir state, shape ``(N_units, N_ens)`` (the output bias row is
            appended internally).

        Returns
        -------
        np.ndarray
            Physical state, shape ``(N_dim, N_ens)``.
        """

        # output bias added
        bias_out = self.bias_out * np.ones((1, r.shape[-1]))
        r_aug = np.concatenate((r, bias_out))

        return np.dot(self.Wout.T, r_aug)

    def normalize_input(self, data):
        r"""Shift-and-scale the input, $(\mathbf{u} - \texttt{shift}) / \texttt{norm}$
        (see `shift`, `norm` and `_set_norm`), before it is fed to `Win`.

        Parameters
        ----------
        data : np.ndarray
            Input data to be normalized, shape ``(N_dim_in, N_ens)``.

        Returns
        -------
        np.ndarray
            Normalized input data, same shape as `data`.
        """
        return (data - self.shift[:, np.newaxis]) / self.norm[:, np.newaxis]


    def outputs_to_inputs(self, full_state):
        """Map a full physical state (e.g. a closed-loop prediction) back to the
        ESN's input space: selects the observed components (`observed_idx`) and, for
        a parametric ESN, appends `input_parameters`.

        Parameters
        ----------
        full_state : np.ndarray
            Full physical state vector, shape ``(N_dim, N_ens)``.

        Returns
        -------
        np.ndarray
            Input state vector, shape ``(N_dim_in, N_ens)``.
        """
        assert full_state.shape[0] == self.N_dim, f'full_state has shape {full_state.shape}, expected first dim to be {self.N_dim}'

        observed_state = full_state[self.observed_idx]

        assert observed_state.shape[0] == len(self.observed_idx), f'observed_state has shape {observed_state.shape}, expected first dim to be {len(self.observed_idx)}'

        if self.input_parameters is None:
            return observed_state
        else:
            return np.concatenate([observed_state, self.input_parameters], axis=0)


    def Jacobian(self, u_in, r_in, open_loop_J=True):
        r"""Analytical Jacobian of the one-step map, $\mathbf{J} = \partial\mathbf{u}_{n+1}/\partial\mathbf{u}_n$,
        obtained by differentiating `step` through $\tanh$:

        $$
        \mathbf{J} = \mathbf{W}_\mathrm{out,1}^\mathrm{T}\,
        \mathrm{diag}\!\left(1-\mathbf{r}_{n+1}^{\,2}\right)\,
        \sigma_\mathrm{in}\,\mathbf{W}_\mathrm{in,1}\,\mathrm{diag}(1/\texttt{norm}),
        $$

        where $\mathbf{W}_\mathrm{out,1}$ and $\mathbf{W}_\mathrm{in,1}$ are `Wout`/`Win`
        with the bias row/column dropped, and $\mathbf{r}_{n+1}$ is the reservoir state
        obtained by stepping from (`u_in`, `r_in`). The $1/\texttt{norm}$ factor comes
        from the chain rule through `normalize_input`. With a leaky reservoir
        (`leak_rate` $\alpha<1$) the middle factor becomes
        $\alpha\,\mathrm{diag}(1-\tilde{\mathbf{x}}^{\,2})$ with $\tilde{\mathbf{x}}$
        the tanh pre-leak value, recovered from the step as
        $(\mathbf{r}_{n+1} - (1-\alpha)\mathbf{r}_n)/\alpha$.

        Parameters
        ----------
        u_in : np.ndarray
            Input state, shape ``(N_dim_in, N_ens)``.
        r_in : np.ndarray
            Reservoir state, shape ``(N_units, N_ens)``.
        open_loop_J : bool
            If True (default), compute the open-loop Jacobian above. The closed-loop
            variant (linearizing through the feedback of `u_out` back into the next
            input) is not implemented -- see Raises.

        Returns
        -------
        np.ndarray
            Jacobian $\partial\mathbf{u}_\mathrm{out}/\partial\mathbf{u}_\mathrm{in}$,
            shape ``(N_dim, N_dim_in)`` if ``N_ens == 1``, else ``(N_dim, N_dim_in, N_ens)``.

        Raises
        ------
        NotImplementedError
            If `open_loop_J` is False (the closed-loop Jacobian is unimplemented; a
            numerical check of the sketched derivation did not pass).
        """
        assert self.trained, 'ESN must be trained before computing the Jacobian. Call ESN.train() first.'


        Wout_1 = self.Wout[:self.N_units, :].T

        # # Option(i) rin function of bin:
        rout = self.step(u_in, r_in)[1]

        if self.leak_rate == 1.0:
            tt = 1. - rout ** 2
        else:
            # d(r_out)/d(pre-activation) = leak_rate * (1 - x_tanh^2), with the
            # tanh pre-leak value recovered from the leaky update
            r_prev = r_in[:, np.newaxis] if r_in.ndim == 1 else \
                (r_in[0] if r_in.ndim == 3 else r_in)
            x_tanh = (rout - (1. - self.leak_rate) * r_prev) / self.leak_rate
            tt = self.leak_rate * (1. - x_tanh ** 2)
        dr_di = self.dr_di
        if not open_loop_J:
            # u_aug = np.concatenate((u_in / self.norm, self.bias_in))
            # rout = np.tanh(self.sigma_in * self.Win.dot(u_aug) + self.rho * np.dot(self.WCout.T, u_in))
            # dr_di = self.sigma_in * Win_1 / self.norm + self.rho * self.WCout.T
            #  Win_G += dr_di ......
            raise NotImplementedError('Numerical test of closed-loop Jacobian did not pass')

        N_ens = tt.shape[-1]
        if N_ens == 1:
            if issparse(dr_di):
                RHS = dr_di.T.multiply(tt[:, 0][np.newaxis, :])
            else:
                RHS = dr_di.T * tt[:, 0][np.newaxis, :]
            return RHS.dot(Wout_1.T).T

        J = np.zeros((self.N_dim, self.N_dim_in, N_ens))
        for ens_i in range(N_ens):
            if issparse(dr_di):
                RHS = dr_di.T.multiply(tt[:, ens_i][np.newaxis, :])
            else:
                RHS = dr_di.T * tt[:, ens_i][np.newaxis, :]
            J[:, :, ens_i] = RHS.dot(Wout_1.T).T

        return J




    # _______________________________________________________________________________________ TRAIN & VALIDATE THE ESN
    def train(self,
              train_data,
              add_noise=True,
              plot_training=True,
              save_ESN_training=False,
              folder=None,
              validation_strategy=None,
              seed=None,
              n_seeds=1,
              **kwargs
              ):
        """Train the ESN: format the data into washout/train/validation/test sets,
        (re)generate `Win`/`W` if not already set, select hyperparameters via
        Bayesian optimization (unless `hyperparameters_to_optimize` is empty), and
        fit `Wout` by ridge regression on the resulting hyperparameters.

        Parameters
        ----------
        train_data : np.ndarray
            Training data, shape ``(L, Nt, N_dim)``, or a ragged list of L
            segments of shape ``(Nt_l, N_dim)``.
        add_noise : bool
            If True, add Gaussian noise (scaled by `noise`) to the training input.
        plot_training : bool
            If True, visualize the training process (BO convergence, `Wout`, and
            post-training test forecasts).
        save_ESN_training : bool
            If True, save the training plots to a PDF (in `folder`).
        folder : str, optional
            Directory to save training plots to. Defaults to `figs_folder`.
        validation_strategy : callable, optional
            Validation function for hyperparameter tuning. Defaults to `_RVC_Noise`.
        seed : int, optional
            Random seed for generating `Win`/`W` if they don't already exist.
            Defaults to `seed`/`rng`.
        n_seeds : int
            If > 1, train this many reservoir realizations (seeds ``base, base+1,
            ...`` with ``base = seed or self.seed``) on the same data and settings,
            in parallel processes, and keep the one with the best validation score;
            per-seed scores are stored in `seed_scores` for statistical comparison.
            Training plots are skipped in this mode.
        **kwargs
            Any existing attribute to override before training (e.g. `N_units`).

        Returns
        -------
        None
            Sets `Wout` (and, if not already present, `Win`/`W`) in place. Also sets
            `bo_results`: a dict with keys ``'func_vals'`` (the per-evaluation
            validation-loss trace), ``'x_iters'`` (evaluated points), ``'x'``/``'fun'``
            (the selected point and its loss), ``'hp_names'``, and ``'n_grid_points'``
            — or None when `hyperparameters_to_optimize` is empty (no BHO ran). This
            is a slimmed copy of the `skopt` result: the raw ``OptimizeResult``
            retains the training corpus and the fitted GP models, which would bloat
            every pickle/deepcopy of a trained ESN.
        """
        if n_seeds > 1:
            return self._train_multi_seed(train_data, n_seeds, add_noise=add_noise,
                                          validation_strategy=validation_strategy,
                                          seed=seed, **kwargs)

        if self.trained:
            print("ESN is already trained. Skipping training.")
            pass #  skip training

        for key, val in kwargs.items():
            if hasattr(self, key):
                print(f'Modifying {key} = {getattr(self, key)} -> {val} at training.')
                setattr(self, key, val)

        # ========================== STEP 1: DATA FORMATTING ==========================
        # Format data into washout, train/validation, and test sets
        U_wtv, Y_wtv, U_test, Y_test = self._split_and_format_data(train_data, add_noise=add_noise)

        # print([xx.shape for xx in [U_wtv, Y_wtv, U_test, Y_test]])

        # Ensure W and Win matrices are initialized
        if not hasattr(self, '_W') or not hasattr(self, '_Win'):
            self._generate_W_Win(seed=seed)

        self.Wout = np.zeros((self.N_units + 1, self.N_dim))  # Initialize Wout with zeros

        # Validation/test runs temporarily overwrite self.input_parameters
        original_input_parameters = self.input_parameters
        try:
            # =================== STEP 2: BAYESIAN HYPERPARAMETER OPTIMIZATION ==============
            self.val_k = 0  # Reset validation counter at the start of training
            # Perform hyperparameter optimization if required
            if self.hyperparameters_to_optimize:
                bo_results = self._optimize_hyperparameters(U_wtv, Y_wtv,
                                                           validation_strategy,
                                                           print_convergence=plot_training)
            else:
                bo_results = None
            # Expose the BHO output for post-training inspection
            if bo_results is None:
                self.bo_results = None
            else:
                res = bo_results['res']
                self.bo_results = dict(
                    func_vals=np.asarray(res.func_vals), x_iters=list(res.x_iters),
                    x=res.x, fun=res.fun,
                    hp_names=bo_results['hp_names'],
                    n_grid_points=bo_results['n_grid_points'])
            # ====================== STEP 3: RIDGE REGRESSION TRAINING =====================
            # Compute the output weight matrix Wout
            self.Wout = self._solve_ridge_regression(U_wtv, Y_wtv)

            print(self.training_summary())

            # ========================== STEP 4: TEST AND PLOTTING ======================
            if plot_training:
                self._plot_training_results(U_test, Y_test, bo_results, save_ESN_training, folder)
        finally:
            self.input_parameters = original_input_parameters

    def _train_multi_seed(self, train_data, n_seeds, add_noise, validation_strategy,
                          seed, **kwargs):
        """Backend of ``train(n_seeds > 1)``: train `n_seeds` reservoir realizations
        of this ESN on the same data and settings, adopt the realization with the
        best validation score in place, and store the per-seed scores in
        `seed_scores` (``{seed: score}``) for statistical comparison. Runs one
        process per seed when the ESN and validation strategy pickle (a closure
        strategy doesn't -- falls back to a serial loop with a note)."""
        base = self.seed if seed is None else seed
        seeds = [base + i for i in range(n_seeds)]
        try:
            pickle.dumps((self, validation_strategy))
            parallel = True
        except Exception:
            print('n_seeds: ESN or validation_strategy is not picklable; '
                  'training the seeds serially instead of in parallel.')
            parallel = False
        if parallel:
            # ponytail: no BLAS-thread throttling in workers; export OMP_NUM_THREADS
            # if n_seeds x BLAS threads oversubscribes the machine.
            with ProcessPoolExecutor(max_workers=min(n_seeds, os.cpu_count() or 1)) as ex:
                results = list(ex.map(_train_one_seed, [self] * n_seeds, seeds,
                                      [train_data] * n_seeds, [add_noise] * n_seeds,
                                      [validation_strategy] * n_seeds,
                                      [kwargs] * n_seeds))
        else:
            results = [_train_one_seed(self.copy(), s, train_data, add_noise,
                                       validation_strategy, kwargs) for s in seeds]
        for _, _, log in results:
            print(log, end='')
        scores = np.array([score for _, score, _ in results])
        best = int(np.nanargmin(scores))
        self.__dict__.clear()
        self.__dict__.update(results[best][0].__dict__)
        self.seed_scores = dict(zip(seeds, scores))
        print(f'n_seeds={n_seeds}: validation score {scores.mean():.4f} +/- '
              f'{scores.std():.4f} (best {scores[best]:.4f} @ seed {seeds[best]}, '
              f'worst {scores.max():.4f}) -> kept seed {seeds[best]}')

    def training_summary(self) -> str:
        """One-line summary of the last `train` call: the data split (from
        `_split_and_format_data`, stored in `split_summary`), the resulting
        train/validation windows, and the selected hyperparameters (with the number
        of Bayesian-optimization evaluations when a search ran)."""
        s = self.split_summary
        if s is None:
            raise RuntimeError('no training summary yet: call train() first.')
        if s['ragged']:
            data = (f"{s['segments_train']}/{s['segments_in']} segments -> {s['pairs']} pairs, "
                    f"{s['segments_test']} held out")
            if s['segments_dropped']:
                data += f", {s['segments_dropped']} dropped (< N_wash+2)"
            if s['t_train_given'] is not None:
                data += " (given t_train ignored: a segmented corpus is never capped)"
        else:
            data = f"{self.N_train}+{self.N_val} train+val steps, {s['n_test']} test"
        hp_names = ('rho', 'sigma_in', 'tikh') + \
            (('leak_rate',) if self.leak_rate != 1.0 else ())
        hps = ', '.join(f'{name}={self._get_hyperparam(name):.3g}' for name in hp_names)
        # realized fold count of the last validation run (strategies may cap or
        # multiply the requested N_folds -- see val_fold_step), so an N_folds
        # override is visible here rather than silently absorbed
        folds = getattr(self, 'n_folds_realized', None)
        if folds is not None:
            hps += f" | {folds} validation fold{'s' if folds != 1 else ''}"
        bo = (f" | BHO: {len(self.bo_results['func_vals'])} evals over "
              f"{self.bo_results['hp_names']}" if self.bo_results is not None else '')
        return (f"trained: {data} | t_train={self.t_train:.3g}, t_val={self.t_val:.3g}"
                + (' (inferred)' if s['t_val_inferred'] else '') + f" | {hps}{bo}")


    def copy(self):
        """Return a deep copy of this `EchoStateNetwork`.

        Returns
        -------
        EchoStateNetwork
            A new, independent instance with the same state and matrices.
        """
        return deepcopy(self)

    # _______________________________________________________________________________________ HELPER METHODS FOR ESN INITIALIZATION & TRAINING



    def _generate_W_Win(self, seed=None):
        """Generate `Win` (random, sparse or dense) and `W` (Erdős–Rényi, rescaled to
        unit spectral radius), if not already set.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Defaults to `seed`/`rng`.

        Raises
        ------
        ValueError
            If `Win_type` is not ``'sparse'`` or ``'dense'``.

        Returns
        -------
        None
            Sets `Win` and `W` in place.
        """
        if seed is None:
            rng0 = self.rng
        else:
            rng0 = np.random.default_rng(seed)

        # Input matrix: Sparse random matrix where only one element per row is different from zero.
        # Parameter columns (if any) are the exception: every neuron connects densely to them,
        # since they carry a single global forcing rather than a per-neuron-selected observation.
        if not hasattr(self, '_Win'):
            Win = lil_matrix((self.N_units,
                              self.N_dim_in + 1))  # +1 accounts for input bias
            if self.Win_type == 'sparse':
                N_param = self._n_param(self.input_parameters) if self.input_parameters is not None else 0
                N_state = self.N_dim_in - N_param
                # columns eligible for the single sparse connection: state columns + bias (last column)
                sparse_cols = np.append(np.arange(N_state), self.N_dim_in)
                for j in range(self.N_units):
                    Win[j, rng0.choice(sparse_cols)] = rng0.uniform(low=-1, high=1)
                if N_param > 0:
                    Win[:, N_state:self.N_dim_in] = rng0.uniform(
                        low=-1, high=1, size=(self.N_units, N_param))
            elif self.Win_type == 'dense':
                for j in range(self.N_units):
                    Win[j, :] = rng0.uniform(low=-1, high=1, size=self.N_dim_in + 1)
            else:
                raise ValueError(f"Win type {self.Win_type} not implemented ['sparse', 'dense']")
            # Store
            self.Win = Win

        # Reservoir state matrix: Erdos-Renyi network
        if not hasattr(self, '_W'):
            W = csr_matrix(rng0.uniform(low=-1, high=1, size=(self.N_units, self.N_units)) *
                        (rng0.random(size=(self.N_units, self.N_units)) < (1 - self.sparsity)))
            # scale W by the spectral radius to have unitary spectral radius
            try:
                spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
            except ArpackNoConvergence:
                # seed-dependent ARPACK stagnation on small reservoirs; dense
                # eigenvalues are exact and cheap at typical N_units
                spectral_radius = np.abs(np.linalg.eigvals(W.toarray())).max()
            self.W = (1. / spectral_radius) * W


    def _compute_RR_terms(self, U_wtv, Y_wtv):
        """
        Computes the Ridge Regression (RR) terms, including left-hand side (LHS) and right-hand side (RHS)
        matrices, for training the output weights.

        Parameters
        ----------
        U_wtv : np.ndarray (L x Nt x N_dim)
            Wash-train-validation input data. L segments, each with Nt time steps and N_dim dimensions.
        Y_wtv : np.ndarray (L x Nt x N_dim)
            Corresponding output labels for input data.

        Returns
        -------
            tuple:
                - LHS (np.ndarray): Left-hand side matrix for ridge regression.
                - RHS (np.ndarray): Right-hand side matrix for ridge regression.
                - U_RR (list): List of input states split by L-segments.
                - R_RR (list): List of reservoir states split by L-segments.
        """

        LHS = np.zeros((self.N_units + 1, self.N_units + 1))
        RHS = np.zeros((self.N_units + 1, self.N_dim))
        R_RR = [None] * len(U_wtv)
        U_RR = [None] * len(U_wtv)


        for ll in range(len(U_wtv)):

            U_wash_l = U_wtv[ll][:self.N_wash]
            # Y_wash_l = Y_wtv[ll][:self.N_wash]
            Uin_l = U_wtv[ll][self.N_wash:]
            Yout_l = Y_wtv[ll][self.N_wash:]

            assert Uin_l.shape[0] == Yout_l.shape[0], \
                f'Inconsistent shapes for training data at segment {ll}: {Uin_l.shape} vs {Yout_l.shape}'

            assert Uin_l.shape[0] > 0, \
                f'Not enough data for training at segment {ll}: {Uin_l.shape}'

            # Washout phase to initialize reservoir state
            N_ens = U_wash_l.shape[-1] if U_wash_l.ndim == 3 else 1
            r = np.zeros((self.N_units, N_ens))
            for u_in in U_wash_l:
                _, r = self.step(u_in, r)

            if Yout_l.ndim == 3:
                assert Yout_l.shape[-1] == 1, f'Yout_l has shape {Yout_l.shape}, only 1 sample at a time is allowed'
                Yout_l = Yout_l[..., 0]

            # Open-loop train phase: one pass over the whole segment, states filled
            # in place. The Gram cost is invariant to batching, so chunking bought
            # no speed, and its per-chunk appends only added copies and peak RAM.
            # ponytail: whole-segment buffers; stream fixed-size chunks (and make
            # the state return opt-in) if RAM on very long single series ever matters.
            r_out = r
            r_open = np.zeros((Uin_l.shape[0], self.N_units, N_ens))
            y_open = np.zeros((Uin_l.shape[0], self.N_dim, N_ens))
            for ii, u_in in enumerate(Uin_l):
                u_out, r_out = self.step(u_in, r_out)
                y_open[ii], r_open[ii] = u_out, r_out

            if y_open.ndim > 2:
                y_open, r_open = y_open.squeeze(axis=-1), r_open.squeeze(axis=-1)

            R_RR[ll] = r_open
            U_RR[ll] = y_open

            # Compute matrices for linear regression system
            bias_out = np.ones([r_open.shape[0], 1]) * self.bias_out
            r_aug = np.hstack((r_open, bias_out))

            LHS += np.dot(r_aug.T, r_aug)
            RHS += np.dot(r_aug.T, Yout_l)

        return LHS, RHS, U_RR, R_RR


    def _solve_ridge_regression(self, U_wtv, Y_wtv):
        """
        Solves the ridge regression problem to compute the output weight matrix (Wout).

        Parameters
        ----------
        U_wtv : np.ndarray (L x Nt x N_dim)
            Input data for ridge regression (train/valiladion). L segments, each with Nt time steps and N_dim dimensions.
        Y_wtv : np.ndarray (L x Nt x N_dim)
            Target labels for ridge regression. L segments, each with Nt time steps and N_dim dimensions.

        Returns
        -------
            np.ndarray: Computed output weight matrix (Wout).
        """
        LHS, RHS = self._compute_RR_terms(U_wtv, Y_wtv)[:2]
        LHS.ravel()[::LHS.shape[1] + 1] += self.tikh  # Add tikhonov to the diagonal
        return np.linalg.solve(LHS, RHS)  # Solve linear regression problem


    def _UY_from_raw_data(self, data, add_noise=True, seed=None):
        """
        Extracts input (U) and output (Y) matrices from raw data.

        Args:
            data (np.ndarray or list[np.ndarray]): Raw time series data, either a
                regular [(L) x Nt x N_dim] array, or a list of L segments of shape
                (Nt_l, N_dim) with possibly different Nt_l (e.g. variable-length
                cluster-dwell chunks) -- see _UY_from_ragged_data.

        Returns:
            tuple: (U, Y). Regular (L, Nt, N_dim_in/N_dim) arrays for ndarray input;
            lists of L arrays of shape (Nt_l, N_dim_in/N_dim) for ragged input.
        """
        if isinstance(data, (list, tuple)):
            return self._UY_from_ragged_data(data, add_noise=add_noise, seed=seed)

        #   APPLY UPSAMPLE AND OBSERVED INDICES ________________________
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        # Set labels always as the full state
        Y = data[:, ::self.upsample].copy()
        # Inputs are the observed components of the state, which can be a subset of the full state
        U = Y[:, :, self.observed_idx].copy()

        assert Y.shape[-1] >= U.shape[-1]

        if self.input_parameters is not None:
            # Broadcast the per-segment parameter (e.g. cluster id) across all Nt steps
            L, Nt = U.shape[0], U.shape[1]
            N_param = self.input_parameters.shape[0]
            assert self.input_parameters.shape == (N_param, L), \
                f'input_parameters must have shape (N_param, L)=({N_param}, {L}), got {self.input_parameters.shape}'
            params_tiled = np.broadcast_to(self.input_parameters.T[:, None, :], (L, Nt, N_param))
            U = np.concatenate([U, params_tiled], axis=-1)

        assert U.shape[-1] == self.N_dim_in

        if add_noise:
            #  ==================== ADD NOISE TO TRAINING INPUT ====================== ##
            # Add noise to the inputs if distinction inputs/labels is not given.
            # Larger noise promotes stability in long term, but hinders time accuracy
            U_std = np.std(U, axis=1, keepdims=True)
            if seed is None:
                rng0 = self.rng
            else:
                rng0 = np.random.default_rng(seed)
            U += rng0.normal(loc=0, scale=self.noise * U_std, size=U.shape)

        return U, Y

    def _UY_from_ragged_data(self, segments, add_noise=True, seed=None):
        """Per-segment version of _UY_from_raw_data for a list of L segments of shape
        (Nt_l, N_dim), Nt_l possibly different per segment (e.g. cluster-dwell chunks
        of different lengths -- no ensemble axis is supported in this path).
        """
        L = len(segments)
        per_step_params = isinstance(self.input_parameters, list)
        if self.input_parameters is not None and not per_step_params:
            N_param = self.input_parameters.shape[0]
            assert self.input_parameters.shape == (N_param, L), \
                f'input_parameters must have shape (N_param, L)=({N_param}, {L}), got {self.input_parameters.shape}'
        elif per_step_params:
            assert len(self.input_parameters) == L, \
                f'input_parameters list must have length L={L}, got {len(self.input_parameters)}'

        rng0 = self.rng if seed is None else np.random.default_rng(seed)
        U, Y = [], []
        for l, seg in enumerate(segments):
            y_l = np.asarray(seg)[::self.upsample]
            u_l = y_l[:, self.observed_idx]
            assert y_l.shape[-1] >= u_l.shape[-1]

            if per_step_params:
                # Explicit per-timestep parameter (e.g. cluster id varying within a
                # segment that spans a transition), already upsample-aligned by the caller.
                params_l = np.asarray(self.input_parameters[l])[::self.upsample]
                assert params_l.shape[0] == y_l.shape[0], \
                    f'input_parameters[{l}] must have {y_l.shape[0]} steps (post-upsample), got {params_l.shape[0]}'
                u_l = np.concatenate([u_l, params_l], axis=-1)
            elif self.input_parameters is not None:
                N_param = self.input_parameters.shape[0]
                params_l = np.broadcast_to(self.input_parameters[:, l], (y_l.shape[0], N_param))
                u_l = np.concatenate([u_l, params_l], axis=-1)
            assert u_l.shape[-1] == self.N_dim_in

            if add_noise:
                u_std = np.std(u_l, axis=0, keepdims=True)
                u_l = u_l + rng0.normal(loc=0, scale=self.noise * u_std, size=u_l.shape)

            U.append(u_l)
            Y.append(y_l)
        return U, Y

    def _split_and_format_data(self, data=None, add_noise=True):
        """Format raw data into washout/train/validation and test sets, adding noise
        and computing/storing `norm`/`shift` (and, for a parametric ESN, tuning the
        parameter normalization -- see the class docstring).

        Parameters
        ----------
        data : np.ndarray
            Input time series data, shape ``(L, Nt, N_dim)`` (or ``(Nt, N_dim)``), or a
            ragged list of L segments of shape ``(Nt_l, N_dim)`` -- see
            `_UY_from_ragged_data` for the segmented-corpus rules.
        add_noise : bool
            Whether to add noise to the training input data. Default True.

        Returns
        -------
        U_wtv : np.ndarray
            Wash-train-validation input data.
        Y_wtv : np.ndarray
            Corresponding labels for train/validation data.
        U_test : np.ndarray
            Test input data.
        Y_test : np.ndarray
            Test labels.

        Raises
        ------
        ValueError
            If `data` is None, or shorter than `N_train + N_val` steps.
        """
        if data is None:
            raise ValueError('No training data provided to format_training_data method.')
        is_ragged = isinstance(data, (list, tuple))
        if not is_ragged and data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        U, Y = self._UY_from_raw_data(data, add_noise=add_noise) # dimensions: L x Nt x N_dim_in/N_dim

        #   SEPARATE INTO WASH/TRAIN/VAL/TEST SETS ______________________
        if is_ragged:
            # Segments may have very different lengths (e.g. cluster-dwell chunks).
            # Two rules decide what enters training:
            #   1. a segment must yield at least one teacher-forced pair past its own
            #      washout (N_wash + 2 raw points). Validation length does NOT gate
            #      training inclusion -- the probing strategies use whatever tail a
            #      segment has (_SegmentRVC_Noise), so tying the drop rule to N_val
            #      (as before) silently wasted training data whenever t_val was long.
            #   2. t_train does NOT cap a segmented corpus. A caller who cut a
            #      trajectory into segments already chose how much data to train on;
            #      t_train is a *window length* for one long trajectory and has no
            #      sensible reading across many short ones -- read as a total pair
            #      budget it silently shrank every segment to a couple of steps
            #      (a per-dwell-sized t_train over hundreds of dwells -> ~2% of each,
            #      i.e. an ESN trained on almost nothing). So every usable segment is
            #      kept in full and t_train is written back from what was kept.
            # U_test/Y_test just reuse the kept segments -- run_test is a visual
            # diagnostic, not the metric BHO optimizes, so an in-sample check is an
            # acceptable trade for not fragmenting already-scarce data further.
            min_len = self.N_wash + 2
            usable = [(U_l, Y_l) for U_l, Y_l in zip(U, Y) if U_l.shape[0] >= min_len]
            if not usable:
                raise ValueError(f'No segment has >= N_wash+2={min_len} steps; reduce N_wash.')

            t_val_inferred = self.t_val is None
            if self.t_val is None:
                # infer the validation window from the dwell (segment) lengths: the
                # median usable segment's closed-loop tail past its own washout
                median_len = int(np.median([U_l.shape[0] for U_l, _ in usable]))
                n_val = max(1, median_len - self.N_wash - 1)
                self.t_val = n_val * self.dt_ESN
                if n_val < 10:
                    warnings.warn(
                        f'inferred validation window is degenerate: N_val={n_val} '
                        f'step(s), because the median segment ({median_len} steps) '
                        f'barely exceeds the washout (N_wash={self.N_wash}). '
                        f'Closed-loop validation over so few steps is meaningless; '
                        f'provide longer segments or set t_val explicitly.',
                        stacklevel=2)

            # Use ALL the input data -- the first 80% of the segments (they arrive in
            # time order) train/validate, the last 20% are held out as the run_test
            # diagnostic window (with a single segment, test falls back to in-sample
            # reuse). t_train is written back from what was kept, so N_train is
            # defined afterwards even though nothing was capped by it.
            t_train_given = self.t_train
            n_wtv = max(1, int(round(0.8 * len(usable))))
            wtv, test = usable[:n_wtv], usable[n_wtv:]
            U_wtv = [U_l[:-1] for U_l, _ in wtv]
            Y_wtv = [Y_l[1:] for _, Y_l in wtv]
            U_test = [U_l[:-1] for U_l, _ in test] or U_wtv
            Y_test = [Y_l[1:] for _, Y_l in test] or Y_wtv
            kept_pairs = sum(u.shape[0] - self.N_wash for u in U_wtv)
            self.t_train = max(kept_pairs - self.N_val, 1) * self.dt_ESN
            # quiet by design: train() prints one compact summary line from this
            self.split_summary = dict(
                ragged=True, segments_in=len(U), segments_dropped=len(U) - len(usable),
                segments_train=len(wtv), segments_test=len(test), pairs=kept_pairs,
                t_val_inferred=t_val_inferred, t_train_given=t_train_given)
        else:
            # a single unsegmented trajectory has no dwell lengths to infer t_val
            # from; when omitted it defaults to 20% of the train/val window, and an
            # omitted t_train consumes all the data (80% train/val, 20% test).
            t_val_inferred = self.t_val is None
            if self.t_train is None:
                n_wtv = max(self.N_wash + 2, int(round(0.8 * U.shape[1])))
                if self.t_val is None:
                    self.t_val = max(1, int(round(0.2 * n_wtv))) * self.dt_ESN
                self.t_train = max(1, n_wtv - self.N_val) * self.dt_ESN
            elif self.t_val is None:
                self.t_val = max(1, int(round(0.2 * self.N_train))) * self.dt_ESN
            # n_test counts usable test PAIRS: U_test = U[:, N_wtv:-1] drops the
            # final step (no successor), hence the -1
            self.split_summary = dict(
                ragged=False, n_steps=U.shape[1],
                n_test=max(U.shape[1] - self.N_train - self.N_val - 1, 0),
                t_val_inferred=t_val_inferred)
            N_wtv = self.N_train + self.N_val
            if U.shape[1] < N_wtv:
                raise ValueError(f'Increase the length of the training data signal. {U.shape} < {N_wtv}')

            U_wtv = U[:, :N_wtv - 1].copy()
            Y_wtv = Y[:, 1:N_wtv].copy()

            U_test = U[:, N_wtv:-1].copy()
            Y_test = Y[:, N_wtv+1:].copy()

            assert U_wtv.shape[1] == Y_wtv.shape[1], \
                f'Inconsistent shapes for train/validation data: {U_wtv.shape} vs {Y_wtv.shape}'
            assert U_test.shape[1] == Y_test.shape[1], \
                f'Inconsistent shapes for test data: {U_test.shape} vs {Y_test.shape}'

            if Y_wtv.ndim not in [2, 3]:
                raise ValueError(f'Inconsistent ensemble size for train/validation data: {Y_wtv.shape}')

        # compute norm (normalize inputs by component range). Parameter columns (e.g.
        # cluster id) are excluded: they are a raw/categorical added input, not a
        # physical quantity to rescale, so they get identity normalization instead.
        N_obs = len(self.observed_idx)
        if is_ragged:
            # U_wtv is a ragged list (variable Nt_l): pool every segment's samples into
            # one long pseudo-trajectory so _set_norm sees a regular array. _set_norm
            # only squeezes its leading (L) axis when L>1, so flatten explicitly --
            # here L=1 by construction (one pooled trajectory).
            obs_pool = np.concatenate([u[:, :N_obs] for u in U_wtv], axis=0)
            norm_obs, shift_obs = EchoStateNetwork._set_norm(obs_pool[np.newaxis], method=self.norm_method)
            norm_obs, shift_obs = norm_obs.reshape(-1), shift_obs.reshape(-1)
        else:
            norm_obs, shift_obs = EchoStateNetwork._set_norm(U_wtv[..., :N_obs], method=self.norm_method)
        if self.input_parameters is not None:
            N_param = self._n_param(self.input_parameters)
            # Identity normalization for the parameter columns, unless the Bayesian
            # search already tuned them: re-formatting after train() (e.g. to slice
            # validation data) must not clobber the tuned values and leave a raw
            # parameter swamping the reservoir.
            if (self.optimize_parameter_normalization and getattr(self, '_norm', None) is not None
                    and self._norm.size == N_obs + N_param):
                norm_param, shift_param = self._norm[-N_param:], self._shift[-N_param:]
            else:
                norm_param, shift_param = np.ones(N_param), np.zeros(N_param)
            norm_obs = np.concatenate([norm_obs, norm_param])
            shift_obs = np.concatenate([shift_obs, shift_param])
            if self.optimize_parameter_normalization:
                # tune shift/norm per parameter via the same BO loop as rho/sigma_in/tikh,
                # instead of leaving them fixed at identity (see class docstring above).
                # Size param_shift_range/param_norm_range from the real, final
                # input_parameters (not the caller's construction-time placeholder --
                # see __init__), unless the caller already customized them.
                if 'param_shift_range' not in vars(self) or 'param_norm_range' not in vars(self):
                    lo, hi = self._param_range(self.input_parameters)
                    half = max((hi - lo) / 2, 1e-6)
                    if 'param_shift_range' not in vars(self):
                        self.param_shift_range = (lo, hi)
                    if 'param_norm_range' not in vars(self):
                        self.param_norm_range = (half / 3, half * 3)
                # Reassign (not .append) so this becomes an instance attribute, since
                # hyperparameters_to_optimize is otherwise a shared mutable class default.
                param_hp_names = [f'param_shift_{i}' for i in range(N_param)] + \
                                  [f'param_norm_{i}' for i in range(N_param)]
                self.hyperparameters_to_optimize = self.hyperparameters_to_optimize + \
                    [name for name in param_hp_names if name not in self.hyperparameters_to_optimize]
        self.norm, self.shift = norm_obs, shift_obs

        return U_wtv, Y_wtv, U_test, Y_test


    # ___________________________________________________________________________________________ BAYESIAN OPTIMIZATION
    def _reset_hyperparams(self, params, names, tikhonov=None):
        """
        Updates specific hyperparameters with new values.

        Parameters
        ----------
        params : list
            List of hyperparameter values to set.
        names : list
            Names of the hyperparameters to update.
        tikhonov : float, optional
            Value to set for the Tikhonov regularization parameter.

        Outputs:
            None. Updates internal hyperparameter values.
        """
        N_obs = len(self.observed_idx)
        for hp, name in zip(params, names):
            if name == 'sigma_in':
                setattr(self, name, 10 ** hp)
            elif name.startswith('param_shift_'):
                self._shift[N_obs + int(name.removeprefix('param_shift_'))] = hp
            elif name.startswith('param_norm_'):
                self._norm[N_obs + int(name.removeprefix('param_norm_'))] = hp
            else:
                setattr(self, name, hp)
        if tikhonov is not None:
            setattr(self, 'tikh', tikhonov)

    def _get_hyperparam(self, name):
        """
        Reads the current value of a hyperparameter by name, including the virtual
        param_shift_i/param_norm_i names that _reset_hyperparams writes directly into
        the shift/norm arrays for (see _reset_hyperparams -- there's no self.param_shift_i
        attribute, unlike e.g. self.rho).
        """
        N_obs = len(self.observed_idx)
        if name.startswith('param_shift_'):
            return self.shift[N_obs + int(name.removeprefix('param_shift_'))]
        if name.startswith('param_norm_'):
            return self.norm[N_obs + int(name.removeprefix('param_norm_'))]
        return getattr(self, name)

    @staticmethod
    def _hp_range_attr(hyper_param):
        """
        Maps a hyperparameter name to the class attribute holding its (min, max) search
        range. Per-parameter names (param_shift_0, param_norm_1, ...) share one range
        attribute (param_shift_range/param_norm_range) rather than one per index.
        """
        if hyper_param.startswith('param_shift_') or hyper_param.startswith('param_norm_'):
            return hyper_param.rsplit('_', 1)[0] + '_range'
        return hyper_param + '_range'


    def _optimize_hyperparameters(self, U_wtv, Y_wtv, validation_strategy=None, print_convergence=True):
        """
        Performs Bayesian hyperparameter optimization to minimize the validation loss.

        Parameters
        ----------
        U_wtv : np.ndarray
            Wash-train-validation input data.
        Y_wtv : np.ndarray
            Corresponding labels for train-validation data.
        validation_strategy : function, optional
            Validation function for hyperparameter tuning.
            Defaults to `_RVC_Noise`.

        Returns
        -------
            OptimizeResult: Results of the Bayesian optimization process.
        """
        from skopt import gp_minimize
        from skopt.learning import GaussianProcessRegressor as GPR
        from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern

        # print("Starting Bayesian hyperparameter optimization...")

        # Prepare search grid, space, and hyperparameter names
        search_grid, search_space, hp_names = self._hyperparameter_search(print_convergence=print_convergence)
        tikh_opt = np.zeros(self.N_func_evals)  # Track optimal Tikhonov regularization

        # Use default or provided validation strategy
        if validation_strategy is None:
            validation_strategy = self._RVC_Noise

        # Prepare the validation function
        val_func = partial(validation_strategy,
                           case=self,
                           U_wtv=U_wtv.copy(),
                           Y_wtv=Y_wtv.copy(),
                           tikh_opt=tikh_opt,
                           hp_names=hp_names,
                           print_convergence=print_convergence
                           )

        # Configure ARD 5/2 Matern Kernel for Gaussian Process
        kernel_ = (ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0)) *
                   Matern(length_scale=[0.2] * len(search_space), nu=2.5, length_scale_bounds=(1e-2, 1e1)))

        # Gaussian Process reconstruction
        gp_estimator = GPR(kernel=kernel_,
                           normalize_y=True,
                           n_restarts_optimizer=3,
                           noise=1e-10,
                           random_state=10)

        # Perform Bayesian Optimization
        result = gp_minimize(val_func,  # function to minimize
                             search_space,  # bounds
                             base_estimator=gp_estimator,  # GP kernel
                             acq_func="gp_hedge",  # acquisition function
                             n_calls=self.N_func_evals,  # number of evaluations
                             x0=search_grid,  # Initial grid points
                             n_random_starts=self.N_initial_rand,  # random initial points
                             n_restarts_optimizer=3,  # tries per acquisition
                             random_state=10)
        assert result is not None, 'gp_minimize retuned a None instance'
        # Process results
        f_iters = np.array(result.func_vals)
        best_idx = np.argmin(f_iters)

        # Update hyperparameters with the best result
        self._reset_hyperparams(result.x, hp_names, tikhonov=tikh_opt[best_idx])

        print(f"seed {self.seed} \t Optimal hyperparameters: {result.x}, {self.tikh}, val score: {result.fun}")  # type: ignore

        return dict(res=result,
                    hp_names=hp_names,
                    n_grid_points=len(search_grid))

    def _hyperparameter_search(self, print_convergence=True):
        """
        Prepares the search grid and search space for Bayesian hyperparameter optimization.
        TODO: add noise to the optional input_parameters to optimize.

        Returns
        -------
            tuple:
                - search_grid (list): List of initial grid points for optimization.
                - search_space (list): Search space objects for each hyperparameter.
                - input_parameters (list): Names of the hyperparameters being optimized.
        """
        from skopt.space import Real

        parameters = [hp for hp in self.hyperparameters_to_optimize if hp != 'tikh']

        if 'tikh' not in self.hyperparameters_to_optimize:
            setattr(self, 'tikh_range', [self.tikh])

        # Grid points per axis, shrunk (never grown) so the full factorial grid fits
        # within N_func_evals. A fixed N_grid blows up combinatorially once many
        # hyperparameters are in play (e.g. optimize_parameter_normalization adds 2
        # dims per parameter) -- N_grid=3 with 8 dims would be a 6561-point grid.
        # ponytail: coarser-per-axis is a blunt way to keep this fast; a real
        # high-dimensional design (e.g. Latin hypercube) would use the eval budget
        # better if this ever needs finer resolution with many parameters.
        n_grid_eff = self.N_grid if not parameters else \
            max(1, min(self.N_grid, int(self.N_func_evals ** (1.0 / len(parameters)))))

        param_grid, search_space = [], []
        for hyper_param in parameters:
            range_ = getattr(self, self._hp_range_attr(hyper_param))  # type: tuple[float,float]
            param_grid.append(np.linspace(*range_, n_grid_eff))
            search_space.append(Real(*range_, name=hyper_param))

        # The first n_grid_eff^len(parameters) points are from grid search
        search_grid = product(*param_grid, repeat=1)
        search_grid = [list(sg) for sg in search_grid]

        # Print optimization header
        if print_convergence:
            print('\n ----------------- HYPERPARAMETER SEARCH ------------------\n '
                  f'{n_grid_eff}^{len(parameters)} grid' +
                  f' and {max(0, self.N_func_evals - len(search_grid))} points with Bayesian Optimization\n\t', end="")
            for kk in self.hyperparameters_to_optimize:
                print(f'\t {kk}', end="")
            print('\t MSE val ')

        return search_grid, search_space, parameters

    # ___________________________________________________________________________________________ NORMALIZATION METHODS

    @staticmethod
    def _set_norm(train_data, method=None):
        """
        Computes the normalization factor for the input data.
        Parameters
        ----------
        train_data : np.ndarray
            Wash-train-validation training input data. (Nens x Nt x Ndim).
        Returns
        -------
            float: Normalization factor based on the range of the input data.
        """
        # assert train_data.ndim in [3, 4], f'U_wtv must be a 3D array, got {train_data.ndim}D: ({train_data.shape})'

        if train_data.ndim == 3:
            L, _, Ndim = train_data.shape
            Nens = 1
        elif train_data.ndim == 4:
            L, _, Ndim, Nens = train_data.shape
        elif train_data.ndim == 2:
            L = 1
            Nens = 1
            Ndim = train_data.shape[1]
        else:
            raise ValueError(f'U_wtv must be a 2D, 3D or 4D array, got {train_data.ndim}D: ({train_data.shape})')

        if method is None:
            return np.ones(Ndim), np.zeros(Ndim)

        shift = np.mean(train_data, axis=1)

        shifted_data  = train_data - shift[:, np.newaxis, :]

        if method == 'std':
            shift = np.mean(train_data, axis=1)
            norm = np.std(shifted_data, axis=1)
        elif method == 'max':
            norm = np.max(shifted_data, axis=1)
        elif method == 'mean':
            norm = np.mean(abs(shifted_data), axis=1)
        elif method == 'range':
            m = np.min(shifted_data, axis=1)
            M = np.max(shifted_data, axis=1)
            norm = M - m
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if L > 1:
            norm = np.mean(norm, axis=0)
            shift = np.mean(shift, axis=0)
        if Nens > 1:
            norm = np.mean(norm, axis=-1)
            shift = np.mean(shift, axis=-1)

        if np.any(abs(norm) < 1e-12):
            norm[abs(norm) < 1e-12] = 1.0  # Prevent division by zero

        return norm, shift

    # ___________________________________________________________________________________________ VALIDATION STRATEGIES

    # Implemented as module-level functions in validation.py; aliased as
    # staticmethods so existing references keep working: self._RVC_Noise
    # (train()'s default), EchoStateNetwork._SegmentRVC_Noise (qlrom's
    # esn_LL/esn_SRC), scripts, tutorials and tests.
    _RVC_Noise = staticmethod(validation.RVC_Noise)
    _SegmentRVC_Noise = staticmethod(validation.SegmentRVC_Noise)
    _single_series_validation = staticmethod(validation.single_series_validation)
    _SSV = staticmethod(validation.SSV)
    _WFV = staticmethod(validation.WFV)
    _KFV = staticmethod(validation.KFV)

    def compute_nMAE(self, Y_true, Y_pred, norm=1.0):
        r"""Error metric used throughout training/validation/testing: the normalized
        mean absolute error
        $\mathrm{mean}(|\mathbf{Y}_\mathrm{true} - \mathbf{Y}_\mathrm{pred}|) /
        \mathrm{mean}(|\texttt{norm}|)$.

        Parameters
        ----------
        Y_true : np.ndarray
            Ground-truth values.
        Y_pred : np.ndarray
            Predicted values, same shape as `Y_true`.
        norm : float or np.ndarray
            Normalization factor (e.g. the data range per component). Default 1.0.

        Returns
        -------
        float
            Normalized error.
        """
        return np.mean(np.abs(Y_true - Y_pred)) / np.mean(np.abs(norm))

    def compute_nRMSE(self, Y_true, Y_pred, norm=1.0):
        """Deprecated alias of `compute_nMAE` -- the metric was never an RMSE."""
        warnings.warn('compute_nRMSE computes a normalized MAE; use compute_nMAE',
                      DeprecationWarning, stacklevel=2)
        return self.compute_nMAE(Y_true, Y_pred, norm)

    # _______________________________________________________________________________________ TEST & PLOTTING FUNCTIONS


    def run_test(self,
                 U_test,
                 Y_test,
                 pdf_file=None,
                 Nt_test=None,
                 max_L_tests=5,
                 nbins=20,
                max_short_tests=10,
                long_term=True,
                short_term=True,
                 ):
        """Evaluate the trained ESN on test data with closed-loop forecasts, printing
        error metrics and building diagnostic figures.

        Parameters
        ----------
        U_test : np.ndarray
            Test input data, shape ``(L, Nt, N_dim)``.
        Y_test : np.ndarray
            Ground-truth outputs for the test data, same leading shape as `U_test`.
        pdf_file : PdfPages, optional
            Currently unused by this method (present for API compatibility).
        Nt_test : int, optional
            Length of each short-term test, in ESN steps. Defaults to `N_val`.
        max_L_tests : int
            Maximum number of segments (of the `L` available) to evaluate. Default 5.
        nbins : int
            Number of bins for the prediction PDFs (long-term test only). Default 20.
        max_short_tests : int
            Maximum number of short-term test figures to draw, across all segments.
            Default 10.
        long_term : bool
            If True, run one long-term (full-length) closed-loop forecast per tested
            segment and plot it. Default True.
        short_term : bool
            If True, run repeated short-term (`Nt_test`-long) closed-loop forecasts
            per tested segment and plot up to `max_short_tests` of them. Default True.

        Returns
        -------
        list
            ``[fig_long] + figures_short``: the long-term test figure (or None if
            `long_term` is False or there is more than one segment) followed by the
            short-term test figures (each possibly None if `short_term` is False).
        """

        if max_L_tests is None and hasattr(self, 'max_L_tests'):
            max_L_tests = self.max_L_tests
        if Nt_test is None:
            Nt_test = self.N_val

        # U_test/Y_test may be a regular ndarray or a list of L segments of possibly
        # different length (e.g. cluster-dwell chunks) -- normalize to a list either way.
        if isinstance(U_test, np.ndarray):
            if U_test.ndim == 1:
                U_test = U_test[np.newaxis, :, np.newaxis]
            elif U_test.ndim == 2:
                U_test = U_test[np.newaxis, :, :]
            U_test = list(U_test)
        if isinstance(Y_test, np.ndarray):
            if Y_test.ndim == 1:
                Y_test = Y_test[np.newaxis, :, np.newaxis]
            elif Y_test.ndim == 2:
                Y_test = Y_test[np.newaxis, :, :]
            Y_test = list(Y_test)

        rng0 = self.rng

        L, Nq = len(U_test), U_test[0].shape[1]

        if Nq > 10:
            nrows, dims = 10, rng0.choice(Nq, 10, replace=False)
        else:
            nrows, dims = self.N_dim, np.arange(Nq)
            if Nq == 1:
                dims = [dims]

        observed_idx_np = np.array(self.observed_idx)

        # Select test cases (with a maximum of max_L_tests), restricted to segments
        # with enough test-tail length: the long-term washout+forecast in predict_Y
        # needs len(Y_test_l) >= 2*N_wash (segments may differ in length -- e.g.
        # cluster-dwell chunks -- so this isn't guaranteed for every Li).
        min_needed = 2 * self.N_wash if long_term else self.N_wash
        testable = [Li for Li in range(L) if len(Y_test[Li]) >= min_needed]
        if not testable:
            print(f'run_test: no segment has >= {min_needed} test steps; skipping.')
            return []

        if len(testable) > 1:
            if max_L_tests != len(testable):
                L_indices = np.sort(rng0.choice(testable, max_L_tests, replace=max_L_tests > len(testable)))
            else:
                L_indices = np.array(testable)
        else:
            L_indices = [testable[0]]

        N_ens = U_test[0].shape[-1] if U_test[0].ndim == 3 else 1
        # Prediction function
        def predict_Y(_input, _target):

            # Perform washout (open-loop without extra forecast step)
            r_out = np.zeros((self.N_units, N_ens))
            u_out = np.zeros((self.N_dim, N_ens))
            u_open = np.zeros_like(_target[:self.N_wash])


            for ii, u_in in enumerate(_input[:self.N_wash]):
                u_out, r_out = self.step(u_in, r_out)
                try:
                    u_open[ii] = u_out.squeeze()
                except Exception:
                    u_open[ii] = u_out.copy()


            Y_closed = np.zeros_like(_target)

            # Reroll input_parameters to the segment's own value for this run, then
            # restore the original (N_param, L) array so callers see the full parameter
            # set again afterwards (e.g. a later re-format of the training data).
            original_input_parameters = self.input_parameters
            if self.input_parameters is not None:
                # Parameter (e.g. cluster id) may vary within _input (a segment straddling
                # a transition); read off its own first step and hold it fixed for this run.
                N_param = self._n_param(self.input_parameters)
                self.input_parameters = _input[0, -N_param:].reshape(N_param, N_ens)

            try:
                for i in range(Y_closed.shape[0]):
                    u_input = self.outputs_to_inputs(full_state=u_out)
                    u_out, r_out = self.step(u_input, r_out)
                    try:
                        Y_closed[i] = u_out.squeeze()
                    except Exception:
                        Y_closed[i] = u_out.copy()
            finally:
                self.input_parameters = original_input_parameters

            return Y_closed, u_open

        # Plotting function
        def plot_time(_axs, _time, _pred_closed, _pred_open, _inputs, _target, _err=None):
            if not isinstance(_axs, (list, np.ndarray)):
                _axs = [_axs]

            t_wash_in = _time[:self.N_wash] - self.dt_ESN
            t_wash_out = _time[:self.N_wash]
            t_out = _time[self.N_wash:]

            for dim_i, _ax in zip(range(self.N_dim), _axs):
                _ax.plot(t_out, _target[:, dim_i], 'k', label=f'truth dim {dim_i}')
                # Plot the input if observed
                if dim_i in self.observed_idx:
                    _i = np.argmin(abs(observed_idx_np-dim_i))
                    _ax.plot(t_wash_in, _inputs[:self.N_wash, _i], 'x', c='C4', ms=5, label='Washout')

                _ax.plot(t_wash_out, _pred_open[:, dim_i], '-co', mfc='none', label='ESN open loop')
                _ax.plot(t_out, _pred_closed[:, dim_i], '--r', dashes=[2, .5],
                         label=[f'ESN closed-loop prediction \n error = {_err:.4}' if _err is not None else 'ESN closed-loop prediction'])
                _ax.set(ylabel=f'$u_{dim_i}$')
                _ax.set(ylim=ylims[dim_i])

        test_counter, errors_all = 0, []
        hist_args = dict(bins=nbins, density=True, orientation='horizontal', stacked=False)


        print('Running test for L=', end=' ')
        for Li in L_indices:
            print(f'{Li}', end=' ')

            # Select dataset
            U_test_l, Y_test_l = U_test[Li], Y_test[Li]


            norm_l = np.max(Y_test_l, axis=0) - np.min(Y_test_l, axis=0)

            t_l = (np.arange(U_test_l.shape[0])) * self.dt_ESN
            # set ylims for plotting
            ylims = [[np.min(Y_test_l[:, dim_i])*1.05, np.max(Y_test_l[:, dim_i])*1.05] for dim_i in range(self.N_dim)]

            # plot tests statistics if the test dataset is long or requested
            if long_term:

                # predict over the entire test set
                Y_closed, U_open = predict_Y(U_test_l[:-1], Y_test_l[self.N_wash:])

                err_long = np.log10(self.compute_nMAE(Y_closed, Y_test_l[self.N_wash:], norm=norm_l))

                fig_long, grid = plt.subplots(nrows=self.N_dim, ncols=2, figsize=[10, 2.5 * self.N_dim],
                                         sharex='col', sharey='row', layout='tight', width_ratios=[5, 1])

                if self.N_dim == 1:
                    axs, axs_pdf = [grid[0]], [grid[1]]
                else:
                    axs, axs_pdf = grid[:, 0], grid[:, 1]

                plot_time(_axs=axs,
                          _time=t_l,
                          _pred_closed=Y_closed,
                           _pred_open=U_open,
                          _inputs=U_test_l,
                          _target=Y_test_l[self.N_wash:],)

                # Plot histograms]
                for dim_i, ax_2 in enumerate(axs_pdf):
                    if dim_i in self.observed_idx:
                        _i = np.argmin(abs(observed_idx_np - dim_i))
                        ax_2.hist(U_test_l[:, _i], color='k', lw=2, alpha=0.6, histtype='step', **hist_args)

                    ax_2.hist(Y_test_l[:, dim_i], color='k', lw=.85, histtype='step', **hist_args)
                    ax_2.hist(Y_closed[:, dim_i], color='r', ls='--', histtype='stepfilled', alpha=0.5, **hist_args)
                    ax_2.hist(Y_closed[:, dim_i], color='r', ls='--', histtype='step', **hist_args)

                # axs[0].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0))
                plt.suptitle(f'Li = {Li}, observed idx = {self.observed_idx}, error = {err_long:.4}')
                axs[-1].set(xlabel='$t/T$')
            else:
                fig_long = None


            if short_term:
                i0 = 0 # reset time index for each Li
                figures_short = []
                short_term_error = 0.
                max_test_time = U_test_l.shape[0]  # per-segment: segments may differ in length

                while i0 + Nt_test < max_test_time:
                    if len(figures_short) >= max_short_tests:
                        break
                    test_counter += 1

                    i1 = i0 + Nt_test + self.N_wash

                    current_input = U_test_l[i0:i1-1].copy()
                    current_target = Y_test_l[i0+self.N_wash:i1].copy()
                    current_time = t_l[i0:i1]

                    # predict
                    Y_closed, U_open = predict_Y(current_input, current_target)

                    current_error = np.log10(self.compute_nMAE(current_target, Y_closed, norm=norm_l))

                    short_term_error += current_error


                    if test_counter <= max_L_tests:
                        fig_short, axs_short = plt.subplots(nrows=nrows, ncols=1, figsize=[8, 1.5 * nrows], sharex='all', layout='tight')
                        if nrows == 1:
                            axs_short = [axs_short]

                        plot_time(_axs=axs_short, _time=current_time, _pred_closed=Y_closed, _pred_open=U_open,
                                  _inputs=current_input, _target=current_target, _err=current_error)


                        axs_short[0].legend(title=f'Test {test_counter}: Li = {Li}', loc='upper left',
                                            bbox_to_anchor=(1, 1), fontsize='x-small')
                        axs_short[-1].set(xlabel='$t/T$')

                        figures_short.append(fig_short)
                    i0 += Nt_test

                errors_all.append(short_term_error / max(1, (i0 // Nt_test)))
            else:
                figures_short = [None]

        else:
            fig_long = None
            figures_short = [None]

        # Compute errors over all Lis
        if test_counter > 0:
            errors_all = np.array(errors_all)
            print(f'Overall tests min, max and mean MSE in {test_counter} tests = {np.min(errors_all):.4}, {np.max(errors_all):.4}, {np.mean(errors_all):.4}.')

        return [fig_long] + figures_short


    def _plot_training_results(self, U_test, Y_test, results, save_ESN_training, folder):
        """
        Plots training results, including Bayesian optimization convergence and test results.
        """
        from skopt.plots import plot_convergence

        all_figs = []

        # Plot Bayesian optimization convergence
        fig1 = plt.figure()
        plot_convergence(results['res'])
        all_figs.append(fig1)
        # Plot Gaussian Process reconstruction
        all_figs.extend(self._plot_BO(results))
        # Plot Wout matrix
        all_figs.append(self.plot_Wout())
        # Plot test results if applicable (run_test itself skips gracefully if no
        # segment has enough test-tail length). Capped by max_L_tests/max_short_tests:
        # run_test makes one long-term figure per tested segment plus one per
        # short-term test, so a multi-segment (e.g. parametric) network otherwise
        # emits dozens of figures.
        max_test_len = max(len(u) for u in U_test) if isinstance(U_test, list) else U_test.shape[1]
        if self.perform_test and max_test_len >= self.N_val:
            test_figs = self.run_test(U_test, Y_test,
                                      long_term=True, short_term=True,
                                      max_L_tests=self.max_L_tests,
                                      max_short_tests=self.max_short_tests)
            all_figs = all_figs + test_figs

        if save_ESN_training:
            if folder is None:
                folder = self.figs_folder
            os.makedirs(folder, exist_ok=True)
            save_pdf = plt_pdf.PdfPages(f'{folder}{self.filename}_Training.pdf')
            [add_pdf_page(save_pdf, fig) for fig in all_figs]
            save_pdf.close()



    def _plot_BO(self, results_bayesian_optimization):
        """Visualize the Bayesian-optimization Gaussian-process reconstruction as one
        subplot per consecutive hyperparameter pair, all pairs sharing a single figure
        (parametric ESNs can optimize a dozen hyperparameters -- one figure each is
        unreadable).

        Parameters
        ----------
        results_bayesian_optimization : dict
            Dictionary with ``hp_names`` (labels of the optimized hyperparameters),
            ``res`` (the `skopt` result carrying the GP reconstruction) and
            ``n_grid_points`` (initial grid evaluations, marked differently from the
            BO-acquired points).

        Returns
        -------
        list
            ``[fig]`` with the shared figure, or ``[]`` when fewer than two
            hyperparameters were optimized (nothing to contour).
        """

        hp_names = results_bayesian_optimization['hp_names']
        res = results_bayesian_optimization['res']
        n_grid_points = results_bayesian_optimization['n_grid_points']

        f_iters = np.array(res.func_vals)

        if len(hp_names) < 2:  # nothing to contour
            return []

        gp = res.models[-1]
        res_x = np.array(res.x_iters)
        best_x = res.x  # best-found point in the full (possibly >2D) hyperparameter space
        best_idx = np.argmin(f_iters)
        amin = np.amin([10, np.max(f_iters)])
        n_len = 100  # points to evaluate the GP at

        # All consecutive hyperparameter pairs share one figure (parametric ESNs can
        # optimize a dozen hyperparameters -- one figure each is unreadable).
        n_pairs = len(hp_names) - 1
        ncols = min(3, n_pairs)
        nrows = int(np.ceil(n_pairs / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), layout='constrained')
        axs = np.atleast_1d(axs).ravel()

        for hpi, ax in enumerate(axs[:n_pairs]):
            range_1 = getattr(self, self._hp_range_attr(hp_names[hpi]))  # type: tuple[float, float]
            range_2 = getattr(self, self._hp_range_attr(hp_names[hpi + 1]))  # type: tuple[float, float]

            xx, yy = np.meshgrid(np.linspace(*range_1, n_len), np.linspace(*range_2, n_len))

            # res.space is the full len(hp_names)-D search space, so every evaluated point
            # needs one value per hyperparameter, not just the 2 being plotted here -- hold
            # every other dimension fixed at its best-found value (a partial-dependence slice)
            x_x = np.tile(best_x, (xx.size, 1)).astype(float)
            x_x[:, hpi] = xx.ravel()
            x_x[:, hpi + 1] = yy.ravel()
            x_gp = res.space.transform(x_x.tolist())  # gp prediction needs norm. format

            # Final GP reconstruction for each realization at the evaluation points
            y_pred = np.clip(-gp.predict(x_gp), a_min=-amin, a_max=-np.min(f_iters)).reshape(n_len, n_len)

            cf = ax.contourf(xx, yy, y_pred, levels=20, cmap='Blues')
            ax.contour(xx, yy, y_pred, levels=20, colors='black', linewidths=1, linestyles='solid', alpha=0.3)
            fig.colorbar(cf, ax=ax, label='-$\\log_{10}$(MSE)')
            #   Plot the n_tot search points (columns hpi/hpi+1 -- the pair being
            #   visualized in *this* subplot, not always the first two hyperparameters)
            for rx, mk in zip([res_x[:n_grid_points], res_x[n_grid_points:]], ['v', 's']):
                ax.plot(rx[:, hpi], rx[:, hpi + 1], mk, c='w', alpha=.8, mec='k', ms=8)
            ax.plot(res_x[best_idx, hpi], res_x[best_idx, hpi + 1], '*r', alpha=.8, mec='r', ms=8)
            ax.set(xlabel=hp_names[hpi], ylabel=hp_names[hpi + 1])

        for ax in axs[n_pairs:]:
            ax.axis('off')

        return [fig]


    def plot_Wout(self):
        """Visualize the trained read-out matrix `Wout`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with a single heat-map axis of ``Wout.T``.
        """
        fig, ax = plt.subplots()
        im = ax.matshow(self.Wout.T, cmap="PRGn", aspect=4., vmin=-np.max(self.Wout), vmax=np.max(self.Wout))
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.colorbar(im, orientation='horizontal', extend='both')
        ax.set(ylabel='$N_u$', xlabel='$N_r$', title='$\\mathbf{W}_\\mathrm{out}$')
        return fig




if __name__ == "__main__":

    print(vars(EchoStateNetwork))

    pass
