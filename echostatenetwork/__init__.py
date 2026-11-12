"""Echo state networks / reservoir computing, pure numpy.

One class, EchoStateNetwork: training (ridge regression with recycle validation,
contiguous or ragged dwell segments), Bayesian hyperparameter search (optional
scikit-optimize), parametric inputs, closed-loop prediction and Jacobians, and a
parallel layout with one reservoir per patch of sites (patch_size, halo).
Shared by the qlrom (qlESN families) and romda (ESN_model, ESN_bias) packages.

The package is silent by default: `set_verbose()` shows its reports (the
hyperparameter search and the summary of `train`) and its diagnostic messages.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import validation  # noqa: E402
from .esn import EchoStateNetwork  # noqa: E402
from .verbose import set_verbose  # noqa: E402

__version__ = "0.5.0"

__all__ = ["EchoStateNetwork", "set_verbose", "validation"]
