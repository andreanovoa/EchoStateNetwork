"""Echo state networks / reservoir computing, pure numpy.

One class, EchoStateNetwork: training (ridge regression with recycle validation,
contiguous or ragged dwell segments), Bayesian hyperparameter search (optional
scikit-optimize), parametric inputs, closed-loop prediction and Jacobians, and a
parallel layout with one reservoir per patch of sites (patch_size, halo).
Shared by the qlrom (qlESN families) and romda (ESN_model, ESN_bias) packages.
"""

from . import validation
from .esn import EchoStateNetwork

__version__ = "0.2.2"

__all__ = ["EchoStateNetwork", "validation"]
