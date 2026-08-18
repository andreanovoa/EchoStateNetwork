"""Echo state networks / reservoir computing, pure numpy.

One class, EchoStateNetwork: training (ridge regression with recycle validation,
contiguous or ragged dwell segments), Bayesian hyperparameter search (optional
scikit-optimize), parametric inputs, closed-loop prediction and Jacobians.
Shared by the qlrom (qlESN families) and romda (ESN_model, ESN_bias) packages.
"""

from .esn import EchoStateNetwork
from . import validation

__version__ = "0.1.2"

__all__ = ["EchoStateNetwork", "validation"]
