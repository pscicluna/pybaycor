"""Deprecated module kept for ``from pybaycor.pybaycor import ...`` imports.

Everything now lives in :mod:`pybaycor.correlation`; import from :mod:`pybaycor`.
"""

import warnings

from .correlation import (
    BayesianCorrelation,
    HierarchicalBayesianCorrelation,
    HierarchicalRobustBayesianCorrelation,
    RobustBayesianCorrelation,
)

__all__ = [
    "BayesianCorrelation",
    "HierarchicalBayesianCorrelation",
    "HierarchicalRobustBayesianCorrelation",
    "RobustBayesianCorrelation",
]

warnings.warn(
    "pybaycor.pybaycor is deprecated; import the classes from the pybaycor package instead",
    DeprecationWarning,
    stacklevel=2,
)
