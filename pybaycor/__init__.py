"""pybaycor: Bayesian inference of (robust) correlation coefficients.

The package infers the correlation matrix of multivariate data under a multivariate
Normal or Student-t model with an LKJ prior, handling measurement uncertainties
(symmetric or asymmetric, on any subset of entries) and censoring (upper limits,
lower limits or intervals, per entry) through a single model and class,
:class:`BayesianCorrelation`.
"""

from .correlation import (
    BayesianCorrelation,
    HierarchicalBayesianCorrelation,
    HierarchicalRobustBayesianCorrelation,
    RobustBayesianCorrelation,
)
from .data import CorrelationData, Standardisation
from .inference import available_nuts_samplers, fit_model
from .model import DEFAULT_PRIORS, build_model
from .stats import bayes_factor, correlation_summary

__version__ = "0.3.0.dev0"


def __getattr__(name: str):
    """Expose the deprecated ``pybaycor.pybaycor`` submodule lazily, with its warning."""
    if name == "pybaycor":
        import importlib

        return importlib.import_module(".pybaycor", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_PRIORS",
    "BayesianCorrelation",
    "CorrelationData",
    "HierarchicalBayesianCorrelation",
    "HierarchicalRobustBayesianCorrelation",
    "RobustBayesianCorrelation",
    "Standardisation",
    "__version__",
    "available_nuts_samplers",
    "bayes_factor",
    "build_model",
    "correlation_summary",
    "fit_model",
]
