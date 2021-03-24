""" pybaycor: a package for estimating (robust) Bayesian correlation coefficients in python

"""


from . import pybaycor
from .pybaycor import BayesianCorrelation
from .pybaycor import RobustBayesianCorrelation
from .pybaycor import HierarchicalBayesianCorrelation
from .pybaycor import HierarchicalRobustBayesianCorrelation

__version__ = "0.2.1"
__all__=["pybaycor"]
