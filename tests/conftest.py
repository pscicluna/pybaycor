"""Shared fixtures and synthetic-data helpers for the pybaycor test suite."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

# Cheap NUTS settings allowed in the fast suite (one or two smoke fits only).
SMOKE_FIT_KWARGS = {
    "draws": 100,
    "tune": 100,
    "chains": 1,
    "nuts_sampler": "pymc",
    "progressbar": False,
    "compute_convergence_checks": False,
}


# ----------------------------------------------------------------------------- data
def make_gaussian_data(
    n: int,
    rho: float,
    seed: int = 0,
    *,
    mean: tuple[float, float] = (0.0, 0.0),
    scales: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Draw ``(n, 2)`` bivariate Normal data with correlation ``rho``."""
    rng = np.random.default_rng(seed)
    s0, s1 = scales
    cov = np.array([[s0**2, rho * s0 * s1], [rho * s0 * s1, s1**2]])
    return rng.multivariate_normal(np.asarray(mean, float), cov, size=n)


def make_gaussian_data_3d(
    n: int, corr: np.ndarray | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``(n, 3)`` Normal data with a known correlation matrix; return (data, corr)."""
    if corr is None:
        corr = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.1], [-0.3, 0.1, 1.0]])
    corr = np.asarray(corr, float)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(corr.shape[0]), corr, size=n), corr


def add_heteroscedastic_noise(
    data: np.ndarray,
    rng: np.random.Generator,
    sigma_range: tuple[float, float] = (0.3, 1.2),
) -> tuple[np.ndarray, np.ndarray]:
    """Add Normal noise with per-entry scales drawn uniformly from ``sigma_range``.

    Returns the noisy data and the ``(n, d)`` array of scales actually used.
    """
    sigma = rng.uniform(*sigma_range, size=data.shape)
    noisy = data + rng.normal(size=data.shape) * sigma
    return noisy, sigma


def add_twopiece_noise(
    data: np.ndarray, rng: np.random.Generator, s_lo: float, s_hi: float
) -> np.ndarray:
    """Add two-piece Normal noise (scale ``s_lo`` below the truth, ``s_hi`` above)."""
    eps = rng.normal(size=data.shape)
    return data + np.where(eps < 0, s_lo, s_hi) * eps


def censor_column(
    data: np.ndarray, col: int, frac: float, kind: str = "upper"
) -> tuple[np.ndarray, np.ndarray]:
    """Censor a fraction of one column, returning ``(values, censoring)``.

    ``kind="upper"``: the lowest ``frac`` of the column are replaced by the
    ``frac``-quantile and flagged ``-1`` (upper limits). ``kind="lower"``: the highest
    ``frac`` are replaced by the ``(1 - frac)``-quantile and flagged ``+1``.
    """
    values = np.array(data, dtype=float, copy=True)
    cens = np.zeros(values.shape, dtype=int)
    if kind == "upper":
        thresh = np.quantile(values[:, col], frac)
        mask = values[:, col] < thresh
        values[mask, col] = thresh
        cens[mask, col] = -1
    elif kind == "lower":
        thresh = np.quantile(values[:, col], 1.0 - frac)
        mask = values[:, col] > thresh
        values[mask, col] = thresh
        cens[mask, col] = 1
    else:
        raise ValueError(kind)
    return values, cens


# -------------------------------------------------------------------------- fixtures
@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def gaussian_data():
    return make_gaussian_data


@pytest.fixture
def gaussian_data_3d():
    return make_gaussian_data_3d


@pytest.fixture
def heteroscedastic_noise():
    return add_heteroscedastic_noise


@pytest.fixture
def censor():
    return censor_column


# ---------------------------------------------------------------------- model utils
def eval_at_point(model, names, point=None):
    """Evaluate the named model variables at ``point`` (default: the initial point).

    Deterministics are expressed through random variables, so they are first rewritten
    in terms of the model's value variables and compiled with those as inputs.
    """
    outs = [model[name] for name in names]
    fn = model.compile_fn(
        model.replace_rvs_by_values(outs),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    if point is None:
        point = model.initial_point()
    return dict(zip(names, fn(point), strict=True))


def free_rv_names(model) -> set[str]:
    return {rv.name for rv in model.free_RVs}
