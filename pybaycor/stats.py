"""Posterior summaries of correlation coefficients.

Functions here operate on the ``posterior`` group of an :class:`xarray.DataTree` as
produced by :func:`~pybaycor.inference.fit_model`, extracting the correlation matrix
``chol_corr`` and summarising each off-diagonal element.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as sps

__all__ = [
    "bayes_factor",
    "correlation_samples",
    "correlation_summary",
    "lkj_marginal_density",
]


def _posterior(idata: xr.DataTree | xr.Dataset) -> xr.Dataset:
    if isinstance(idata, xr.DataTree):
        return idata["posterior"].to_dataset()
    return idata


def correlation_samples(idata: xr.DataTree | xr.Dataset, i: int, j: int) -> np.ndarray:
    """Return the flattened posterior samples of the correlation between ``i`` and ``j``."""
    post = _posterior(idata)
    da = post["chol_corr"]
    sample_dims = [dd for dd in ("chain", "draw") if dd in da.dims]
    other = [dd for dd in da.dims if dd not in sample_dims]
    return da.isel({other[0]: i, other[1]: j}).stack(sample=sample_dims).to_numpy().ravel()


def lkj_marginal_density(r: Any, eta: float, n_dim: int) -> np.ndarray:
    """Prior density of a single correlation coefficient under an LKJ(``eta``) prior.

    For a ``n_dim`` x ``n_dim`` correlation matrix with an LKJ prior of shape ``eta``,
    each off-diagonal element has the marginal distribution
    ``(r + 1) / 2 ~ Beta(a, a)`` with ``a = eta - 1 + n_dim / 2`` (Lewandowski,
    Kurowicka & Joe 2009).

    Parameters
    ----------
    r : array_like
        Correlation values in ``(-1, 1)``.
    eta : float
        LKJ shape parameter.
    n_dim : int
        Dimension of the correlation matrix.
    """
    a = eta - 1.0 + n_dim / 2.0
    r = np.asarray(r, dtype=float)
    return 0.5 * sps.beta(a, a).pdf(0.5 * (r + 1.0))


def bayes_factor(
    samples: np.ndarray,
    *,
    eta: float,
    n_dim: int,
    ref_val: float = 0.0,
) -> float:
    """Savage-Dickey Bayes factor ``BF10`` against ``rho = ref_val``.

    The Savage-Dickey density ratio compares the prior and posterior densities at the
    point null: ``BF01 = p(rho = ref | data) / p(rho = ref)``, so that ``BF10 = 1/BF01``.
    Values above 1 favour a non-zero correlation. The posterior density is estimated with
    a Gaussian kernel density estimate on the samples, which is adequate unless the
    posterior is concentrated very close to ``±1``.

    Parameters
    ----------
    samples : ndarray
        Posterior samples of the correlation coefficient.
    eta, n_dim : float, int
        Hyper-parameters defining the LKJ prior (see :func:`lkj_marginal_density`).
    ref_val : float
        The null value.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size < 10:
        return float("nan")
    prior = float(lkj_marginal_density(ref_val, eta, n_dim))
    try:
        post = float(sps.gaussian_kde(samples).evaluate([ref_val])[0])
    except np.linalg.LinAlgError:  # degenerate posterior (e.g. MAP)
        return float("nan")
    if post <= 0:
        return float("inf")
    return prior / post


def correlation_summary(
    idata: xr.DataTree | xr.Dataset,
    *,
    names: list[str] | None = None,
    eta: float = 2.0,
    ci_prob: float | None = None,
    ci_kind: str | None = None,
    ref_val: float = 0.0,
) -> pd.DataFrame:
    """Tidy per-pair summary of the posterior correlation coefficients.

    Parameters
    ----------
    idata : xarray.DataTree or xarray.Dataset
        Posterior containing ``chol_corr``.
    names : list of str, optional
        Variable names (defaults to the ``variable`` coordinate of the posterior).
    eta : float
        LKJ shape used in the prior, needed for the Bayes factor.
    ci_prob : float, optional
        Probability mass of the credible interval. Defaults to ArviZ's
        ``rcParams["stats.ci_prob"]`` (0.89 in ArviZ 1), the same default that
        :func:`arviz.summary` uses, so both tables report the same interval.
    ci_kind : {"hdi", "eti"}, optional
        Highest-density or equal-tailed interval; defaults to
        ``rcParams["stats.ci_kind"]`` ("eti" in ArviZ 1).
    ref_val : float
        Null value for ``prob_greater`` and the Bayes factor.

    Returns
    -------
    pandas.DataFrame
        One row per pair ``(i, j)`` with ``i < j`` and columns ``mean``, ``median``,
        ``sd``, the interval bounds, ``prob_positive`` (``P(rho > ref_val)``),
        ``bf10`` (Savage-Dickey Bayes factor for ``rho != ref_val``) and the raw
        indices ``i``, ``j``.
    """
    import arviz as az

    if ci_prob is None:
        ci_prob = float(az.rcParams["stats.ci_prob"])
    if ci_kind is None:
        ci_kind = str(az.rcParams["stats.ci_kind"])
    post = _posterior(idata)
    corr = post["chol_corr"]
    sample_dims = [dd for dd in ("chain", "draw") if dd in corr.dims]
    n_dim = corr.shape[-1]
    if names is None:
        var_dim = next(dd for dd in corr.dims if dd not in sample_dims)
        names = [str(v) for v in corr[var_dim].to_numpy()] if var_dim in corr.coords else None
    if names is None:
        names = [f"x{j}" for j in range(n_dim)]

    rows = []
    lo_label = f"{ci_kind}_{100 * (1 - ci_prob) / 2:g}%"
    hi_label = f"{ci_kind}_{100 * (1 + ci_prob) / 2:g}%"
    for i, j in combinations(range(n_dim), 2):
        s = correlation_samples(post, i, j)
        da = xr.DataArray(s.reshape(1, -1), dims=("chain", "draw"))
        if ci_kind == "hdi":
            interval = az.hdi(da, prob=ci_prob)
        elif ci_kind == "eti":
            interval = az.eti(da, prob=ci_prob)
        else:
            raise ValueError("ci_kind must be 'hdi' or 'eti'")
        interval = np.asarray(interval).ravel()
        rows.append(
            {
                "pair": f"{names[i]}~{names[j]}",
                "i": i,
                "j": j,
                "mean": float(np.mean(s)),
                "median": float(np.median(s)),
                "sd": float(np.std(s, ddof=1)) if s.size > 1 else 0.0,
                lo_label: float(interval[0]),
                hi_label: float(interval[1]),
                "prob_positive": float(np.mean(s > ref_val)),
                "bf10": bayes_factor(s, eta=eta, n_dim=n_dim, ref_val=ref_val),
            }
        )
    return pd.DataFrame(rows).set_index("pair")
