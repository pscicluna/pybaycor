"""Inference back-ends.

:func:`fit_model` runs one of several algorithms on a model produced by
:func:`~pybaycor.model.build_model` and always returns an :class:`xarray.DataTree`
with a ``posterior`` group, so downstream code does not care which algorithm ran.

Available methods
-----------------
``"nuts"``
    Hamiltonian Monte Carlo (NUTS). The ``nuts_sampler`` argument picks the
    implementation: ``"pymc"`` (pure Python/numba), ``"nutpie"`` (Rust; the fastest CPU
    option and PyMC's default when installed), ``"numpyro"`` or ``"blackjax"`` (JAX;
    run on CPU, GPU or TPU depending on the installed ``jax`` build).
``"advi"`` / ``"fullrank_advi"``
    Automatic differentiation variational inference. Fast approximations; mean-field
    ADVI systematically under-estimates posterior correlations and widths and should
    be used for exploration only.
``"smc"``
    Sequential Monte Carlo with independent Metropolis-Hastings kernels. Robust to
    multimodality and provides the log marginal likelihood (``log_marginal_likelihood``
    in ``sample_stats``).
``"pathfinder"``
    Pathfinder variational inference from ``pymc-extras`` (optional dependency).
``"map"``
    Maximum a posteriori point estimate (returned as a one-draw posterior).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pymc as pm
import xarray as xr

__all__ = ["METHODS", "NUTS_SAMPLERS", "available_nuts_samplers", "fit_model"]

METHODS = ("nuts", "advi", "fullrank_advi", "smc", "pathfinder", "map")
NUTS_SAMPLERS = ("pymc", "nutpie", "numpyro", "blackjax")

Method = Literal["nuts", "advi", "fullrank_advi", "smc", "pathfinder", "map"]
NutsSampler = Literal["pymc", "nutpie", "numpyro", "blackjax"]


def _importable(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def available_nuts_samplers() -> list[str]:
    """Return the NUTS back-ends that can be used in this environment."""
    out = ["pymc"]
    for name in ("nutpie", "numpyro", "blackjax"):
        if _importable(name):
            out.append(name)
    return out


def _map_to_datatree(point: dict[str, np.ndarray], model: pm.Model) -> xr.DataTree:
    """Wrap a MAP point as a one-chain, one-draw posterior with the model's dims."""
    import arviz as az

    data = {}
    dims = {}
    coords = {k: np.asarray(v) for k, v in model.coords.items()}
    for name, value in point.items():
        if name not in model.named_vars:
            continue
        value = np.asarray(value)
        data[name] = value[None, None, ...]
        var_dims = model.named_vars_to_dims.get(name)
        if var_dims is not None:
            dims[name] = [str(dd) for dd in var_dims]
    return az.from_dict({"posterior": data}, coords=coords, dims=dims)


def fit_model(
    model: pm.Model,
    *,
    method: Method = "nuts",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    nuts_sampler: NutsSampler | None = None,
    target_accept: float = 0.9,
    random_seed: int | np.random.Generator | None = None,
    progressbar: bool = True,
    vi_iterations: int = 30_000,
    **kwargs: Any,
) -> tuple[xr.DataTree, Any]:
    """Run inference on ``model``.

    Parameters
    ----------
    model : pymc.Model
        Model from :func:`~pybaycor.model.build_model`.
    method : {"nuts", "advi", "fullrank_advi", "smc", "pathfinder", "map"}
        Inference algorithm (see the module docstring).
    draws : int
        Posterior draws per chain (NUTS, SMC) or in total (variational methods).
    tune : int
        Warm-up iterations per chain (NUTS only).
    chains : int
        Number of chains (NUTS, SMC).
    nuts_sampler : {"pymc", "nutpie", "numpyro", "blackjax"}, optional
        NUTS implementation. ``None`` uses ``nutpie`` if installed, else ``pymc``.
    target_accept : float
        Target acceptance statistic for NUTS step-size adaptation.
    random_seed : int or numpy.random.Generator, optional
        Seed for reproducibility.
    progressbar : bool
        Show progress bars.
    vi_iterations : int
        Optimisation steps for the variational methods.
    **kwargs
        Passed through to the underlying PyMC function (:func:`pymc.sample`,
        :func:`pymc.fit`, :func:`pymc.sample_smc`, ...).

    Returns
    -------
    idata : xarray.DataTree
        The posterior (and sample statistics where available).
    extra : object or None
        Algorithm-specific object: the :class:`pymc.Approximation` for variational
        methods, the raw point for ``"map"``, ``None`` otherwise.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")

    if method == "nuts":
        if nuts_sampler is None:
            nuts_sampler = "nutpie" if _importable("nutpie") else "pymc"
        if nuts_sampler not in NUTS_SAMPLERS:
            raise ValueError(f"nuts_sampler must be one of {NUTS_SAMPLERS}; got {nuts_sampler!r}")
        if nuts_sampler != "pymc" and not _importable(nuts_sampler):
            extra = "jax" if nuts_sampler in ("numpyro", "blackjax") else nuts_sampler
            raise ImportError(
                f"nuts_sampler={nuts_sampler!r} requires the '{nuts_sampler}' package; "
                f"install pybaycor[{extra}]"
            )
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                nuts_sampler=nuts_sampler,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                **kwargs,
            )
        return idata, None

    if method in ("advi", "fullrank_advi"):
        if method == "advi":
            warnings.warn(
                "Mean-field ADVI under-estimates posterior correlations; prefer "
                "method='fullrank_advi' or NUTS for final results.",
                UserWarning,
                stacklevel=2,
            )
        with model:
            approx = pm.fit(
                n=vi_iterations,
                method=method,
                random_seed=random_seed,
                progressbar=progressbar,
                **kwargs,
            )
            idata = approx.sample(draws, random_seed=random_seed)
        return idata, approx

    if method == "smc":
        with model:
            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                **kwargs,
            )
        return idata, None

    if method == "pathfinder":
        if not _importable("pymc_extras"):
            raise ImportError("method='pathfinder' requires pymc-extras; install pybaycor[extras]")
        import pymc_extras as pmx

        with model:
            idata = pmx.fit(
                method="pathfinder",
                num_draws=draws,
                random_seed=random_seed,
                progressbar=progressbar,
                **kwargs,
            )
        return idata, None

    # method == "map"
    with model:
        point = pm.find_MAP(progressbar=progressbar, **kwargs)
    return _map_to_datatree(point, model), point
