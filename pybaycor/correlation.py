"""User-facing classes for Bayesian correlation inference.

:class:`BayesianCorrelation` is the single entry point. The other classes exist for
backwards compatibility with pybaycor ≤ 0.2 and only pre-set its options.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from .data import CorrelationData, Standardisation
from .inference import Method, NutsSampler, fit_model
from .model import ModelInfo, build_model, set_model_values
from .stats import bayes_factor, correlation_samples, correlation_summary

__all__ = [
    "BayesianCorrelation",
    "HierarchicalBayesianCorrelation",
    "HierarchicalRobustBayesianCorrelation",
    "RobustBayesianCorrelation",
]


class BayesianCorrelation:
    """Infer the correlation matrix of a multivariate data set.

    The true values of the data are modelled with a multivariate Normal or, with
    ``robust=True``, a multivariate Student-t distribution whose correlation matrix has
    an LKJ prior. Each entry of the data set may be exact or carry a (possibly
    asymmetric) measurement uncertainty, and may be observed or censored (upper limit,
    lower limit or interval), independently of every other entry. See
    :class:`~pybaycor.data.CorrelationData` for the input conventions and
    :func:`~pybaycor.model.build_model` for the model.

    Parameters
    ----------
    data : array_like of shape (n_points, n_dim), optional
        The data. Either ``data`` or both ``x`` and ``y`` must be given.
    x, y : array_like of shape (n_points,), optional
        Convenience inputs for two-variable data sets.
    sigma : array_like, optional
        Symmetric measurement uncertainties: scalar, ``(n_dim,)`` (one per variable),
        or ``(n_points, n_dim)``. ``0``/``NaN`` entries are exact. Omit for data without
        measurement errors.
    sigma_lower, sigma_upper : array_like, optional
        Asymmetric uncertainties below and above each value (two-piece Normal errors),
        same shape rules as ``sigma``; give both, and not ``sigma``.
    x_err, y_err : array_like, optional
        With ``x``/``y``: symmetric uncertainties per axis (scalar or ``(n_points,)``).
        For asymmetric per-axis errors build the data with
        :meth:`CorrelationData.from_xy` and pass the result as ``data``.
    censoring : array_like, optional
        Per-entry censoring flags: ``0``/``"observed"``, ``-1``/``"upper"`` (value is an
        upper limit), ``1``/``"lower"`` (value is a lower limit), ``2``/``"interval"``.
        A boolean array marks upper limits.
    lower, upper : array_like of shape (n_points, n_dim), optional
        Explicit censoring bounds (``NaN``/``±inf`` = unbounded); any finite bound makes
        the entry censored.
    robust : bool, default False
        Use the multivariate Student-t likelihood (recommended in the presence of
        outliers); adds the ``nu`` parameter.
    names : sequence of str, optional
        Variable names used in summaries and plots.
    priors : dict, optional
        Hyper-parameter overrides, see :data:`~pybaycor.model.DEFAULT_PRIORS`.
    error_scale : bool, default False
        Infer a per-variable multiplicative correction to the quoted uncertainties.
    standardize : bool, default True
        Fit in standardised coordinates (recommended).
    ndim : int, optional
        Deprecated; the number of variables is taken from the data.
    mu_prior, sigma_prior : optional
        Deprecated pybaycor ≤ 0.2 prior specification: ``[mean, sd]`` (or a ``(2, n_dim)``
        array of per-variable means and sds) for the Normal prior on ``mu``, and
        ``beta`` (scalar or one per variable) for the half-Cauchy prior on the standard
        deviations, in data units. Giving either disables standardisation.

    Attributes
    ----------
    data : CorrelationData
        The validated data.
    model : pymc.Model
        The PyMC model.
    info : ModelInfo
        How the model was built.
    trace : xarray.DataTree or None
        The posterior, once :meth:`fit` has run (``idata`` is an alias).
    summary : pandas.DataFrame or None
        The last summary table produced.

    Examples
    --------
    >>> import numpy as np, pybaycor as pbc
    >>> rng = np.random.default_rng(0)
    >>> data = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=50)
    >>> bc = pbc.BayesianCorrelation(data, robust=True)
    >>> bc
    <BayesianCorrelation robust Student-t, 50 points x 2 variables, observed model, not fitted>
    >>> print(bc.describe())
    CorrelationData: 50 points x 2 variables
      x0: 0 uncertain
      x1: 0 uncertain
    Model: observed, 0 latent values, robust (Student-t), standardised

    Fitting runs the sampler, which takes a while:

    >>> idata = bc.fit(draws=500, tune=500)  # doctest: +SKIP
    >>> bc.correlation_summary()  # doctest: +SKIP
    """

    def __init__(
        self,
        data: ArrayLike | None = None,
        *,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        sigma: ArrayLike | None = None,
        sigma_lower: ArrayLike | None = None,
        sigma_upper: ArrayLike | None = None,
        x_err: ArrayLike | None = None,
        y_err: ArrayLike | None = None,
        censoring: ArrayLike | None = None,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        robust: bool = False,
        names: list[str] | None = None,
        priors: dict[str, Any] | None = None,
        error_scale: bool = False,
        standardize: bool = True,
        ndim: int | None = None,
        mu_prior: ArrayLike | None = None,
        sigma_prior: float | ArrayLike | None = None,
    ) -> None:
        if isinstance(data, CorrelationData):
            extras = {
                "x": x,
                "y": y,
                "sigma": sigma,
                "sigma_lower": sigma_lower,
                "sigma_upper": sigma_upper,
                "x_err": x_err,
                "y_err": y_err,
                "censoring": censoring,
                "lower": lower,
                "upper": upper,
                "names": names,
            }
            given = [k for k, v in extras.items() if v is not None]
            if given:
                raise ValueError(
                    "data is already a CorrelationData, so these arguments would be ignored: "
                    f"{given}; build the CorrelationData with them instead"
                )
            cdata = data
        else:
            if data is None:
                if x is None or y is None:
                    raise ValueError("Either data must be given as input, or both x and y")
                x = np.asarray(x, float).ravel()
                y = np.asarray(y, float).ravel()
                if x.shape != y.shape:
                    raise ValueError("x and y must have the same length")
                data = np.column_stack((x, y))
                if names is None:
                    names = ["x", "y"]
                if x_err is not None or y_err is not None:
                    if sigma is not None or sigma_lower is not None or sigma_upper is not None:
                        raise ValueError("Give either x_err/y_err or sigma, not both")
                    n = x.size
                    xe = np.asarray(0.0 if x_err is None else x_err, float)
                    ye = np.asarray(0.0 if y_err is None else y_err, float)
                    sigma = np.column_stack((np.broadcast_to(xe, (n,)), np.broadcast_to(ye, (n,))))
            elif x is not None or y is not None or x_err is not None or y_err is not None:
                raise ValueError("x/y/x_err/y_err cannot be combined with data")
            if ndim is not None:
                arr = np.asarray(data)
                if arr.ndim != 2 or arr.shape[1] != ndim:
                    raise ValueError("Data must have the same number of features and ndim")
            cdata = CorrelationData(
                data,
                sigma=sigma,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                censoring=censoring,
                lower=lower,
                upper=upper,
                names=names,
            )

        priors = dict(priors or {})
        if mu_prior is not None or sigma_prior is not None:
            warnings.warn(
                "mu_prior/sigma_prior are deprecated; use priors={'mu': (mean, sd), "
                "'sigma': beta}. They are interpreted in data units, so standardize=False "
                "is implied.",
                DeprecationWarning,
                stacklevel=2,
            )
            standardize = False
            if mu_prior is not None:
                mp = np.asarray(mu_prior, dtype=float)
                if mp.shape == (2,):
                    priors.setdefault("mu", (float(mp[0]), float(mp[1])))
                elif mp.ndim == 2 and mp.shape[0] == 2 and mp.shape[1] == cdata.n_dim:
                    priors.setdefault("mu", (mp[0], mp[1]))
                else:
                    raise ValueError(
                        "mu_prior must be [mean, sd] or a (2, n_dim) array of per-variable "
                        f"means and sds; got shape {mp.shape}"
                    )
            if sigma_prior is not None:
                sp = np.asarray(sigma_prior, dtype=float)
                if sp.size == 1:
                    priors.setdefault("sigma", float(sp.ravel()[0]))
                elif sp.shape == (cdata.n_dim,):
                    priors.setdefault("sigma", sp)
                else:
                    raise ValueError(
                        "sigma_prior must be a scalar or have one entry per variable; "
                        f"got shape {sp.shape}"
                    )

        self.data: CorrelationData = cdata
        self.robust = bool(robust)
        self.error_scale = bool(error_scale)
        self.standardize = bool(standardize)
        self.priors = priors
        self.model, self.info = build_model(
            cdata,
            robust=self.robust,
            priors=priors,
            error_scale=self.error_scale,
            standardize=self.standardize,
        )
        # build_model drops the error-scale factors when there are no uncertainties.
        self.error_scale = self.info.error_scale
        self.trace: xr.DataTree | None = None
        self.summary: pd.DataFrame | None = None
        self.approx: Any = None
        self.fit_kwargs: dict[str, Any] = {}
        self.fitted = False

    # ------------------------------------------------------------------ properties
    @property
    def ndim(self) -> int:
        """Number of variables."""
        return self.data.n_dim

    @property
    def npoints(self) -> int:
        """Number of data points."""
        return self.data.n_points

    @property
    def names(self) -> list[str]:
        """Variable names."""
        return list(self.data.names)

    @property
    def idata(self) -> xr.DataTree | None:
        """Alias of :attr:`trace`."""
        return self.trace

    @property
    def transform(self) -> Standardisation:
        """The standardisation applied before fitting."""
        return self.info.transform

    def __repr__(self) -> str:
        kind = "robust Student-t" if self.robust else "Gaussian"
        state = "fitted" if self.fitted else "not fitted"
        return (
            f"<BayesianCorrelation {kind}, {self.npoints} points x {self.ndim} variables, "
            f"{self.info.case} model, {state}>"
        )

    def describe(self) -> str:
        """Return a description of the data and model configuration."""
        extras = []
        if self.robust:
            extras.append("robust (Student-t)")
        if self.error_scale:
            extras.append("error-scale factors")
        if self.standardize:
            extras.append("standardised")
        return (
            self.data.describe()
            + f"\nModel: {self.info.case}, {self.data.n_latent} latent values"
            + (", " + ", ".join(extras) if extras else "")
        )

    # ------------------------------------------------------------------ inference
    def fit(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        *,
        method: Method = "nuts",
        nuts_sampler: NutsSampler | None = None,
        target_accept: float = 0.9,
        random_seed: int | np.random.Generator | None = None,
        progressbar: bool = True,
        summarise: bool = False,
        steps: int | None = None,
        **kwargs: Any,
    ) -> xr.DataTree | tuple[xr.DataTree, pd.DataFrame]:
        """Run inference and store the posterior in :attr:`trace`.

        Parameters
        ----------
        draws : int, default 1000
            Posterior draws per chain.
        tune : int, default 1000
            Warm-up iterations per chain (NUTS).
        chains : int, default 4
            Number of chains.
        method : {"nuts", "advi", "fullrank_advi", "smc", "pathfinder", "map"}
            Inference algorithm; see :mod:`pybaycor.inference`.
        nuts_sampler : {"pymc", "nutpie", "numpyro", "blackjax"}, optional
            NUTS implementation (``nutpie`` if installed, else ``pymc``). The JAX samplers
            use whatever accelerator ``jax`` is configured for.
        target_accept : float, default 0.9
            NUTS acceptance target.
        random_seed : int or Generator, optional
            Seed.
        progressbar : bool, default True
            Show progress.
        summarise : bool, default False
            Also print and return the summary table.
        steps : int, optional
            Deprecated alias of ``draws``.
        **kwargs
            Passed to the PyMC sampler.

        Returns
        -------
        xarray.DataTree, or (DataTree, DataFrame) if ``summarise`` is True
        """
        if steps is not None:
            warnings.warn("steps= is deprecated, use draws=", DeprecationWarning, stacklevel=2)
            draws = int(steps)
        self.trace, self.approx = fit_model(
            self.model,
            method=method,
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler=nuts_sampler,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            **kwargs,
        )
        self.fit_kwargs = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "method": method,
            "nuts_sampler": nuts_sampler,
            "target_accept": target_accept,
        }
        self.fitted = True
        self.summary = None
        if summarise:
            summary = self.summarise()
            return self.trace, summary
        return self.trace

    def _require_fit(self) -> xr.DataTree:
        if self.trace is None:
            raise RuntimeError("Please run fit() before using the results")
        return self.trace

    # ------------------------------------------------------------------ summaries
    def summarise(
        self,
        var_names: list[str] | None = None,
        *,
        printout: bool = True,
        round_to: int | str = 3,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Tabulate the posterior of the population parameters.

        Parameters
        ----------
        var_names : list of str, optional
            Variables to include; defaults to ``mu``, ``chol_stds``, ``chol_corr`` and,
            where present, ``nu`` and ``error_scale``.
        printout : bool, default True
            Print the table.
        round_to : int or "auto"
            Decimal places.
        **kwargs
            Passed to :func:`arviz.summary` (e.g. ``ci_prob``, ``ci_kind``).

        Returns
        -------
        pandas.DataFrame
        """
        import arviz as az

        idata = self._require_fit()
        if var_names is None:
            var_names = list(self.info.summary_vars)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # single-draw / single-chain
            summary = az.summary(idata, var_names=var_names, round_to=round_to, **kwargs)
        self.summary = summary
        if printout:
            print(summary)
        return summary

    summarize = summarise

    def data_summary(self, printout: bool = True, **kwargs: Any) -> pd.DataFrame:
        """Tabulate the inferred true values ``vals`` (latent models only)."""
        import arviz as az

        idata = self._require_fit()
        if not self.info.has_latent:
            raise ValueError("This model has no latent values: every entry is exact and observed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary = az.summary(idata, var_names=["vals"], **kwargs)
        if printout:
            print(summary)
        return summary

    def correlation_summary(self, **kwargs: Any) -> pd.DataFrame:
        """Per-pair posterior summary of the correlation coefficients.

        See :func:`pybaycor.stats.correlation_summary`; includes the posterior
        probability of a positive correlation and the Savage-Dickey Bayes factor
        against zero correlation.
        """
        idata = self._require_fit()
        kwargs.setdefault("eta", float(self.info.priors["eta"]))
        kwargs.setdefault("names", self.names)
        return correlation_summary(idata, **kwargs)

    def correlation(self, i: int = 0, j: int = 1) -> np.ndarray:
        """Posterior samples of the correlation between variables ``i`` and ``j``."""
        return correlation_samples(self._require_fit(), i, j)

    def bayes_factor(self, i: int = 0, j: int = 1, ref_val: float = 0.0) -> float:
        """Savage-Dickey Bayes factor ``BF10`` for ``corr[i, j] != ref_val``."""
        return bayes_factor(
            self.correlation(i, j),
            eta=float(self.info.priors["eta"]),
            n_dim=self.ndim,
            ref_val=ref_val,
        )

    def inferred_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of the true values (latent models).

        Returns
        -------
        mean, sd : ndarray of shape (n_points, n_dim)
        """
        idata = self._require_fit()
        if not self.info.has_latent:
            return self.data.values.copy(), np.zeros(self.data.shape)
        vals = idata["posterior"]["vals"]
        dims = [dd for dd in ("chain", "draw") if dd in vals.dims]
        return vals.mean(dims).to_numpy(), vals.std(dims).to_numpy()

    # ------------------------------------------------------------------ p-values
    def null_pvalue(
        self,
        i: int = 0,
        j: int = 1,
        *,
        n_sim: int = 50,
        statistic: str = "mean",
        random_seed: int | None = None,
        draws: int = 300,
        tune: int = 300,
        chains: int = 2,
        method: Method = "nuts",
        nuts_sampler: NutsSampler | None = None,
        progressbar: bool = False,
        return_null_distribution: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """Parametric-bootstrap p-value for the correlation between ``i`` and ``j``.

        The procedure follows issue #6: the model is refitted with the correlation
        fixed at zero, synthetic data sets with the same size, uncertainties and
        censoring pattern are drawn from that null posterior, each is refitted with the
        full model, and the p-value is the fraction of simulations whose summary
        statistic of ``|rho|`` is at least as large as the one observed.

        This is expensive (``n_sim`` complete fits). The Bayes factor and posterior
        probability in :meth:`correlation_summary` are cheaper alternatives.

        Parameters
        ----------
        i, j : int
            The pair of variables.
        n_sim : int
            Number of simulated data sets.
        statistic : {"mean", "median"}
            Posterior statistic of ``|rho|`` to compare.
        random_seed : int, optional
            Seed.
        draws, tune, chains, method, nuts_sampler, progressbar
            Settings for the fits of the simulated data sets.
        return_null_distribution : bool
            Also return the simulated statistics.

        Returns
        -------
        float, or (float, ndarray)
        """
        self._require_fit()
        rng = np.random.default_rng(random_seed)
        stat_fn = {"mean": np.mean, "median": np.median}[statistic]
        observed = float(stat_fn(np.abs(self.correlation(i, j))))

        null_model, _ = build_model(
            self.data,
            robust=self.robust,
            priors=self.priors,
            error_scale=self.error_scale,
            standardize=self.standardize,
            transform=self.info.transform,
            independent=True,
        )
        null_idata, _ = fit_model(
            null_model,
            method=method,
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler=nuts_sampler,
            random_seed=int(rng.integers(2**31)),
            progressbar=progressbar,
        )
        post = null_idata["posterior"]
        sample_dims = [dd for dd in ("chain", "draw") if dd in post["mu"].dims]
        mu_s = post["mu"].stack(sample=sample_dims).transpose("sample", ...).to_numpy()
        sd_s = post["chol_stds"].stack(sample=sample_dims).transpose("sample", ...).to_numpy()
        nu_s = post["nu"].stack(sample=sample_dims).to_numpy() if "nu" in post else None
        n_samp = mu_s.shape[0]

        d = self.data
        n, k = d.shape
        stats = np.empty(n_sim)
        sim_model: Any = None
        sim_info: ModelInfo | None = None
        for s in range(n_sim):
            idx = int(rng.integers(n_samp))
            true = rng.normal(size=(n, k)) * sd_s[idx] + mu_s[idx]
            if nu_s is not None:
                w = np.sqrt(nu_s[idx] / rng.chisquare(nu_s[idx], size=(n, 1)))
                true = mu_s[idx] + (true - mu_s[idx]) * w
            # Two-piece Normal measurement noise (reduces to Normal for symmetric errors).
            below = rng.uniform(size=(n, k)) < d.sigma_lower / np.maximum(
                d.sigma_lower + d.sigma_upper, 1e-300
            )
            noise = np.abs(rng.normal(size=(n, k)))
            sim = true + np.where(below, -noise * d.sigma_lower, noise * d.sigma_upper)
            # Keep the censoring pattern of the real data set.
            sim = np.where(d.is_censored, d.values, sim)
            sim_data = CorrelationData(
                sim,
                sigma_lower=d.sigma_lower,
                sigma_upper=d.sigma_upper,
                lower=np.where(np.isfinite(d.lower), d.lower, np.nan),
                upper=np.where(np.isfinite(d.upper), d.upper, np.nan),
                names=d.names,
            )
            if sim_model is None:
                # The simulated data sets share the uncertainty and censoring pattern of
                # the real data, so one model with mutable values serves all of them.
                sim_model, sim_info = build_model(
                    sim_data,
                    robust=self.robust,
                    priors=self.priors,
                    error_scale=self.error_scale,
                    standardize=self.standardize,
                    transform=self.info.transform,
                    mutable_values=True,
                )
            else:
                set_model_values(sim_model, sim_info, sim_data)
            sim_idata, _ = fit_model(
                sim_model,
                method=method,
                draws=draws,
                tune=tune,
                chains=chains,
                nuts_sampler=nuts_sampler,
                random_seed=int(rng.integers(2**31)),
                progressbar=progressbar,
            )
            stats[s] = stat_fn(np.abs(correlation_samples(sim_idata, i, j)))
        pvalue = float((np.sum(stats >= observed) + 1) / (n_sim + 1))
        if return_null_distribution:
            return pvalue, stats
        return pvalue

    # ------------------------------------------------------------------ plotting
    def plot_trace(
        self,
        plotfile: str | None = None,
        show: bool = False,
        var_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Trace and marginal plots of the population parameters.

        Parameters
        ----------
        plotfile : str, optional
            Save the figure to this file.
        show : bool, default False
            Call :func:`matplotlib.pyplot.show`.
        var_names : list of str, optional
            Trace variables to show; defaults to ``mu``, ``chol_corr`` and, where present,
            ``nu`` and ``error_scale``.
        **kwargs
            Passed to :func:`pybaycor.plotting.plot_trace`.
        """
        from . import plotting

        if var_names is None:
            var_names = [v for v in self.info.summary_vars if v != "chol_stds"]
        return plotting.plot_trace(
            self._require_fit(), var_names=var_names, plotfile=plotfile, show=show, **kwargs
        )

    def plot_data(
        self,
        plot_input: bool = True,
        plot_fitted: bool = True,
        plotfile: str | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Scatter plots of every pair of variables with the inferred 95% ellipse.

        Uncertain points carry error bars, censored points are drawn as arrows, and for
        latent models the inferred true values can be overlaid.

        Parameters
        ----------
        plot_input : bool, default True
            Draw the input data.
        plot_fitted : bool, default True
            Draw the inferred true values (latent models only).
        plotfile : str, optional
            Save the figure to this file.
        show : bool, default False
            Call :func:`matplotlib.pyplot.show`.
        **kwargs
            Passed to :func:`pybaycor.plotting.plot_data`.
        """
        from . import plotting

        return plotting.plot_data(
            self.data,
            self._require_fit(),
            self.info,
            plot_input=plot_input,
            plot_fitted=plot_fitted,
            plotfile=plotfile,
            show=show,
            **kwargs,
        )

    def plot_corner(
        self,
        point_estimate: str = "mean",
        plotfile: str | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Corner plot of the correlation coefficients (and ``nu`` for robust models).

        Parameters
        ----------
        point_estimate : {"mean", "median", "mode"}
            Point estimate to mark.
        plotfile : str, optional
            Save the figure to this file.
        show : bool, default False
            Call :func:`matplotlib.pyplot.show`.
        **kwargs
            Passed to :func:`pybaycor.plotting.plot_corner`.
        """
        from . import plotting

        return plotting.plot_corner(
            self._require_fit(),
            self.info,
            point_estimate=point_estimate,
            plotfile=plotfile,
            show=show,
            **kwargs,
        )


class RobustBayesianCorrelation(BayesianCorrelation):
    """:class:`BayesianCorrelation` with ``robust=True`` (multivariate Student-t).

    Kept for backwards compatibility; all arguments are forwarded.
    """

    def __init__(self, data: ArrayLike | None = None, **kwargs: Any) -> None:
        kwargs["robust"] = True
        super().__init__(data, **kwargs)


class HierarchicalBayesianCorrelation(BayesianCorrelation):
    """:class:`BayesianCorrelation` for data with measurement uncertainties.

    Kept for backwards compatibility: ``sigma`` is required. The same model is
    obtained from :class:`BayesianCorrelation` by passing ``sigma``.
    """

    def __init__(self, data: ArrayLike, sigma: ArrayLike, **kwargs: Any) -> None:
        asym = kwargs.get("sigma_lower") is not None and kwargs.get("sigma_upper") is not None
        if sigma is None and not asym:
            raise ValueError(
                "sigma is required for the hierarchical models (or give sigma_lower and "
                "sigma_upper for asymmetric uncertainties)"
            )
        for s in (sigma, kwargs.get("sigma_lower"), kwargs.get("sigma_upper")):
            if s is not None and np.any(np.asarray(s, dtype=float) <= 0):
                raise ValueError("Uncertainties must be positive real numbers!")
        super().__init__(data, sigma=sigma, **kwargs)


class HierarchicalRobustBayesianCorrelation(HierarchicalBayesianCorrelation):
    """:class:`HierarchicalBayesianCorrelation` with ``robust=True``."""

    def __init__(self, data: ArrayLike, sigma: ArrayLike, **kwargs: Any) -> None:
        kwargs["robust"] = True
        super().__init__(data, sigma, **kwargs)
