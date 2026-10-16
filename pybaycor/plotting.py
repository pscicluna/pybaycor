"""Plotting helpers for :class:`~pybaycor.BayesianCorrelation`.

Three public functions are provided, each returning a :class:`matplotlib.figure.Figure`:

* :func:`plot_trace`: trace and marginal plots of the population parameters,
* :func:`plot_data`: scatter plots of every pair of variables with the input data (error
  bars, censoring arrows), the inferred true values and the posterior density ellipse,
* :func:`plot_corner`: a corner (pair) plot of the correlation coefficients.

The ArviZ-based functions use the ArviZ 1.x plotting API (``arviz_plots``), whose plotting
functions return a ``PlotCollection``/``PlotMatrix`` rather than matplotlib axes; the
figure is extracted from that object with :func:`_figure_from`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .data import CorrelationData
from .model import ModelInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_corner", "plot_data", "plot_trace"]

_SAMPLE_DIMS = ("chain", "draw")


# --------------------------------------------------------------------------- helpers
def _posterior(idata: Any) -> xr.Dataset:
    """Return the posterior group of ``idata`` as an :class:`xarray.Dataset`."""
    if isinstance(idata, xr.Dataset):
        return idata
    if isinstance(idata, xr.DataTree):
        node = idata["posterior"] if "posterior" in idata.children else idata
        return node.to_dataset()
    if hasattr(idata, "posterior"):  # legacy arviz.InferenceData
        return idata.posterior
    raise TypeError("idata must be an xarray.DataTree with a 'posterior' group")


def _sample_dims(da: xr.DataArray) -> list[str]:
    """Return the sample dimensions (``chain``/``draw``) present in ``da``."""
    return [d for d in _SAMPLE_DIMS if d in da.dims]


def _feature_names(post: xr.Dataset, info: ModelInfo | None = None) -> list[str]:
    """Variable names, from ``info`` if available, else from the ``feature`` coordinate."""
    if info is not None and info.names:
        return list(info.names)
    if "feature" in post.coords:
        return [str(v) for v in post["feature"].to_numpy()]
    return [f"x{j}" for j in range(post.sizes.get("feature", 0))]


def _pair_dataset(post: xr.Dataset, names: Sequence[str], fmt: str) -> xr.Dataset:
    """One scalar variable per off-diagonal (``i < j``) element of ``chol_corr``.

    The variable for the pair ``(i, j)`` is named ``fmt.format(a=names[i], b=names[j])``
    and has only the sample dimensions.
    """
    corr = post["chol_corr"]
    n = corr.sizes["feature"]
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            da = corr.isel(feature=i, feature2=j, drop=True)
            out[fmt.format(a=names[i], b=names[j])] = da
    return xr.Dataset(out)


def _figure_from(pc: Any) -> Figure:
    """Extract the matplotlib figure from an ArviZ ``PlotCollection``/``PlotMatrix``.

    Falls back to the figure of the first axes for the legacy array-of-axes return type.
    """
    import matplotlib.pyplot as plt

    viz = getattr(pc, "viz", None)
    if viz is not None:
        try:
            fig = viz["figure"].item()
            if fig is not None:
                return fig
        except (KeyError, AttributeError, ValueError):
            pass
        try:  # older layouts stored the figure as an attribute of a child node
            for child in viz.children.values():
                if "figure" in child.data_vars:
                    return child["figure"].item()
        except AttributeError:
            pass
    axes = np.asarray(pc, dtype=object).ravel()
    for ax in axes:
        if hasattr(ax, "figure"):
            return ax.figure
    return plt.gcf()


def _finish(fig: Figure, plotfile: str | None, show: bool) -> Figure:
    """Save and/or show ``fig`` and return it."""
    if plotfile:
        fig.savefig(plotfile, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return fig


# --------------------------------------------------------------------------- trace plot
def plot_trace(
    idata: Any,
    var_names: Sequence[str] | None = None,
    *,
    plotfile: str | None = None,
    show: bool = False,
    **kwargs: Any,
) -> Figure:
    """Trace and marginal-density plots of the population parameters.

    The correlation matrix ``chol_corr`` is expanded into one scalar variable per pair
    of variables (``corr[a~b]`` for the strictly upper-triangular elements) so that the
    constant diagonal and the duplicated lower triangle are not plotted; per-variable
    vectors such as ``mu`` are likewise expanded into one row per variable (``mu[a]``).

    Parameters
    ----------
    idata : xarray.DataTree
        The posterior, as returned by :meth:`~pybaycor.BayesianCorrelation.fit`.
    var_names : sequence of str, optional
        Trace variables to plot. Defaults to ``mu``, ``chol_corr`` and, for robust
        models, ``nu``.
    plotfile : str, optional
        Save the figure to this path.
    show : bool, default False
        Call :func:`matplotlib.pyplot.show`.
    **kwargs
        Passed to :func:`arviz.plot_trace_dist` (or :func:`arviz.plot_trace`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import arviz as az

    post = _posterior(idata)
    if var_names is None:
        var_names = ["mu", "chol_corr"] + (["nu"] if "nu" in post else [])

    names = _feature_names(post)
    pieces: dict[str, xr.DataArray] = {}
    for var in var_names:
        if var == "chol_corr":
            pieces.update(_pair_dataset(post, names, "corr[{a}~{b}]").data_vars)
        elif var not in post:
            raise KeyError(f"Variable {var!r} is not in the posterior")
        elif "feature" in post[var].dims and post[var].ndim == len(_sample_dims(post[var])) + 1:
            # One labelled row per variable instead of overlaid, unlabelled colours.
            for k, name in enumerate(names):
                pieces[f"{var}[{name}]"] = post[var].isel(feature=k, drop=True)
        else:
            pieces[var] = post[var]
    ds = xr.Dataset(pieces)

    kwargs.setdefault("backend", "matplotlib")
    plotter = getattr(az, "plot_trace_dist", None)
    if plotter is None:  # pragma: no cover - older ArviZ
        plotter = az.plot_trace
    pc = plotter(ds, **kwargs)
    return _finish(_figure_from(pc), plotfile, show)


# --------------------------------------------------------------------------- data plot
def _ellipse_params(
    mu: NDArray[np.float64], cov: NDArray[np.float64], prob: float
) -> tuple[tuple[float, float], float, float, float]:
    """Centre, width, height and angle (degrees) of the ``prob`` ellipse of a 2-D Gaussian."""
    from scipy.stats import chi2

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    angle = float(np.degrees(np.arctan2(*eigvecs[:, 0][::-1])))
    semi = np.sqrt(chi2.ppf(prob, 2) * eigvals)
    return (float(mu[0]), float(mu[1])), 2.0 * float(semi[0]), 2.0 * float(semi[1]), angle


def _plot_positions(data: CorrelationData, col: int) -> NDArray[np.float64]:
    """Return the plotting position of each entry of variable ``col``.

    Interval-censored entries, whose stored value is the upper bound, are placed at the
    midpoint of their interval.
    """
    pos = data.values[:, col].copy()
    iv = data.is_interval[:, col]
    pos[iv] = 0.5 * (data.lower[iv, col] + data.upper[iv, col])
    return pos


def _arrow_length_points(ax: Axes, frac: float = 0.05) -> float:
    """Length of a censoring arrow: ``frac`` of the axes width, in points (at least 12)."""
    fig = ax.figure
    width_in = ax.get_position().width * fig.get_figwidth()
    return max(12.0, frac * width_in * 72.0)


def _draw_censoring(
    ax: Axes,
    data: CorrelationData,
    i: int,
    j: int,
    color: str,
) -> None:
    """Arrows for one-sided limits and capped segments for interval-censored entries."""
    x = _plot_positions(data, i)
    y = _plot_positions(data, j)
    length = _arrow_length_points(ax)
    arrow = {"arrowstyle": "<-", "color": color, "lw": 1.0, "shrinkA": 0, "shrinkB": 0}
    # (axis index, mask, offset direction in points)
    specs = (
        (data.is_upper_limit[:, i], (-length, 0.0)),  # x upper limit: point left
        (data.is_lower_limit[:, i], (length, 0.0)),  # x lower limit: point right
        (data.is_upper_limit[:, j], (0.0, -length)),  # y upper limit: point down
        (data.is_lower_limit[:, j], (0.0, length)),  # y lower limit: point up
    )
    for mask, offset in specs:
        for k in np.nonzero(mask)[0]:
            ax.annotate(
                "",
                xy=(x[k], y[k]),
                xycoords="data",
                xytext=offset,
                textcoords="offset points",
                arrowprops=arrow,
                annotation_clip=False,
            )
    for axis, mask in ((0, data.is_interval[:, i]), (1, data.is_interval[:, j])):
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            continue
        col = i if axis == 0 else j
        lo = data.lower[idx, col]
        hi = data.upper[idx, col]
        mid = 0.5 * (lo + hi)
        err = np.vstack((mid - lo, hi - mid))
        if axis == 0:
            cont = ax.errorbar(mid, y[idx], xerr=err, fmt="none", ecolor=color, capsize=3, lw=1)
        else:
            cont = ax.errorbar(x[idx], mid, yerr=err, fmt="none", ecolor=color, capsize=3, lw=1)
        for line in cont[2]:
            line.set_linestyle("--")


@dataclass
class _PosteriorSummary:
    """Posterior quantities needed by every panel of :func:`plot_data`, computed once."""

    mu: NDArray[np.float64]  # posterior mean of mu, (n_dim,)
    cov: NDArray[np.float64]  # posterior mean of cov, (n_dim, n_dim)
    vals_mean: NDArray[np.float64] | None  # posterior mean of vals, (n_points, n_dim)
    vals_sd: NDArray[np.float64] | None
    mu_draws: NDArray[np.float64] | None  # (n_draws, n_dim) subset for the extra ellipses
    cov_draws: NDArray[np.float64] | None

    @classmethod
    def from_posterior(
        cls, post: xr.Dataset, *, with_vals: bool, n_ellipses: int, rng: np.random.Generator
    ) -> _PosteriorSummary:
        sdims = _sample_dims(post["mu"])
        mu = post["mu"].mean(sdims).to_numpy()
        cov = post["cov"].mean(sdims).to_numpy()
        vals_mean = vals_sd = None
        if with_vals and "vals" in post:
            vals_mean = post["vals"].mean(sdims).to_numpy()
            vals_sd = post["vals"].std(sdims).to_numpy()
        mu_draws = cov_draws = None
        if n_ellipses > 0:
            mu_s = post["mu"].stack(sample=sdims).transpose("sample", ...)
            cov_s = post["cov"].stack(sample=sdims).transpose("sample", ...)
            n_samples = mu_s.sizes["sample"]
            picks = np.sort(rng.choice(n_samples, size=min(n_ellipses, n_samples), replace=False))
            mu_draws = mu_s.isel(sample=picks).to_numpy()
            cov_draws = cov_s.isel(sample=picks).to_numpy()
        return cls(mu, cov, vals_mean, vals_sd, mu_draws, cov_draws)


def _draw_panel(
    ax: Axes,
    data: CorrelationData,
    summ: _PosteriorSummary,
    info: ModelInfo,
    i: int,
    j: int,
    *,
    plot_input: bool,
    plot_fitted: bool,
    prob: float,
    color_input: str,
    color_fitted: str,
    scatter_kwargs: dict[str, Any],
) -> None:
    """Draw the pair (x = variable ``i``, y = variable ``j``) into ``ax``."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Ellipse

    x = _plot_positions(data, i)
    y = _plot_positions(data, j)

    if plot_input:
        censored = data.is_censored[:, i] | data.is_censored[:, j]
        # User keyword arguments override the defaults, including the colour.
        kw = {"s": 22, "zorder": 3, "color": color_input, **scatter_kwargs}
        if np.any(~censored):
            ax.scatter(x[~censored], y[~censored], label="data", **kw)
        if np.any(censored):
            kw_open = dict(kw)
            edge = kw_open.pop("color")
            kw_open.setdefault("facecolors", "none")
            kw_open.setdefault("edgecolors", edge)
            ax.scatter(x[censored], y[censored], label="censored (limit)", **kw_open)
        unc = data.is_uncertain[:, i] | data.is_uncertain[:, j]
        if np.any(unc):
            xerr = np.vstack((data.sigma_lower[unc, i], data.sigma_upper[unc, i]))
            yerr = np.vstack((data.sigma_lower[unc, j], data.sigma_upper[unc, j]))
            ax.errorbar(
                x[unc],
                y[unc],
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=color_input,
                elinewidth=0.8,
                alpha=0.7,
                zorder=2,
            )
        if data.has_censoring:
            _draw_censoring(ax, data, i, j, color_input)

    if plot_fitted and info.has_latent and summ.vals_mean is not None:
        mean = summ.vals_mean
        sd = summ.vals_sd
        latent = data.is_latent[:, i] | data.is_latent[:, j]
        if np.any(latent):
            mx, my = mean[latent, i], mean[latent, j]
            if plot_input:
                segs = np.stack(
                    [np.column_stack((x[latent], y[latent])), np.column_stack((mx, my))], axis=1
                )
                ax.add_collection(
                    LineCollection(segs, colors="0.6", linewidths=0.6, alpha=0.8, zorder=1)
                )
            ax.errorbar(
                mx,
                my,
                xerr=sd[latent, i],
                yerr=sd[latent, j],
                fmt="o",
                ms=3.5,
                color=color_fitted,
                ecolor=color_fitted,
                elinewidth=0.8,
                alpha=0.6,
                zorder=4,
                label="inferred true values",
            )

    # Posterior-mean density ellipse (+ optional draws to show its uncertainty).
    sel = [i, j]
    centre, width, height, angle = _ellipse_params(summ.mu[sel], summ.cov[np.ix_(sel, sel)], prob)
    if summ.mu_draws is not None and summ.cov_draws is not None:
        for mu_k, cov_k in zip(summ.mu_draws, summ.cov_draws, strict=True):
            c, w, h, a = _ellipse_params(mu_k[sel], cov_k[np.ix_(sel, sel)], prob)
            ax.add_patch(
                Ellipse(c, w, h, angle=a, fill=False, ec=color_fitted, lw=0.5, alpha=0.3, zorder=0)
            )
    label = f"{prob:.0%} region (Gaussian approx.)" if info.robust else f"{prob:.0%} density region"
    ax.add_patch(
        Ellipse(
            centre,
            width,
            height,
            angle=angle,
            fc=color_fitted,
            ec=color_fitted,
            alpha=0.25,
            lw=1.5,
            zorder=0,
            label=label,
        )
    )
    ax.autoscale_view()
    ax.margins(0.08)


def plot_data(
    data: CorrelationData,
    idata: Any,
    info: ModelInfo,
    *,
    plot_input: bool = True,
    plot_fitted: bool = True,
    prob: float = 0.95,
    n_ellipses: int = 0,
    axes: Axes | NDArray[Any] | None = None,
    plotfile: str | None = None,
    show: bool = False,
    color_input: str = "C0",
    color_fitted: str = "C1",
    random_seed: int | None = 0,
    **kwargs: Any,
) -> Figure:
    """Scatter plots of every pair of variables with the fitted density ellipse.

    For two variables a single panel is drawn; for more, a lower-triangular grid where
    the panel in row ``r`` and column ``c`` (``c <= r``) shows variable ``c`` on the x axis
    against variable ``r + 1`` on the y axis.

    In every panel the input data are drawn as points with (asymmetric) error bars for
    uncertain entries; censored entries are drawn as open markers with an arrow pointing
    in the direction of the allowed values (down for a *y* upper limit, up for a lower
    limit, left/right for *x*) or, for interval censoring, a dashed segment spanning the
    interval. For latent models the posterior mean and standard deviation of the inferred
    true values are overlaid and joined to the input points by thin grey lines. The
    ellipse containing probability ``prob`` of the fitted Gaussian with the posterior
    mean ``mu`` and ``cov`` is shaded; for robust (Student-t) models this is the
    corresponding region of a Gaussian with the fitted scale matrix.

    Parameters
    ----------
    data : CorrelationData
        The input data.
    idata : xarray.DataTree
        The posterior from :meth:`~pybaycor.BayesianCorrelation.fit`.
    info : ModelInfo
        How the model was built.
    plot_input : bool, default True
        Draw the input data.
    plot_fitted : bool, default True
        Draw the inferred true values (latent models only).
    prob : float, default 0.95
        Probability content of the ellipse.
    n_ellipses : int, default 0
        Also draw this many thin ellipses from random posterior draws to visualise the
        uncertainty of the fitted distribution.
    axes : matplotlib.axes.Axes or array of Axes, optional
        Draw into these axes: a single ``Axes`` for two variables, or an array of shape
        ``(n_dim - 1, n_dim - 1)`` for more. A new figure is created when omitted.
    plotfile : str, optional
        Save the figure to this path.
    show : bool, default False
        Call :func:`matplotlib.pyplot.show`.
    color_input, color_fitted : str
        Colours of the input data and of the inferred values/ellipse. A ``color=`` entry
        in ``kwargs`` overrides ``color_input`` for the input points.
    random_seed : int, optional
        Seed for the choice of posterior draws when ``n_ellipses > 0``.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.scatter` for the input data.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    post = _posterior(idata)
    n_dim = data.n_dim
    names = list(data.names)
    rng = np.random.default_rng(random_seed)
    summ = _PosteriorSummary.from_posterior(
        post, with_vals=plot_fitted and info.has_latent, n_ellipses=n_ellipses, rng=rng
    )
    common = {
        "plot_input": plot_input,
        "plot_fitted": plot_fitted,
        "prob": prob,
        "color_input": color_input,
        "color_fitted": color_fitted,
        "scatter_kwargs": kwargs,
    }

    if n_dim == 2:
        if axes is None:
            fig, ax = plt.subplots(figsize=(5.5, 4.8), layout="constrained")
        else:
            ax = np.asarray(axes, dtype=object).ravel()[0]
            fig = ax.figure
        _draw_panel(ax, data, summ, info, 0, 1, **common)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.legend(loc="best", fontsize="small")
        return _finish(fig, plotfile, show)

    m = n_dim - 1
    if axes is None:
        fig, grid = plt.subplots(
            m,
            m,
            figsize=(3.2 * m + 1, 3.0 * m + 1),
            sharex="col",
            sharey="row",
            layout="constrained",
            squeeze=False,
        )
    else:
        grid = np.asarray(axes, dtype=object).reshape(m, m)
        fig = grid[0, 0].figure
    for r in range(m):
        for c in range(m):
            ax = grid[r, c]
            if c > r:
                ax.set_axis_off()
                continue
            _draw_panel(ax, data, summ, info, c, r + 1, **common)
            if r == m - 1:
                ax.set_xlabel(names[c])
            if c == 0:
                ax.set_ylabel(names[r + 1])
    # One legend entry per distinct label, collected over all panels (a given kind of
    # entry, e.g. a censored point, may only occur in some of them).
    entries: dict[str, Any] = {}
    for ax in grid[np.tril_indices(m)]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    if entries:
        grid[0, m - 1].legend(
            entries.values(), entries.keys(), loc="center", fontsize="small", frameon=False
        )
    return _finish(fig, plotfile, show)


# --------------------------------------------------------------------------- corner plot
def plot_corner(
    idata: Any,
    info: ModelInfo,
    *,
    point_estimate: str = "mean",
    plotfile: str | None = None,
    show: bool = False,
    **kwargs: Any,
) -> Figure:
    """Corner (pair) plot of the correlation coefficients.

    The off-diagonal elements of the correlation matrix (``i < j``, named
    ``"{names[i]}~{names[j]}"``) are plotted against each other with their marginal
    densities on the diagonal, together with ``nu`` for robust models and the
    ``error_scale`` factors when present. With a single coefficient (two variables, no
    extra parameters) only its marginal density is drawn.

    Parameters
    ----------
    idata : xarray.DataTree
        The posterior from :meth:`~pybaycor.BayesianCorrelation.fit`.
    info : ModelInfo
        How the model was built (used for the variable names and the model options).
    point_estimate : {"mean", "median", "mode"}, default "mean"
        Point estimate marked on the marginal densities.
    plotfile : str, optional
        Save the figure to this path.
    show : bool, default False
        Call :func:`matplotlib.pyplot.show`.
    **kwargs
        Passed to :func:`arviz.plot_pair` (or :func:`arviz.plot_dist` for a single
        variable), e.g. ``marginal_kind="hist"`` or ``visuals={"contour": True}``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import arviz as az
    from arviz_base import rc_context

    post = _posterior(idata)
    names = _feature_names(post, info)
    pieces = dict(_pair_dataset(post, names, "{a}~{b}").data_vars)
    if info.robust and "nu" in post:
        pieces["nu"] = post["nu"]
    if info.error_scale and "error_scale" in post:
        es = post["error_scale"]
        for k, name in enumerate(names):
            pieces[f"error_scale[{name}]"] = es.isel(feature=k, drop=True)
    ds = xr.Dataset(pieces)

    kwargs.setdefault("backend", "matplotlib")
    with rc_context({"stats.point_estimate": point_estimate}):
        if len(ds.data_vars) == 1:
            kwargs.setdefault("point_estimate", point_estimate)
            pc = az.plot_dist(ds, **kwargs)
        else:
            kwargs.setdefault("marginal", True)
            kwargs.setdefault("triangle", "lower")
            visuals = dict(kwargs.pop("visuals", None) or {})
            try:
                pc = az.plot_pair(ds, visuals={"point_estimate": True, **visuals}, **kwargs)
            except (TypeError, ValueError, KeyError):  # marker unsupported by this ArviZ
                pc = az.plot_pair(ds, visuals=visuals or None, **kwargs)
    return _finish(_figure_from(pc), plotfile, show)
