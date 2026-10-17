"""Rendering tests for the multi-variable plotting paths (3 variables, robust, error scale)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.collections import LineCollection

from conftest import make_gaussian_data_3d
from pybaycor import BayesianCorrelation

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@pytest.fixture(scope="module")
def fitted_3d():
    data, _ = make_gaussian_data_3d(30, seed=4)
    sig = np.zeros(data.shape)
    sig[:, 0] = np.linspace(0.1, 0.4, 30)  # uncertain first variable, exact others
    cens = np.zeros(data.shape, dtype=int)
    cens[:3, 2] = -1  # three upper limits on the third variable
    bc = BayesianCorrelation(
        data, sigma=sig, censoring=cens, robust=True, error_scale=True, names=["a", "b", "c"]
    )
    bc.fit(
        draws=60,
        tune=60,
        chains=1,
        nuts_sampler="pymc",
        progressbar=False,
        compute_convergence_checks=False,
        random_seed=4,
    )
    return bc


def test_plot_corner_pair_grid(fitted_3d):
    fig = fitted_3d.plot_corner()
    # 3 coefficients + nu + 3 error scales -> a 7x7 lower-triangular pair grid.
    assert len(fig.axes) >= 7
    plt.close(fig)
    fig = fitted_3d.plot_corner(point_estimate="median", marginal_kind="hist")
    plt.close(fig)


def test_plot_data_grid_and_legend(fitted_3d):
    fig = fitted_3d.plot_data(n_ellipses=3)
    axes = np.asarray(fig.axes)
    assert axes.size == 4  # (n_dim - 1) ** 2 panels
    legends = [ax.get_legend() for ax in axes if ax.get_legend() is not None]
    assert len(legends) == 1
    labels = {t.get_text() for t in legends[0].get_texts()}
    assert {"data", "censored (limit)", "inferred true values"} <= labels
    assert any("Gaussian approx" in lab for lab in labels)
    plt.close(fig)


def test_plot_data_into_user_axes(fitted_3d):
    fig, grid = plt.subplots(2, 2)
    out = fitted_3d.plot_data(axes=grid, plot_fitted=False)
    assert out is fig
    assert grid[0, 0].get_xlabel() == "" and grid[1, 0].get_xlabel() == "a"
    plt.close(fig)


def test_plot_data_colour_applies_to_error_bars_and_arrows(fitted_3d):
    fig = fitted_3d.plot_data(color="red", plot_fitted=False)
    ax = fig.axes[0]  # panel (0, 0): a vs b, with error bars on a
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert lines, "error bars expected"
    assert np.allclose(lines[0].get_color()[0][:3], [1.0, 0.0, 0.0])
    arrows = [a for a in fig.axes[2].get_children() if type(a).__name__ == "Annotation"]
    assert arrows and all(
        np.allclose(a.arrow_patch.get_edgecolor()[:3], [1.0, 0.0, 0.0]) for a in arrows
    )
    plt.close(fig)


def test_missing_coordinates_have_axis_rugs():
    from pybaycor.model import ModelInfo
    from pybaycor.plotting import _draw_panel, _PosteriorSummary

    from pybaycor import CorrelationData, Standardisation

    values = np.array([[np.nan, 0.5], [0.6, np.nan], [1.0, 1.2], [1.5, 1.0]])
    sigma = np.zeros(values.shape)
    sigma[2, 0] = 0.1
    data = CorrelationData(values, sigma=sigma)
    summary = _PosteriorSummary(
        np.zeros(2), np.eye(2), np.nan_to_num(values), np.zeros_like(values), None, None
    )
    info = ModelInfo(Standardisation.identity(2), False, False, "augmented")
    fig, ax = plt.subplots()
    _draw_panel(
        ax,
        data,
        summary,
        info,
        0,
        1,
        plot_input=True,
        plot_fitted=True,
        prob=0.95,
        color_input="black",
        color_fitted="blue",
        scatter_kwargs={},
    )
    assert {"missing x", "missing y"} <= {line.get_label() for line in ax.lines}
    for collection in ax.collections:
        if isinstance(collection, LineCollection):
            assert all(np.isfinite(segment).all() for segment in collection.get_segments())
    plt.close(fig)


def test_plot_trace_rows(fitted_3d):
    fig = fitted_3d.plot_trace()
    # mu[a..c], corr[a~b], corr[a~c], corr[b~c], nu, error_scale[a..c] -> 10 rows x 2
    assert len(fig.axes) == 20
    plt.close(fig)
