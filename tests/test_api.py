"""Tests of the :class:`pybaycor.BayesianCorrelation` API (no MCMC: MAP fits only)."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr

import pybaycor as pbc
from conftest import make_gaussian_data
from pybaycor import (
    BayesianCorrelation,
    CorrelationData,
    HierarchicalBayesianCorrelation,
    HierarchicalRobustBayesianCorrelation,
    RobustBayesianCorrelation,
)

N = 30
RHO = 0.6


@pytest.fixture(scope="module", autouse=True)
def fast_compile():
    """No MCMC here: skip the C compilation of the few MAP fits."""
    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        yield


@pytest.fixture(scope="module")
def xy():
    return make_gaussian_data(N, RHO, seed=11)


@pytest.fixture(scope="module")
def map_observed(xy):
    bc = BayesianCorrelation(xy, names=["a", "b"])
    bc.fit(method="map", progressbar=False)
    return bc


@pytest.fixture(scope="module")
def map_latent(xy):
    bc = BayesianCorrelation(xy, sigma=0.25, names=["a", "b"])
    bc.fit(method="map", progressbar=False)
    return bc


# ----------------------------------------------------------------- construction
class TestConstruction:
    def test_from_data(self, xy):
        bc = BayesianCorrelation(xy)
        assert bc.ndim == 2
        assert bc.npoints == N
        assert bc.names == ["x0", "x1"]
        assert isinstance(bc.data, CorrelationData)
        assert bc.info.case == "observed"
        assert not bc.fitted
        assert bc.trace is None and bc.idata is None
        assert bc.summary is None
        assert bc.transform is bc.info.transform

    def test_from_xy(self, xy):
        bc = BayesianCorrelation(x=xy[:, 0], y=xy[:, 1])
        assert bc.names == ["x", "y"]
        np.testing.assert_array_equal(bc.data.values, xy)
        bc = BayesianCorrelation(x=xy[:, 0], y=xy[:, 1], names=["u", "v"])
        assert bc.names == ["u", "v"]

    def test_missing_input_raises(self, xy):
        with pytest.raises(ValueError, match="Either data must be given"):
            BayesianCorrelation()
        with pytest.raises(ValueError, match="Either data must be given"):
            BayesianCorrelation(x=xy[:, 0])

    def test_correlation_data_instance(self, xy):
        cd = CorrelationData(xy, sigma=0.2, names=["p", "q"])
        bc = BayesianCorrelation(cd)
        assert bc.data is cd
        assert bc.names == ["p", "q"]
        assert bc.info.case == "latent"

    def test_ndim_mismatch_raises(self, xy):
        with pytest.raises(ValueError, match="same number of features and ndim"):
            BayesianCorrelation(xy, ndim=3)
        assert BayesianCorrelation(xy, ndim=2).ndim == 2

    def test_options_forwarded(self, xy):
        bc = BayesianCorrelation(xy, sigma=0.2, robust=True, error_scale=True, priors={"eta": 3.0})
        assert bc.robust and bc.info.robust
        assert bc.error_scale and bc.info.error_scale
        assert bc.info.priors["eta"] == 3.0
        assert bc.priors == {"eta": 3.0}
        assert "nu" in bc.model.named_vars
        assert "error_scale" in bc.model.named_vars

    def test_censoring_and_bounds_forwarded(self, xy):
        cens = np.zeros(xy.shape, dtype=int)
        cens[:3, 1] = -1
        lo = np.full(xy.shape, np.nan)
        lo[4, 0] = xy[4, 0] - 1
        hi = np.full(xy.shape, np.nan)
        hi[4, 0] = xy[4, 0] + 1
        bc = BayesianCorrelation(xy, censoring=cens, lower=lo, upper=hi)
        assert bc.data.n_censored == 4
        assert bc.info.case == "augmented"

    def test_unknown_prior_raises(self, xy):
        with pytest.raises(ValueError, match="Unknown prior keys"):
            BayesianCorrelation(xy, priors={"nope": 1})

    def test_deprecated_priors_warn_and_disable_standardisation(self, xy):
        with pytest.warns(DeprecationWarning, match="mu_prior/sigma_prior are deprecated"):
            bc = BayesianCorrelation(xy, mu_prior=[1.0, 3.0], sigma_prior=4.0)
        assert bc.standardize is False
        assert bc.info.priors["mu"] == (1.0, 3.0)
        assert bc.info.priors["sigma"] == 4.0
        np.testing.assert_array_equal(bc.transform.loc, 0.0)
        np.testing.assert_array_equal(bc.transform.scale, 1.0)

    def test_deprecated_sigma_prior_alone(self, xy):
        with pytest.warns(DeprecationWarning):
            bc = BayesianCorrelation(xy, sigma_prior=[2.0])
        assert bc.info.priors["sigma"] == 2.0
        assert bc.info.priors["mu"] == pbc.DEFAULT_PRIORS["mu"]
        assert not bc.standardize

    def test_explicit_priors_win_over_deprecated(self, xy):
        with pytest.warns(DeprecationWarning):
            bc = BayesianCorrelation(xy, priors={"sigma": 9.0}, sigma_prior=4.0)
        assert bc.info.priors["sigma"] == 9.0

    def test_standardize_false(self, xy):
        bc = BayesianCorrelation(xy, standardize=False)
        assert not bc.standardize
        np.testing.assert_array_equal(bc.transform.scale, 1.0)


# ------------------------------------------------------------------- strings
class TestStrings:
    def test_repr(self, xy):
        bc = BayesianCorrelation(xy)
        r = repr(bc)
        assert r.startswith("<BayesianCorrelation Gaussian")
        assert f"{N} points x 2 variables" in r
        assert "observed model" in r
        assert "not fitted" in r
        assert "robust Student-t" in repr(BayesianCorrelation(xy, robust=True))

    def test_repr_fitted(self, map_observed):
        assert repr(map_observed).endswith("fitted>")
        assert "not fitted" not in repr(map_observed)

    def test_describe(self, xy):
        bc = BayesianCorrelation(xy, sigma=0.2, robust=True, error_scale=True, names=["a", "b"])
        text = bc.describe()
        assert text.startswith(bc.data.describe())
        assert f"Model: latent, {2 * N} latent values" in text
        assert "robust (Student-t)" in text
        assert "error-scale factors" in text
        assert "standardised" in text
        plain = BayesianCorrelation(xy, standardize=False).describe()
        assert "Model: observed, 0 latent values" in plain
        assert "standardised" not in plain


# ----------------------------------------------------------------- before fit
class TestRequireFit:
    @pytest.mark.parametrize(
        "call",
        [
            lambda bc: bc.summarise(printout=False),
            lambda bc: bc.summarize(printout=False),
            lambda bc: bc.data_summary(printout=False),
            lambda bc: bc.correlation_summary(),
            lambda bc: bc.correlation(),
            lambda bc: bc.bayes_factor(),
            lambda bc: bc.inferred_values(),
            lambda bc: bc.null_pvalue(n_sim=1),
        ],
    )
    def test_methods_raise_before_fit(self, xy, call):
        bc = BayesianCorrelation(xy)
        with pytest.raises(RuntimeError, match="run fit\\(\\)"):
            call(bc)


# ------------------------------------------------------------------ fit (MAP)
class TestMapFit:
    def test_returns_datatree(self, map_observed):
        idata = map_observed.trace
        assert isinstance(idata, xr.DataTree)
        assert map_observed.idata is idata
        assert map_observed.fitted
        post = idata["posterior"]
        assert "chol_corr" in post.data_vars
        assert post["chol_corr"].dims == ("chain", "draw", "feature", "feature2")
        assert post["chol_corr"].shape == (1, 1, 2, 2)
        assert list(post["feature"].values) == ["a", "b"]
        for name in ("mu", "chol_stds", "cov"):
            assert name in post.data_vars
        assert map_observed.fit_kwargs["method"] == "map"
        assert map_observed.approx is not None  # the raw MAP point

    def test_map_is_sensible(self, map_observed, xy):
        corr = float(map_observed.trace["posterior"]["chol_corr"].values[0, 0, 0, 1])
        naive = np.corrcoef(xy.T)[0, 1]
        assert abs(corr - naive) < 0.15
        mu = map_observed.trace["posterior"]["mu"].values[0, 0]
        np.testing.assert_allclose(mu, xy.mean(axis=0), atol=0.3)

    def test_summarise(self, map_observed):
        df = map_observed.summarise(printout=False)
        assert isinstance(df, pd.DataFrame)
        assert map_observed.summary is df
        assert "chol_corr[a, b]" in df.index
        assert "mu[a]" in df.index and "chol_stds[b]" in df.index
        assert "mean" in df.columns
        assert df.loc["chol_corr[a, a]", "mean"] == pytest.approx(1.0)

    def test_summarise_var_names_and_round(self, map_observed):
        df = map_observed.summarise(var_names=["mu"], printout=False, round_to=2)
        assert list(df.index) == ["mu[a]", "mu[b]"]

    def test_summarise_prints(self, map_observed, capsys):
        map_observed.summarise()
        assert "chol_corr" in capsys.readouterr().out

    def test_summarize_alias(self):
        assert BayesianCorrelation.summarize is BayesianCorrelation.summarise

    def test_correlation_summary(self, map_observed):
        df = map_observed.correlation_summary()
        assert list(df.index) == ["a~b"]
        assert df.loc["a~b", "sd"] == 0.0
        assert df.loc["a~b", "mean"] == pytest.approx(map_observed.correlation()[0])
        assert np.isnan(df.loc["a~b", "bf10"])  # degenerate single draw
        df2 = map_observed.correlation_summary(names=["u", "v"], ci_kind="eti")
        assert list(df2.index) == ["u~v"]
        assert "eti_3%" in df2.columns

    def test_correlation_and_bayes_factor(self, map_observed):
        s = map_observed.correlation()
        assert s.shape == (1,)
        np.testing.assert_allclose(map_observed.correlation(1, 0), s)
        assert np.isnan(map_observed.bayes_factor())

    def test_inferred_values_observed(self, map_observed, xy):
        mean, sd = map_observed.inferred_values()
        np.testing.assert_array_equal(mean, xy)
        np.testing.assert_array_equal(sd, 0.0)
        assert mean is not map_observed.data.values

    def test_data_summary_raises_for_observed(self, map_observed):
        with pytest.raises(ValueError, match="no latent values"):
            map_observed.data_summary(printout=False)

    def test_latent_fit(self, map_latent, xy):
        post = map_latent.trace["posterior"]
        assert "vals" in post.data_vars
        assert post["vals"].dims == ("chain", "draw", "point", "feature")
        mean, sd = map_latent.inferred_values()
        assert mean.shape == xy.shape and sd.shape == xy.shape
        # MAP of the true values shrinks slightly towards the data: close to it.
        np.testing.assert_allclose(mean, xy, atol=0.6)
        np.testing.assert_array_equal(sd, 0.0)

    def test_data_summary_latent(self, map_latent):
        df = map_latent.data_summary(printout=False)
        assert len(df) == 2 * N
        assert "vals[0, a]" in df.index

    def test_fit_summarise_flag(self, map_latent, capsys):
        out = map_latent.fit(method="map", progressbar=False, summarise=True)
        assert isinstance(out, tuple) and len(out) == 2
        idata, summary = out
        assert isinstance(idata, xr.DataTree)
        assert isinstance(summary, pd.DataFrame)
        assert map_latent.summary is summary
        assert "chol_corr" in capsys.readouterr().out

    def test_refit_resets_summary(self, map_observed):
        map_observed.summarise(printout=False)
        assert map_observed.summary is not None
        map_observed.fit(method="map", progressbar=False)
        assert map_observed.summary is None

    def test_steps_deprecated(self, xy):
        bc = BayesianCorrelation(xy)
        with pytest.warns(DeprecationWarning, match="steps= is deprecated"):
            bc.fit(method="map", progressbar=False, steps=5)
        assert bc.fit_kwargs["draws"] == 5


# -------------------------------------------------------------- fit arguments
class TestFitArguments:
    def test_invalid_method(self, xy):
        with pytest.raises(ValueError, match="method must be one of"):
            BayesianCorrelation(xy).fit(method="bogus")

    def test_invalid_nuts_sampler(self, xy):
        with pytest.raises(ValueError, match="nuts_sampler must be one of"):
            BayesianCorrelation(xy).fit(nuts_sampler="stan")

    @pytest.mark.parametrize("sampler", ["nutpie", "numpyro", "blackjax"])
    def test_missing_sampler_raises_import_error(self, xy, monkeypatch, sampler):
        monkeypatch.setattr(pbc.inference, "_importable", lambda name: False)
        with pytest.raises(ImportError, match=f"nuts_sampler='{sampler}' requires"):
            BayesianCorrelation(xy).fit(nuts_sampler=sampler)

    def test_missing_pathfinder_raises_import_error(self, xy, monkeypatch):
        monkeypatch.setattr(pbc.inference, "_importable", lambda name: False)
        with pytest.raises(ImportError, match="pymc-extras"):
            BayesianCorrelation(xy).fit(method="pathfinder")

    def test_fit_model_rejects_bad_method(self, xy):
        bc = BayesianCorrelation(xy)
        with pytest.raises(ValueError):
            pbc.fit_model(bc.model, method="bogus")


# ------------------------------------------------------------- legacy classes
class TestLegacyClasses:
    def test_robust(self, xy):
        bc = RobustBayesianCorrelation(xy)
        assert isinstance(bc, BayesianCorrelation)
        assert bc.robust
        assert "nu" in bc.model.named_vars
        assert RobustBayesianCorrelation(x=xy[:, 0], y=xy[:, 1]).names == ["x", "y"]

    def test_hierarchical_requires_sigma(self, xy):
        with pytest.raises(TypeError):
            HierarchicalBayesianCorrelation(xy)
        with pytest.raises(ValueError, match="sigma is required"):
            HierarchicalBayesianCorrelation(xy, None)

    def test_hierarchical_rejects_nonpositive_sigma(self, xy):
        with pytest.raises(ValueError, match="Uncertainties must be positive real numbers!"):
            HierarchicalBayesianCorrelation(xy, 0.0)
        sig = np.full(xy.shape, 0.2)
        sig[0, 0] = -1.0
        with pytest.raises(ValueError, match="positive real numbers"):
            HierarchicalBayesianCorrelation(xy, sig)

    def test_hierarchical(self, xy):
        bc = HierarchicalBayesianCorrelation(xy, 0.2, names=["a", "b"])
        assert not bc.robust
        assert bc.info.case == "latent"
        assert bc.data.has_uncertainties
        np.testing.assert_array_equal(bc.data.sigma, 0.2)

    def test_hierarchical_accepts_tuple_sigma(self, xy):
        bc = HierarchicalBayesianCorrelation(xy, (0.1, 0.3))
        assert bc.data.is_asymmetric.all()

    def test_hierarchical_robust(self, xy):
        bc = HierarchicalRobustBayesianCorrelation(xy, 0.2)
        assert bc.robust
        assert bc.info.case == "latent"
        assert "nu" in bc.model.named_vars
        with pytest.raises(ValueError, match="positive real numbers"):
            HierarchicalRobustBayesianCorrelation(xy, -0.2)


# ------------------------------------------------------------------- package
class TestPackage:
    def test_version(self):
        assert isinstance(pbc.__version__, str)
        assert pbc.__version__[0].isdigit()

    def test_all_exports(self):
        for name in pbc.__all__:
            assert hasattr(pbc, name), name

    def test_available_nuts_samplers(self):
        avail = pbc.available_nuts_samplers()
        assert avail[0] == "pymc"
        assert set(avail) <= set(pbc.inference.NUTS_SAMPLERS)

    def test_available_nuts_samplers_minimal(self, monkeypatch):
        monkeypatch.setattr(pbc.inference, "_importable", lambda name: False)
        assert pbc.available_nuts_samplers() == ["pymc"]

    def test_deprecated_module(self):
        sys.modules.pop("pybaycor.pybaycor", None)
        with pytest.warns(DeprecationWarning, match="pybaycor.pybaycor is deprecated"):
            legacy = importlib.import_module("pybaycor.pybaycor")
        assert legacy.BayesianCorrelation is BayesianCorrelation
        assert legacy.RobustBayesianCorrelation is RobustBayesianCorrelation
        assert legacy.HierarchicalBayesianCorrelation is HierarchicalBayesianCorrelation
        assert legacy.HierarchicalRobustBayesianCorrelation is HierarchicalRobustBayesianCorrelation
        assert set(legacy.__all__) == {
            "BayesianCorrelation",
            "RobustBayesianCorrelation",
            "HierarchicalBayesianCorrelation",
            "HierarchicalRobustBayesianCorrelation",
        }
