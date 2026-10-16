"""Tests for :mod:`pybaycor.stats`."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pybaycor.stats import (
    bayes_factor,
    correlation_samples,
    correlation_summary,
    lkj_marginal_density,
)
from scipy import stats as sps
from scipy.integrate import quad


# -------------------------------------------------------------------- LKJ prior
class TestLKJMarginal:
    @pytest.mark.parametrize("eta", [1.0, 2.0, 5.0, 0.5])
    @pytest.mark.parametrize("n_dim", [2, 3, 6])
    def test_integrates_to_one(self, eta, n_dim):
        total, _ = quad(lambda r: float(lkj_marginal_density(r, eta, n_dim)), -1.0, 1.0)
        assert total == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("eta", [1.0, 2.0, 5.0])
    def test_symmetric(self, eta):
        r = np.linspace(-0.95, 0.95, 39)
        d = lkj_marginal_density(r, eta, 3)
        np.testing.assert_allclose(d, d[::-1])
        assert (d >= 0).all()

    def test_uniform_for_eta_one_in_2d(self):
        # LKJ(1) on a 2x2 matrix is uniform on (-1, 1).
        r = np.linspace(-0.9, 0.9, 7)
        np.testing.assert_allclose(lkj_marginal_density(r, 1.0, 2), 0.5)

    def test_larger_eta_concentrates_at_zero(self):
        assert lkj_marginal_density(0.0, 5.0, 2) > lkj_marginal_density(0.0, 2.0, 2)
        assert lkj_marginal_density(0.8, 5.0, 2) < lkj_marginal_density(0.8, 2.0, 2)

    def test_matches_beta_formula(self):
        eta, n_dim = 2.0, 4
        a = eta - 1 + n_dim / 2
        r = np.array([-0.5, 0.0, 0.3])
        np.testing.assert_allclose(
            lkj_marginal_density(r, eta, n_dim), 0.5 * sps.beta(a, a).pdf((r + 1) / 2)
        )

    def test_scalar_and_array_inputs(self):
        assert np.ndim(lkj_marginal_density(0.1, 2.0, 2)) == 0
        assert lkj_marginal_density(np.zeros((2, 3)), 2.0, 2).shape == (2, 3)


# ----------------------------------------------------------------- Bayes factor
class TestBayesFactor:
    def test_far_from_zero_is_large(self, rng):
        samples = rng.normal(0.8, 0.03, size=4000)
        bf = bayes_factor(samples, eta=2.0, n_dim=2)
        assert bf > 100 or np.isinf(bf)

    def test_prior_like_posterior_is_near_one(self, rng):
        # Samples drawn from the prior itself give BF ~ 1 (up to KDE error).
        eta, n_dim = 2.0, 2
        a = eta - 1 + n_dim / 2
        samples = 2 * rng.beta(a, a, size=20000) - 1
        bf = bayes_factor(samples, eta=eta, n_dim=n_dim)
        assert 0.7 < bf < 1.4

    def test_concentrated_at_zero_favours_null(self, rng):
        samples = rng.normal(0.0, 0.05, size=4000)
        assert bayes_factor(samples, eta=2.0, n_dim=2) < 0.2

    def test_tiny_sample_is_nan(self, rng):
        assert np.isnan(bayes_factor(rng.normal(size=9), eta=2.0, n_dim=2))
        assert np.isnan(bayes_factor(np.array([]), eta=2.0, n_dim=2))

    def test_degenerate_sample_is_nan(self):
        assert np.isnan(bayes_factor(np.full(50, 0.3), eta=2.0, n_dim=2))

    def test_ref_val(self, rng):
        samples = rng.normal(0.5, 0.05, size=4000)
        assert bayes_factor(samples, eta=2.0, n_dim=2, ref_val=0.5) < 1.0
        far = bayes_factor(samples, eta=2.0, n_dim=2, ref_val=0.0)
        assert far > 10 or np.isinf(far)

    def test_accepts_2d_input(self, rng):
        samples = rng.normal(0.5, 0.05, size=(4, 1000))
        assert np.isfinite(bayes_factor(samples, eta=2.0, n_dim=2, ref_val=0.5))


# ----------------------------------------------------------- correlation summary
def _make_idata(rng, chains=2, draws=200, names=("a", "b", "c"), centre=(0.6, -0.2, 0.0)):
    k = len(names)
    corr = np.zeros((chains, draws, k, k))
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    for c in range(chains):
        for d in range(draws):
            m = np.eye(k)
            for (i, j), mu in zip(pairs, centre, strict=True):
                r = np.clip(mu + rng.normal(scale=0.08), -0.99, 0.99)
                m[i, j] = m[j, i] = r
            corr[c, d] = m
    return az.from_dict(
        {"posterior": {"chol_corr": corr}},
        coords={"feature": list(names), "feature2": list(names)},
        dims={"chol_corr": ["feature", "feature2"]},
    )


@pytest.fixture(scope="module")
def idata():
    return _make_idata(np.random.default_rng(1))


class TestCorrelationSamples:
    def test_shape_and_values(self, idata):
        s = correlation_samples(idata, 0, 1)
        assert s.shape == (400,)
        post = idata["posterior"].to_dataset()
        expected = post["chol_corr"].values[:, :, 0, 1].ravel()
        np.testing.assert_allclose(np.sort(s), np.sort(expected))
        np.testing.assert_allclose(correlation_samples(idata, 1, 0), s)

    def test_accepts_dataset(self, idata):
        post = idata["posterior"].to_dataset()
        np.testing.assert_allclose(
            correlation_samples(post, 0, 2), correlation_samples(idata, 0, 2)
        )


class TestCorrelationSummary:
    def test_structure(self, idata):
        df = correlation_summary(idata)
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "pair"
        assert list(df.index) == ["a~b", "a~c", "b~c"]
        expected_cols = {
            "i",
            "j",
            "mean",
            "median",
            "sd",
            "hdi_3%",
            "hdi_97%",
            "prob_positive",
            "bf10",
        }
        assert expected_cols == set(df.columns)
        assert df.loc["a~c", ["i", "j"]].tolist() == [0, 2]

    def test_values(self, idata):
        df = correlation_summary(idata)
        s = correlation_samples(idata, 0, 1)
        assert df.loc["a~b", "mean"] == pytest.approx(s.mean())
        assert df.loc["a~b", "median"] == pytest.approx(np.median(s))
        assert df.loc["a~b", "sd"] == pytest.approx(s.std(ddof=1))
        assert df.loc["a~b", "prob_positive"] == pytest.approx(np.mean(s > 0))
        assert df.loc["a~b", "hdi_3%"] < df.loc["a~b", "mean"] < df.loc["a~b", "hdi_97%"]
        assert 0.4 < df.loc["a~b", "hdi_3%"] and df.loc["a~b", "hdi_97%"] < 0.8
        assert df.loc["a~b", "prob_positive"] == 1.0
        assert df.loc["a~c", "prob_positive"] < 0.1
        assert 0.3 < df.loc["b~c", "prob_positive"] < 0.7
        # Bayes factors: strong for a~b, near/below one for b~c.
        assert df.loc["a~b", "bf10"] > 10
        assert df.loc["b~c", "bf10"] < 2
        assert df.loc["a~b", "bf10"] == pytest.approx(bayes_factor(s, eta=2.0, n_dim=3))

    def test_names_override(self, idata):
        df = correlation_summary(idata, names=["x", "y", "z"])
        assert list(df.index) == ["x~y", "x~z", "y~z"]

    def test_eti(self, idata):
        df = correlation_summary(idata, ci_kind="eti", ci_prob=0.9)
        assert {"eti_5%", "eti_95%"} <= set(df.columns)
        s = correlation_samples(idata, 0, 1)
        assert df.loc["a~b", "eti_5%"] == pytest.approx(np.quantile(s, 0.05), abs=0.02)
        assert df.loc["a~b", "eti_95%"] == pytest.approx(np.quantile(s, 0.95), abs=0.02)

    def test_ci_prob_labels(self, idata):
        df = correlation_summary(idata, ci_prob=0.5)
        assert {"hdi_25%", "hdi_75%"} <= set(df.columns)

    def test_invalid_ci_kind_raises(self, idata):
        with pytest.raises(ValueError, match="ci_kind"):
            correlation_summary(idata, ci_kind="bogus")

    def test_ref_val(self, idata):
        df = correlation_summary(idata, ref_val=0.6)
        assert 0.3 < df.loc["a~b", "prob_positive"] < 0.7
        assert df.loc["a~b", "bf10"] < 1.0

    def test_eta_changes_bf(self, idata):
        bf2 = correlation_summary(idata, eta=2.0).loc["b~c", "bf10"]
        bf5 = correlation_summary(idata, eta=5.0).loc["b~c", "bf10"]
        # Larger eta puts more prior mass at zero, so BF10 grows.
        assert bf5 > bf2

    def test_without_coords_uses_default_names(self, rng):
        corr = _make_idata(rng)["posterior"]["chol_corr"].values
        plain = xr.Dataset({"chol_corr": (("chain", "draw", "feature", "feature2"), corr)})
        df = correlation_summary(plain)
        assert list(df.index) == ["x0~x1", "x0~x2", "x1~x2"]

    def test_from_dict_integer_coords_become_names(self, rng):
        corr = _make_idata(rng)["posterior"]["chol_corr"].values
        df = correlation_summary(az.from_dict({"posterior": {"chol_corr": corr}}))
        assert list(df.index) == ["0~1", "0~2", "1~2"]

    def test_single_draw(self):
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])[None, None]
        plain = xr.Dataset({"chol_corr": (("chain", "draw", "feature", "feature2"), corr)})
        df = correlation_summary(plain)
        assert df.loc["x0~x1", "sd"] == 0.0
        assert df.loc["x0~x1", "mean"] == pytest.approx(0.3)
        assert np.isnan(df.loc["x0~x1", "bf10"])
