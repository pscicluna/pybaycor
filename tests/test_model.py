"""Tests for :func:`pybaycor.model.build_model` and the two-piece Normal helpers."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor
import pytest
from pybaycor.model import ModelInfo, twopiece_logcdf, twopiece_logp, twopiece_logsf
from scipy import stats as sps

from conftest import eval_at_point, free_rv_names, make_gaussian_data
from pybaycor import DEFAULT_PRIORS, CorrelationData, Standardisation, build_model

N = 20


@pytest.fixture(scope="module", autouse=True)
def fast_compile():
    """Skip the C compilation: these tests evaluate graphs once, they never sample."""
    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        yield


@pytest.fixture(scope="module")
def xdata():
    return make_gaussian_data(N, 0.5, seed=3, mean=(10.0, -2.0), scales=(3.0, 0.5))


def _cens_upper(n_rows, col, shape):
    cens = np.zeros(shape, dtype=int)
    cens[:n_rows, col] = -1
    return cens


def _finite_logps(model):
    logps = model.point_logps()
    assert all(np.isfinite(v) for v in logps.values()), logps
    return logps


# ----------------------------------------------------------------------- cases
class TestCases:
    def test_observed_case(self, xdata):
        model, info = build_model(CorrelationData(xdata, names=["a", "b"]))
        assert isinstance(model, pm.Model)
        assert isinstance(info, ModelInfo)
        assert info.case == "observed"
        assert not info.has_latent
        assert info.n_points == N and info.n_dim == 2 and info.names == ["a", "b"]
        assert info.summary_vars == ["mu", "chol_stds", "chol_corr"]
        assert info.priors == DEFAULT_PRIORS
        names = free_rv_names(model)
        assert "chol_z" in names and "mu_z" in names
        assert not any(n.startswith("theta") for n in names)
        assert "vals" not in model.named_vars
        assert {"mu", "chol_stds", "chol_corr", "cov"} <= set(model.named_vars)
        assert model.named_vars_to_dims["chol_corr"] == ("feature", "feature2")
        assert list(model.coords["feature"]) == ["a", "b"]
        assert len(model.observed_RVs) == 1
        _finite_logps(model)

    def test_latent_case(self, xdata):
        model, info = build_model(CorrelationData(xdata, sigma=0.3))
        assert info.case == "latent"
        assert info.has_latent
        names = free_rv_names(model)
        assert "theta_z" in names
        assert "theta_free" not in names
        assert "vals" in model.named_vars
        assert model.named_vars_to_dims["vals"] == ("point", "feature")
        assert len(model.potentials) == 0
        _finite_logps(model)

    def test_augmented_free(self, xdata):
        sig = np.zeros(xdata.shape)
        sig[:, 0] = 0.3
        model, info = build_model(CorrelationData(xdata, sigma=sig))
        assert info.case == "augmented"
        names = free_rv_names(model)
        assert "theta_free" in names
        assert "theta_z" not in names
        assert not any(n.endswith("_raw") for n in names)
        assert {p.name for p in model.potentials} == {"latent_logp"}
        assert model["theta_free"].type.shape == (N,)
        _finite_logps(model)

    @pytest.mark.parametrize(
        ("kind", "rv"),
        [("upper", "theta_upper_raw"), ("lower", "theta_lower_raw")],
    )
    def test_augmented_exact_limits(self, xdata, kind, rv):
        cens = np.zeros(xdata.shape, dtype=object)
        cens[:3, 1] = kind
        model, info = build_model(CorrelationData(xdata, censoring=cens))
        assert info.case == "augmented"
        names = free_rv_names(model)
        assert rv in names
        assert model[rv].type.shape == (3,)
        assert "theta_free" not in names
        assert "obs_z" not in model.named_vars  # no uncertain entries, no obs channel
        _finite_logps(model)

    def test_augmented_exact_interval(self, xdata):
        lo = np.full(xdata.shape, np.nan)
        hi = np.full(xdata.shape, np.nan)
        lo[:2, 0] = xdata[:2, 0] - 1.0
        hi[:2, 0] = xdata[:2, 0] + 1.0
        model, info = build_model(CorrelationData(xdata, lower=lo, upper=hi))
        assert info.case == "augmented"
        assert "theta_interval_raw" in free_rv_names(model)
        assert model["theta_interval_raw"].type.shape == (2,)
        _finite_logps(model)

    def test_augmented_all_kinds_at_once(self, xdata):
        sig = np.zeros(xdata.shape)
        sig[:5, 0] = 0.2  # uncertain observed
        sig[6, 1] = 0.3  # uncertain upper limit
        cens = np.zeros(xdata.shape, dtype=int)
        cens[6, 1] = -1  # uncertain upper limit
        cens[7, 1] = -1  # exact upper limit
        cens[8, 0] = 1  # exact lower limit
        lo = np.full(xdata.shape, np.nan)
        hi = np.full(xdata.shape, np.nan)
        lo[9, 1], hi[9, 1] = xdata[9, 1] - 1, xdata[9, 1] + 1  # exact interval
        model, info = build_model(
            CorrelationData(xdata, sigma=sig, censoring=cens, lower=lo, upper=hi)
        )
        assert info.case == "augmented"
        names = free_rv_names(model)
        assert {"theta_free", "theta_upper_raw", "theta_lower_raw", "theta_interval_raw"} <= names
        assert model["theta_free"].type.shape == (6,)
        pots = {p.name for p in model.potentials}
        assert pots == {"latent_logp", "obs_upper_logp"}
        assert "obs_z" in model.named_vars
        _finite_logps(model)


# --------------------------------------------------------------------- options
class TestOptions:
    def test_robust_adds_nu(self, xdata):
        model, info = build_model(CorrelationData(xdata), robust=True)
        assert info.robust
        assert "nu_minus_one" in free_rv_names(model)
        assert "nu" in model.named_vars
        assert info.summary_vars[-1] == "nu"
        _finite_logps(model)

    def test_robust_latent_and_augmented(self, xdata):
        model, _ = build_model(CorrelationData(xdata, sigma=0.3), robust=True)
        assert {"theta_z", "nu_minus_one"} <= free_rv_names(model)
        _finite_logps(model)
        cens = _cens_upper(2, 1, xdata.shape)
        model, _ = build_model(CorrelationData(xdata, censoring=cens), robust=True)
        assert {"theta_upper_raw", "nu_minus_one"} <= free_rv_names(model)
        _finite_logps(model)

    def test_error_scale(self, xdata):
        model, info = build_model(CorrelationData(xdata, sigma=0.3), error_scale=True)
        assert info.error_scale
        assert "error_scale" in free_rv_names(model)
        assert model.named_vars_to_dims["error_scale"] == ("feature",)
        assert "error_scale" in info.summary_vars
        _finite_logps(model)

    def test_error_scale_without_uncertainties(self, xdata):
        # Nothing to scale in the observed case: the model builds without the factor.
        model, info = build_model(CorrelationData(xdata), error_scale=True)
        assert info.case == "observed"
        assert "error_scale" not in free_rv_names(model)
        assert "error_scale" not in info.summary_vars

    def test_independent(self, xdata):
        model, info = build_model(CorrelationData(xdata), independent=True)
        assert info.independent
        names = free_rv_names(model)
        assert "chol_z" not in names
        assert "stds_z" in names
        out = eval_at_point(model, ["chol_corr", "cov", "chol_stds"])
        np.testing.assert_allclose(out["chol_corr"], np.eye(2))
        np.testing.assert_allclose(out["cov"], np.diag(out["chol_stds"] ** 2))
        _finite_logps(model)

    def test_priors_override(self, xdata):
        model, info = build_model(CorrelationData(xdata), priors={"eta": 1.0, "mu": (1.0, 2.0)})
        assert info.priors["eta"] == 1.0
        assert info.priors["mu"] == (1.0, 2.0)
        assert info.priors["sigma"] == DEFAULT_PRIORS["sigma"]
        _finite_logps(model)

    def test_priors_do_not_mutate_defaults(self, xdata):
        before = dict(DEFAULT_PRIORS)
        build_model(CorrelationData(xdata), priors={"eta": 7.0})
        assert DEFAULT_PRIORS == before

    def test_unknown_prior_key_raises(self, xdata):
        with pytest.raises(ValueError, match="Unknown prior keys \\['bogus'\\]"):
            build_model(CorrelationData(xdata), priors={"bogus": 1.0})

    def test_standardize_false_is_identity(self, xdata):
        _, info = build_model(CorrelationData(xdata), standardize=False)
        np.testing.assert_array_equal(info.transform.loc, 0.0)
        np.testing.assert_array_equal(info.transform.scale, 1.0)

    def test_standardize_true_uses_data(self, xdata):
        cd = CorrelationData(xdata)
        _, info = build_model(cd)
        expected = Standardisation.from_data(cd)
        np.testing.assert_allclose(info.transform.loc, expected.loc)
        np.testing.assert_allclose(info.transform.scale, expected.scale)

    def test_explicit_transform(self, xdata):
        tr = Standardisation(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        model, info = build_model(CorrelationData(xdata), transform=tr)
        assert info.transform is tr
        out = eval_at_point(model, ["mu"])
        # mu_z starts at the prior mean (0), so mu == loc at the initial point.
        np.testing.assert_allclose(out["mu"], tr.loc)

    def test_three_dimensions(self, rng):
        x = rng.normal(size=(15, 3))
        model, info = build_model(CorrelationData(x, names=list("abc")))
        assert info.n_dim == 3
        out = eval_at_point(model, ["chol_corr", "cov"])
        assert out["chol_corr"].shape == (3, 3)
        assert out["cov"].shape == (3, 3)
        _finite_logps(model)


# ------------------------------------------------------------------- finiteness
class TestFiniteLogp:
    def test_asymmetric_sigma(self, xdata):
        lo = np.full(xdata.shape, 0.2)
        hi = np.full(xdata.shape, 0.5)
        model, _ = build_model(CorrelationData(xdata, sigma=(lo, hi)))
        assert {p.name for p in model.potentials} == {"obs_asym_logp"}
        _finite_logps(model)

    def test_asymmetric_sigma_augmented(self, xdata):
        lo = np.zeros(xdata.shape)
        hi = np.zeros(xdata.shape)
        lo[:4, 0], hi[:4, 0] = 0.2, 0.5
        model, info = build_model(CorrelationData(xdata, sigma=(lo, hi)))
        assert info.case == "augmented"
        assert "obs_asym_logp" in {p.name for p in model.potentials}
        _finite_logps(model)

    def test_uncertain_limits(self, xdata):
        sig = np.full(xdata.shape, 0.3)
        cens = np.zeros(xdata.shape, dtype=int)
        cens[:3, 1] = -1
        cens[3:5, 0] = 1
        model, info = build_model(CorrelationData(xdata, sigma=sig, censoring=cens))
        assert info.case == "latent"
        assert {p.name for p in model.potentials} == {"obs_upper_logp", "obs_lower_logp"}
        _finite_logps(model)

    def test_uncertain_interval(self, xdata):
        sig = np.full(xdata.shape, 0.3)
        lo = np.full(xdata.shape, np.nan)
        hi = np.full(xdata.shape, np.nan)
        lo[:3, 1] = xdata[:3, 1] - 0.5
        hi[:3, 1] = xdata[:3, 1] + 0.5
        model, _ = build_model(CorrelationData(xdata, sigma=sig, lower=lo, upper=hi))
        assert "obs_interval_logp" in {p.name for p in model.potentials}
        _finite_logps(model)

    def test_uncertain_asymmetric_interval_with_error_scale(self, xdata):
        lo_s = np.full(xdata.shape, 0.2)
        hi_s = np.full(xdata.shape, 0.4)
        lo = np.full(xdata.shape, np.nan)
        hi = np.full(xdata.shape, np.nan)
        lo[:3, 1] = xdata[:3, 1] - 0.5
        hi[:3, 1] = xdata[:3, 1] + 0.5
        cd = CorrelationData(xdata, sigma=(lo_s, hi_s), lower=lo, upper=hi)
        model, _ = build_model(cd, error_scale=True, robust=True)
        _finite_logps(model)

    def test_exact_interval(self, xdata):
        lo = np.full(xdata.shape, np.nan)
        hi = np.full(xdata.shape, np.nan)
        lo[:3, 1] = xdata[:3, 1] - 0.5
        hi[:3, 1] = xdata[:3, 1] + 0.5
        model, _ = build_model(CorrelationData(xdata, lower=lo, upper=hi))
        _finite_logps(model)

    def test_compiled_logp_and_gradient_finite(self, xdata):
        sig = np.zeros(xdata.shape)
        sig[:5, 0] = 0.2
        cens = _cens_upper(3, 1, xdata.shape)
        model, _ = build_model(CorrelationData(xdata, sigma=sig, censoring=cens), robust=True)
        ip = model.initial_point()
        logp = model.compile_logp()(ip)
        assert np.isfinite(logp)
        dlogp = model.compile_dlogp()(ip)
        assert np.isfinite(dlogp).all()


# ------------------------------------------------------------- back-transform
class TestBackTransform:
    @pytest.mark.parametrize("standardize", [True, False])
    def test_cov_is_consistent(self, xdata, standardize):
        model, _ = build_model(CorrelationData(xdata), standardize=standardize)
        out = eval_at_point(model, ["mu", "chol_stds", "cov", "chol_corr"])
        stds, corr, cov = out["chol_stds"], out["chol_corr"], out["cov"]
        np.testing.assert_allclose(np.diag(corr), 1.0)
        np.testing.assert_allclose(corr, corr.T)
        np.testing.assert_allclose(cov, np.diag(stds) @ corr @ np.diag(stds), rtol=1e-10)
        assert (stds > 0).all()

    def test_mu_and_stds_in_data_units(self, xdata):
        cd = CorrelationData(xdata)
        model, info = build_model(cd)
        out = eval_at_point(model, ["mu", "chol_stds"])
        tr = info.transform
        ip = model.initial_point()
        np.testing.assert_allclose(out["mu"], ip["mu_z"] * tr.scale + tr.loc)
        # chol_stds = stds_z * scale; recover stds_z from the cov in z units.
        zmodel, _ = build_model(cd, standardize=False)
        zout = eval_at_point(zmodel, ["chol_stds"])
        np.testing.assert_allclose(out["chol_stds"] / tr.scale, zout["chol_stds"])

    def test_vals_fixed_entries_equal_data(self, xdata):
        sig = np.zeros(xdata.shape)
        sig[:4, 0] = 0.3
        cens = _cens_upper(3, 1, xdata.shape)
        cd = CorrelationData(xdata, sigma=sig, censoring=cens)
        model, _ = build_model(cd)
        vals = eval_at_point(model, ["vals"])["vals"]
        assert vals.shape == xdata.shape
        np.testing.assert_allclose(vals[cd.is_fixed], xdata[cd.is_fixed], rtol=1e-10, atol=1e-10)
        # Uncertain entries start at the data value.
        np.testing.assert_allclose(vals[:4, 0], xdata[:4, 0], rtol=1e-8)

    def test_vals_respect_limits_at_initial_point(self, xdata):
        cens = np.zeros(xdata.shape, dtype=int)
        cens[:3, 1] = -1
        cens[3:6, 0] = 1
        lo = np.full(xdata.shape, np.nan)
        hi = np.full(xdata.shape, np.nan)
        lo[6:8, 1] = xdata[6:8, 1] - 0.7
        hi[6:8, 1] = xdata[6:8, 1] + 0.3
        cd = CorrelationData(xdata, censoring=cens, lower=lo, upper=hi)
        model, _ = build_model(cd)
        vals = eval_at_point(model, ["vals"])["vals"]
        assert (vals[cd.is_upper_limit] < cd.upper[cd.is_upper_limit]).all()
        assert (vals[cd.is_lower_limit] > cd.lower[cd.is_lower_limit]).all()
        assert (vals[cd.is_interval] > cd.lower[cd.is_interval]).all()
        assert (vals[cd.is_interval] < cd.upper[cd.is_interval]).all()
        np.testing.assert_allclose(vals[cd.is_fixed], xdata[cd.is_fixed])

    def test_vals_respect_limits_at_random_points(self, xdata, rng):
        cens = np.zeros(xdata.shape, dtype=int)
        cens[:3, 1] = -1
        cens[3:6, 0] = 1
        lo = np.full(xdata.shape, np.nan)
        hi = np.full(xdata.shape, np.nan)
        lo[6:8, 1] = xdata[6:8, 1] - 0.7
        hi[6:8, 1] = xdata[6:8, 1] + 0.3
        cd = CorrelationData(xdata, censoring=cens, lower=lo, upper=hi)
        model, _ = build_model(cd)
        point = model.initial_point()
        for _ in range(5):
            for key in ("theta_upper_raw", "theta_lower_raw", "theta_interval_raw"):
                point[key] = rng.normal(scale=3.0, size=point[key].shape)
            vals = eval_at_point(model, ["vals"], point)["vals"]
            assert (vals[cd.is_upper_limit] < cd.upper[cd.is_upper_limit]).all()
            assert (vals[cd.is_lower_limit] > cd.lower[cd.is_lower_limit]).all()
            assert (vals[cd.is_interval] > cd.lower[cd.is_interval]).all()
            assert (vals[cd.is_interval] < cd.upper[cd.is_interval]).all()

    def test_latent_vals_at_initial_point(self, xdata):
        cd = CorrelationData(xdata, sigma=0.3)
        model, _ = build_model(cd)
        vals = eval_at_point(model, ["vals"])["vals"]
        np.testing.assert_allclose(vals, xdata, rtol=1e-8)


# ----------------------------------------------------------- augmented density
class TestAugmentedDensity:
    """The augmented potential must equal the plain multivariate density on the rows."""

    def test_pseudo_prior_cancels(self, xdata, rng):
        sig = np.zeros(xdata.shape)
        sig[:4, 0] = 0.3
        cd = CorrelationData(xdata, sigma=sig)
        model, _ = build_model(cd, standardize=False)
        # The pseudo-prior on theta_free plus the potential must equal the plain
        # multivariate Normal density of the assembled rows.
        point = model.initial_point()
        point["theta_free"] = xdata[:4, 0] + rng.normal(size=4)
        fn = model.compile_fn(
            model.replace_rvs_by_values([model["mu_z"], model["cov"], model["vals"]]),
            inputs=model.value_vars,
            on_unused_input="ignore",
        )
        mu_z, cov, vals = fn(point)
        logps = model.point_logps(point, round_vals=10)
        got = float(logps["latent_logp"]) + float(logps["theta_free"])
        expected = sps.multivariate_normal(mu_z, cov).logpdf(vals).sum()
        assert got == pytest.approx(expected, rel=1e-8)


# ----------------------------------------------------------- two-piece Normal
TP_MU, TP_S_LO, TP_S_HI = 0.5, 1.0, 2.0


@pytest.fixture(scope="module")
def grid():
    x = np.linspace(-25.0, 25.0, 100_001)
    lp = twopiece_logp(x, TP_MU, TP_S_LO, TP_S_HI).eval()
    return x, lp


class TestTwoPiece:
    MU, S_LO, S_HI = TP_MU, TP_S_LO, TP_S_HI

    def test_integrates_to_one(self, grid):
        x, lp = grid
        assert np.trapezoid(np.exp(lp), x) == pytest.approx(1.0, abs=1e-6)

    def test_continuous_at_mode(self):
        eps = 1e-9
        below = twopiece_logp(self.MU - eps, self.MU, self.S_LO, self.S_HI).eval()
        above = twopiece_logp(self.MU + eps, self.MU, self.S_LO, self.S_HI).eval()
        assert below == pytest.approx(above, abs=1e-8)

    def test_reduces_to_normal(self):
        x = np.linspace(-6.0, 6.0, 101)
        for s in (0.3, 1.0, 2.5):
            lp = twopiece_logp(x, self.MU, s, s).eval()
            np.testing.assert_allclose(lp, sps.norm.logpdf(x, self.MU, s), rtol=1e-6)
            lc = twopiece_logcdf(x, self.MU, s, s).eval()
            np.testing.assert_allclose(lc, sps.norm.logcdf(x, self.MU, s), rtol=1e-6)
            ls = twopiece_logsf(x, self.MU, s, s).eval()
            np.testing.assert_allclose(ls, sps.norm.logsf(x, self.MU, s), rtol=1e-6)

    def test_logcdf_matches_numerical_integration(self, grid):
        x, lp = grid
        cdf = np.cumsum(np.exp(lp)) * (x[1] - x[0])
        pts = np.array([-4.0, -1.0, 0.0, self.MU, 1.0, 3.0, 8.0])
        lc = twopiece_logcdf(pts, self.MU, self.S_LO, self.S_HI).eval()
        np.testing.assert_allclose(np.exp(lc), np.interp(pts, x, cdf), atol=2e-4)
        # Analytic value at the mode: mass below mu is s_lo / (s_lo + s_hi).
        at_mode = twopiece_logcdf(self.MU, self.MU, self.S_LO, self.S_HI).eval()
        assert np.exp(at_mode) == pytest.approx(self.S_LO / (self.S_LO + self.S_HI))

    def test_logsf_is_log1m_cdf(self):
        pts = np.linspace(-5.0, 8.0, 27)
        lc = twopiece_logcdf(pts, self.MU, self.S_LO, self.S_HI).eval()
        ls = twopiece_logsf(pts, self.MU, self.S_LO, self.S_HI).eval()
        np.testing.assert_allclose(ls, np.log1p(-np.exp(lc)), rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(np.exp(lc) + np.exp(ls), 1.0, rtol=1e-10)

    def test_monotone_cdf(self):
        pts = np.linspace(-6.0, 8.0, 200)
        lc = twopiece_logcdf(pts, self.MU, self.S_LO, self.S_HI).eval()
        assert (np.diff(lc) > 0).all()

    def test_broadcasts_over_arrays(self):
        x = np.array([-1.0, 0.0, 2.0])
        mu = np.array([0.0, 0.5, 1.0])
        s_lo = np.array([1.0, 0.5, 2.0])
        s_hi = np.array([2.0, 0.5, 1.0])
        lp = twopiece_logp(x, mu, s_lo, s_hi).eval()
        expected = [twopiece_logp(x[i], mu[i], s_lo[i], s_hi[i]).eval() for i in range(3)]
        np.testing.assert_allclose(lp, expected)


# -------------------------------------------------- censored likelihood terms
class TestCensoredLikelihood:
    """The censored observation channels of the model equal scipy's Normal values."""

    def _potential(self, model, name, point):
        fn = model.compile_fn(
            model.replace_rvs_by_values([model[name]]),
            inputs=model.value_vars,
            on_unused_input="ignore",
        )
        return float(fn(point)[0])

    def test_symmetric_upper_and_lower_limits(self, xdata, rng):
        sig = np.full(xdata.shape, 0.4)
        cens = np.zeros(xdata.shape, dtype=int)
        cens[:3, 1] = -1
        cens[3:6, 0] = 1
        cd = CorrelationData(xdata, sigma=sig, censoring=cens)
        model, _ = build_model(cd, standardize=False)  # data units == model units
        point = model.initial_point()
        point["theta_z"] = xdata + rng.normal(scale=0.5, size=xdata.shape)
        theta = point["theta_z"]
        up = self._potential(model, "obs_upper_logp", point)
        expected_up = sps.norm.logcdf(cd.upper[:3, 1], theta[:3, 1], 0.4).sum()
        assert up == pytest.approx(expected_up, rel=1e-6)
        lo = self._potential(model, "obs_lower_logp", point)
        expected_lo = sps.norm.logsf(cd.lower[3:6, 0], theta[3:6, 0], 0.4).sum()
        assert lo == pytest.approx(expected_lo, rel=1e-6)

    def test_symmetric_interval(self, xdata, rng):
        sig = np.full(xdata.shape, 0.4)
        lo = np.full(xdata.shape, np.nan)
        hi = np.full(xdata.shape, np.nan)
        lo[:3, 1] = xdata[:3, 1] - 0.5
        hi[:3, 1] = xdata[:3, 1] + 0.8
        cd = CorrelationData(xdata, sigma=sig, lower=lo, upper=hi)
        model, _ = build_model(cd, standardize=False)
        point = model.initial_point()
        point["theta_z"] = xdata + rng.normal(scale=0.5, size=xdata.shape)
        theta = point["theta_z"]
        got = self._potential(model, "obs_interval_logp", point)
        expected = np.log(
            sps.norm.cdf(hi[:3, 1], theta[:3, 1], 0.4) - sps.norm.cdf(lo[:3, 1], theta[:3, 1], 0.4)
        ).sum()
        assert got == pytest.approx(expected, rel=1e-6)

    def test_symmetric_observed_channel(self, xdata, rng):
        sig = np.full(xdata.shape, 0.4)
        cd = CorrelationData(xdata, sigma=sig)
        model, _ = build_model(cd, standardize=False)
        point = model.initial_point()
        point["theta_z"] = xdata + rng.normal(scale=0.5, size=xdata.shape)
        logps = model.point_logps(point, round_vals=12)
        expected = sps.norm.logpdf(xdata, point["theta_z"], 0.4).sum()
        assert float(logps["obs_z"]) == pytest.approx(expected, rel=1e-8)

    def test_asymmetric_observed_channel(self, xdata, rng):
        lo_s = np.full(xdata.shape, 0.2)
        hi_s = np.full(xdata.shape, 0.6)
        cd = CorrelationData(xdata, sigma=(lo_s, hi_s))
        model, _ = build_model(cd, standardize=False)
        point = model.initial_point()
        point["theta_z"] = xdata + rng.normal(scale=0.5, size=xdata.shape)
        got = self._potential(model, "obs_asym_logp", point)
        expected = twopiece_logp(xdata, point["theta_z"], lo_s, hi_s).eval().sum()
        assert got == pytest.approx(expected, rel=1e-8)

    def test_censoring_is_informative(self, xdata):
        # Moving a latent value far above its upper limit must lower the likelihood.
        sig = np.full(xdata.shape, 0.4)
        cens = _cens_upper(3, 1, xdata.shape)
        cd = CorrelationData(xdata, sigma=sig, censoring=cens)
        model, _ = build_model(cd, standardize=False)
        point = model.initial_point()
        base = self._potential(model, "obs_upper_logp", point)
        point["theta_z"][:3, 1] = cd.upper[:3, 1] + 3.0
        moved = self._potential(model, "obs_upper_logp", point)
        assert moved < base - 10


def test_latent_case_exact_censored_entries_are_constrained(xdata):
    """Exact censored entries force the bounded reparameterisation even if all is latent."""
    sig = np.full(xdata.shape, 0.3)
    sig[:3, 1] = 0.0  # censored entries quoted without an error bar
    cens = _cens_upper(3, 1, xdata.shape)
    cd = CorrelationData(xdata, sigma=sig, censoring=cens)
    assert cd.is_latent.all()
    model, info = build_model(cd, standardize=False)
    assert info.case == "augmented"
    names = free_rv_names(model)
    assert "theta_upper_raw" in names
    assert "theta_z" not in names
    rng = np.random.default_rng(0)
    point = model.initial_point()
    for _ in range(5):
        point["theta_upper_raw"] = rng.normal(scale=3.0, size=point["theta_upper_raw"].shape)
        vals = eval_at_point(model, ["vals"], point)["vals"]
        assert np.all(vals[:3, 1] < cd.upper[:3, 1])
