"""Integration tests: parameter recovery with the sampling back-ends.

Everything that samples with NUTS/SMC/VI is marked ``slow`` except one cheap smoke fit.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import (
    SMOKE_FIT_KWARGS,
    add_heteroscedastic_noise,
    add_twopiece_noise,
    censor_column,
    make_gaussian_data,
    make_gaussian_data_3d,
)
from pybaycor import BayesianCorrelation

slow = pytest.mark.slow

NUTS = {"nuts_sampler": "pymc", "progressbar": False, "compute_convergence_checks": False}


def fit_nuts(bc, seed, draws=400, tune=400, chains=2, **kwargs):
    bc.fit(draws=draws, tune=tune, chains=chains, random_seed=seed, **NUTS, **kwargs)
    return bc


def corr_stats(bc):
    df = bc.correlation_summary(ci_prob=0.94, ci_kind="hdi")
    row = df.iloc[0]
    return float(row["mean"]), float(row["sd"]), float(row["hdi_3%"]), float(row["hdi_97%"])


# ------------------------------------------------------------------- smoke fit
@pytest.fixture(scope="module")
def smoke_fit():
    data = make_gaussian_data(40, 0.6, seed=5)
    bc = BayesianCorrelation(data, names=["a", "b"])
    bc.fit(random_seed=0, **SMOKE_FIT_KWARGS)
    return bc


def test_smoke_nuts_fit(smoke_fit):
    idata = smoke_fit.trace
    assert isinstance(idata, xr.DataTree)
    assert {"posterior", "sample_stats"} <= set(idata.children)
    corr = idata["posterior"]["chol_corr"]
    assert corr.shape == (1, 100, 2, 2)
    assert smoke_fit.fit_kwargs["draws"] == 100
    mean, sd, lo, hi = corr_stats(smoke_fit)
    assert abs(mean - 0.6) < 0.3
    assert 0 < sd < 0.3
    assert lo < mean < hi
    assert smoke_fit.correlation().shape == (100,)
    assert smoke_fit.bayes_factor() > 1
    summary = smoke_fit.summarise(printout=False)
    assert "chol_corr[a, b]" in summary.index
    assert np.isfinite(summary["mean"]).all()
    mean_vals, sd_vals = smoke_fit.inferred_values()
    np.testing.assert_array_equal(mean_vals, smoke_fit.data.values)
    np.testing.assert_array_equal(sd_vals, 0.0)


# --------------------------------------------------------------- (a) exact 2-D
@pytest.fixture(scope="module")
def exact_fit():
    data = make_gaussian_data(60, 0.7, seed=1, mean=(5.0, -1.0), scales=(2.0, 0.5))
    return fit_nuts(BayesianCorrelation(data, names=["a", "b"]), seed=1)


@slow
def test_exact_recovery(exact_fit):
    mean, sd, lo, hi = corr_stats(exact_fit)
    assert abs(mean - 0.7) < 0.15
    assert lo < 0.7 < hi
    assert 0.02 < sd < 0.15
    df = exact_fit.correlation_summary()
    assert df.loc["a~b", "prob_positive"] == 1.0
    assert df.loc["a~b", "bf10"] > 10
    # mu and chol_stds come back in data units.
    post = exact_fit.trace["posterior"]
    mu = post["mu"].mean(("chain", "draw")).values
    stds = post["chol_stds"].mean(("chain", "draw")).values
    x = exact_fit.data.values
    np.testing.assert_allclose(mu, x.mean(axis=0), atol=0.5)
    np.testing.assert_allclose(stds, x.std(axis=0), rtol=0.3)
    cov = post["cov"].mean(("chain", "draw")).values
    assert cov[0, 1] == pytest.approx(cov[1, 0])
    assert abs(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) - 0.7) < 0.15


@slow
def test_three_dimensional_recovery():
    data, corr = make_gaussian_data_3d(60, seed=7)
    bc = fit_nuts(BayesianCorrelation(data, names=list("abc")), seed=7)
    df = bc.correlation_summary()
    assert list(df.index) == ["a~b", "a~c", "b~c"]
    for (i, j), (_, row) in zip([(0, 1), (0, 2), (1, 2)], df.iterrows(), strict=True):
        assert abs(row["mean"] - corr[i, j]) < 0.2
        assert row["hdi_3%"] - 0.05 < corr[i, j] < row["hdi_97%"] + 0.05
    post = bc.trace["posterior"]["chol_corr"].mean(("chain", "draw")).values
    np.testing.assert_allclose(post, post.T)
    np.testing.assert_allclose(np.diag(post), 1.0)


# ---------------------------------------------------------- (b) robust outliers
@slow
def test_robust_outliers():
    data = make_gaussian_data(60, 0.7, seed=2)
    data[:3] = [[3.5, -3.5], [4.0, -3.0], [-3.5, 3.5]]  # gross outliers against the trend
    gauss = fit_nuts(BayesianCorrelation(data), seed=2)
    robust = fit_nuts(BayesianCorrelation(data, robust=True), seed=2)
    g_mean = corr_stats(gauss)[0]
    r_mean = corr_stats(robust)[0]
    assert abs(r_mean - 0.7) < abs(g_mean - 0.7)
    assert abs(r_mean - 0.7) < 0.2
    nu = float(robust.trace["posterior"]["nu"].mean())
    assert nu < 30
    assert "nu" in robust.summarise(printout=False).index


# ---------------------------------------------------- (c) hierarchical (latent)
@slow
def test_hierarchical_heteroscedastic():
    rng = np.random.default_rng(1003)  # independent of the data seed
    truth = make_gaussian_data(60, 0.8, seed=3)
    noisy, sigma = add_heteroscedastic_noise(truth, rng, sigma_range=(0.6, 1.4))
    naive = np.corrcoef(noisy.T)[0, 1]
    assert naive < 0.65  # attenuated by the measurement noise
    bc = fit_nuts(BayesianCorrelation(noisy, sigma=sigma), seed=3)
    assert bc.info.case == "latent"
    mean, _, lo, hi = corr_stats(bc)
    assert abs(mean - 0.8) < abs(naive - 0.8)
    assert lo - 0.05 < 0.8 < hi + 0.05
    # The inferred true values are shrunk towards the trend: closer to the truth.
    est, est_sd = bc.inferred_values()
    assert est.shape == truth.shape
    assert (est_sd > 0).all()
    assert np.mean((est - truth) ** 2) < np.mean((noisy - truth) ** 2)
    assert len(bc.data_summary(printout=False)) == truth.size


# ----------------------------------------------- (d) partial uncertainty (aug.)
@slow
def test_partial_uncertainty_augmented():
    rng = np.random.default_rng(1004)
    truth = make_gaussian_data(60, 0.7, seed=4)
    data = truth.copy()
    sig = np.zeros(data.shape)
    sig[:, 0] = 0.5
    data[:, 0] += rng.normal(scale=0.5, size=60)
    bc = BayesianCorrelation(data, sigma=sig)
    assert bc.info.case == "augmented"
    fit_nuts(bc, seed=4)
    assert abs(corr_stats(bc)[0] - 0.7) < 0.2
    est, est_sd = bc.inferred_values()
    np.testing.assert_allclose(est[:, 1], data[:, 1], rtol=1e-10)  # exact column is fixed
    np.testing.assert_allclose(est_sd[:, 1], 0.0, atol=1e-10)
    assert (est_sd[:, 0] > 0).all()


# ------------------------------------------------------------- (e) censoring
@pytest.fixture(scope="module")
def censored_setup():
    truth = make_gaussian_data(80, 0.6, seed=6)
    values, cens = censor_column(truth, 1, 0.3, kind="upper")
    mask = cens[:, 1] == -1
    assert 0.2 < mask.mean() < 0.4
    return truth, values, cens, mask


@pytest.fixture(scope="module")
def dropped_fit(censored_setup):
    _, values, _, mask = censored_setup
    return fit_nuts(BayesianCorrelation(values[~mask]), seed=6)


@slow
def test_exact_censoring_augmented(censored_setup, dropped_fit):
    _, values, cens, mask = censored_setup
    bc = BayesianCorrelation(values, censoring=cens)
    assert bc.info.case == "augmented"
    assert "theta_upper_raw" in {rv.name for rv in bc.model.free_RVs}
    fit_nuts(bc, seed=6)
    mean, sd, _, _ = corr_stats(bc)
    assert abs(mean - 0.6) < 0.2
    est, est_sd = bc.inferred_values()
    limits = bc.data.upper[mask, 1]
    assert (est[mask, 1] < limits).all()
    assert (est_sd[mask, 1] > 0).all()
    np.testing.assert_allclose(est[~mask], values[~mask], rtol=1e-10)
    # Using the limits must not be (much) less informative than dropping them.
    assert sd <= 2 * corr_stats(dropped_fit)[1]


@slow
def test_uncertain_censoring(censored_setup, dropped_fit):
    rng = np.random.default_rng(1066)
    truth = censored_setup[0]
    noisy = truth.copy()
    noisy[:, 1] += rng.normal(scale=0.2, size=80)
    values, cens = censor_column(noisy, 1, 0.3, kind="upper")
    mask = cens[:, 1] == -1
    sig = np.zeros(values.shape)
    sig[:, 1] = 0.2
    bc = BayesianCorrelation(values, sigma=sig, censoring=cens)
    assert bc.info.case == "augmented"
    assert "obs_upper_logp" in {p.name for p in bc.model.potentials}
    fit_nuts(bc, seed=66)
    mean, sd, _, _ = corr_stats(bc)
    assert abs(mean - 0.6) < 0.2
    est, est_sd = bc.inferred_values()
    limits = bc.data.upper[mask, 1]
    # The true value is only constrained up to the error bar.
    assert (est[mask, 1] < limits + 2 * 0.2).all()
    assert np.mean(est[mask, 1] < limits) > 0.8
    assert (est_sd[mask, 1] > 0).all()
    assert sd <= 2 * corr_stats(dropped_fit)[1]


@slow
def test_lower_limits():
    truth = make_gaussian_data(60, 0.6, seed=8)
    values, cens = censor_column(truth, 0, 0.25, kind="lower")
    mask = cens[:, 0] == 1
    bc = fit_nuts(BayesianCorrelation(values, censoring=cens), seed=8)
    assert abs(corr_stats(bc)[0] - 0.6) < 0.2
    est, _ = bc.inferred_values()
    assert (est[mask, 0] > bc.data.lower[mask, 0]).all()


# ------------------------------------------------------- (f) interval censoring
@slow
def test_interval_censoring():
    truth = make_gaussian_data(60, 0.6, seed=9)
    values = truth.copy()
    lo = np.full(values.shape, np.nan)
    hi = np.full(values.shape, np.nan)
    idx = np.arange(0, 60, 6)
    lo[idx, 0] = truth[idx, 0] - 0.4
    hi[idx, 0] = truth[idx, 0] + 0.6
    values[idx, 0] = np.nan  # only the bounds are known
    bc = BayesianCorrelation(values, lower=lo, upper=hi)
    assert bc.data.is_interval[idx, 0].all()
    assert bc.info.case == "augmented"
    fit_nuts(bc, seed=9)
    assert abs(corr_stats(bc)[0] - 0.6) < 0.2
    est, est_sd = bc.inferred_values()
    assert (est[idx, 0] > lo[idx, 0]).all()
    assert (est[idx, 0] < hi[idx, 0]).all()
    assert (est_sd[idx, 0] > 0).all()
    other = np.ones(60, dtype=bool)
    other[idx] = False
    np.testing.assert_allclose(est[other], truth[other], rtol=1e-10)


# ------------------------------------------------------- (g) asymmetric errors
@slow
def test_asymmetric_errors():
    rng = np.random.default_rng(1010)
    truth = make_gaussian_data(60, 0.7, seed=10)
    s_lo, s_hi = 0.2, 0.5
    noisy = add_twopiece_noise(truth, rng, s_lo, s_hi)
    bc = BayesianCorrelation(noisy, sigma_lower=s_lo, sigma_upper=s_hi)
    assert bc.data.is_asymmetric.all()
    assert "obs_asym_logp" in {p.name for p in bc.model.potentials}
    fit_nuts(bc, seed=10)
    assert abs(corr_stats(bc)[0] - 0.7) < 0.25
    assert np.isfinite(bc.inferred_values()[0]).all()


# ------------------------------------------------------------------ (h) SMC
@slow
def test_smc():
    data = make_gaussian_data(60, 0.7, seed=12)
    bc = BayesianCorrelation(data)
    idata = bc.fit(
        method="smc",
        draws=200,
        chains=2,
        random_seed=12,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert isinstance(idata, xr.DataTree)
    assert idata["posterior"]["chol_corr"].shape == (2, 200, 2, 2)
    assert "log_marginal_likelihood" in idata["sample_stats"].data_vars
    assert abs(corr_stats(bc)[0] - 0.7) < 0.25


# ------------------------------------------------------------- (i, j) ADVI
@slow
def test_fullrank_advi():
    data = make_gaussian_data(60, 0.7, seed=13)
    bc = BayesianCorrelation(data)
    idata = bc.fit(
        method="fullrank_advi", vi_iterations=20_000, draws=500, random_seed=13, progressbar=False
    )
    assert bc.approx is not None
    corr = idata["posterior"]["chol_corr"]
    assert corr.sizes["draw"] == 500
    assert np.isfinite(corr.values).all()
    mean, sd, lo, hi = corr_stats(bc)
    assert np.isfinite([mean, sd, lo, hi]).all()
    assert -1 < mean < 1 and sd > 0


@slow
def test_advi_warns():
    data = make_gaussian_data(40, 0.7, seed=14)
    bc = BayesianCorrelation(data)
    with pytest.warns(UserWarning, match="Mean-field ADVI under-estimates"):
        bc.fit(method="advi", vi_iterations=500, draws=50, random_seed=14, progressbar=False)
    assert bc.trace["posterior"]["chol_corr"].sizes["draw"] == 50


# ------------------------------------------------------------ (k) error_scale
@slow
def test_error_scale():
    rng = np.random.default_rng(1015)
    truth = make_gaussian_data(60, 0.7, seed=15)
    noisy, sigma = add_heteroscedastic_noise(truth, rng, sigma_range=(0.2, 0.8))
    bc = BayesianCorrelation(noisy, sigma=sigma, error_scale=True)
    fit_nuts(bc, seed=15)
    post = bc.trace["posterior"]
    assert post["error_scale"].dims == ("chain", "draw", "feature")
    f = post["error_scale"].mean(("chain", "draw")).values
    assert f.shape == (2,)
    assert ((f > 0.3) & (f < 3.0)).all()
    assert "error_scale[x0]" in bc.summarise(printout=False).index
    assert abs(corr_stats(bc)[0] - 0.7) < 0.25


# ------------------------------------------------------------- (l) null p-value
@slow
def test_null_pvalue():
    data = make_gaussian_data(30, 0.0, seed=16)
    bc = BayesianCorrelation(data)
    bc.fit(random_seed=16, **SMOKE_FIT_KWARGS)
    out = bc.null_pvalue(
        n_sim=3,
        draws=100,
        tune=100,
        chains=1,
        nuts_sampler="pymc",
        random_seed=16,
        return_null_distribution=True,
    )
    pvalue, null_stats = out
    assert isinstance(pvalue, float)
    assert 0.0 <= pvalue <= 1.0
    assert null_stats.shape == (3,)
    assert np.isfinite(null_stats).all()
    assert (null_stats >= 0).all()


# ------------------------------------------------------ (m) other NUTS samplers
@slow
@pytest.mark.nutpie
def test_nutpie_sampler(exact_fit):
    pytest.importorskip("nutpie")
    data = exact_fit.data.values
    bc = BayesianCorrelation(data)
    idata = bc.fit(
        draws=400,
        tune=400,
        chains=2,
        nuts_sampler="nutpie",
        random_seed=1,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "chol_corr" in idata["posterior"].data_vars
    assert idata["posterior"]["chol_corr"].shape == (2, 400, 2, 2)
    assert abs(corr_stats(bc)[0] - corr_stats(exact_fit)[0]) < 0.1


@slow
@pytest.mark.jax
def test_numpyro_sampler(exact_fit):
    pytest.importorskip("jax")
    pytest.importorskip("numpyro")
    data = exact_fit.data.values
    bc = BayesianCorrelation(data)
    idata = bc.fit(
        draws=400,
        tune=400,
        chains=2,
        nuts_sampler="numpyro",
        random_seed=1,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "chol_corr" in idata["posterior"].data_vars
    assert idata["posterior"]["chol_corr"].shape == (2, 400, 2, 2)
    assert abs(corr_stats(bc)[0] - corr_stats(exact_fit)[0]) < 0.1
    # The latent model also runs on JAX.
    latent = BayesianCorrelation(data, sigma=0.3)
    latent.fit(
        draws=200,
        tune=200,
        chains=1,
        nuts_sampler="numpyro",
        random_seed=1,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert "vals" in latent.trace["posterior"].data_vars
    assert np.isfinite(latent.correlation()).all()
