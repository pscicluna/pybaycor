"""PyMC model factory for Bayesian correlation inference.

The single entry point, :func:`build_model`, turns a :class:`~pybaycor.data.CorrelationData`
into a :class:`pymc.Model`. The same function serves every use case of the package:

* no uncertainties (the classic "Bayesian First Aid" correlation),
* measurement uncertainties on every entry (the hierarchical model of Matzke et al. 2017),
* any mixture of exact and uncertain entries, symmetric or asymmetric errors,
* arbitrary per-entry censoring (upper limits, lower limits, intervals),
* Gaussian or (robust) multivariate Student-t likelihood,
* an optional per-variable rescaling of the quoted uncertainties.

Variables stored in the trace
-----------------------------
``mu``
    Mean of each variable, in data units (``dims=("feature",)``).
``chol_stds``
    Standard deviation of each variable, in data units.
``chol_corr``
    The correlation matrix (``dims=("feature", "feature2")``); the quantity of interest.
``cov``
    The covariance matrix in data units.
``nu``
    Degrees of freedom of the Student-t (robust models only).
``error_scale``
    Multiplicative factor on the quoted uncertainties (only with ``error_scale=True``).
``vals``
    The inferred *true* values of every entry in data units
    (``dims=("point", "feature")``; only when the model has latent values).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.dist_math import normal_lccdf, normal_lcdf

from .data import CorrelationData, Standardisation

__all__ = ["DEFAULT_PRIORS", "ModelInfo", "build_model", "set_model_values"]

#: Default hyper-parameters, expressed in *standardised* units (see
#: :class:`~pybaycor.data.Standardisation`). Keys:
#:
#: ``mu``: ``(mean, sd)`` of the Normal prior on each variable's mean; each may be a
#: scalar or have one entry per variable;
#: ``sigma``: scale ``beta`` of the half-Cauchy prior on each variable's standard
#: deviation (scalar or one entry per variable);
#: ``eta``: shape of the LKJ prior on the correlation matrix (``eta > 1`` shrinks towards
#: zero correlation, ``eta = 1`` is uniform over correlation matrices);
#: ``nu_rate``: rate of the shifted-exponential prior on the Student-t degrees of freedom,
#: ``nu = 1 + Exponential(nu_rate)`` (mean ``1 + 1/nu_rate``);
#: ``error_scale_sd``: standard deviation of the log-normal prior on the error-scale factor.
DEFAULT_PRIORS: dict[str, Any] = {
    "mu": (0.0, 5.0),
    "sigma": 2.5,
    "eta": 2.0,
    "nu_rate": 1.0 / 29.0,
    "error_scale_sd": 0.5,
}

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


@dataclass
class ModelInfo:
    """Book-keeping about how a model was built.

    Attributes
    ----------
    transform : Standardisation
        The affine transform applied to the data before fitting.
    robust : bool
        Whether the multivariate Student-t likelihood was used.
    error_scale : bool
        Whether the per-variable error-scale factors are in the model.
    case : {"observed", "latent", "augmented"}
        ``"observed"``: no latent values (all entries exact and uncensored);
        ``"latent"``: every entry has a latent true value (classic hierarchical model);
        ``"augmented"``: a mixture of fixed and latent entries, joined through a potential.
    summary_vars : list of str
        Trace variables that :meth:`~pybaycor.BayesianCorrelation.summarise` reports.
    priors : dict
        The hyper-parameters actually used.
    pattern : dict or None
        For models built with ``mutable_values=True``: the uncertainties and censoring
        of the original data (in data units), which any replacement data must match.
    """

    transform: Standardisation
    robust: bool
    error_scale: bool
    case: str
    summary_vars: list[str] = field(default_factory=list)
    priors: dict[str, Any] = field(default_factory=dict)
    n_points: int = 0
    n_dim: int = 0
    names: list[str] = field(default_factory=list)
    independent: bool = False
    mutable_values: bool = False
    pattern: dict[str, np.ndarray] | None = None

    @property
    def has_latent(self) -> bool:
        """Whether the trace contains the inferred true values ``vals``."""
        return self.case != "observed"


# --------------------------------------------------------------------------- densities
def twopiece_logp(x: Any, mu: Any, s_lo: Any, s_hi: Any) -> Any:
    """Log-density of the two-piece ("split") Normal with scales ``s_lo`` below ``mu``.

    The density is continuous at ``mu`` and integrates to one:
    ``f(x) = 2 / (s_lo + s_hi) * phi((x - mu) / s)`` with ``s = s_lo`` for ``x < mu`` and
    ``s = s_hi`` otherwise. It reduces to the Normal when ``s_lo == s_hi``.
    """
    s = pt.switch(pt.lt(x, mu), s_lo, s_hi)
    z = (x - mu) / s
    return pt.log(2.0) - pt.log(s_lo + s_hi) - 0.5 * z**2 - np.log(_SQRT_2PI)


def twopiece_logcdf(x: Any, mu: Any, s_lo: Any, s_hi: Any) -> Any:
    """Log of ``P(X <= x)`` for the two-piece Normal (see :func:`twopiece_logp`)."""
    w_lo = 2.0 * s_lo / (s_lo + s_hi)
    w_hi = 2.0 * s_hi / (s_lo + s_hi)
    below = pt.log(w_lo) + normal_lcdf(mu, s_lo, x)
    above = pt.log1p(-w_hi * pt.exp(normal_lccdf(mu, s_hi, x)))
    return pt.switch(pt.lt(x, mu), below, above)


def twopiece_logsf(x: Any, mu: Any, s_lo: Any, s_hi: Any) -> Any:
    """Log of ``P(X > x)`` for the two-piece Normal (see :func:`twopiece_logp`)."""
    w_lo = 2.0 * s_lo / (s_lo + s_hi)
    w_hi = 2.0 * s_hi / (s_lo + s_hi)
    below = pt.log1p(-w_lo * pt.exp(normal_lcdf(mu, s_lo, x)))
    above = pt.log(w_hi) + normal_lccdf(mu, s_hi, x)
    return pt.switch(pt.lt(x, mu), below, above)


def _prior_array(key: str, value: Any, n_dim: int, positive: bool) -> np.ndarray:
    """Validate a scalar-or-per-variable hyper-parameter and return it as an array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim > 1 or (arr.ndim == 1 and arr.size != n_dim):
        raise ValueError(
            f"priors[{key!r}] must be a scalar or have one entry per variable ({n_dim}); "
            f"got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"priors[{key!r}] must be finite")
    if positive and np.any(arr <= 0):
        raise ValueError(f"priors[{key!r}] must be strictly positive")
    return arr


def _validate_priors(hp: dict[str, Any], n_dim: int) -> dict[str, Any]:
    """Check the hyper-parameters and normalise them to arrays/floats."""
    try:
        mu_mean, mu_sd = hp["mu"]
    except (TypeError, ValueError) as err:
        raise ValueError("priors['mu'] must be a (mean, sd) pair") from err
    out = dict(hp)
    out["mu"] = (
        _prior_array("mu (mean)", mu_mean, n_dim, positive=False),
        _prior_array("mu (sd)", mu_sd, n_dim, positive=True),
    )
    out["sigma"] = _prior_array("sigma", hp["sigma"], n_dim, positive=True)
    for key in ("eta", "nu_rate", "error_scale_sd"):
        val = np.asarray(hp[key], dtype=float)
        if val.ndim != 0 or not np.isfinite(val) or val <= 0:
            raise ValueError(f"priors[{key!r}] must be a positive finite scalar")
        out[key] = float(val)
    return out


def _value_arrays(z: CorrelationData) -> dict[str, np.ndarray]:
    """Return the data-dependent arrays of the model graph, in standardised units.

    These are the only arrays that change between data sets with the same
    uncertainty/censoring pattern, so they are the ones :func:`set_model_values` can
    replace when the model was built with ``mutable_values=True``.
    """
    free = z.is_latent & ~(z.is_exact & z.is_censored)
    unc = z.is_uncertain
    sym_obs = unc & z.is_observed & ~z.is_asymmetric
    asym_obs = unc & z.is_observed & z.is_asymmetric
    return {
        "values_z": z.values,
        "fixed_values_z": np.where(z.is_latent, 0.0, z.values),
        "free_init_z": _initial_latent(z)[np.nonzero(free)],
        "obs_values_z": z.values[np.nonzero(sym_obs)],
        "asym_values_z": z.values[np.nonzero(asym_obs)],
    }


_PATTERN_KEYS = ("sigma_lower", "sigma_upper", "censoring", "lower", "upper", "is_missing")


def _data_pattern(data: CorrelationData) -> dict[str, np.ndarray]:
    return {key: np.array(getattr(data, key), copy=True) for key in _PATTERN_KEYS}


def set_model_values(model: pm.Model, info: ModelInfo, data: CorrelationData) -> None:
    """Replace the observed values of a model built with ``mutable_values=True``.

    Only the *values* of the data may differ from those the model was built from: the
    uncertainties and the censoring (flags and limits) are compiled into the graph, so
    ``data`` must have identical ``sigma_lower``, ``sigma_upper``, ``censoring``,
    ``lower``, ``upper`` and ``is_missing`` arrays, and a mismatch raises ``ValueError``. Used to refit
    many simulated data sets without rebuilding and recompiling the model.
    """
    if not info.mutable_values or info.pattern is None:
        raise ValueError("The model was not built with mutable_values=True")
    if data.shape != (info.n_points, info.n_dim):
        raise ValueError("The new data must have the same shape as the original data")
    for key in _PATTERN_KEYS:
        if not np.array_equal(getattr(data, key), info.pattern[key], equal_nan=True):
            raise ValueError(
                f"The new data's {key} differs from the data the model was built from; "
                "only the values may change between calls to set_model_values"
            )
    arrays = _value_arrays(data.standardised(info.transform))
    pm.set_data({k: v for k, v in arrays.items() if k in model.named_vars}, model=model)


def _initial_latent(data: CorrelationData) -> np.ndarray:
    """Sensible starting values for the latent true values (standardised units)."""
    centre = Standardisation.from_data(data).loc
    init = np.where(data.is_missing, centre, data.values)
    up, lo, iv = data.is_upper_limit, data.is_lower_limit, data.is_interval
    init[up] = data.upper[up] - 0.5
    init[lo] = data.lower[lo] + 0.5
    init[iv] = 0.5 * (data.lower[iv] + data.upper[iv])
    return init


# --------------------------------------------------------------------------- builder
def build_model(
    data: CorrelationData,
    *,
    robust: bool = False,
    priors: dict[str, Any] | None = None,
    error_scale: bool = False,
    standardize: bool = True,
    transform: Standardisation | None = None,
    independent: bool = False,
    mutable_values: bool = False,
) -> tuple[pm.Model, ModelInfo]:
    """Build the PyMC model for ``data``.

    Parameters
    ----------
    data : CorrelationData
        The data set, in its original units.
    robust : bool, default False
        Use a multivariate Student-t distribution for the true values instead of a
        multivariate Normal. The degrees of freedom ``nu`` get a ``1 + Exponential``
        prior and are inferred; small ``nu`` means the data contain outliers.
    priors : dict, optional
        Overrides for :data:`DEFAULT_PRIORS`. Values are interpreted in standardised
        units when ``standardize=True`` and in data units otherwise.
    error_scale : bool, default False
        Infer a multiplicative factor ``error_scale[j]`` on the quoted uncertainties of
        each variable (log-normal prior centred on 1). This can absorb globally under-
        or over-estimated error bars, but it is only identifiable when the quoted
        uncertainties of a variable differ between points; with constant uncertainties
        it is degenerate with the variable's standard deviation.
    standardize : bool, default True
        Fit in per-variable standardised coordinates (robust location and scale from
        the observed data). This makes the default priors scale-free and improves
        sampling geometry. Correlations are invariant; ``mu``, ``chol_stds``, ``cov`` and
        ``vals`` are reported in the original units either way.
    transform : Standardisation, optional
        Use this transform instead of the one derived from the data.
    independent : bool, default False
        Fix the correlation matrix to the identity (the null model of no correlation).
        Used by :meth:`~pybaycor.BayesianCorrelation.null_pvalue` to simulate data under
        the null hypothesis.
    mutable_values : bool, default False
        Store the observed values in :func:`pymc.Data` containers so that
        :func:`set_model_values` can swap in another data set with the same
        uncertainty and censoring pattern without rebuilding the model.

    Returns
    -------
    model : pymc.Model
    info : ModelInfo
    """
    hp = dict(DEFAULT_PRIORS)
    if priors:
        unknown = set(priors) - set(DEFAULT_PRIORS)
        if unknown:
            raise ValueError(
                f"Unknown prior keys {sorted(unknown)}; valid: {sorted(DEFAULT_PRIORS)}"
            )
        hp.update(priors)

    n, d = data.shape
    hp = _validate_priors(hp, d)
    if error_scale and not data.has_uncertainties:
        warnings.warn(
            "error_scale=True has no effect because the data have no uncertainties; "
            "the error-scale factors are not added to the model",
            UserWarning,
            stacklevel=2,
        )
        error_scale = False
    if transform is None:
        transform = Standardisation.from_data(data) if standardize else Standardisation.identity(d)
    z = data.standardised(transform)
    loc = pt.as_tensor(transform.loc)
    scale = pt.as_tensor(transform.scale)
    scale_outer = np.outer(transform.scale, transform.scale)

    exact_censored = data.is_exact & data.is_censored
    if not data.is_latent.any():
        case = "observed"
    elif data.is_latent.all() and not exact_censored.any():
        case = "latent"
    else:
        # Mixed fixed/latent entries, or exact censored entries that need the bounded
        # reparameterisation below.
        case = "augmented"

    coords = {
        "point": np.arange(n),
        "feature": list(data.names),
        "feature2": list(data.names),
    }
    summary_vars = ["mu", "chol_stds", "chol_corr"]
    values = _value_arrays(z)

    def value_tensor(name: str) -> Any:
        """Return the named data array, as a :func:`pymc.Data` container when mutable."""
        return pm.Data(name, values[name]) if mutable_values else values[name]

    with pm.Model(coords=coords) as model:
        # ---- population-level parameters (standardised units) -------------------
        mu_mean, mu_sd = hp["mu"]
        mu_z = pm.Normal("mu_z", mu=mu_mean, sigma=mu_sd, dims="feature")
        if independent:
            stds_z = pm.HalfCauchy("stds_z", beta=hp["sigma"], dims="feature")
            chol = pt.diag(stds_z)
            corr = pt.eye(d)
        else:
            chol, corr, stds_z = pm.LKJCholeskyCov(
                "chol_z",
                n=d,
                eta=float(hp["eta"]),
                sd_dist=pm.HalfCauchy.dist(beta=hp["sigma"], shape=(d,)),
                compute_corr=True,
                store_in_trace=False,
            )
        cov_z = chol.dot(chol.T)

        # Public, data-unit versions of everything.
        pm.Deterministic("mu", mu_z * scale + loc, dims="feature")
        pm.Deterministic("chol_stds", stds_z * scale, dims="feature")
        pm.Deterministic("chol_corr", corr, dims=("feature", "feature2"))
        pm.Deterministic("cov", cov_z * scale_outer, dims=("feature", "feature2"))

        nu = None
        if robust:
            nu_minus_one = pm.Exponential("nu_minus_one", lam=float(hp["nu_rate"]))
            nu = pm.Deterministic("nu", nu_minus_one + 1.0)
            summary_vars.append("nu")

        def mv_dist(shape: tuple[int, ...]) -> Any:
            if robust:
                return pm.MvStudentT.dist(nu=nu, mu=mu_z, chol=chol, shape=shape)
            return pm.MvNormal.dist(mu=mu_z, chol=chol, shape=shape)

        # ---- the true values ------------------------------------------------------
        if case == "observed":
            kwargs = {"nu": nu} if robust else {}
            cls = pm.MvStudentT if robust else pm.MvNormal
            cls(
                "obs_z",
                mu=mu_z,
                chol=chol,
                observed=value_tensor("values_z"),
                dims=("point", "feature"),
                **kwargs,
            )
            info = ModelInfo(
                transform,
                robust,
                False,
                case,
                summary_vars,
                hp,
                n,
                d,
                list(data.names),
                independent,
                mutable_values=mutable_values,
                pattern=_data_pattern(data) if mutable_values else None,
            )
            return model, info

        init = _initial_latent(z)
        if case == "latent":
            # Every entry is latent: a proper multivariate RV per point.
            theta = (
                pm.MvStudentT(
                    "theta_z", nu=nu, mu=mu_z, chol=chol, dims=("point", "feature"), initval=init
                )
                if robust
                else pm.MvNormal(
                    "theta_z", mu=mu_z, chol=chol, dims=("point", "feature"), initval=init
                )
            )
        else:
            # Mixture of fixed and latent entries. The latent entries get a proper
            # "pseudo-prior" (so prior sampling and SMC work) whose density is subtracted
            # again inside the potential, leaving exactly the multivariate density on the
            # assembled rows. Entries that are exact *and* censored are reparameterised
            # so that they always respect their censoring bounds.
            theta = pt.as_tensor(value_tensor("fixed_values_z"))
            extra_logp = []

            free = z.is_latent & ~(z.is_exact & z.is_censored)
            if free.any():
                idx = np.nonzero(free)
                col_scale = Standardisation.from_data(z).scale
                pseudo_sd = np.where(
                    z.is_missing[idx], col_scale[idx[1]], np.maximum(z.sigma[idx], 1e-3)
                )
                raw = pm.Normal(
                    "theta_free",
                    mu=value_tensor("free_init_z"),
                    sigma=pseudo_sd,
                    shape=int(free.sum()),
                    initval=init[idx],
                )
                theta = pt.set_subtensor(theta[idx], raw)
                extra_logp.append(-pm.logp(raw, raw).sum())

            bounded = z.is_latent & z.is_exact & z.is_censored
            for kind, mask in (
                ("upper", bounded & z.is_upper_limit),
                ("lower", bounded & z.is_lower_limit),
                ("interval", bounded & z.is_interval),
            ):
                if not mask.any():
                    continue
                idx = np.nonzero(mask)
                raw = pm.Normal(f"theta_{kind}_raw", mu=0.0, sigma=1.0, shape=int(mask.sum()))
                extra_logp.append(-pm.logp(raw, raw).sum())
                if kind == "upper":
                    val = z.upper[idx] - pt.exp(raw)
                    jac = raw
                elif kind == "lower":
                    val = z.lower[idx] + pt.exp(raw)
                    jac = raw
                else:
                    width = z.upper[idx] - z.lower[idx]
                    val = z.lower[idx] + width * pt.sigmoid(raw)
                    jac = pt.log(width) - pt.softplus(-raw) - pt.softplus(raw)
                theta = pt.set_subtensor(theta[idx], val)
                extra_logp.append(jac.sum())

            pm.Potential(
                "latent_logp", pm.logp(mv_dist((n, d)), theta).sum() + pt.sum(pt.stack(extra_logp))
            )

        pm.Deterministic("vals", theta * scale + loc, dims=("point", "feature"))

        # ---- error scaling ---------------------------------------------------------
        if error_scale:
            f = pm.LogNormal(
                "error_scale", mu=0.0, sigma=float(hp["error_scale_sd"]), dims="feature"
            )
            summary_vars.append("error_scale")

        def scaled(s: np.ndarray, cols: np.ndarray) -> Any:
            return s * f[cols] if error_scale else pt.as_tensor(s)

        # ---- observation channels for uncertain entries ---------------------------
        unc = z.is_uncertain
        sym_obs = unc & z.is_observed & ~z.is_asymmetric
        if sym_obs.any():
            idx = np.nonzero(sym_obs)
            pm.Normal(
                "obs_z",
                mu=theta[idx],
                sigma=scaled(z.sigma[idx], idx[1]),
                observed=value_tensor("obs_values_z"),
            )

        asym_obs = unc & z.is_observed & z.is_asymmetric
        if asym_obs.any():
            idx = np.nonzero(asym_obs)
            pm.Potential(
                "obs_asym_logp",
                twopiece_logp(
                    value_tensor("asym_values_z"),
                    theta[idx],
                    scaled(z.sigma_lower[idx], idx[1]),
                    scaled(z.sigma_upper[idx], idx[1]),
                ).sum(),
            )

        up = unc & z.is_upper_limit
        if up.any():
            idx = np.nonzero(up)
            pm.Potential(
                "obs_upper_logp",
                twopiece_logcdf(
                    z.upper[idx],
                    theta[idx],
                    scaled(z.sigma_lower[idx], idx[1]),
                    scaled(z.sigma_upper[idx], idx[1]),
                ).sum(),
            )

        lo = unc & z.is_lower_limit
        if lo.any():
            idx = np.nonzero(lo)
            pm.Potential(
                "obs_lower_logp",
                twopiece_logsf(
                    z.lower[idx],
                    theta[idx],
                    scaled(z.sigma_lower[idx], idx[1]),
                    scaled(z.sigma_upper[idx], idx[1]),
                ).sum(),
            )

        iv = unc & z.is_interval
        if iv.any():
            idx = np.nonzero(iv)
            s_lo = scaled(z.sigma_lower[idx], idx[1])
            s_hi = scaled(z.sigma_upper[idx], idx[1])
            lcdf_hi = twopiece_logcdf(z.upper[idx], theta[idx], s_lo, s_hi)
            lcdf_lo = twopiece_logcdf(z.lower[idx], theta[idx], s_lo, s_hi)
            pm.Potential("obs_interval_logp", pm.math.logdiffexp(lcdf_hi, lcdf_lo).sum())

    info = ModelInfo(
        transform,
        robust,
        error_scale,
        case,
        summary_vars,
        hp,
        n,
        d,
        list(data.names),
        independent,
        mutable_values=mutable_values,
        pattern=_data_pattern(data) if mutable_values else None,
    )
    return model, info
