"""Input validation and representation of (possibly uncertain, censored) multivariate data.

The :class:`CorrelationData` container normalises the many ways a user can describe
a data set into a single canonical form that the model builder consumes:

* ``values``: ``(n_points, n_dim)`` observed values (``NaN`` where an entry is only
  known through its censoring limits),
* ``sigma_lower`` / ``sigma_upper``: ``(n_points, n_dim)`` measurement uncertainties
  below and above the value (equal for symmetric errors; ``0`` means *exact*),
* ``lower`` / ``upper``: ``(n_points, n_dim)`` censoring bounds (``-inf``/``+inf`` when
  unbounded).

Each entry of the data set is independently classified as exact or uncertain, and
as observed or censored (upper limit, lower limit or interval).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["CorrelationData", "Standardisation"]

# Integer codes used in the canonical ``censoring`` array.
OBSERVED = 0
UPPER_LIMIT = -1  # true value <= quoted value (left-censored)
LOWER_LIMIT = 1  # true value >= quoted value (right-censored)
INTERVAL = 2  # lower <= true value <= upper

_CENSORING_ALIASES: dict[Any, int] = {
    0: OBSERVED,
    "0": OBSERVED,
    "none": OBSERVED,
    "observed": OBSERVED,
    "obs": OBSERVED,
    "detected": OBSERVED,
    -1: UPPER_LIMIT,
    "-1": UPPER_LIMIT,
    "upper": UPPER_LIMIT,
    "upper_limit": UPPER_LIMIT,
    "upperlimit": UPPER_LIMIT,
    "left": UPPER_LIMIT,
    "<": UPPER_LIMIT,
    1: LOWER_LIMIT,
    "1": LOWER_LIMIT,
    "lower": LOWER_LIMIT,
    "lower_limit": LOWER_LIMIT,
    "lowerlimit": LOWER_LIMIT,
    "right": LOWER_LIMIT,
    ">": LOWER_LIMIT,
    2: INTERVAL,
    "2": INTERVAL,
    "interval": INTERVAL,
}


def _as_2d(name: str, arr: ArrayLike, shape: tuple[int, int] | None = None) -> NDArray[np.float64]:
    """Convert ``arr`` to a float ``(n_points, n_dim)`` array, broadcasting if allowed."""
    out = np.asarray(arr, dtype=float)
    if shape is not None:
        if out.ndim == 0:
            out = np.full(shape, float(out))
        elif out.ndim == 1:
            if out.shape[0] == shape[1]:
                out = np.broadcast_to(out[None, :], shape).copy()
            elif out.shape[0] == shape[0] and shape[1] == 1:
                out = out[:, None].copy()
            else:
                raise ValueError(
                    f"{name} has length {out.shape[0]} but should have one entry per "
                    f"variable ({shape[1]}) or shape {shape}"
                )
        elif out.shape != shape:
            raise ValueError(f"{name} has shape {out.shape} but data has shape {shape}")
    elif out.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (n_points, n_dim) array; got ndim={out.ndim}")
    return np.array(out, dtype=float, copy=True)


def _parse_censoring(censoring: ArrayLike, shape: tuple[int, int]) -> NDArray[np.int_]:
    """Map user-facing censoring flags onto the canonical integer codes."""
    raw = np.asarray(censoring, dtype=object)
    if raw.ndim == 0:
        raw = np.full(shape, raw.item(), dtype=object)
    elif raw.ndim == 1 and raw.shape[0] == shape[0] and shape[1] == 1:
        raw = raw[:, None]
    elif raw.shape != shape:
        raise ValueError(f"censoring has shape {raw.shape} but data has shape {shape}")
    out = np.empty(shape, dtype=int)
    for idx in np.ndindex(shape):
        val = raw[idx]
        if isinstance(val, (bool, np.bool_)):
            # A boolean mask is interpreted as "this value is an upper limit".
            out[idx] = UPPER_LIMIT if val else OBSERVED
            continue
        if isinstance(val, str):
            key: Any = val.strip().lower()
        elif isinstance(val, (int, np.integer)):
            key = int(val)
        elif isinstance(val, (float, np.floating)):
            if np.isnan(val):
                key = 0
            elif float(val).is_integer():
                key = int(val)
            else:
                key = val
        else:
            key = val
        try:
            out[idx] = _CENSORING_ALIASES[key]
        except (KeyError, TypeError) as err:
            raise ValueError(
                f"Unrecognised censoring flag {val!r} at {idx}; use 0/'observed', "
                "-1/'upper' (value is an upper limit), 1/'lower' (value is a lower limit) "
                "or 2/'interval' (with lower= and upper= bounds)."
            ) from err
    return out


@dataclass(frozen=True)
class Standardisation:
    """Per-variable affine transform ``z = (x - loc) / scale`` applied before fitting.

    Attributes
    ----------
    loc : ndarray of shape (n_dim,)
        Location subtracted from each variable.
    scale : ndarray of shape (n_dim,)
        Scale each variable is divided by (strictly positive).
    """

    loc: NDArray[np.float64]
    scale: NDArray[np.float64]

    @classmethod
    def identity(cls, n_dim: int) -> Standardisation:
        """Return the transform that leaves the data untouched."""
        return cls(np.zeros(n_dim), np.ones(n_dim))

    @classmethod
    def from_data(cls, data: CorrelationData) -> Standardisation:
        """Robust location/scale from the observed, uncensored entries of ``data``.

        The location is the median and the scale is ``1.4826 * MAD`` (falling back to
        the standard deviation, then to the typical uncertainty or censoring limit, then
        to 1) so that outliers do not dominate the standardisation.
        """
        loc = np.zeros(data.n_dim)
        scale = np.ones(data.n_dim)
        for j in range(data.n_dim):
            good = data.is_observed[:, j] & np.isfinite(data.values[:, j])
            vals = data.values[good, j]
            if vals.size == 0:
                # Only censored entries: centre on the limits instead.
                lim = np.where(np.isfinite(data.upper[:, j]), data.upper[:, j], data.lower[:, j])
                vals = lim[np.isfinite(lim)]
            if vals.size == 0:
                continue
            loc[j] = float(np.median(vals))
            s = 1.4826 * float(np.median(np.abs(vals - loc[j])))
            if not np.isfinite(s) or s <= 0:
                s = float(np.std(vals)) if vals.size > 1 else 0.0
            if not np.isfinite(s) or s <= 0:
                unc = data.sigma_upper[good, j]
                s = float(np.median(unc[unc > 0])) if np.any(unc > 0) else 0.0
            scale[j] = s if s > 0 else 1.0
        return cls(loc, scale)

    def forward(self, x: ArrayLike) -> NDArray[np.float64]:
        """Map data-space values of shape ``(..., n_dim)`` to standardised space."""
        return (np.asarray(x, dtype=float) - self.loc) / self.scale

    def inverse(self, z: ArrayLike) -> NDArray[np.float64]:
        """Map standardised values of shape ``(..., n_dim)`` back to data space."""
        return np.asarray(z, dtype=float) * self.scale + self.loc


@dataclass
class CorrelationData:
    """Canonical container for a multivariate data set with uncertainties and censoring.

    Parameters
    ----------
    data : array_like of shape (n_points, n_dim)
        Observed values. For a censored entry this is the limit (for one-sided
        censoring) or may be ``NaN`` (when ``lower``/``upper`` give the bounds).
    sigma : array_like, optional
        Measurement uncertainties. May be a scalar, an ``(n_dim,)`` vector (one value
        per variable), or an ``(n_points, n_dim)`` array. ``0`` or ``NaN`` marks an entry
        as *exact* (no measurement error), so variables with and without uncertainties
        can be mixed freely, as can points with and without uncertainties within a
        variable. A 2-tuple ``(sigma_lower, sigma_upper)`` of such arrays gives
        asymmetric uncertainties (two-piece Normal errors).
    censoring : array_like, optional
        Per-entry censoring flags with the same shape as ``data`` (or a scalar applied
        to everything). Each flag is one of ``0``/``"observed"``; ``-1``/``"upper"``
        (the quoted value is an *upper limit*: the true value lies below it);
        ``1``/``"lower"`` (the quoted value is a *lower limit*); or ``2``/``"interval"``
        (the true value lies between ``lower`` and ``upper``). A boolean array is read
        as a mask of upper limits.
    lower, upper : array_like, optional
        Explicit censoring bounds, same shape as ``data``; ``NaN`` or ``±inf`` means
        unbounded. Any entry with a finite bound here is treated as censored, which is
        the most general way of specifying arbitrary per-point, per-variable censoring.
    names : sequence of str, optional
        Variable names, used for labelling. Defaults to ``x0, x1, ...``.

    Attributes
    ----------
    values, sigma_lower, sigma_upper, lower, upper : ndarray of shape (n_points, n_dim)
        Canonical arrays described in the module docstring.
    censoring : ndarray of int, shape (n_points, n_dim)
        ``0`` observed, ``-1`` upper limit, ``1`` lower limit, ``2`` interval.

    Examples
    --------
    >>> import numpy as np
    >>> from pybaycor import CorrelationData
    >>> x = np.array([[1.0, 2.0], [2.0, 3.5], [0.5, 1.0]])
    >>> sig = np.array([[0.1, 0.0], [0.1, 0.0], [0.1, 0.0]])  # second variable exact
    >>> cd = CorrelationData(x, sigma=sig, censoring=[[0, 0], [0, 0], [-1, 0]])
    >>> cd.n_censored, cd.n_exact
    (1, 3)
    """

    values: NDArray[np.float64]
    sigma_lower: NDArray[np.float64] = field(repr=False)
    sigma_upper: NDArray[np.float64] = field(repr=False)
    lower: NDArray[np.float64] = field(repr=False)
    upper: NDArray[np.float64] = field(repr=False)
    censoring: NDArray[np.int_] = field(repr=False)
    names: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ construction
    def __init__(
        self,
        data: ArrayLike,
        sigma: ArrayLike | tuple[ArrayLike, ArrayLike] | None = None,
        censoring: ArrayLike | None = None,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        names: list[str] | None = None,
    ) -> None:
        values = np.asarray(data, dtype=float)
        if values.ndim == 1:
            raise ValueError(
                "data must be a 2-D (n_points, n_dim) array; for two 1-D arrays use "
                "np.column_stack((x, y)) or the x=/y= arguments of BayesianCorrelation"
            )
        if values.ndim != 2:
            raise ValueError(f"data must be 2-D (n_points, n_dim); got ndim={values.ndim}")
        n, d = values.shape
        if d < 2:
            raise ValueError("At least two variables are needed to infer a correlation")
        if n < 3:
            raise ValueError("At least three data points are needed to infer a correlation")
        values = values.copy()

        # --- uncertainties -------------------------------------------------------
        if sigma is None:
            s_lo = np.zeros((n, d))
            s_hi = np.zeros((n, d))
        elif isinstance(sigma, tuple) and len(sigma) == 2:
            s_lo = _as_2d("sigma[0]", sigma[0], (n, d))
            s_hi = _as_2d("sigma[1]", sigma[1], (n, d))
        else:
            s_lo = _as_2d("sigma", sigma, (n, d))
            s_hi = s_lo.copy()
        for name, s in (("sigma_lower", s_lo), ("sigma_upper", s_hi)):
            s[np.isnan(s)] = 0.0
            if np.any(s < 0):
                raise ValueError(f"{name} must be non-negative (0 or NaN means exact)")
            if np.any(~np.isfinite(s)):
                raise ValueError(f"{name} must be finite")
        # An entry is exact only if both scales are 0; a single 0 with a positive partner
        # is almost certainly an input error.
        mixed = (s_lo == 0) != (s_hi == 0)
        if np.any(mixed):
            raise ValueError(
                "Asymmetric uncertainties must be either both zero (exact) or both positive"
            )

        # --- censoring -----------------------------------------------------------
        cens = (
            np.full((n, d), OBSERVED, dtype=int)
            if censoring is None
            else _parse_censoring(censoring, (n, d))
        )
        lo = np.full((n, d), -np.inf) if lower is None else _as_2d("lower", lower, (n, d))
        hi = np.full((n, d), np.inf) if upper is None else _as_2d("upper", upper, (n, d))
        lo[np.isnan(lo)] = -np.inf
        hi[np.isnan(hi)] = np.inf

        # Flags -> bounds.
        is_up = cens == UPPER_LIMIT
        is_lo = cens == LOWER_LIMIT
        is_iv = cens == INTERVAL
        if np.any(~np.isfinite(values[is_up | is_lo])):
            raise ValueError("Entries flagged as upper/lower limits must have a finite value")
        hi = np.where(is_up, np.minimum(hi, values), hi)
        lo = np.where(is_lo, np.maximum(lo, values), lo)
        if np.any(is_iv & ~(np.isfinite(lo) & np.isfinite(hi))):
            raise ValueError("Interval-censored entries need finite lower= and upper= bounds")

        # Bounds -> flags (an explicit finite bound always implies censoring).
        has_lo = np.isfinite(lo)
        has_hi = np.isfinite(hi)
        cens = np.where(has_lo & has_hi, INTERVAL, cens)
        cens = np.where(has_hi & ~has_lo, UPPER_LIMIT, cens)
        cens = np.where(has_lo & ~has_hi, LOWER_LIMIT, cens)
        if np.any(lo >= hi):
            raise ValueError("Censoring bounds must satisfy lower < upper")

        # Observed entries must be finite; censored entries may be NaN.
        observed = cens == OBSERVED
        if np.any(~np.isfinite(values[observed])):
            raise ValueError(
                "data contains NaN/inf for an entry that is not censored; mark it as "
                "censored (censoring=, lower=, upper=) or remove the point"
            )
        # For censored entries without a point value, record the limit for plotting.
        missing = ~observed & ~np.isfinite(values)
        values = np.where(missing & has_hi, hi, values)
        values = np.where(missing & has_lo & ~has_hi, lo, values)

        self.values = values
        self.sigma_lower = s_lo
        self.sigma_upper = s_hi
        self.lower = lo
        self.upper = hi
        self.censoring = cens.astype(int)
        if names is None:
            self.names = [f"x{j}" for j in range(d)]
        else:
            self.names = [str(v) for v in names]
            if len(self.names) != d:
                raise ValueError(f"names has {len(self.names)} entries but data has {d} variables")

    # ------------------------------------------------------------------ properties
    @property
    def n_points(self) -> int:
        """Number of observations."""
        return int(self.values.shape[0])

    @property
    def n_dim(self) -> int:
        """Number of variables."""
        return int(self.values.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_points, n_dim)``."""
        return (self.n_points, self.n_dim)

    @property
    def sigma(self) -> NDArray[np.float64]:
        """Symmetric uncertainty (mean of the lower and upper scales)."""
        return 0.5 * (self.sigma_lower + self.sigma_upper)

    @property
    def is_asymmetric(self) -> NDArray[np.bool_]:
        """Mask of entries whose lower and upper uncertainties differ."""
        return self.sigma_lower != self.sigma_upper

    @property
    def is_exact(self) -> NDArray[np.bool_]:
        """Mask of entries without measurement uncertainty."""
        return (self.sigma_lower == 0) & (self.sigma_upper == 0)

    @property
    def is_uncertain(self) -> NDArray[np.bool_]:
        """Mask of entries with measurement uncertainty."""
        return ~self.is_exact

    @property
    def is_censored(self) -> NDArray[np.bool_]:
        """Mask of censored entries (any kind)."""
        return self.censoring != OBSERVED

    @property
    def is_observed(self) -> NDArray[np.bool_]:
        """Mask of uncensored entries."""
        return self.censoring == OBSERVED

    @property
    def is_upper_limit(self) -> NDArray[np.bool_]:
        """Mask of entries that are upper limits (true value below the quoted value)."""
        return self.censoring == UPPER_LIMIT

    @property
    def is_lower_limit(self) -> NDArray[np.bool_]:
        """Mask of entries that are lower limits (true value above the quoted value)."""
        return self.censoring == LOWER_LIMIT

    @property
    def is_interval(self) -> NDArray[np.bool_]:
        """Mask of interval-censored entries."""
        return self.censoring == INTERVAL

    @property
    def is_fixed(self) -> NDArray[np.bool_]:
        """Mask of entries that need no latent parameter (exact and observed)."""
        return self.is_exact & self.is_observed

    @property
    def is_latent(self) -> NDArray[np.bool_]:
        """Mask of entries whose true value is a parameter of the model."""
        return ~self.is_fixed

    @property
    def n_exact(self) -> int:
        """Number of exact entries."""
        return int(self.is_exact.sum())

    @property
    def n_censored(self) -> int:
        """Number of censored entries."""
        return int(self.is_censored.sum())

    @property
    def n_latent(self) -> int:
        """Number of latent true values the model must sample."""
        return int(self.is_latent.sum())

    @property
    def has_uncertainties(self) -> bool:
        """Whether any entry carries a measurement uncertainty."""
        return bool(self.is_uncertain.any())

    @property
    def has_censoring(self) -> bool:
        """Whether any entry is censored."""
        return bool(self.is_censored.any())

    # ------------------------------------------------------------------ helpers
    def describe(self) -> str:
        """Return a short human-readable description of the data set."""
        lines = [f"CorrelationData: {self.n_points} points x {self.n_dim} variables"]
        for j, name in enumerate(self.names):
            unc = int(self.is_uncertain[:, j].sum())
            asym = int(self.is_asymmetric[:, j].sum())
            up = int(self.is_upper_limit[:, j].sum())
            lo = int(self.is_lower_limit[:, j].sum())
            iv = int(self.is_interval[:, j].sum())
            bits = [f"{unc} uncertain"]
            if asym:
                bits.append(f"{asym} asymmetric")
            if up:
                bits.append(f"{up} upper limits")
            if lo:
                bits.append(f"{lo} lower limits")
            if iv:
                bits.append(f"{iv} interval-censored")
            lines.append(f"  {name}: " + ", ".join(bits))
        return "\n".join(lines)

    def standardised(self, transform: Standardisation) -> CorrelationData:
        """Return a copy of the data expressed in the standardised coordinates."""
        new = CorrelationData.__new__(CorrelationData)
        new.values = transform.forward(self.values)
        new.sigma_lower = self.sigma_lower / transform.scale
        new.sigma_upper = self.sigma_upper / transform.scale
        new.lower = transform.forward(self.lower)
        new.upper = transform.forward(self.upper)
        new.censoring = self.censoring.copy()
        new.names = list(self.names)
        return new

    @classmethod
    def from_xy(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        *,
        x_err: ArrayLike | None = None,
        y_err: ArrayLike | None = None,
        x_censoring: ArrayLike | None = None,
        y_censoring: ArrayLike | None = None,
        names: list[str] | None = None,
    ) -> CorrelationData:
        """Build a two-variable data set from separate ``x`` and ``y`` arrays.

        Parameters
        ----------
        x, y : array_like of shape (n_points,)
            The two variables.
        x_err, y_err : array_like, optional
            Uncertainties on ``x`` and ``y``; scalar, ``(n_points,)`` or a 2-tuple of
            such arrays for asymmetric errors. ``None`` means exact.
        x_censoring, y_censoring : array_like, optional
            Per-point censoring flags (see :class:`CorrelationData`).
        names : list of str, optional
            Variable names; defaults to ``["x", "y"]``.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")
        n = x.size
        values = np.column_stack((x, y))

        def _col(err: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            if err is None:
                z = np.zeros(n)
                return z, z.copy()
            if isinstance(err, tuple) and len(err) == 2:
                return (
                    np.broadcast_to(np.asarray(err[0], float), (n,)).copy(),
                    np.broadcast_to(np.asarray(err[1], float), (n,)).copy(),
                )
            e = np.broadcast_to(np.asarray(err, float), (n,)).copy()
            return e, e.copy()

        xl, xu = _col(x_err)
        yl, yu = _col(y_err)
        sigma = (np.column_stack((xl, yl)), np.column_stack((xu, yu)))
        censoring = None
        if x_censoring is not None or y_censoring is not None:
            xc = (
                np.zeros(n, dtype=object)
                if x_censoring is None
                else np.asarray(x_censoring, object)
            )
            yc = (
                np.zeros(n, dtype=object)
                if y_censoring is None
                else np.asarray(y_censoring, object)
            )
            censoring = np.column_stack((np.broadcast_to(xc, (n,)), np.broadcast_to(yc, (n,))))
        return cls(values, sigma=sigma, censoring=censoring, names=names or ["x", "y"])
