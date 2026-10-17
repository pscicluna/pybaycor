"""Unit tests for :class:`pybaycor.data.CorrelationData` and :class:`Standardisation`."""

from __future__ import annotations

import numpy as np
import pytest
from pybaycor.data import (
    INTERVAL,
    LOWER_LIMIT,
    OBSERVED,
    UPPER_LIMIT,
    CorrelationData,
    Standardisation,
)


@pytest.fixture
def x(rng):
    return rng.normal(size=(12, 3))


@pytest.fixture
def x2(rng):
    return rng.normal(size=(10, 2))


# ------------------------------------------------------------------------- sigma
class TestSigma:
    def test_no_sigma_is_exact(self, x):
        cd = CorrelationData(x)
        assert cd.shape == (12, 3)
        assert cd.n_points == 12
        assert cd.n_dim == 3
        assert cd.is_exact.all()
        assert not cd.is_uncertain.any()
        assert not cd.has_uncertainties
        assert cd.n_exact == 36
        assert cd.n_latent == 0
        np.testing.assert_array_equal(cd.sigma, 0.0)
        np.testing.assert_array_equal(cd.values, x)

    def test_values_are_copied(self, x):
        cd = CorrelationData(x)
        x[0, 0] = 99.0
        assert cd.values[0, 0] != 99.0

    def test_scalar_sigma(self, x):
        cd = CorrelationData(x, sigma=0.5)
        assert cd.sigma_lower.shape == (12, 3)
        np.testing.assert_array_equal(cd.sigma_lower, 0.5)
        np.testing.assert_array_equal(cd.sigma_upper, 0.5)
        assert cd.is_uncertain.all()
        assert not cd.is_asymmetric.any()
        assert cd.has_uncertainties

    def test_per_variable_sigma(self, x):
        cd = CorrelationData(x, sigma=[0.1, 0.0, 0.3])
        np.testing.assert_array_equal(cd.sigma[:, 0], 0.1)
        np.testing.assert_array_equal(cd.sigma[:, 1], 0.0)
        np.testing.assert_array_equal(cd.sigma[:, 2], 0.3)
        assert cd.is_exact[:, 1].all()
        assert cd.is_uncertain[:, [0, 2]].all()
        assert cd.n_exact == 12
        assert cd.n_latent == 24

    def test_full_sigma(self, x, rng):
        sig = rng.uniform(0.1, 1.0, size=x.shape)
        cd = CorrelationData(x, sigma=sig)
        np.testing.assert_array_equal(cd.sigma_lower, sig)
        assert cd.sigma_lower is not sig

    def test_wrong_sigma_length_raises(self, x):
        with pytest.raises(ValueError, match="sigma has length 4"):
            CorrelationData(x, sigma=np.ones(4))

    def test_wrong_sigma_shape_raises(self, x):
        with pytest.raises(ValueError, match="sigma has shape"):
            CorrelationData(x, sigma=np.ones((5, 3)))

    def test_nan_and_zero_sigma_are_exact(self, x):
        sig = np.full(x.shape, 0.2)
        sig[0, 0] = np.nan
        sig[1, 1] = 0.0
        cd = CorrelationData(x, sigma=sig)
        assert cd.is_exact[0, 0]
        assert cd.is_exact[1, 1]
        assert cd.sigma_lower[0, 0] == 0.0
        assert cd.n_exact == 2
        assert cd.is_uncertain.sum() == 34
        assert cd.is_latent.sum() == 34

    def test_negative_sigma_raises(self, x):
        with pytest.raises(ValueError, match="non-negative"):
            CorrelationData(x, sigma=-0.1)

    def test_infinite_sigma_raises(self, x):
        sig = np.full(x.shape, 0.2)
        sig[0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            CorrelationData(x, sigma=sig)

    def test_tuple_sigma_is_asymmetric(self, x):
        lo = np.full(x.shape, 0.1)
        hi = np.full(x.shape, 0.3)
        hi[0, 0] = 0.1  # one symmetric entry
        cd = CorrelationData(x, sigma_lower=lo, sigma_upper=hi)
        np.testing.assert_array_equal(cd.sigma_lower, lo)
        np.testing.assert_array_equal(cd.sigma_upper, hi)
        assert cd.is_asymmetric.sum() == 35
        assert not cd.is_asymmetric[0, 0]
        np.testing.assert_allclose(cd.sigma, 0.5 * (lo + hi))
        assert cd.is_uncertain.all()

    def test_asymmetric_sigma_broadcasts(self, x):
        cd = CorrelationData(x, sigma_lower=0.1, sigma_upper=[0.2, 0.3, 0.4])
        np.testing.assert_array_equal(cd.sigma_lower, 0.1)
        np.testing.assert_array_equal(cd.sigma_upper[:, 2], 0.4)

    def test_tuple_sigma_is_symmetric_per_variable(self, x):
        # A tuple is just a sequence: one symmetric value per variable, never (lo, hi).
        cd = CorrelationData(x, sigma=(0.1, 0.2, 0.3))
        assert not cd.is_asymmetric.any()
        np.testing.assert_array_equal(cd.sigma[:, 2], 0.3)

    def test_asymmetric_requires_both_bounds(self, x):
        with pytest.raises(ValueError, match="given together"):
            CorrelationData(x, sigma_lower=0.1)
        with pytest.raises(ValueError, match="given together"):
            CorrelationData(x, sigma_upper=0.1)

    def test_asymmetric_excludes_symmetric(self, x):
        with pytest.raises(ValueError, match="not both"):
            CorrelationData(x, sigma=0.1, sigma_lower=0.1, sigma_upper=0.2)

    def test_1d_sigma_ambiguous_when_square(self, rng):
        square = rng.normal(size=(3, 3))
        with pytest.raises(ValueError, match="2-D"):
            CorrelationData(square, sigma=[0.1, 1.0, 10.0])
        cd = CorrelationData(square, sigma=np.full((3, 3), 0.1))
        assert cd.is_uncertain.all()

    def test_mixed_zero_positive_asymmetric_raises(self, x):
        lo = np.zeros(x.shape)
        hi = np.full(x.shape, 0.3)
        with pytest.raises(ValueError, match=r"both zero .* or both positive"):
            CorrelationData(x, sigma_lower=lo, sigma_upper=hi)

    def test_tuple_sigma_both_zero_is_exact(self, x):
        lo = np.full(x.shape, 0.2)
        hi = np.full(x.shape, 0.4)
        lo[2, 1] = hi[2, 1] = 0.0
        cd = CorrelationData(x, sigma_lower=lo, sigma_upper=hi)
        assert cd.is_exact[2, 1]
        assert cd.n_exact == 1


# --------------------------------------------------------------------- censoring
class TestCensoring:
    def test_default_observed(self, x):
        cd = CorrelationData(x)
        assert cd.censoring.dtype.kind == "i"
        np.testing.assert_array_equal(cd.censoring, OBSERVED)
        assert cd.is_observed.all()
        assert not cd.has_censoring
        assert cd.n_censored == 0
        np.testing.assert_array_equal(cd.lower, -np.inf)
        np.testing.assert_array_equal(cd.upper, np.inf)

    def test_integer_flags(self, x):
        cens = np.zeros(x.shape, dtype=int)
        cens[0, 0] = -1
        cens[1, 1] = 1
        cd = CorrelationData(x, censoring=cens)
        assert cd.censoring[0, 0] == UPPER_LIMIT
        assert cd.censoring[1, 1] == LOWER_LIMIT
        assert cd.is_upper_limit[0, 0]
        assert cd.is_lower_limit[1, 1]
        assert cd.n_censored == 2
        assert cd.has_censoring
        # Upper limit: the quoted value becomes the upper bound.
        assert cd.upper[0, 0] == x[0, 0]
        assert cd.lower[0, 0] == -np.inf
        # Lower limit: the quoted value becomes the lower bound.
        assert cd.lower[1, 1] == x[1, 1]
        assert cd.upper[1, 1] == np.inf
        # The quoted value is kept for plotting.
        np.testing.assert_array_equal(cd.values, x)

    def test_float_integer_flags(self, x):
        cens = np.zeros(x.shape)
        cens[0, 0] = -1.0
        cens[1, 1] = np.nan  # NaN -> observed
        cd = CorrelationData(x, censoring=cens)
        assert cd.censoring[0, 0] == UPPER_LIMIT
        assert cd.censoring[1, 1] == OBSERVED

    @pytest.mark.parametrize(
        ("flag", "code"),
        [
            ("upper", UPPER_LIMIT),
            ("<", UPPER_LIMIT),
            ("UPPER_LIMIT", UPPER_LIMIT),
            ("left", UPPER_LIMIT),
            ("lower", LOWER_LIMIT),
            (">", LOWER_LIMIT),
            (" Lower ", LOWER_LIMIT),
            ("right", LOWER_LIMIT),
            ("observed", OBSERVED),
            ("none", OBSERVED),
            ("detected", OBSERVED),
            ("0", OBSERVED),
            ("-1", UPPER_LIMIT),
            ("1", LOWER_LIMIT),
        ],
    )
    def test_string_flags(self, x, flag, code):
        cens = np.full(x.shape, "observed", dtype=object)
        cens[3, 2] = flag
        cd = CorrelationData(x, censoring=cens)
        assert cd.censoring[3, 2] == code
        assert cd.censoring.sum() == code

    def test_interval_string_flag(self, x):
        cens = np.full(x.shape, "observed", dtype=object)
        cens[3, 2] = "interval"
        lo = np.full(x.shape, np.nan)
        hi = np.full(x.shape, np.nan)
        lo[3, 2], hi[3, 2] = -1.0, 1.0
        cd = CorrelationData(x, censoring=cens, lower=lo, upper=hi)
        assert cd.censoring[3, 2] == INTERVAL
        assert cd.is_interval[3, 2]

    def test_bool_mask_marks_upper_limits(self, x):
        mask = np.zeros(x.shape, dtype=bool)
        mask[0, 1] = True
        cd = CorrelationData(x, censoring=mask)
        assert cd.censoring[0, 1] == UPPER_LIMIT
        assert cd.n_censored == 1
        assert cd.upper[0, 1] == x[0, 1]

    def test_scalar_flag_applies_everywhere(self, x):
        cd = CorrelationData(x, censoring="upper")
        assert (cd.censoring == UPPER_LIMIT).all()
        assert cd.n_censored == x.size
        np.testing.assert_array_equal(cd.upper, x)

    def test_unknown_flag_raises(self, x):
        cens = np.zeros(x.shape, dtype=object)
        cens[0, 0] = "bogus"
        with pytest.raises(ValueError, match="Unrecognised censoring flag 'bogus'"):
            CorrelationData(x, censoring=cens)

    def test_wrong_censoring_shape_raises(self, x):
        with pytest.raises(ValueError, match="censoring has shape"):
            CorrelationData(x, censoring=np.zeros((3, 3), dtype=int))

    def test_limit_flag_needs_finite_value(self, x):
        xx = x.copy()
        xx[0, 0] = np.nan
        cens = np.zeros(x.shape, dtype=int)
        cens[0, 0] = -1
        with pytest.raises(ValueError, match="finite value"):
            CorrelationData(xx, censoring=cens)

    def test_lower_upper_arrays_imply_censoring(self, x):
        lo = np.full(x.shape, np.nan)
        hi = np.full(x.shape, np.nan)
        hi[0, 0] = 0.5  # upper limit
        lo[1, 1] = -0.5  # lower limit
        lo[2, 2], hi[2, 2] = -1.0, 1.0  # interval
        cd = CorrelationData(x, lower=lo, upper=hi)
        assert cd.censoring[0, 0] == UPPER_LIMIT
        assert cd.censoring[1, 1] == LOWER_LIMIT
        assert cd.censoring[2, 2] == INTERVAL
        assert cd.n_censored == 3
        assert cd.upper[0, 0] == 0.5
        assert cd.lower[1, 1] == -0.5
        assert (cd.lower[2, 2], cd.upper[2, 2]) == (-1.0, 1.0)
        # Everything else stays unbounded and observed.
        assert np.isinf(cd.lower[0, 0]) and np.isinf(cd.upper[1, 1])
        assert cd.is_observed.sum() == x.size - 3

    def test_infinite_bounds_mean_unbounded(self, x):
        lo = np.full(x.shape, -np.inf)
        hi = np.full(x.shape, np.inf)
        cd = CorrelationData(x, lower=lo, upper=hi)
        assert not cd.has_censoring

    def test_flag_and_explicit_bound_combine(self, x):
        # An upper-limit flag plus an explicit bound takes the tighter of the two.
        cens = np.zeros(x.shape, dtype=int)
        cens[0, 0] = -1
        hi = np.full(x.shape, np.nan)
        hi[0, 0] = x[0, 0] - 1.0
        cd = CorrelationData(x, censoring=cens, upper=hi)
        assert cd.upper[0, 0] == x[0, 0] - 1.0
        assert cd.censoring[0, 0] == UPPER_LIMIT

    def test_interval_flag_needs_both_bounds(self, x):
        cens = np.zeros(x.shape, dtype=int)
        cens[0, 0] = 2
        with pytest.raises(ValueError, match="Interval-censored entries need finite"):
            CorrelationData(x, censoring=cens)
        hi = np.full(x.shape, np.nan)
        hi[0, 0] = 1.0
        with pytest.raises(ValueError, match="Interval-censored entries need finite"):
            CorrelationData(x, censoring=cens, upper=hi)

    def test_lower_ge_upper_raises(self, x):
        lo = np.full(x.shape, np.nan)
        hi = np.full(x.shape, np.nan)
        lo[0, 0], hi[0, 0] = 1.0, 1.0
        with pytest.raises(ValueError, match="lower < upper"):
            CorrelationData(x, lower=lo, upper=hi)
        lo[0, 0], hi[0, 0] = 2.0, 1.0
        with pytest.raises(ValueError, match="lower < upper"):
            CorrelationData(x, lower=lo, upper=hi)

    def test_nan_in_observed_entry_is_missing(self, x):
        xx = x.copy()
        xx[4, 1] = np.nan
        cd = CorrelationData(xx)
        assert cd.is_missing[4, 1]
        assert cd.n_missing == 1
        assert cd.n_latent == 1
        assert cd.n_exact == x.size - 1
        assert not cd.is_observed[4, 1]
        assert not cd.is_uncertain[4, 1]
        assert not cd.is_censored[4, 1]
        assert np.isnan(cd.values[4, 1])

    def test_missing_entries_ignore_uncertainties(self, x):
        xx = x.copy()
        xx[0, 0] = np.nan
        low = np.zeros(x.shape)
        high = np.zeros(x.shape)
        low[0, 0], high[0, 0] = -1.0, np.inf
        cd = CorrelationData(xx, sigma_lower=low, sigma_upper=high)
        assert cd.n_missing == 1
        assert cd.sigma_lower[0, 0] == cd.sigma_upper[0, 0] == 0.0
        assert not cd.is_asymmetric[0, 0]

    def test_missing_with_bounds_remains_censored(self, x):
        xx = x.copy()
        xx[0, 0] = np.nan
        bounds = np.full(x.shape, np.nan)
        bounds[0, 0] = 2.0
        cd = CorrelationData(xx, upper=bounds, sigma=0.3)
        assert cd.n_missing == 0
        assert cd.is_upper_limit[0, 0]
        assert cd.is_uncertain[0, 0]

    def test_few_non_missing_entries_warn(self, x):
        xx = x.copy()
        xx[:-2, 1] = np.nan
        with pytest.warns(UserWarning, match="Variable x1.*weakly identified"):
            cd = CorrelationData(xx)
        assert cd.n_missing == x.shape[0] - 2
        assert f"{x.shape[0] - 2} missing" in cd.describe()
        assert np.isfinite(Standardisation.from_data(cd).loc).all()

    def test_inf_in_observed_entry_raises(self, x):
        xx = x.copy()
        xx[4, 1] = np.inf
        with pytest.raises(ValueError, match="not censored"):
            CorrelationData(xx)

    def test_nan_in_censored_entry_replaced_by_limit(self, x):
        xx = x.copy()
        xx[0, 0] = np.nan  # upper limit
        xx[1, 1] = np.nan  # lower limit
        xx[2, 2] = np.nan  # interval
        lo = np.full(x.shape, np.nan)
        hi = np.full(x.shape, np.nan)
        hi[0, 0] = 0.5
        lo[1, 1] = -0.5
        lo[2, 2], hi[2, 2] = -1.0, 1.0
        cd = CorrelationData(xx, lower=lo, upper=hi)
        assert np.isfinite(cd.values).all()
        assert cd.values[0, 0] == 0.5
        assert cd.values[1, 1] == -0.5
        assert cd.values[2, 2] in (-1.0, 1.0)

    def test_fixed_and_latent_masks(self, x):
        sig = np.zeros(x.shape)
        sig[:, 0] = 0.1
        cens = np.zeros(x.shape, dtype=int)
        cens[0, 1] = -1
        cd = CorrelationData(x, sigma=sig, censoring=cens)
        # Fixed = exact & observed; latent = everything else.
        assert cd.is_fixed[1, 1] and cd.is_fixed[0, 2]
        assert not cd.is_fixed[0, 1]  # exact but censored
        assert not cd.is_fixed[0, 0]  # uncertain
        np.testing.assert_array_equal(cd.is_latent, ~cd.is_fixed)
        assert cd.n_latent == 12 + 1
        assert cd.n_exact == 24
        assert cd.n_censored == 1


# ------------------------------------------------------------------- validation
class TestValidation:
    def test_1d_data_raises_with_hint(self, rng):
        with pytest.raises(ValueError, match=r"column_stack|x=/y="):
            CorrelationData(rng.normal(size=10))

    def test_3d_data_raises(self, rng):
        with pytest.raises(ValueError, match="2-D"):
            CorrelationData(rng.normal(size=(3, 3, 3)))

    def test_too_few_points_raises(self, rng):
        with pytest.raises(ValueError, match="three data points"):
            CorrelationData(rng.normal(size=(2, 2)))

    def test_one_variable_raises(self, rng):
        with pytest.raises(ValueError, match="two variables"):
            CorrelationData(rng.normal(size=(10, 1)))

    def test_names_default_and_custom(self, x):
        assert CorrelationData(x).names == ["x0", "x1", "x2"]
        cd = CorrelationData(x, names=("a", "b", 3))
        assert cd.names == ["a", "b", "3"]

    def test_names_length_mismatch_raises(self, x):
        with pytest.raises(ValueError, match="names has 2 entries but data has 3"):
            CorrelationData(x, names=["a", "b"])

    def test_list_input(self):
        cd = CorrelationData([[1, 2], [3, 4], [5, 6.5]])
        assert cd.values.dtype == float
        assert cd.shape == (3, 2)

    def test_repr_mentions_values_only(self, x):
        r = repr(CorrelationData(x))
        assert "CorrelationData" in r
        assert "sigma_lower" not in r


# --------------------------------------------------------------------- describe
class TestDescribe:
    def test_describe_counts(self, x):
        lo = np.full(x.shape, 0.1)
        hi = np.full(x.shape, 0.1)
        hi[:2, 0] = 0.3  # two asymmetric entries in x0
        cens = np.zeros(x.shape, dtype=object)
        cens[5, 0] = "upper"
        cens[6, 1] = "lower"
        cens[7, 1] = "lower"
        cens[8, 2] = "interval"
        lob = np.full(x.shape, np.nan)
        hib = np.full(x.shape, np.nan)
        lob[8, 2], hib[8, 2] = -1.0, 1.0
        cd = CorrelationData(
            x,
            sigma_lower=lo,
            sigma_upper=hi,
            censoring=cens,
            lower=lob,
            upper=hib,
            names=["a", "b", "c"],
        )
        text = cd.describe()
        assert "12 points x 3 variables" in text
        assert "a: 12 uncertain, 2 asymmetric, 1 upper limits" in text
        assert "b: 12 uncertain, 2 lower limits" in text
        assert "c: 12 uncertain, 1 interval-censored" in text

    def test_describe_exact(self, x):
        text = CorrelationData(x).describe()
        assert "x0: 0 uncertain" in text
        assert "limits" not in text


class TestCensoringParser:
    """The vectorised flag parser accepts the same spellings as the old per-entry loop."""

    def test_object_array_with_mixed_spellings(self, x):
        flags = np.zeros(x.shape, dtype=object)
        flags[0, 0] = "Upper "
        flags[1, 0] = -1.0
        flags[2, 1] = "1"
        flags[3, 1] = True
        flags[4, 2] = None
        flags[5, 2] = np.nan
        flags[6, 2] = "<"
        cd = CorrelationData(x, censoring=flags)
        assert cd.censoring[0, 0] == UPPER_LIMIT
        assert cd.censoring[1, 0] == UPPER_LIMIT
        assert cd.censoring[2, 1] == LOWER_LIMIT
        assert cd.censoring[3, 1] == UPPER_LIMIT
        assert cd.censoring[4, 2] == OBSERVED
        assert cd.censoring[5, 2] == OBSERVED
        assert cd.censoring[6, 2] == UPPER_LIMIT
        assert cd.n_censored == 5

    def test_integer_array_fast_path_validates(self, x):
        flags = np.zeros(x.shape, dtype=int)
        flags[2, 1] = 7
        with pytest.raises(ValueError, match=r"Unrecognised censoring flag 7 at \(2, 1\)"):
            CorrelationData(x, censoring=flags)

    def test_unknown_string_reports_position(self, x):
        flags = np.full(x.shape, "observed", dtype=object)
        flags[4, 0] = "maybe"
        with pytest.raises(ValueError, match=r"'maybe' at \(4, 0\)"):
            CorrelationData(x, censoring=flags)

    def test_large_all_zero_is_fast(self):
        big = np.random.default_rng(0).normal(size=(50_000, 4))
        cd = CorrelationData(big, censoring=np.zeros(big.shape, dtype=int))
        assert cd.n_censored == 0
        cd = CorrelationData(big, censoring=np.full(big.shape, "observed"))
        assert cd.n_censored == 0


# ---------------------------------------------------------------------- from_xy
class TestFromXY:
    def test_basic(self, rng):
        xv, yv = rng.normal(size=10), rng.normal(size=10)
        cd = CorrelationData.from_xy(xv, yv)
        assert cd.names == ["x", "y"]
        np.testing.assert_array_equal(cd.values, np.column_stack((xv, yv)))
        assert cd.is_exact.all()

    def test_errors_scalar_and_vector(self, rng):
        xv, yv = rng.normal(size=10), rng.normal(size=10)
        yerr = rng.uniform(0.1, 0.5, size=10)
        cd = CorrelationData.from_xy(xv, yv, x_err=0.2, y_err=yerr, names=["u", "v"])
        assert cd.names == ["u", "v"]
        np.testing.assert_array_equal(cd.sigma[:, 0], 0.2)
        np.testing.assert_array_equal(cd.sigma[:, 1], yerr)
        assert not cd.is_asymmetric.any()

    def test_asymmetric_errors(self, rng):
        xv, yv = rng.normal(size=10), rng.normal(size=10)
        cd = CorrelationData.from_xy(xv, yv, x_err_lower=0.1, x_err_upper=0.3, y_err=None)
        np.testing.assert_array_equal(cd.sigma_lower[:, 0], 0.1)
        np.testing.assert_array_equal(cd.sigma_upper[:, 0], 0.3)
        assert cd.is_asymmetric[:, 0].all()
        assert cd.is_exact[:, 1].all()

    def test_asymmetric_errors_validation(self, rng):
        xv, yv = rng.normal(size=10), rng.normal(size=10)
        with pytest.raises(ValueError, match="given together"):
            CorrelationData.from_xy(xv, yv, y_err_lower=0.1)
        with pytest.raises(ValueError, match="not both"):
            CorrelationData.from_xy(xv, yv, x_err=0.2, x_err_lower=0.1, x_err_upper=0.3)

    def test_censoring(self, rng):
        xv, yv = rng.normal(size=10), rng.normal(size=10)
        ycens = ["upper", 0, 0, 0, "lower", 0, 0, 0, 0, 0]
        cd = CorrelationData.from_xy(xv, yv, y_censoring=ycens)
        assert cd.censoring[0, 1] == UPPER_LIMIT
        assert cd.censoring[4, 1] == LOWER_LIMIT
        assert cd.is_observed[:, 0].all()
        assert cd.n_censored == 2
        mask = np.zeros(10, dtype=bool)
        mask[1] = True
        cd = CorrelationData.from_xy(xv, yv, x_censoring=mask)
        assert cd.censoring[1, 0] == UPPER_LIMIT
        assert cd.n_censored == 1

    def test_length_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="same length"):
            CorrelationData.from_xy(rng.normal(size=10), rng.normal(size=9))


# -------------------------------------------------------------- standardisation
class TestStandardisation:
    def test_identity(self, x):
        st = Standardisation.identity(3)
        np.testing.assert_array_equal(st.loc, 0.0)
        np.testing.assert_array_equal(st.scale, 1.0)
        np.testing.assert_array_equal(st.forward(x), x)
        np.testing.assert_array_equal(st.inverse(x), x)

    def test_from_data_median_mad(self, x):
        cd = CorrelationData(x)
        st = Standardisation.from_data(cd)
        for j in range(3):
            med = np.median(x[:, j])
            mad = np.median(np.abs(x[:, j] - med))
            assert st.loc[j] == pytest.approx(med)
            assert st.scale[j] == pytest.approx(1.4826 * mad)
        assert (st.scale > 0).all()

    def test_from_data_uses_only_observed_entries(self, x):
        cens = np.zeros(x.shape, dtype=int)
        cens[:4, 0] = -1
        cd = CorrelationData(x, censoring=cens)
        st = Standardisation.from_data(cd)
        good = x[4:, 0]
        med = np.median(good)
        assert st.loc[0] == pytest.approx(med)
        assert st.scale[0] == pytest.approx(1.4826 * np.median(np.abs(good - med)))
        # Other columns are unaffected.
        assert st.loc[1] == pytest.approx(np.median(x[:, 1]))

    def test_from_data_ignores_uncertainties_for_location(self, x):
        cd = CorrelationData(x, sigma=100.0)
        st = Standardisation.from_data(cd)
        assert st.loc[0] == pytest.approx(np.median(x[:, 0]))

    def test_all_censored_column_gets_finite_scale(self, x):
        cens = np.zeros(x.shape, dtype=int)
        cens[:, 1] = -1
        cd = CorrelationData(x, censoring=cens)
        st = Standardisation.from_data(cd)
        assert np.isfinite(st.loc).all()
        assert np.isfinite(st.scale).all()
        assert (st.scale > 0).all()
        # The limits are used as a stand-in for the values.
        assert st.loc[1] == pytest.approx(np.median(x[:, 1]))

    def test_zero_mad_falls_back_to_std(self, x):
        xx = x.copy()
        xx[:, 2] = 1.0
        xx[0, 2] = 3.0  # MAD is 0, std is not
        st = Standardisation.from_data(CorrelationData(xx))
        assert st.scale[2] == pytest.approx(np.std(xx[:, 2]))
        assert st.scale[2] > 0

    def test_constant_column_falls_back_to_uncertainty_then_one(self, x):
        xx = x.copy()
        xx[:, 2] = 1.0
        st = Standardisation.from_data(CorrelationData(xx))
        assert st.scale[2] == 1.0
        sig = np.zeros(x.shape)
        sig[:, 2] = 0.25
        st = Standardisation.from_data(CorrelationData(xx, sigma=sig))
        assert st.scale[2] == pytest.approx(0.25)

    def test_round_trip(self, x):
        st = Standardisation.from_data(CorrelationData(x))
        np.testing.assert_allclose(st.inverse(st.forward(x)), x)
        np.testing.assert_allclose(st.forward(st.inverse(x)), x)
        # Broadcasting over leading dimensions.
        stacked = np.stack([x, x])
        np.testing.assert_allclose(st.inverse(st.forward(stacked)), stacked)

    def test_forward_is_affine(self):
        st = Standardisation(np.array([1.0, -2.0]), np.array([2.0, 4.0]))
        np.testing.assert_allclose(st.forward([[3.0, 2.0]]), [[1.0, 1.0]])
        np.testing.assert_allclose(st.inverse([[1.0, 1.0]]), [[3.0, 2.0]])

    def test_standardised_scales_everything_consistently(self, x):
        sig_lo = np.full(x.shape, 0.2)
        sig_hi = np.full(x.shape, 0.4)
        lo = np.full(x.shape, np.nan)
        hi = np.full(x.shape, np.nan)
        hi[0, 0] = 0.5
        lo[1, 1] = -0.5
        lo[2, 2], hi[2, 2] = -1.0, 1.0
        cd = CorrelationData(
            x, sigma_lower=sig_lo, sigma_upper=sig_hi, lower=lo, upper=hi, names=["a", "b", "c"]
        )
        st = Standardisation(np.array([1.0, -2.0, 0.5]), np.array([2.0, 4.0, 0.5]))
        z = cd.standardised(st)
        assert isinstance(z, CorrelationData)
        np.testing.assert_allclose(z.values, (x - st.loc) / st.scale)
        np.testing.assert_allclose(z.sigma_lower, sig_lo / st.scale)
        np.testing.assert_allclose(z.sigma_upper, sig_hi / st.scale)
        assert z.upper[0, 0] == pytest.approx((0.5 - 1.0) / 2.0)
        assert z.lower[1, 1] == pytest.approx((-0.5 + 2.0) / 4.0)
        assert z.lower[2, 2] == pytest.approx((-1.0 - 0.5) / 0.5)
        assert z.upper[2, 2] == pytest.approx((1.0 - 0.5) / 0.5)
        # Unbounded entries stay unbounded.
        assert np.isneginf(z.lower[0, 0]) and np.isposinf(z.upper[1, 1])
        assert np.isneginf(z.lower[z.is_observed]).all()
        assert np.isposinf(z.upper[z.is_observed]).all()
        # Classification, names and asymmetry are preserved.
        np.testing.assert_array_equal(z.censoring, cd.censoring)
        np.testing.assert_array_equal(z.is_asymmetric, cd.is_asymmetric)
        np.testing.assert_array_equal(z.is_exact, cd.is_exact)
        assert z.names == ["a", "b", "c"]
        assert z.shape == cd.shape
        # The original is untouched.
        assert cd.upper[0, 0] == 0.5

    def test_standardised_is_a_copy(self, x):
        cd = CorrelationData(x)
        z = cd.standardised(Standardisation.identity(3))
        z.values[0, 0] = 123.0
        z.censoring[0, 0] = -1
        assert cd.values[0, 0] != 123.0
        assert cd.censoring[0, 0] == OBSERVED
