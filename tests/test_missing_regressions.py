"""Regression tests for missing-at-random inputs."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pybaycor import BayesianCorrelation, CorrelationData
import pybaycor.correlation as correlation_module


def test_null_pvalue_preserves_missing_mask(monkeypatch):
    rng = np.random.default_rng(17)
    data = rng.normal(size=(12, 2))
    data[[1, 4, 8], 1] = np.nan
    bc = BayesianCorrelation(data)
    bc.fit(method="map", progressbar=False)

    original_build_model = correlation_module.build_model
    simulated_masks = []

    def recording_build_model(input_data, **kwargs):
        if kwargs.get("mutable_values"):
            simulated_masks.append(input_data.is_missing.copy())
        return original_build_model(input_data, **kwargs)

    monkeypatch.setattr(correlation_module, "build_model", recording_build_model)
    bc.null_pvalue(n_sim=1, method="map", progressbar=False, random_seed=3)

    assert len(simulated_masks) == 1
    np.testing.assert_array_equal(simulated_masks[0], bc.data.is_missing)


def test_fully_missing_variable_raises():
    data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    with pytest.raises(ValueError, match="x1 has no non-missing entries"):
        CorrelationData(data)


def test_disjoint_observations_warn_pairwise():
    data = np.array([[1.0, np.nan], [2.0, np.nan], [np.nan, 3.0], [np.nan, 4.0]])
    with pytest.warns(UserWarning, match="both present in only 0 points"):
        CorrelationData(data)


def test_complete_data_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CorrelationData(np.arange(12.0).reshape(6, 2))
