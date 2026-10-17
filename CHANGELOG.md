# Changelog

All notable changes to pybaycor are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - unreleased

A ground-up modernisation. See `ROADMAP.md` for the analysis and the plan.

### Changed (breaking)

- Requires Python ≥ 3.12 and PyMC ≥ 6.1 / PyTensor 3 / ArviZ ≥ 1.1 instead of
  PyMC3 3.10 / ArviZ 0.11 ([#7](https://github.com/pscicluna/pybaycor/issues/7)).
- `fit()` returns an `xarray.DataTree` (ArviZ 1 replaced `InferenceData`); the
  `steps=` argument is deprecated in favour of `draws=`.
- Summary tables use ArviZ 1 defaults (89% equal-tailed intervals) and label the
  correlation entries by variable name, e.g. `chol_corr[x, y]` instead of
  `chol_corr[0, 1]`. Pass `names=` to choose the labels.
- Data are standardised internally before fitting and the default priors are
  expressed in standardised units; `mu`, `chol_stds`, `cov` and `vals` are reported
  in data units. The old `mu_prior`/`sigma_prior` arguments still work (with a
  deprecation warning) and imply fitting in raw units.
- Plotting functions return the matplotlib figure, never call `plt.show()` unless
  asked, and save with `savefig` (the old code called the non-existent `plt.save`).
- The package version is derived from git tags at build time (`hatch-vcs`); a
  tagged commit `vX.Y.Z` builds as `X.Y.Z`, later commits as `X.Y.(Z+1).devN+g<hash>`.
  `CITATION.cff` follows the latest release tag (`scripts/sync_citation.py`, run by
  CI on tag pushes and checked on every build).
- `seaborn` is no longer a dependency.
- The module `pybaycor.pybaycor` is deprecated; import from `pybaycor`.
- All constructor arguments after `data` (after `sigma` for the hierarchical classes)
  are keyword-only; positional calls such as `BayesianCorrelation(None, x, y)` raise
  `TypeError` and must become `BayesianCorrelation(x=x, y=y)`.
- A 1-D `sigma` always means one value per variable; when `n_points == n_dim` a 2-D
  array is required to avoid ambiguity.

### Added

- A single `BayesianCorrelation` class covering every model: `robust=True` selects
  the multivariate Student-t; `sigma=` adds measurement uncertainties to any subset of
  entries (variables with and without uncertainties, and points with and without
  uncertainties, can be mixed); heteroscedastic (per-point) uncertainties; asymmetric
  uncertainties via the explicit `sigma_lower=`/`sigma_upper=` pair
  ([#5](https://github.com/pscicluna/pybaycor/issues/5)); `x_err`/`y_err` for the
  `x=`/`y=` form; per-variable `priors["mu"]`/`priors["sigma"]`.
- Censored data: upper limits, lower limits and interval censoring, specified per
  entry with `censoring=` flags or `lower=`/`upper=` bound arrays, with or without
  measurement uncertainty on the censored entries.
- Fully missing (`NaN`) data entries are treated as unobserved latent values;
  `CorrelationData` exposes `is_missing`/`n_missing`, warns when a variable has fewer
  than three non-missing entries, and `plot_data` marks missing coordinates on axes
  ([#9](https://github.com/pscicluna/pybaycor/issues/9)).
- `error_scale=True` infers a per-variable correction factor for the quoted
  uncertainties.
- `CorrelationData` container with validation and `describe()`.
- Inference back-ends: NUTS via `pymc`, `nutpie`, `numpyro` or `blackjax`
  (`nuts_sampler=`; blackjax is experimental until PyMC's wrapper is fixed
  upstream), plus `method="advi"`, `"fullrank_advi"`, `"smc"`, `"pathfinder"`
  (approximate; less accurate for models with many latent values) and `"map"`.
- `correlation_summary()` with per-pair posterior mean, median, credible interval,
  probability of a positive correlation and Savage-Dickey Bayes factor;
  `bayes_factor()`, `correlation()` sample access, `inferred_values()`,
  `null_pvalue()` (parametric bootstrap, [#6](https://github.com/pscicluna/pybaycor/issues/6)).
- Plotting for any number of variables: data with error bars, censoring arrows,
  inferred values and posterior ellipses; corner plots of the correlation
  coefficients; trace plots ([#4](https://github.com/pscicluna/pybaycor/issues/4)).
- Test suite (unit and integration, [#1](https://github.com/pscicluna/pybaycor/issues/1),
  [#2](https://github.com/pscicluna/pybaycor/issues/2)), Sphinx documentation
  skeleton ([#3](https://github.com/pscicluna/pybaycor/issues/3)), `pyproject.toml`
  with hatchling, pixi environments and lock file, ruff, pre-commit and GitHub
  Actions CI.

### Fixed

- `np.int`, swapped `isinstance` arguments, `fig.add_gridSpec`, wrong axes in the
  posterior-mean computation for more than two variables, `summary` used before
  assignment, and the corner plot silently dropping pairs in more than two dimensions.

## [0.2.0] - 2021-03

- Hierarchical (uncertain-data) models, plotting for 2-D data, PyPI release.
