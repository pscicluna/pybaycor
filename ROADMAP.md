# pybaycor modernisation plan

This document is the plan for bringing pybaycor up to current software-management
standards and for extending its statistical scope. It records the analysis of the
`v0.2.0` code base, the decisions taken, what has been implemented on this branch,
and what remains.

## 1. Where the package stood (v0.2.0)

* One 750-line module (`pybaycor/pybaycor.py`) with four classes that duplicate
  nearly all of their code (`BayesianCorrelation`, `RobustBayesianCorrelation`,
  `HierarchicalBayesianCorrelation`, `HierarchicalRobustBayesianCorrelation`).
* Pinned to `pymc3==3.10.0` and `arviz==0.11.0`. PyMC3 is unmaintained, does not
  install on current Python, and cannot use JAX/numba/GPU backends. This is
  [issue #7](https://github.com/pscicluna/pybaycor/issues/7).
* No tests ([#1](https://github.com/pscicluna/pybaycor/issues/1),
  [#2](https://github.com/pscicluna/pybaycor/issues/2)), no docs beyond README
  ([#3](https://github.com/pscicluna/pybaycor/issues/3)), no linting, no CI,
  `setup.py` with a duplicated `xarray` requirement and no lock file.
* Latent bugs: `plt.save` (should be `savefig`), `np.int`, `isinstance(int, self.ndim)`
  (arguments swapped), `fig.add_gridSpec`, `plot_corner` coordinate selection
  that silently drops pairs in more than two dimensions, `mu_post = ...mean(axis=(i, j))`
  averaging over the wrong axes, `self.summary` used before assignment, and
  `plot_trace` silently continuing when the model is unfitted.
* Priors in absolute units (`Normal(0, 1000)`, `HalfCauchy(200)`) that are only
  weakly informative for data of order 1–100 and make sampling slow or unstable for
  data on other scales.
* Uncertainties had to be given for every entry of every variable, so a feature
  without measurement error could not be mixed with one that has it; censored
  (upper/lower-limit) data could not be used at all.

## 2. Target architecture

```
pybaycor/
  __init__.py        public API and version
  data.py            CorrelationData: validation, censoring spec, partial sigma, standardisation
  model.py           build_model(): PyMC 6 model factory (Gaussian / Student-t, latent values,
                     censoring, asymmetric errors, error scaling)
  inference.py       fit(): dispatch to NUTS backends, VI, SMC, Pathfinder, MAP
  stats.py           pairwise correlation summaries, Savage-Dickey Bayes factors, direction probs
  plotting.py        trace, data/ellipse and corner plots for any dimensionality
  correlation.py     BayesianCorrelation user class + backwards-compatible subclasses
  pybaycor.py        deprecated shim re-exporting the old names
tests/               pytest unit + (marked) integration tests
docs/                Sphinx + MyST + AutoAPI
pyproject.toml       hatchling build, ruff, pytest, mypy and pixi configuration
pixi.lock            reproducible multi-platform environments
.github/workflows/   lint, test matrix (py3.12/3.13 × linux/macos), docs build
```

### 2.1 Dependency and environment management (pixi)

* `pyproject.toml` is the single manifest: PEP 621 metadata, optional extras
  (`jax`, `nutpie`, `blackjax`, `extras`, `dev`, `docs`, `all`), and the
  `[tool.pixi]` workspace with conda-forge channels, features and environments
  (`default`, `dev`, `jax`, `full`, `docs`, `py312`, `py313`) and tasks
  (`test`, `test-all`, `test-cov`, `lint`, `fmt`, `typecheck`, `docs`).
* `pixi.lock` pins every environment for linux-64, osx-64, osx-arm64 and win-64.
* `pip install pybaycor[jax]` remains supported for users who do not use pixi.

### 2.2 Code quality

* `ruff` for linting (pyflakes, pycodestyle, isort, pyupgrade, bugbear, numpy,
  pydocstyle/numpy convention) and formatting; `mypy` in permissive mode.
* `pre-commit` hooks for ruff and the standard hygiene hooks.
* GitHub Actions CI with a `lint` job, a `test` matrix and a `docs` build.
* `pytest` with markers: fast unit tests always run; `slow` MCMC recovery tests,
  `jax` and `nutpie` backend tests are opt-in.

### 2.3 Documentation

* Numpy-style docstrings on every public object, rendered by `sphinx-autoapi`.
* MyST pages: installation, usage, roadmap (this file), changelog.
* README rewritten around the unified API with migration notes.
* CI builds the site with warnings as errors and runs the docstring examples as
  doctests; Read the Docs publishes it (<https://pybaycor.readthedocs.io/>) from the locked pixi `docs`
  environment, with `latest` following `main` and one version per release tag.

## 3. Statistical model

All classes now share one model family. For observation $i$ and variable $j$ the
*true* value $\theta_{ij}$ follows a multivariate Normal (or, with `robust=True`,
multivariate Student-t with $\nu \sim 1 + \mathrm{Exponential}(1/29)$):

$$\boldsymbol\theta_i \sim \mathcal{MVN}(\boldsymbol\mu, \boldsymbol\Sigma), \qquad
\boldsymbol\Sigma = \mathrm{diag}(\boldsymbol s)\, \mathbf R\, \mathrm{diag}(\boldsymbol s),
\qquad \mathbf R \sim \mathrm{LKJ}(\eta = 2).$$

The correlation matrix $\mathbf R$ (`chol_corr` in the trace) is the quantity of
interest. Each entry is then observed through one of the following channels,
chosen independently per entry from the inputs `sigma`, `censoring`, `lower`,
`upper`:

| Entry type | Input | Treatment |
| --- | --- | --- |
| exact, observed | `sigma == 0`/NaN, not censored | $\theta_{ij} = x_{ij}$ fixed (no latent parameter) |
| uncertain, observed | `sigma_ij > 0` | $x_{ij} \sim \mathcal N(\theta_{ij}, \sigma_{ij})$ |
| uncertain, asymmetric | `sigma_lower=`, `sigma_upper=` | two-piece Normal with scales $\sigma^-_{ij}, \sigma^+_{ij}$ ([#5](https://github.com/pscicluna/pybaycor/issues/5)) |
| uncertain, censored | limit + `sigma_ij > 0` | $P(x < U) = \Phi\big((U-\theta_{ij})/\sigma_{ij}\big)$ (and the lower-limit / interval analogues) |
| exact, censored | limit, `sigma == 0` | $\theta_{ij}$ sampled as a latent parameter constrained to the censoring interval (data augmentation) |

Because the treatment is chosen per entry, a data set can freely mix variables with
and without uncertainties, points with and without uncertainties within a variable,
heteroscedastic (point-dependent) uncertainties, and arbitrary censoring: upper
limits, lower limits or interval censoring, with different limits for every point
and every variable.

Exact and uncensored entries are compiled into the graph as constants, so the
classical (no-uncertainty) case has the same number of free parameters as before and
the fully-uncertain case reproduces the Matzke et al. (2017) hierarchical model.
Where some entries of a row are fixed and others are latent, the joint density is
evaluated through a `pm.Potential`; the latent parameters carry a proper
pseudo-prior that the potential cancels exactly, so prior-predictive sampling and SMC
still work.

### 3.1 Heteroscedasticity

* Measurement errors are heteroscedastic by construction (`sigma` is `(n_points, n_dim)`;
  a scalar or `(n_dim,)` input is broadcast).
* `error_scale=True` adds a per-variable multiplicative factor $f_j$ on the quoted
  uncertainties ($\sigma_{ij} \to f_j \sigma_{ij}$) with a log-normal prior, to absorb
  globally under- or over-estimated error bars. This is identifiable only when the
  quoted uncertainties vary from point to point; the docstring says so.
* An additive per-variable "intrinsic scatter" term is *not* offered: in a pure
  correlation model it is exactly degenerate with the diagonal of $\boldsymbol\Sigma$.
  Scatter that depends on a covariate (a location–scale model) changes the question
  being asked and is left for a future `scale_model` extension (section 6).

### 3.2 Standardisation and priors

Data are standardised internally (per-variable robust location and scale computed
from the observed, uncensored values), the model is fitted in standardised space
with scale-free weakly-informative priors ($\mu_j \sim \mathcal N(0, 5)$,
$s_j \sim \mathrm{HalfCauchy}(2.5)$), and all reported quantities (`mu`, `chol_stds`,
`cov`, `vals`) are transformed back to the original units. Correlations are invariant
under this transformation. `standardize=False` with a `priors` dictionary reproduces
fitting in raw units (the old `mu_prior`/`sigma_prior` arguments still work and
imply raw units).

## 4. Inference

`fit()` exposes several algorithms through one call; the `method`/`nuts_sampler`
arguments select them:

| `method` | Backend | Notes |
| --- | --- | --- |
| `"nuts"` (default) | `nuts_sampler="pymc"`, `"nutpie"`, `"numpyro"`, `"blackjax"` | `nutpie` (Rust, default when installed) and `numpyro` (JAX) run on CPU, GPU or TPU without code changes; `blackjax` is experimental (see section 6) |
| `"advi"`, `"fullrank_advi"` | PyMC variational inference | fast approximation; mean-field ADVI is known to underestimate correlations and is flagged as such |
| `"smc"` | Sequential Monte Carlo | useful for multimodal posteriors and marginal likelihoods |
| `"pathfinder"` | `pymc-extras` | fast quasi-Newton variational approximation; needs the `extras` extra |
| `"map"` | optimisation | point estimate for quick checks |

PyTensor 3 compiles the model graph with numba by default; the JAX samplers compile
to XLA. The package itself contains no backend-specific code, so acceleration is a
matter of installing the optional extra.

## 5. Open-issue map

| Issue | Status on this branch |
| --- | --- |
| #1 unit tests | `tests/test_data.py`, `tests/test_model.py`, `tests/test_stats.py`, `tests/test_api.py` |
| #2 integration tests | `tests/test_inference.py` recovery tests (marked `slow`) incl. censoring and partial uncertainty |
| #3 documentation | docstrings (with doctested examples) on all public objects, Sphinx/AutoAPI site published on Read the Docs, rewritten README; worked examples and a model page still to come |
| #4 plotting | `plotting.py` rewritten: any dimensionality, error bars, censoring arrows, posterior ellipses, corner plots of the correlation coefficients only, every function returns the figure and accepts `ax` |
| #5 asymmetric errors | two-piece Normal observation channel (`sigma_lower=`/`sigma_upper=`) |
| #6 p-values | `correlation_summary()` reports $P(\rho>0)$ and the Savage-Dickey Bayes factor for $\rho_{ij}=0$; the parametric-bootstrap p-value from the issue is provided by `null_pvalue()` and is documented as expensive |
| #7 PyMC3 → modern stack | PyMC ≥ 6.1 / PyTensor 3 / ArviZ ≥ 1.1 with JAX and nutpie backends |

## 6. Remaining work (not on this branch)

Each item below is tracked as a GitHub issue so it can be picked up independently.


1. **Location–scale extension** ([#8](https://github.com/pscicluna/pybaycor/issues/8)): allow $\boldsymbol\Sigma$ (or the error scale) to
   depend on a covariate for data whose intrinsic scatter is genuinely heteroscedastic.
2. **Missing data** ([#9](https://github.com/pscicluna/pybaycor/issues/9)): `NaN` entries that are neither observed nor censored could be
   treated as fully latent; the machinery exists, only the input validation needs to
   allow it and the identifiability warning needs writing.
3. **Posterior predictive checks** ([#10](https://github.com/pscicluna/pybaycor/issues/10)): `plot_ppc` using ArviZ's `plot_ppc_*` family once
   the observed-data group is populated for the mixed exact/latent case.
4. **Publishing** ([#11](https://github.com/pscicluna/pybaycor/issues/11)): enable the release workflow (trusted publishing to PyPI from a tag),
   Zenodo DOI, and a conda-forge feedstock.
5. **Performance** ([#12](https://github.com/pscicluna/pybaycor/issues/12)): for very large `n_points` the per-entry latent parameters make the
   model high-dimensional; a marginalised (closed-form) likelihood for the Gaussian,
   uncensored, uncertain case ($x_i \sim \mathcal N(\mu, \Sigma + \mathrm{diag}(\sigma_i^2))$)
   would remove them entirely.
6. **Blackjax** ([#13](https://github.com/pscicluna/pybaycor/issues/13)): PyMC 6.3's blackjax wrapper is broken against released blackjax
   versions (it forwards a `progress_bar` keyword that the NUTS kernel rejects, and
   its chain-mapping code fails on current JAX). `pybaycor.inference` shims the first
   problem and turns the second into a descriptive error that points at `numpyro`
   and `nutpie`. Re-test when PyMC or blackjax ship a fix, then drop the shim.
