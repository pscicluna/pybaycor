# pybaycor
<sub><sup>*It's all you knead*</sup></sub>
---

Pybaycor ("Pie Baker") estimates Bayesian correlation coefficients in Python. It
started as a re-implementation of the ["Bayesian First Aid"](http://www.sumsar.net/blog/2014/03/bayesian-first-aid-pearson-correlation-test/)
robust and non-robust correlation tests, extended with the hierarchical treatment of
measurement uncertainty of Matzke et al. (2017), and now provides one model that
handles:

* any number of variables, with a multivariate Normal or (robust) multivariate
  Student-t distribution and an LKJ prior on the correlation matrix;
* measurement uncertainties on **any subset of entries**: variables with and without
  errors can be mixed, and errors can differ from point to point (heteroscedastic) and
  be asymmetric;
* **censored data**: upper limits, lower limits and interval censoring, with a
  different limit for every point and every variable, with or without measurement
  error on the censored entries;
* several inference engines on top of [PyMC](https://www.pymc.io) ≥ 6.1: NUTS through
  `pymc`, `nutpie` (Rust) or `numpyro` (JAX, so GPU/TPU-capable; `blackjax` is wired up
  but currently blocked by an upstream PyMC incompatibility), as well as variational
  inference, sequential Monte Carlo, Pathfinder and MAP;
* posterior summaries per pair of variables, including the probability that a
  correlation is positive and a Savage-Dickey Bayes factor against zero correlation.

The full documentation, including the API reference, is at
<https://pybaycor.readthedocs.io/>.

## Installation

```
pip install pybaycor            # core (PyMC, ArviZ, numpy, scipy, pandas, xarray, matplotlib)
pip install pybaycor[nutpie]    # fastest CPU sampler
pip install pybaycor[jax]       # numpyro sampler; install a GPU jaxlib to use a GPU
pip install pybaycor[all]
```

To develop, or to get a fully reproducible environment, use [pixi](https://pixi.sh):

```
git clone https://github.com/pscicluna/pybaycor && cd pybaycor
pixi install -e dev     # or -e full for the JAX and nutpie back-ends
pixi run -e dev test    # fast unit tests and docstring examples
pixi run -e dev test-all    # including MCMC recovery tests
pixi run -e dev lint && pixi run -e dev fmt-check
pixi run -e docs docs   # warnings are errors, as in CI
```

The `default` pixi environment holds only the runtime dependencies, so the
development tasks need `-e dev` (or `-e full`), and the documentation `-e docs`.

pybaycor requires Python ≥ 3.12. The package version comes from the git tags
(`hatch-vcs`), and `pybaycor.__version__` reports the resolved version. Releases are
published to PyPI by pushing a tag; see [RELEASING.md](RELEASING.md).

## Quick start

```python
import numpy as np
import pybaycor as pbc

rng = np.random.default_rng(0)
data = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=50)  # (n_points, n_dim)

bc = pbc.BayesianCorrelation(data, names=["x", "y"], robust=True)
bc.fit(draws=1000, tune=1000)  # NUTS; uses nutpie if installed, else the PyMC sampler
bc.summarise()  # table of mu, chol_stds, chol_corr, nu
bc.correlation_summary()  # one row per pair: mean, interval, P(rho > 0), BF10
bc.plot_data(plotfile="data.png")  # data with the posterior 95% ellipse
bc.plot_corner(plotfile="corner.png")  # posterior of the correlation coefficient(s)
bc.plot_trace(plotfile="trace.png")
```

`bc.trace` is an `xarray.DataTree` (ArviZ 1 format) holding the posterior; the
correlation matrix is the variable `chol_corr`, and `bc.correlation(i, j)` returns the
flattened samples of one coefficient.

### Measurement uncertainties

```python
bc = pbc.BayesianCorrelation(data, sigma=sigma)  # sigma: (n_points, n_dim), scalar or (n_dim,)
```

Entries of `sigma` that are `0` or `NaN` denote exact measurements (zero uncertainty); unlike `sigma`, a `NaN` in `data` with no censoring bounds represents a missing observation. Thus a variable without
uncertainties is simply a column of zeros, and a data set where only some points have
error bars needs no special treatment. A 1-D `sigma` always means one value per
variable (when the data have as many points as variables a 2-D array is required).
Asymmetric uncertainties use a two-piece Normal error model:

```python
bc = pbc.BayesianCorrelation(data, sigma_lower=sigma_lower, sigma_upper=sigma_upper)
```

Both arrays follow the same shape rules as `sigma`; `sigma` and the asymmetric pair are
mutually exclusive, so the choice of error model is always explicit. For two 1-D
arrays, `BayesianCorrelation(x=x, y=y, x_err=..., y_err=...)` or
`CorrelationData.from_xy(x, y, x_err_lower=..., x_err_upper=..., ...)` do the stacking.

If the quoted error bars may be globally under- or over-estimated, `error_scale=True`
infers a correction factor per variable (identifiable only when the quoted
uncertainties vary from point to point).

### Missing data

A `NaN` in `data` with no censoring bounds is treated as **missing at random** (MAR): its true value is inferred as a latent variable, and the filled-in values are available in the posterior `vals` and through `inferred_values()` after fitting. This assumption is important: **non-detections are censored measurements, not missing-at-random values**. Specify their limits using `censoring=` or `lower=`/`upper=`; leaving them as plain `NaN` will impute them and can bias the inferred correlation coefficient.

### Censored data

Each entry can be flagged independently:

```python
# -1: the value in `data` is an upper limit (true value below it)
#  1: the value is a lower limit;  0: observed
censoring = np.zeros_like(data, dtype=int)
censoring[upper_limit_mask, 1] = -1
bc = pbc.BayesianCorrelation(data, sigma=sigma, censoring=censoring)
```

or with explicit bounds, which also allows interval censoring:

```python
lower = np.full(data.shape, np.nan)
upper = np.full(data.shape, np.nan)
upper[3, 0] = 2.5  # x_3 < 2.5
lower[7, 1], upper[7, 1] = 0.1, 0.4  # 0.1 < y_7 < 0.4
bc = pbc.BayesianCorrelation(data, lower=lower, upper=upper)
```

A censored entry with `sigma > 0` is modelled through the probability that a noisy
measurement would fall below (above, between) the limit(s); a censored entry with
`sigma == 0` is treated as an unknown true value constrained to the allowed range.
The inferred true values of all latent entries are available as `bc.inferred_values()`
and as the trace variable `vals`.

### Inference options

```python
bc.fit(method="nuts", nuts_sampler="numpyro")  # JAX; CPU/GPU/TPU depending on jaxlib
bc.fit(method="nuts", nuts_sampler="nutpie")  # Rust sampler, fastest on CPU
bc.fit(method="smc")  # sequential Monte Carlo
bc.fit(method="fullrank_advi")  # variational approximation
bc.fit(method="pathfinder")  # needs pybaycor[extras]
bc.fit(method="map")  # point estimate
```

Hyper-parameters can be overridden with `priors={"eta": 1.0, "sigma": 5.0, ...}`; see
`pybaycor.DEFAULT_PRIORS`. By default the data are standardised internally and all
reported quantities are transformed back to data units.

### Hypothesis testing

`correlation_summary()` reports `prob_positive`, the posterior probability that the
correlation exceeds zero, and `bf10`, the Savage-Dickey Bayes factor for a non-zero
correlation. Its interval is ArviZ's default, an 89% equal-tailed interval; pass e.g.
`correlation_summary(ci_prob=0.95, ci_kind="hdi")` for another. A parametric-bootstrap p-value (refit under the null of no correlation,
simulate, refit) is available through `bc.null_pvalue()`; it is expensive.

## Example notebook

[`examples/dustpedia_dust_scaling.ipynb`](examples/dustpedia_dust_scaling.ipynb)
walks through the package on real data: the dust, stellar, gas and star-formation
properties of nearby galaxies from DustPedia. It starts with two variables without
uncertainties and builds up to a five-dimensional model with measurement errors, HI
upper limits and missing HI data, then tests which correlations are significant. Run
it with `pixi run -e examples examples`.

## Migrating from pybaycor 0.2

* `RobustBayesianCorrelation`, `HierarchicalBayesianCorrelation` and
  `HierarchicalRobustBayesianCorrelation` still exist and behave as before, but are
  thin wrappers: `BayesianCorrelation(data, sigma=..., robust=True)` is equivalent.
* All constructor arguments after `data` (and `sigma` for the hierarchical classes)
  are keyword-only; positional calls such as `BayesianCorrelation(None, x, y)` must
  become `BayesianCorrelation(x=x, y=y)`.
* `fit(steps=...)` is now `fit(draws=...)`; the return value is an `xarray.DataTree`.
* Summary rows are labelled by variable name (`chol_corr[x, y]`).
* `mu_prior`/`sigma_prior` still work but are deprecated in favour of `priors=`.
* Plot methods return figures and no longer show them unless `show=True`.

## Model

The true values $\boldsymbol\theta_i$ of each data point follow
$\mathcal{MVN}(\boldsymbol\mu, \boldsymbol\Sigma)$ or
$\mathcal{MVT}_\nu(\boldsymbol\mu, \boldsymbol\Sigma)$ with
$\boldsymbol\Sigma = \mathrm{diag}(\boldsymbol s)\,\mathbf R\,\mathrm{diag}(\boldsymbol s)$,
$\mathbf R \sim \mathrm{LKJ}(\eta=2)$, $\nu \sim 1 + \mathrm{Exponential}(1/29)$.
Each observed entry is then either equal to its true value, a Normal (or two-piece
Normal) measurement around it, or a censoring statement about it. See `ROADMAP.md`
for the full description and the development plan.

## Relevant citations

```
@inbook{inbook,
author = {Gelman, Andrew and Hill, Jennifer},
year = {2006},
month = {11},
pages = {},
title = {Data Analysis Using Regression And Multilevel/Hierarchical Models},
volume = {3},
isbn = {0521867061},
journal = {Cambridge Universty Press},
doi = {10.1017/CBO9780511790942}
}

@article{doi:10.1152/jn.90727.2008,
author = {Behseta, Sam and Berdyyeva, Tamara and Olson, Carl R. and Kass, Robert E.},
title = {Bayesian Correction for Attenuation of Correlation in Multi-Trial Spike Count Data},
journal = {Journal of Neurophysiology},
volume = {101},
number = {4},
pages = {2186-2193},
year = {2009},
doi = {10.1152/jn.90727.2008},
note ={PMID: 19129297},
URL = {https://doi.org/10.1152/jn.90727.2008},
}

@article{10.1525/collabra.78,
    author = {Matzke, Dora and Ly, Alexander and Selker, Ravi and Weeda, Wouter D. and Scheibehenne, Benjamin and Lee, Michael D. and Wagenmakers, Eric-Jan},
    title = "{Bayesian Inference for Correlations in the Presence of Measurement Error and Estimation Uncertainty}",
    journal = {Collabra: Psychology},
    volume = {3},
    number = {1},
    year = {2017},
    month = {10},
    issn = {2474-7394},
    doi = {10.1525/collabra.78},
    url = {https://doi.org/10.1525/collabra.78},
    note = {25},
}

@article{lewandowski2009,
    author = {Lewandowski, Daniel and Kurowicka, Dorota and Joe, Harry},
    title = {Generating random correlation matrices based on vines and extended onion method},
    journal = {Journal of Multivariate Analysis},
    volume = {100},
    number = {9},
    pages = {1989-2001},
    year = {2009},
    doi = {10.1016/j.jmva.2009.04.008},
}
```
