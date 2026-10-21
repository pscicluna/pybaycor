# Usage

## One class, one model

Everything goes through {py:class}`pybaycor.BayesianCorrelation`. The inputs decide
which parts of the model are switched on:

| Input | Effect |
| --- | --- |
| `data` only | classic correlation on exact data |
| `NaN` in `data` without a censoring bound | missing at random; inferred as a latent value (not appropriate for non-detections) |
| `sigma=` | measurement uncertainties; `0`/`NaN` entries stay exact |
| `sigma_lower=`, `sigma_upper=` | asymmetric (two-piece Normal) uncertainties; exclusive with `sigma` |
| `censoring=` or `lower=`/`upper=` | per-entry censoring (upper limit, lower limit, interval) |
| `robust=True` | multivariate Student-t instead of Normal |
| `error_scale=True` | per-variable rescaling of the quoted uncertainties |
| `priors=` | override hyper-parameters (`pybaycor.DEFAULT_PRIORS`) |

```python
import pybaycor as pbc

# data: (n_points, 3) array; sigma and censoring: arrays of the same shape
bc = pbc.BayesianCorrelation(
    data, sigma=sigma, censoring=censoring, robust=True, names=["log L", "log M", "z"]
)
print(bc.describe())
```

`describe()` lists how many entries of each variable are missing, uncertain, asymmetric or
censored, and which model structure will be used.

## Fitting

```python
idata = bc.fit(draws=1000, tune=1000, chains=4, random_seed=1)
```

NUTS is the default; it runs through `nutpie` when that is installed (`pybaycor[nutpie]`)
and PyMC's own sampler otherwise. Pick one explicitly with `nuts_sampler=`.

The result is an `xarray.DataTree` (also stored as `bc.trace`). Trace variables:

| Variable | Dims | Meaning |
| --- | --- | --- |
| `mu` | `feature` | mean of each variable |
| `chol_stds` | `feature` | standard deviation of each variable |
| `chol_corr` | `feature, feature2` | the correlation matrix |
| `cov` | `feature, feature2` | covariance matrix |
| `nu` | | Student-t degrees of freedom (`robust=True`) |
| `error_scale` | `feature` | uncertainty scale factors (`error_scale=True`) |
| `vals` | `point, feature` | inferred true values (including imputed missing-at-random entries, if present) |

The posterior also holds the internal parameters the sampler works with (`mu_z`,
`chol_z`, `nu_minus_one`, `theta_z`, `theta_free`, ...), in standardised units; use the
variables above instead.

Other algorithms: `method="smc"`, `"fullrank_advi"`, `"advi"` (mean-field; it
under-estimates correlations, so use it only for exploration), `"pathfinder"` (needs
`pybaycor[extras]`) and `"map"`. The NUTS back-end `"numpyro"` runs through JAX and
uses a GPU or TPU when the installed `jaxlib` supports one. `"blackjax"` is wired up
but currently fails because of an upstream PyMC incompatibility
([#13](https://github.com/pscicluna/pybaycor/issues/13)); pybaycor then raises an
error pointing to `numpyro` and `nutpie`.

## Summaries

```python
bc.summarise()  # ArviZ summary of the population parameters
bc.correlation_summary()  # per pair: mean, median, sd, interval, P(rho>0), BF10
bc.correlation(0, 1)  # posterior samples of one coefficient
bc.bayes_factor(0, 1)  # Savage-Dickey BF10 against rho = 0
bc.inferred_values()  # posterior mean and sd of the true values
bc.null_pvalue(0, 1, n_sim=100)  # parametric-bootstrap p-value (slow)
```

`correlation_summary()` has one row per pair, indexed by `"name_i~name_j"`. Its
interval follows ArviZ's defaults (`rcParams["stats.ci_prob"]` and
`rcParams["stats.ci_kind"]`, an 89% equal-tailed interval in ArviZ 1), as does
`summarise()`; pass e.g. `correlation_summary(ci_prob=0.95, ci_kind="hdi")` for
another interval.

## Plots

```python
bc.plot_data(n_ellipses=30, plotfile="data.png")
bc.plot_corner(plotfile="corner.png")
bc.plot_trace(plotfile="trace.png")
```

All plotting methods return the matplotlib figure and accept `show=True`.

## Censoring conventions

* `censoring` flags: `0`/`"observed"`, `-1`/`"upper"` (the quoted value is an upper
  limit, i.e. the true value is below it), `1`/`"lower"` (lower limit), `2`/`"interval"`
  (bounds given through `lower=`/`upper=`). A boolean array is read as a mask of upper
  limits.
* `lower`/`upper` arrays: `NaN` or `±inf` means unbounded; any finite bound makes the
  entry censored. Both finite gives interval censoring.
* A censored entry may also have `sigma > 0`, in which case the limit is treated as
  the outcome of a noisy measurement; with `sigma == 0` the true value is constrained
  to the allowed range exactly.

## Standardisation and priors

By default the data are standardised per variable (median and 1.4826 × MAD of the
observed, uncensored entries) and the priors below act in those units:

```python
pbc.DEFAULT_PRIORS
# {'mu': (0.0, 5.0), 'sigma': 2.5, 'eta': 2.0, 'nu_rate': 0.0344... (= 1/29), 'error_scale_sd': 0.5}
```

`eta` is the LKJ shape (`1` = uniform over correlation matrices, larger values shrink
towards zero correlation). Pass `standardize=False` to fit in raw units, in which
case the priors should be adapted to the scale of the data.
