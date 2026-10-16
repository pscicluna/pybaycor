# Usage

## One class, one model

Everything goes through {py:class}`pybaycor.BayesianCorrelation`. The inputs decide
which parts of the model are switched on:

| Input | Effect |
| --- | --- |
| `data` only | classic correlation on exact data |
| `sigma=` | measurement uncertainties; `0`/`NaN` entries stay exact |
| `sigma_lower=`, `sigma_upper=` | asymmetric (two-piece Normal) uncertainties; exclusive with `sigma` |
| `censoring=` or `lower=`/`upper=` | per-entry censoring (upper limit, lower limit, interval) |
| `robust=True` | multivariate Student-t instead of Normal |
| `error_scale=True` | per-variable rescaling of the quoted uncertainties |
| `priors=` | override hyper-parameters (`pybaycor.DEFAULT_PRIORS`) |

```python
import numpy as np
import pybaycor as pbc

bc = pbc.BayesianCorrelation(
    data, sigma=sigma, censoring=censoring, robust=True, names=["log L", "log M", "z"]
)
print(bc.describe())
```

`describe()` lists how many entries of each variable are uncertain, asymmetric or
censored, and which model structure will be used.

## Fitting

```python
idata = bc.fit(draws=1000, tune=1000, chains=4, nuts_sampler="nutpie", random_seed=1)
```

The result is an `xarray.DataTree` (also stored as `bc.trace`). Trace variables:

| Variable | Dims | Meaning |
| --- | --- | --- |
| `mu` | `feature` | mean of each variable |
| `chol_stds` | `feature` | standard deviation of each variable |
| `chol_corr` | `feature, feature2` | the correlation matrix |
| `cov` | `feature, feature2` | covariance matrix |
| `nu` | | Student-t degrees of freedom (`robust=True`) |
| `error_scale` | `feature` | uncertainty scale factors (`error_scale=True`) |
| `vals` | `point, feature` | inferred true values (models with uncertainties or censoring) |

Other algorithms: `method="smc"`, `"fullrank_advi"`, `"advi"`, `"pathfinder"`,
`"map"`. The NUTS back-ends `"numpyro"` and `"blackjax"` run through JAX and use a
GPU or TPU when the installed `jaxlib` supports one.

## Summaries

```python
bc.summarise()  # ArviZ summary of the population parameters
bc.correlation_summary()  # per pair: mean, median, sd, interval, P(rho>0), BF10
bc.correlation(0, 1)  # posterior samples of one coefficient
bc.bayes_factor(0, 1)  # Savage-Dickey BF10 against rho = 0
bc.inferred_values()  # posterior mean and sd of the true values
bc.null_pvalue(0, 1, n_sim=100)  # parametric-bootstrap p-value (slow)
```

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
# {'mu': (0.0, 5.0), 'sigma': 2.5, 'eta': 2.0, 'nu_rate': 1/29, 'error_scale_sd': 0.5}
```

`eta` is the LKJ shape (`1` = uniform over correlation matrices, larger values shrink
towards zero correlation). Pass `standardize=False` to fit in raw units, in which
case the priors should be adapted to the scale of the data.
