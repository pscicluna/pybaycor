# Installation

pybaycor requires Python 3.12 or newer and depends on PyMC (>= 6.1) and ArviZ
(>= 1.1).

## With pip

```bash
pip install pybaycor
```

### Optional extras

Extra sampling backends and tooling are available as extras:

| Extra      | Installs                     | Purpose                                            |
| ---------- | ---------------------------- | -------------------------------------------------- |
| `jax`      | `jax`, `numpyro`             | NumPyro NUTS sampler via `nuts_sampler="numpyro"`   |
| `nutpie`   | `nutpie`                     | Rust NUTS sampler via `nuts_sampler="nutpie"`       |
| `blackjax` | `blackjax`                   | Experimental: BlackJAX NUTS via `nuts_sampler="blackjax"`; currently fails with PyMC 6.3 (upstream incompatibility, [#13](https://github.com/pscicluna/pybaycor/issues/13)), so use `numpyro` for a JAX sampler |
| `extras`   | `pymc-extras`                | Pathfinder variational inference via `fit(method="pathfinder")` |
| `all`      | everything above             |                                                    |
| `dev`      | pytest, ruff, mypy, ...      | Development tooling                                |
| `docs`     | sphinx, furo, ...            | Building this documentation                        |

```bash
pip install "pybaycor[jax]"
pip install "pybaycor[nutpie]"
pip install "pybaycor[all]"
```

With no extras, `fit()` uses PyMC's own NUTS sampler, which works everywhere; if
`nutpie` is installed it is used automatically. `nutpie` is usually the fastest
option for the models in pybaycor; `jax`/`numpyro` is a good choice if you already
have a JAX installation (including GPU builds).

## With pixi (recommended for development)

The repository ships a [pixi](https://pixi.sh) workspace (in `pyproject.toml`)
that resolves the full scientific stack from conda-forge and installs pybaycor
in editable mode:

```bash
git clone https://github.com/pscicluna/pybaycor
cd pybaycor
pixi install            # default environment: runtime dependencies only
pixi shell              # drop into an activated shell
```

The `default` environment has no test or documentation tooling, so the development
tasks are run in a named environment with `-e`:

```bash
pixi install -e dev     # + pytest, ruff, mypy, pre-commit
pixi install -e jax     # + dev tools, jax, numpyro
pixi install -e full    # + dev tools, jax, numpyro, nutpie
pixi install -e docs    # + sphinx toolchain

pixi run -e dev test    # fast unit tests and docstring examples
pixi run -e dev lint
pixi run -e full test-all
pixi run -e docs docs   # warnings are errors, as in CI
```

## From source with pip

```bash
git clone https://github.com/pscicluna/pybaycor
cd pybaycor
pip install -e ".[dev]"
```
