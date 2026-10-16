# Installation

pybaycor requires Python 3.12 or newer and depends on PyMC (>= 6) and ArviZ
(>= 1).

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
| `blackjax` | `blackjax`                   | BlackJAX NUTS sampler via `nuts_sampler="blackjax"` |
| `extras`   | `pymc-extras`                | Experimental PyMC features                         |
| `all`      | everything above             |                                                    |
| `dev`      | pytest, ruff, mypy, ...      | Development tooling                                |
| `docs`     | sphinx, furo, ...            | Building this documentation                        |

```bash
pip install "pybaycor[jax]"
pip install "pybaycor[nutpie]"
pip install "pybaycor[all]"
```

The default PyMC sampler works everywhere with no extras. `nutpie` is usually
the fastest option for the models in pybaycor; `jax`/`numpyro` is a good choice
if you already have a JAX installation (including GPU builds).

## With pixi (recommended for development)

The repository ships a [pixi](https://pixi.sh) workspace (in `pyproject.toml`)
that resolves the full scientific stack from conda-forge and installs pybaycor
in editable mode:

```bash
git clone https://github.com/pscicluna/pybaycor
cd pybaycor
pixi install            # default environment
pixi run test           # run the fast test suite
pixi shell              # drop into an activated shell
```

Named environments add optional backends and tooling:

```bash
pixi install -e dev     # + pytest, ruff, mypy, pre-commit
pixi install -e jax     # + jax, numpyro
pixi install -e full    # + jax, numpyro, nutpie
pixi install -e docs    # + sphinx toolchain

pixi run -e dev lint
pixi run -e full test-all
pixi run -e docs docs
```

## From source with pip

```bash
git clone https://github.com/pscicluna/pybaycor
cd pybaycor
pip install -e ".[dev]"
```
