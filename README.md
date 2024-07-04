# metalearners

[![CI](https://github.com/Quantco/metalearners/actions/workflows/ci.yml/badge.svg)](https://github.com/Quantco/metalearners/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/metalearners/badge/?version=latest)](https://metalearners.readthedocs.io/en/latest/?badge=latest)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/metalearners?logoColor=white&logo=conda-forge)](https://anaconda.org/conda-forge/metalearners)
[![PypiVersion](https://img.shields.io/pypi/v/metalearners.svg?logo=pypi&logoColor=white)](https://pypi.org/project/metalearners)
[![codecov.io](https://codecov.io/github/QuantCo/metalearners/coverage.svg?branch=main)](https://codecov.io/github/QuantCo/metalearners?branch=main)

MetaLearners for CATE estimation

## Installation

`metalearners` can either be installed via PyPI with

```console
$ pip install metalearners
```

or via conda-forge with

```bash
$ conda install metalearners -c conda-forge
```

## Development

The `metalearners` repository can be cloned as follows

```bash
$ git clone https://github.com/Quantco/metalearners.git
```

The dependencies are managed via [pixi](https://pixi.sh/latest/). Please make sure to install `pixi` on
your system. Once pixi is installed, you can install and run the pre-commit hooks as follows:

```bash
$ pixi run pre-commit-install
$ pixi run pre-commit-run
```

You can run the tests as follows:

```bash
$ pixi run postinstall
$ pixi run pytest tests
```

You can build the documentation locally by running

```console
$ pixi run -e docs postinstall
$ pixi run -e docs docs
```

You can then inspect the locally built docs by opening `docs/_build/index.html`.

You can find all pixi tasks in `pixi.toml`.
