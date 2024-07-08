# metalearners

[![CI](https://github.com/Quantco/metalearners/actions/workflows/ci.yml/badge.svg)](https://github.com/Quantco/metalearners/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/metalearners/badge/?version=latest)](https://metalearners.readthedocs.io/en/latest/?badge=latest)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/metalearners?logoColor=white&logo=conda-forge)](https://anaconda.org/conda-forge/metalearners)
[![PypiVersion](https://img.shields.io/pypi/v/metalearners.svg?logo=pypi&logoColor=white)](https://pypi.org/project/metalearners)
[![codecov.io](https://codecov.io/github/QuantCo/metalearners/coverage.svg?branch=main)](https://codecov.io/github/QuantCo/metalearners?branch=main)

MetaLearners for Conditional Average Treatment Effect (CATE) estimation

The library focuses on providing

- Methodologically sound cross-fitting
- Convenient access to and reuse of base models
- Consistent APIs across Metalearners
- Support for more than binary treatment variants
- Integrations with `pandas`, `shap`, `lime`, `optuna` and soon `onnx`

## Example

```python

df = ...

from metalearners import RLearner
from lightgbm import LGBMClassifier, LGBMRegressor

rlearner = RLearner(
    nuisance_model_factory=LGBMRegressor,
    propensity_model_factory=LGBMClassifier,
    treatment_model_factory=LGBMRegressor,
    is_classification=False,
    n_variants=2,
)

features = ["age", "weight", "height"]
rlearner.fit(df[features], df["treatment"], df["outcomes"])
cate_estimates = rlearner.predict(df[features], is_oos=False)
```

Please refer to our
[docs](https://metalearners.readthedocs.io/en/latest/?badge=latest)
for many more in-depth and reproducible examples.

## Installation

`metalearners` can either be installed via PyPI with

```bash
$ pip install metalearners
```

or via conda-forge with

```bash
$ conda install metalearners -c conda-forge
```

## Development

Development instructions can be found [here](https://metalearners.readthedocs.io/en/latest/development.html).
