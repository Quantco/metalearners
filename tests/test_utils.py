# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from lightgbm import LGBMRegressor

from metalearners.metalearner import MetaLearner
from metalearners.utils import metalearner_factory, simplify_output


@pytest.mark.parametrize("prefix", ["T"])
def test_metalearner_factory_smoke(prefix):
    factory = metalearner_factory(prefix)
    model = factory(
        nuisance_model_factory=LGBMRegressor, is_classification=False, n_variants=2
    )
    assert isinstance(model, MetaLearner)


@pytest.mark.parametrize("prefix", ["", "H", None])
def test_metalearner_factory_raises(prefix):
    with pytest.raises(ValueError, match="No MetaLearner implementation found"):
        metalearner_factory(prefix)


@pytest.mark.parametrize(
    "input,expected",
    [
        (np.zeros((5, 1, 1)), np.zeros(5)),
        (np.zeros((5, 1, 2)), np.zeros(5)),
        (np.zeros((5, 1, 3)), np.zeros((5, 3))),
        (np.zeros((5, 2, 1)), np.zeros((5, 2))),
        (np.zeros((5, 2, 2)), np.zeros((5, 2))),
        (np.zeros((5, 2, 3)), np.zeros((5, 2, 3))),
    ],
)
def test_simplify_output(input, expected):
    actual = simplify_output(input)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "input",
    [
        np.zeros(5),
        np.zeros((5,)),
        np.zeros((5, 2)),
        np.zeros((5, 2, 2, 2)),
    ],
)
def test_simplify_output_raises(input):
    with pytest.raises(ValueError, match="needs to be 3-dimensional"):
        simplify_output(input)
