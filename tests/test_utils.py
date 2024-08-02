# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from metalearners.metalearner import MetaLearner
from metalearners.utils import (
    FixedBinaryPropensity,
    metalearner_factory,
    simplify_output,
)


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


@pytest.mark.parametrize("use_pd", [True, False])
def test_fixed_binary_propensity(use_pd):
    propensity_score = 0.3
    dominant_class = propensity_score >= 0.5

    model = FixedBinaryPropensity(propensity_score=propensity_score)

    n_samples = 5
    X_train = np.ones((n_samples, 5))
    y_train = np.ones(n_samples)
    if use_pd:
        X_train = pd.DataFrame(X_train)
        y_train = pd.Series(y_train)

    model.fit(X_train, y_train)

    n_test_samples = 3
    X_test = np.zeros(n_test_samples)

    class_predictions = model.predict(X_test)
    assert np.array_equal(
        class_predictions, np.array(np.ones(n_test_samples) * dominant_class)
    )

    probability_estimates = model.predict_proba(X_test)
    assert np.array_equal(
        probability_estimates,
        np.column_stack(
            (
                np.ones(n_test_samples) * (1 - propensity_score),
                np.ones(n_test_samples) * propensity_score,
            )
        ),
    )


@pytest.mark.parametrize("propensity_score", [-1, 100, 1.1])
def test_fixed_binary_propensity_not_a_propbability(propensity_score):
    with pytest.raises(ValueError, match="between 0 and 1 but got"):
        FixedBinaryPropensity(propensity_score=propensity_score)


def test_fixed_binary_propensity_non_binary():
    propensity_score = 0.3

    model = FixedBinaryPropensity(propensity_score=propensity_score)

    n_samples = 5
    X_train = np.ones((n_samples, 5))
    y_train = np.fromiter(range(n_samples), dtype=int)
    with pytest.raises(ValueError, match="only supports binary outcomes"):
        model.fit(X_train, y_train)
