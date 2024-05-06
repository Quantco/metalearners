# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from metalearners._utils import (
    check_probability,
    check_propensity_score,
    get_linear_dimension,
)
from metalearners.data_generation import generate_covariates


@pytest.mark.parametrize("n_numericals", [0, 5, 10])
@pytest.mark.parametrize("n_categoricals, n_categories", [(5, 5), (3, [3, 4, 10])])
def test_get_linear_dimension(n_numericals, n_categoricals, n_categories, rng):
    features, _, _ = generate_covariates(
        1000,
        n_numericals + n_categoricals,
        n_categoricals=n_categoricals,
        n_categories=n_categories,
        rng=rng,
    )
    dim = get_linear_dimension(features)
    if isinstance(n_categories, int):
        total_categories = n_categoricals * n_categories
    else:
        total_categories = sum(n_categories)
    assert dim == n_numericals + total_categories


@pytest.mark.parametrize(
    "p, expected",
    [(np.array([2, 3]), (2,)), (np.array([[2, 3, 4], [34, 35, 66]]), (2, 3))],
)
def test_check_propensity_score_shape(p, expected):
    with pytest.raises(ValueError) as e:
        check_propensity_score(p)
    assert (
        e.value.args[0]
        == f"One propensity score must be provided for each variant. There are 2 but "
        f"the shape of the propensity scores is {expected}."
    )


@pytest.mark.parametrize("check_kwargs", [None, {"force_all_finite": "allow-nan"}])
def test_check_propensity_score_handle_nan(check_kwargs):
    if check_kwargs is None:
        with pytest.raises(ValueError) as e:
            check_propensity_score(
                np.array([[0.2, 0.8], [0.4, 0.6]]),
                np.array([[np.nan, 1], [2.0, 1]]),
                check_kwargs=check_kwargs,
            )
        assert "contains NaN" in e.value.args[0]
    else:
        check_propensity_score(
            np.array([[0.2, 0.8], [0.4, 0.6]]),
            np.array([[np.nan, 1], [2.0, 1]]),
            check_kwargs=check_kwargs,
        )


@pytest.mark.parametrize(
    "p, expected",
    [
        (np.array([[-0.2, 0.4], [0.4, 0.6], [0.9, 0.1]]), (-0.2, 0.9)),
        (np.array([[0.2, 0.4], [0.4, 0.6], [0.9, 1.1]]), (0.2, 1.1)),
    ],
)
def test_check_propensity_score_min_max(p, expected):
    with pytest.raises(ValueError) as e:
        check_propensity_score(p)
    assert (
        e.value.args[0] == f"Propensity scores have to be between 0 and 1. Minimum is "
        f"{expected[0]:.4f} and maximum is {expected[1]:.4f}."
    )


@pytest.mark.parametrize(
    "p, expected",
    [
        (np.array([[0.2, 0.4], [0.4, 0.6], [0.9, 0.1]]), (0.6, 1)),
        (np.array([[0.2, 0.8], [0.4, 0.6], [0.9, 0.4]]), (1, 1.3)),
    ],
)
def test_check_propensity_score_sum_to_one(p, expected):
    with pytest.raises(ValueError) as e:
        check_propensity_score(p, sum_to_one=True)
    assert (
        e.value.args[0]
        == f"Propensity scores for all observations must sum to 1. Minimum is "
        f"{expected[0]:.4f} and maximum is {expected[1]:.4f}."
    )


@pytest.mark.parametrize("value", [np.nan, -0.5, 0, 0.5, 1, 1.5])
@pytest.mark.parametrize("zero_included", [False, True])
@pytest.mark.parametrize("one_included", [False, True])
def test_check_probability(value, zero_included, one_included):
    if np.isnan(value):
        context = pytest.raises(
            ValueError, match="Invalid input! Probability p should not be NaN."
        )
    elif zero_included and value < 0:
        context = pytest.raises(
            ValueError, match="Probability p must be greater than or equal to 0."
        )
    elif not zero_included and value <= 0:
        context = pytest.raises(
            ValueError, match="Probability p must be greater than or equal to 0."
        )
    elif one_included and value > 1:
        context = pytest.raises(
            ValueError, match="Probability p must be less than or equal to 1."
        )
    elif not one_included and value >= 1:
        context = pytest.raises(
            ValueError, match="Probability p must be less than or equal to 1."
        )
    else:
        context = does_not_raise()  # type: ignore
    with context:
        check_probability(value, zero_included, one_included)
