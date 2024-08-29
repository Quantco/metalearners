# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor

from metalearners._utils import (
    are_pd_indices_equal,
    check_probability,
    check_propensity_score,
    clip_element_absolute_value_to_epsilon,
    convert_treatment,
    function_has_argument,
    get_linear_dimension,
    index_matrix,
    index_vector,
    supports_categoricals,
    validate_all_vectors_same_index,
    validate_model_and_predict_method,
    validate_valid_treatment_variant_not_control,
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


@pytest.mark.parametrize(
    "treatment",
    [
        np.array([0, 1, 0, 2]),
        np.array([0.0, 1.0, 2.0]),
        np.array([False, True, False]),
    ],
)
@pytest.mark.parametrize("use_pd", [False, True])
def test_convert_treatment(treatment, use_pd):
    if use_pd:
        treatment = pd.Series(treatment)
    new_treatment = convert_treatment(treatment)
    assert isinstance(new_treatment, np.ndarray)
    assert new_treatment.dtype == int


@pytest.mark.parametrize("use_pd", [False, True])
def test_convert_treatment_raise(use_pd):
    if use_pd:
        treatment = pd.Series([1.2, 0.5])
    else:
        treatment = np.array([1.2, 0.5])
    with pytest.raises(
        TypeError,
        match="Treatment must be boolean, integer or float with integer values.",
    ):
        convert_treatment(treatment)


@pytest.mark.parametrize(
    "model, expected",
    [
        (LinearRegression(), False),
        (
            HistGradientBoostingClassifier(),
            False,
        ),  # The default for categorical_features will change to "from_dtype" in v1.6
        (HistGradientBoostingClassifier(categorical_features="from_dtype"), True),
        (LGBMRegressor(), True),
        (XGBRegressor(), False),
        (XGBClassifier(enable_categorical=True), True),
        (GeneralizedLinearRegressor(), True),
        (GeneralizedLinearRegressorCV(), True),
    ],
)
def test_supports_categoricals(model, expected):
    assert supports_categoricals(model) == expected


def _foo1(sample_weight):
    return None


def _foo2(qeight, sample_weight):
    return None


def _foo3():
    return None


def _foo4(weight):
    return None


def _foo5(weight, sample_weight=None):
    return None


@pytest.mark.parametrize(
    "foo, result",
    [
        (lambda sample_weight: None, True),
        (lambda sample_qeight: None, False),
        (_foo1, True),
        (_foo2, True),
        (_foo3, False),
        (_foo4, False),
        (_foo5, True),
        (LinearRegression.fit, True),
        (LGBMRegressor.fit, True),
    ],
)
def test_function_has_argument(foo, result):
    assert function_has_argument(foo, "sample_weight") == result


@pytest.mark.parametrize(
    "data,result",
    [
        ([pd.Series([10, 11, 12]), pd.Series(data=[10, 11, 12])], True),
        (
            [pd.Series([10, 11, 12]), pd.Series(index=[0, 1, 2], data=[10, 11, 12])],
            True,
        ),
        (
            [pd.Series([10, 11, 12]), pd.Series(index=[1, 2, 3], data=[10, 11, 12])],
            False,
        ),
        (
            [
                pd.Series([10, 11, 12], index=[1, 2, 3]),
                pd.Series(index=[1, 2, 3], data=[10, 11, 12]),
            ],
            True,
        ),
    ],
)
def test_are_pd_indices_equal(data, result):
    assert are_pd_indices_equal(*data) == result


@pytest.mark.parametrize(
    "data, result",
    [
        ([pd.Series([10, 11, 12]), pd.Series(data=[10, 11, 12])], "valid"),
        (
            [pd.Series([10, 11, 12]), pd.Series(index=[0, 1, 2], data=[10, 11, 12])],
            "valid",
        ),
        (
            [pd.Series([10, 11, 12]), pd.Series(index=[1, 2, 3], data=[10, 11, 12])],
            "pd_invalid",
        ),
        (
            [
                pd.Series([10, 11, 12], index=[1, 2, 3]),
                pd.Series(index=[1, 2, 3], data=[10, 11, 12]),
            ],
            "valid",
        ),
        ([pd.Series([10, 11, 12])], "valid"),
        ([pd.Series([10, 11, 12]), np.ndarray([10, 11, 12])], "valid"),
        (
            [pd.Series([10, 11, 12], index=[1, 2, 3]), np.ndarray([10, 11, 12])],
            "mixed_invalid",
        ),
        (
            [pd.Series([10, 11, 12], index=[0, 1, 3]), np.ndarray([10, 11, 12])],
            "mixed_invalid",
        ),
    ],
)
def test_validate_all_vectors_same_index(data, result):
    if result == "valid":
        validate_all_vectors_same_index(*data)
    elif result == "pd_invalid":
        with pytest.raises(
            ValueError,
            match="are expected to rely on the same index",
        ):
            validate_all_vectors_same_index(*data)
    elif result == "mixed_invalid":
        with pytest.raises(
            ValueError,
            match="should have an index of 0 to n-1",
        ):
            validate_all_vectors_same_index(*data)


@pytest.mark.parametrize(
    "factory,predict_method,success",
    [
        (LGBMRegressor, "predict", True),
        (LGBMRegressor, "predict_proba", False),
        (LGBMClassifier, "predict", False),
        (LGBMClassifier, "predict_proba", True),
    ],
)
def test_validate_model_and_predict_method(factory, predict_method, success):
    if success:
        validate_model_and_predict_method(factory, predict_method)
    else:
        with pytest.raises(
            ValueError, match="supposed to be used with the predict method"
        ):
            validate_model_and_predict_method(factory, predict_method)


def test_increase_element_absolute_value_by_epsilon():
    vector = np.array([0, 1, -1, 0.1, -0.1, 0.05, -0.05])
    epsilon = 0.1
    result = clip_element_absolute_value_to_epsilon(vector, epsilon)
    assert all(result == np.array([epsilon, 1, -1, 0.1, -0.1, epsilon, -epsilon]))


@pytest.mark.parametrize(
    "treatment_variant,n_variants,success",
    [
        (0, 2, False),
        (-1, 2, False),
        (1, 2, True),
        (2, 2, False),
    ],
)
def test_validate_valid_treatment_variant_not_control(
    treatment_variant, n_variants, success
):
    if success:
        validate_valid_treatment_variant_not_control(treatment_variant, n_variants)
    else:
        with pytest.raises(ValueError, match="variant"):
            validate_valid_treatment_variant_not_control(treatment_variant, n_variants)


@pytest.mark.parametrize("matrix_backend", [np.ndarray, pd.DataFrame, csr_matrix])
@pytest.mark.parametrize("rows_backend", [np.array, pd.Series])
def test_index_matrix(matrix_backend, rows_backend):
    n_samples = 10
    if matrix_backend == np.ndarray:
        matrix = np.array(list(range(n_samples))).reshape((-1, 1))
    elif matrix_backend == pd.DataFrame:
        # We make sure that the index is not equal to the row number.
        matrix = pd.DataFrame(
            list(range(n_samples)), index=list(range(20, 20 + n_samples))
        )
    elif matrix_backend == csr_matrix:
        matrix = csr_matrix(np.array(list(range(n_samples))).reshape((-1, 1)))
    else:
        raise ValueError()
    rows = rows_backend([1, 4, 5])
    result = index_matrix(matrix=matrix, rows=rows)

    assert isinstance(result, matrix_backend)
    assert result.shape[1] == matrix.shape[1]

    if isinstance(result, pd.DataFrame):
        processed_result = result.values[:, 0]
    else:
        processed_result = result[:, 0]

    expected = np.array([1, 4, 5])
    assert (processed_result == expected).sum() == len(expected)


@pytest.mark.parametrize("vector_backend", [np.ndarray, pd.Series])
@pytest.mark.parametrize("rows_backend", [np.array, pd.Series])
def test_index_vector(vector_backend, rows_backend):
    n_samples = 10
    if vector_backend == np.ndarray:
        vector = np.array(list(range(n_samples)))
    elif vector_backend == pd.Series:
        # We make sure that the index is not equal to the row number.
        vector = pd.Series(
            list(range(n_samples)), index=list(range(20, 20 + n_samples))
        )
    else:
        raise ValueError()

    rows = rows_backend([1, 4, 5])

    result = index_vector(vector=vector, rows=rows)
    assert isinstance(result, vector_backend)

    if isinstance(result, pd.Series):
        result = result.values

    expected = np.array([1, 4, 5])
    assert (result == expected).all()
