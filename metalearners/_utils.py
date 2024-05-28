# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import operator
from collections.abc import Callable
from inspect import signature
from operator import le, lt
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import check_array, check_X_y, is_classifier, is_regressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from metalearners._typing import PredictMethod, _ScikitModel

# ruff is not happy about the usage of Union.
Vector = Union[pd.Series, np.ndarray]  # noqa
Matrix = Union[pd.DataFrame, np.ndarray]  # noqa

_PREDICT = "predict"
_PREDICT_PROBA = "predict_proba"


default_rng = np.random.default_rng()


def index_matrix(matrix: Matrix, rows: Vector) -> Matrix:
    """Subselect certain rows from a matrix."""
    if isinstance(rows, pd.Series):
        rows = rows.to_numpy()
    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[rows]
    return matrix[rows, :]


def are_pd_indices_equal(*args: pd.DataFrame | pd.Series) -> bool:
    if len(args) < 2:
        return True
    reference_index = args[0].index
    for data_structure in args[1:]:
        if any(data_structure.index != reference_index):
            return False
    return True


def is_pd_df_or_series(arg) -> bool:
    return isinstance(arg, pd.DataFrame) or isinstance(arg, pd.Series)


def validate_all_vectors_same_index(*args: Vector) -> None:
    if len(args) < 2:
        return None
    pd_args = list(filter(is_pd_df_or_series, args))
    if len(pd_args) > 1:
        if not are_pd_indices_equal(*pd_args):
            raise ValueError(
                "All inputs provided as pandas data structures are expected to rely on "
                "the same index. Yet, at least two data structures have a different index."
            )
    if 0 < len(pd_args) < len(args):
        if any(pd_args[0].index != range(0, len(pd_args[0]))):  # type: ignore
            raise ValueError(
                "In order to mix numpy np.ndarray  and pd.Series objects, the pd.Series objects "
                "should have an index of 0 to n-1."
            )


def validate_number_positive(
    value: int | float, name: str, strict: bool = False
) -> None:
    if strict:
        comparison = operator.lt
    else:
        comparison = operator.le
    if comparison(value, 0):
        raise ValueError(f"{name} was expected to be positive but was {value}.")


def check_propensity_score(
    propensity_scores: Matrix,
    features: Matrix | None = None,
    n_variants: int = 2,
    sum_to_one: bool = False,
    check_kwargs: dict | None = None,
    sum_tolerance: float = 0.001,
) -> None:
    """Ensure propensity scores match assumptions.

    The function ensures that ``propensity_scores`` and, if provided, ``features`` are:
     * shape-compatible and contain no missing or invalid entries;
     * the shape of ``propensity_scores`` is (nobs, ``n_variants``);
     * propensity scores are between 0 and 1;
     * if ``sum_to_one`` is ``True``, ``propensity_scores`` sums up to 1 for each observation.
    Shape compatibility is checked through scikit-learn's :func:`check_X_y` and
    :func:`check_array`; optional parameters for that method can be supplied via
    ``check_kwargs``.
    """
    if check_kwargs is None:
        check_kwargs = {}

    if len(propensity_scores.shape) < 2 or propensity_scores.shape[1] != n_variants:
        raise ValueError(
            f"One propensity score must be provided for each variant. "
            f"There are {n_variants} but the shape of the propensity "
            f"scores is {propensity_scores.shape}."
        )

    if features is not None:
        check_X_y(features, propensity_scores, multi_output=True, **check_kwargs)
    else:
        check_array(propensity_scores, **check_kwargs)

    if not 0.0 < np.min(propensity_scores) <= np.max(propensity_scores) < 1.0:
        raise ValueError(
            f"Propensity scores have to be between 0 and 1. Minimum is "
            f"{np.min(propensity_scores):.4f} and maximum is "
            f"{np.max(propensity_scores):.4f}."
        )

    if sum_to_one:
        min_sum = np.min(np.sum(propensity_scores, axis=1))
        max_sum = np.max(np.sum(propensity_scores, axis=1))
        if not 1 - sum_tolerance < min_sum <= max_sum < 1 + sum_tolerance:
            raise ValueError(
                f"Propensity scores for all observations must sum to 1. "
                f"Minimum is {min_sum:.4f} and maximum is {max_sum:.4f}."
            )


def convert_and_pad_propensity_score(
    propensity_scores: Vector | Matrix, n_variants: int
) -> np.ndarray:
    """Convert to ``np.ndarray`` and pad propensity scores, if necessary.

    Taking in a matrix or vector of propensity scores ``propensity_scores``, the function
    performs two things:
    * convert ``propensity_scores`` to an ``np.ndarray``.
    * if ``propensity_scores`` is a vector, the function will expand it to a matrix in
     case the number of variants ``n_variants`` is 2. That ensures there  is one
     propensity score per variant. The expansion assumes that the provided scores are
     those for the second variant.
    """
    if isinstance(propensity_scores, pd.Series) or isinstance(
        propensity_scores, pd.DataFrame
    ):
        propensity_scores = propensity_scores.to_numpy()
    p_is_1d = len(propensity_scores.shape) == 1 or propensity_scores.shape[1] == 1
    if n_variants == 2 and p_is_1d:
        propensity_scores = np.c_[1 - propensity_scores, propensity_scores]
    return propensity_scores


def get_n_variants(propensity_scores: Matrix) -> int:
    """Returns the number of treatment variants based on the shape of
    ``propensity_scores``.

    If the propensity scores array has a single dimension (i.e., it's a 1D array) or
    only one column in the second dimension, the function returns 2 (representing a binary
    treatment scenario, typically treated vs. not-treated). Otherwise, the number of treatment
    variants is assumed to be the size of the second dimension of the ``propensity_scores``
    array.

    propensity_score can be either a ``np.ndarray`` or a ``pd.DataFrame`` of propensity
    scores. The expected shape is ``(n_obs,)`` for binary treatment, or
    ``(n_obs, n_variants)`` for multiple treatments.
    """
    n_variants = (
        2
        if len(propensity_scores.shape) == 1 or propensity_scores.shape[1] == 1
        else propensity_scores.shape[1]
    )
    return n_variants


def get_linear_dimension(X: Matrix) -> int:
    """Calculates the required dimensionality of a vector in order to perform a linear
    transformation of the given data matrix.

    If the matrix consists only of numerical variables, the linear dimension is the
    number of features in the matrix. However, if there are categorical variables, they
    are expanded into dummy/indicator variables (one-hot encoding) for each category.
    Thus, the dimension, in this case, is the total number of numerical features plus
    the number of individual categories across all categorical features.
    """
    if isinstance(X, pd.DataFrame):
        categorical_features = X.select_dtypes(include="category")
        n_categories = 0
        for c in categorical_features.columns:
            n_categories += len(X[c].cat.categories)
        return len(set(X.columns) - set(categorical_features)) + n_categories
    return X.shape[1]


def sigmoid(x: np.ndarray) -> np.ndarray:
    r"""Sigmoid function.

    .. math::
        \sigma (x) = \frac{1}{1+e^{-x}}
    """
    return 1 / (1 + np.exp(-x))


def check_probability(p: float, zero_included=False, one_included=False) -> None:
    r"""Checks whether the provided probability p lies within a valid range.

    The valid range, from 0 to 1, which can be inclusive or exclusive, depending on
    the flags ``zero_included`` and ``one_included`` respectively.
    """

    if np.isnan(p):
        raise ValueError("Invalid input! Probability p should not be NaN.")

    left_operator = le if zero_included else lt
    right_operator = le if one_included else lt

    if not left_operator(0, p):
        raise ValueError("Probability p must be greater than or equal to 0.")
    if not right_operator(p, 1):
        raise ValueError("Probability p must be less than or equal to 1.")


def convert_treatment(treatment: Vector) -> np.ndarray:
    """Convert to ``np.ndarray`` and adapt dtype, if necessary."""
    if isinstance(treatment, np.ndarray):
        new_treatment = treatment.copy()
    elif isinstance(treatment, pd.Series):
        new_treatment = treatment.to_numpy()
    if new_treatment.dtype == bool:
        return new_treatment.astype(int)
    elif new_treatment.dtype == float and all(x.is_integer() for x in new_treatment):
        return new_treatment.astype(int)
    elif new_treatment.dtype != int:
        raise TypeError(
            "Treatment must be boolean, integer or float with integer values."
        )
    return new_treatment


def supports_categoricals(model: _ScikitModel) -> bool:
    if (
        isinstance(model, HistGradientBoostingClassifier)
        or isinstance(model, HistGradientBoostingRegressor)
    ) and model.categorical_features == "from_dtype":
        return True
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        if isinstance(model, LGBMClassifier | LGBMRegressor):
            return True
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier, XGBRegressor

        if isinstance(model, XGBClassifier | XGBRegressor) and model.enable_categorical:
            return True
    except (ImportError, AttributeError):  # enable_categorical was added in v1.5.0
        pass
    try:
        from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV

        if isinstance(model, GeneralizedLinearRegressor | GeneralizedLinearRegressorCV):
            return True
    except ImportError:
        pass
    # TODO: Add support for Catboost? The problem is that we need the cat features names and reinit the model
    return False


def function_has_argument(func: Callable, argument: str) -> bool:
    return argument in signature(func).parameters


def validate_model_and_predict_method(
    model_factory: type[_ScikitModel],
    predict_method: PredictMethod,
    name: str = "model",
) -> None:
    if is_classifier(model_factory) and predict_method == _PREDICT:
        raise ValueError(
            f"The {name} is supposed to be used with the predict "
            "method 'predict' but it is a classifier."
        )
    if is_regressor(model_factory) and predict_method == _PREDICT_PROBA:
        raise ValueError(
            f"The {name} is supposed to be used with the predict "
            "method 'predict_proba' but it is not a classifier."
        )


def clip_element_absolute_value_to_epsilon(vector: Vector, epsilon: float) -> Vector:
    """Clip in accordance with the sign of each element if the element's absolute value
    is below ``epsilon``.

    For instance, if ``vector`` equals ``[0, 1, -1, .09, -.09]`` and ``epsilon`` equals ``.1``,
    this function will return ``[0, 1, -1, .1, -.1]``.

    Typically, this is done in order to avoid numerical problems when there is an element-wise
    division by ``vector`` and that the elements of ``vector`` are very close to 0.
    """
    bound = np.where(vector < 0, -1, 1) * epsilon
    return np.where(np.abs(vector) < epsilon, bound, vector)


def validate_valid_treatment_variant_not_control(
    treatment_variant: int, n_variants: int
) -> None:
    if treatment_variant >= n_variants:
        raise ValueError(
            f"MetaLearner was initialized to have {n_variants} 0-index variants but tried "
            f"to index variant {treatment_variant}."
        )
    if treatment_variant < 1:
        raise ValueError(
            "pseudo outcomes can only be computed for a treatment variant which isn't "
            "considered to be control."
        )
