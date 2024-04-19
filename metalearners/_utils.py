# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import operator
from operator import le, lt
from typing import Optional, Protocol, Union

import numpy as np
import pandas as pd
from sklearn.base import check_array, check_X_y

Vector = Union[pd.Series, np.ndarray]
Matrix = Union[pd.DataFrame, np.ndarray]

default_rng = np.random.default_rng()


class _ScikitModel(Protocol):
    # https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model/60542986#60542986
    def fit(self, X, y, *params, **kwargs): ...

    def predict(self, X, *params, **kwargs): ...

    def score(self, X, y, **kwargs): ...

    def set_params(self, **params): ...


def index_matrix(matrix: Matrix, rows: Vector) -> Matrix:
    """Subselect certain rows from a matrix."""
    if isinstance(rows, pd.Series):
        rows = rows.to_numpy()
    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[rows]
    return matrix[rows, :]


def validate_number_positive(
    value: Union[int, float], name: str, strict: bool = False
) -> None:
    if strict:
        comparison = operator.lt
    else:
        comparison = operator.le
    if comparison(value, 0):
        raise ValueError(f"{name} was expected to be positive but was {value}.")


def check_propensity_score(
    propensity_scores: Matrix,
    features: Optional[Matrix] = None,
    n_variants: int = 2,
    sum_to_one: bool = False,
    check_kwargs: Optional[dict] = None,
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
    propensity_scores: Union[Vector, Matrix], n_variants: int
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


def get_linear_dimension(X: Matrix):
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
