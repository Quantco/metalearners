# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import operator
from typing import Protocol, Union

import numpy as np
import pandas as pd

Vector = Union[pd.Series, np.ndarray]
Matrix = Union[pd.DataFrame, np.ndarray]


class _ScikitModel(Protocol):
    # https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model/60542986#60542986
    def fit(self, X, y, *params, **kwargs): ...

    def predict(self, X, *params, **kwargs): ...

    def score(self, X, y, **kwargs): ...

    def set_params(self, **params): ...


def index_matrix(matrix: Matrix, rows: Vector) -> Matrix:
    """Subselect certain rows from a matrix."""
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
