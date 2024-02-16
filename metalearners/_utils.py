# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from typing import Protocol, Union

import numpy as np
import pandas as pd

Vector = Union[pd.Series, np.ndarray]
Matrix = Union[pd.DataFrame, np.ndarray]


class _ScikitModel(Protocol):
    # https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model/60542986#60542986
    def __call__(self, **kwargs): ...

    def fit(self, X, y, sample_weight=None, **kwargs): ...

    def predict(self, X, **kwargs): ...

    def score(self, X, y, sample_weight=None, **kwargs): ...

    def set_params(self, **params): ...


def index_matrix(matrix: Matrix, rows: Vector) -> Matrix:
    """Subselect certain rows from a matrix."""
    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[rows]
    return matrix[rows, :]
