# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Literal, Protocol, Union

import numpy as np
import pandas as pd
import scipy.sparse as sps

PredictMethod = Literal["predict", "predict_proba"]

# As of 24/01/19, no convenient way of dynamically creating a literal collection that
# mypy can deal with seems to exist. Therefore we duplicate the values.
# See https://stackoverflow.com/questions/64522040/typing-dynamically-create-literal-alias-from-list-of-valid-values
# As of 24/04/25 there is no way either to reuse variables inside a Literal definition, see
# https://mypy.readthedocs.io/en/stable/literal_types.html#limitations
OosMethod = Literal["overall", "median", "mean"]

Params = Mapping[str, int | float | str]
Features = Collection[str] | Collection[int] | None

# ruff is not happy about the usage of Union.
Vector = Union[pd.Series, np.ndarray]  # noqa
Matrix = Union[pd.DataFrame, np.ndarray, sps.csr_matrix]  # noqa


class _ScikitModel(Protocol):
    _estimator_type: str

    # https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model/60542986#60542986
    def fit(self, X, y, *params, **kwargs): ...
    def predict(self, X, *params, **kwargs): ...

    def score(self, X, y, **kwargs): ...

    def set_params(self, **params): ...


ModelFactory = type[_ScikitModel] | dict[str, type[_ScikitModel]]

# List of (train_indices, test_indices) tuples where each
# list item corresponds to one way of splitting or folding data.
# For instance, if converting the Generator resulting from a call to
# sklearn.model_selection.KFold.split to a list we obtain this type.
SplitIndices = list[tuple[np.ndarray, np.ndarray]]

Scorer = str | Callable
Scorers = Sequence[Scorer]
Scoring = Mapping[str, Scorers]
