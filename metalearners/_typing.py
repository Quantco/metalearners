# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from typing import Literal, Protocol

PredictMethod = Literal["predict", "predict_proba"]

# As of 24/01/19, no convenient way of dynamically creating a literal collection that
# mypy can deal with seems to exist. Therefore we duplicate the values.
# See https://stackoverflow.com/questions/64522040/typing-dynamically-create-literal-alias-from-list-of-valid-values
# As of 24/04/25 there is no way either to reuse variables inside a Literal definition, see
# https://mypy.readthedocs.io/en/stable/literal_types.html#limitations
OosMethod = Literal["overall", "median", "mean"]


class _ScikitModel(Protocol):
    _estimator_type: str

    # https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model/60542986#60542986
    def fit(self, X, y, *params, **kwargs): ...

    def predict(self, X, *params, **kwargs): ...

    def score(self, X, y, **kwargs): ...

    def set_params(self, **params): ...
