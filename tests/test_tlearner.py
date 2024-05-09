# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression

from metalearners.tlearner import TLearner


def test_validate_models():
    with pytest.raises(
        ValueError,
        match="is_classification is set to True but the treatment_model is not a classifier.",
    ):
        TLearner(LGBMRegressor, True)
    with pytest.raises(
        ValueError,
        match="is_classification is set to False but the treatment_model is not a regressor.",
    ):
        TLearner(LGBMClassifier, False)


def test_check_treatment_error_multi():
    tlearner = TLearner(LinearRegression, False)
    covariates = np.zeros((10, 1))
    w = np.array(range(10))
    y = np.zeros(10)
    with pytest.raises(NotImplementedError, match="Current implementation of TLearner"):
        tlearner.fit(covariates, y, w)


def test_check_treatment_error_encoding():
    tlearner = TLearner(LinearRegression, False)
    covariates = np.zeros((10, 1))
    w = np.array([1, 2] * 5)
    y = np.zeros(10)
    with pytest.raises(ValueError, match="Treatment variant should be encoded"):
        tlearner.fit(covariates, y, w)
