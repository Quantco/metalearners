# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import numpy as np
import pytest
from lightgbm import LGBMRegressor

from metalearners.metalearner import MetaLearner


class _TestMetaLearner(MetaLearner):
    @classmethod
    def nuisance_model_names(cls):
        return ["nuisance1", "nuisance2"]

    @classmethod
    def treatment_model_names(cls):
        return ["treatment1", "treatment2"]

    def fit(self, X, y, w):
        for model_kind in self.__class__.nuisance_model_names():
            self._nuisance_models[model_kind].fit(X, y)
        for model_kind in self.__class__.treatment_model_names():
            self._treatment_models[model_kind].fit(X, y)
        return self

    def predict(self, X, is_oos, oos_method=None):
        return np.zeros(len(X))

    def evaluate(self, X, y, w, is_regression, is_oos, oos_method=None):
        return {}

    def predict_potential_outcomes(self, X, is_oos, oos_method=None):
        return np.zeros(len(X)), np.zeros(len(X))

    def _pseudo_outcome(self, X):
        return np.zeros(len(X))


@pytest.mark.parametrize("nuisance_model_factory", [LGBMRegressor])
@pytest.mark.parametrize("treatment_model_factory", [LGBMRegressor])
@pytest.mark.parametrize("nuisance_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize("treatment_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize("feature_set", [None])
@pytest.mark.parametrize("n_folds", [5])
def test_metalearner_init(
    mindset_data,
    nuisance_model_factory,
    treatment_model_factory,
    nuisance_model_params,
    treatment_model_params,
    feature_set,
    n_folds,
):
    _TestMetaLearner(
        nuisance_model_factory=nuisance_model_factory,
        treatment_model_factory=treatment_model_factory,
        nuisance_model_params=nuisance_model_params,
        treatment_model_params=treatment_model_params,
        feature_set=feature_set,
        n_folds=n_folds,
    )
