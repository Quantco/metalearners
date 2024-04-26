# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from itertools import chain

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from metalearners.data_generation import insert_missing
from metalearners.metalearner import MetaLearner, _validate_nuisance_predict_methods


@pytest.mark.parametrize(
    "set1,set2,success",
    [
        ({"a", "b"}, {"a", "b"}, True),
        ({"a"}, {"a", "b"}, False),
        ({"a", "b"}, {"a"}, False),
    ],
)
def test_validate_nuisance_predict_methods(set1, set2, success):
    if success:
        _validate_nuisance_predict_methods(set1, set2)
    else:
        "Mapping from nuisance model"
        with pytest.raises(
            ValueError, match=r"Mapping from nuisance model kind to predict method"
        ):
            _validate_nuisance_predict_methods(set1, set2)


class _TestMetaLearner(MetaLearner):
    @classmethod
    def nuisance_model_names(cls):
        return {"nuisance1", "nuisance2"}

    @classmethod
    def treatment_model_names(cls):
        return {"treatment1", "treatment2"}

    @property
    def _nuisance_predict_methods(self):
        return {"nuisance1": "predict", "nuisance2": "predict"}

    def fit(self, X, y, w):
        for model_kind in self.__class__.nuisance_model_names():
            self._nuisance_models[model_kind].fit(X, y)
        for model_kind in self.__class__.treatment_model_names():
            self._treatment_models[model_kind].fit(X, y)
        return self

    def predict(self, X, is_oos, oos_method=None):
        return np.zeros(len(X))

    def evaluate(self, X, y, w, is_oos, oos_method=None):
        return {}

    def predict_potential_outcomes(self, X, is_oos, oos_method=None):
        return np.zeros((len(X), 1))

    def _pseudo_outcome(self, X):
        return np.zeros(len(X))


@pytest.mark.parametrize("nuisance_model_factory", [LGBMRegressor])
@pytest.mark.parametrize("treatment_model_factory", [LGBMRegressor])
@pytest.mark.parametrize("is_classification", [True, False])
@pytest.mark.parametrize("nuisance_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize("treatment_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize("feature_set", [None])
@pytest.mark.parametrize("n_folds", [5])
def test_metalearner_init(
    nuisance_model_factory,
    treatment_model_factory,
    is_classification,
    nuisance_model_params,
    treatment_model_params,
    feature_set,
    n_folds,
):
    _TestMetaLearner(
        nuisance_model_factory=nuisance_model_factory,
        treatment_model_factory=treatment_model_factory,
        is_classification=is_classification,
        nuisance_model_params=nuisance_model_params,
        treatment_model_params=treatment_model_params,
        feature_set=feature_set,
        n_folds=n_folds,
    )


@pytest.mark.parametrize("implementation", [_TestMetaLearner])
def test_metalearner_categorical(
    mixed_experiment_dataset_continuous_outcome, implementation
):
    covariates, _, treatment, observed_outcomes, _, _ = (
        mixed_experiment_dataset_continuous_outcome
    )
    ml = implementation(
        LGBMRegressor,
        LGBMRegressor,
        is_classification=False,
        nuisance_model_params={"n_estimators": 1},  # Just to make the test faster
        treatment_model_params={"n_estimators": 1},
    )
    ml.fit(X=covariates, y=observed_outcomes, w=treatment)
    categorical_columns = [
        covariates.columns.get_loc(col)
        for col in covariates.select_dtypes(include="category").columns
    ]
    for cf_estimator in chain(
        ml._nuisance_models.values(), ml._treatment_models.values()
    ):
        assert (
            categorical_columns
            == cf_estimator._estimators[0]._Booster.params["categorical_column"]
        )
        if cf_estimator.enable_overall:
            assert (
                categorical_columns
                == cf_estimator._overall_estimator._Booster.params["categorical_column"]
            )


@pytest.mark.parametrize("implementation", [_TestMetaLearner])
def test_metalearner_missing_data_smoke(
    mixed_experiment_dataset_continuous_outcome, implementation, rng
):
    covariates, _, treatment, observed_outcomes, _, _ = (
        mixed_experiment_dataset_continuous_outcome
    )

    covariates_with_missing = insert_missing(
        covariates, missing_probability=0.25, rng=rng
    )
    ml = implementation(
        LGBMRegressor,
        LGBMRegressor,
        is_classification=False,
        nuisance_model_params={"n_estimators": 1},  # Just to make the test faster
        treatment_model_params={"n_estimators": 1},
    )
    ml.fit(X=covariates_with_missing, y=observed_outcomes, w=treatment)


@pytest.mark.parametrize("implementation", [_TestMetaLearner])
def test_metalearner_missing_data_error(
    numerical_experiment_dataset_continuous_outcome, implementation, rng
):
    covariates, _, treatment, observed_outcomes, _, _ = (
        numerical_experiment_dataset_continuous_outcome
    )
    covariates_with_missing = insert_missing(
        covariates, missing_probability=0.25, rng=rng
    )

    ml = implementation(LinearRegression, LinearRegression, is_classification=False)
    with pytest.raises(
        ValueError, match=r"LinearRegression does not accept missing values*"
    ):
        ml.fit(X=covariates_with_missing, y=observed_outcomes, w=treatment)


@pytest.mark.parametrize("implementation", [_TestMetaLearner])
def test_metalearner_format_consistent(
    numerical_experiment_dataset_continuous_outcome, implementation
):
    covariates, _, treatment, observed_outcomes, _, _ = (
        numerical_experiment_dataset_continuous_outcome
    )

    np_ml = implementation(
        LGBMRegressor,
        LGBMRegressor,
        is_classification=False,
        nuisance_model_params={"n_estimators": 1},  # Just to make the test faster
        treatment_model_params={"n_estimators": 1},
    )
    pd_ml = implementation(
        LGBMRegressor,
        LGBMRegressor,
        is_classification=False,
        nuisance_model_params={"n_estimators": 1},  # Just to make the test faster
        treatment_model_params={"n_estimators": 1},
    )

    np_ml.fit(X=covariates, y=observed_outcomes, w=treatment)
    pd_ml.fit(
        X=pd.DataFrame(covariates),
        y=pd.Series(observed_outcomes),
        w=pd.Series(treatment),
    )

    np_cate_estimates = np_ml.predict(covariates, is_oos=False)
    pd_cate_estimates = np_ml.predict(pd.DataFrame(covariates), is_oos=False)
    np.testing.assert_allclose(np_cate_estimates, pd_cate_estimates)


@pytest.mark.parametrize("implementation", [_TestMetaLearner])
def test_metalearner_model_names(implementation):
    set1 = implementation.nuisance_model_names()
    set2 = implementation.treatment_model_names()
    assert len(set1 | set2) == len(set1) + len(set2)
