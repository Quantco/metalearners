# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from itertools import chain

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression

from metalearners.data_generation import insert_missing
from metalearners.metalearner import (
    MetaLearner,
    _combine_propensity_and_nuisance_specs,
    _ModelSpecifications,
)
from metalearners.rlearner import RLearner
from metalearners.slearner import SLearner
from metalearners.tlearner import TLearner
from metalearners.xlearner import XLearner


class _TestMetaLearner(MetaLearner):
    @classmethod
    def nuisance_model_specifications(cls):
        return {
            "nuisance1": _ModelSpecifications(
                cardinality=lambda ml: ml.n_variants, predict_method=lambda _: "predict"
            ),
            "nuisance2": _ModelSpecifications(
                cardinality=lambda _: 1, predict_method=lambda _: "predict"
            ),
        }

    @classmethod
    def treatment_model_specifications(cls):
        return {
            "treatment1": _ModelSpecifications(
                cardinality=lambda _: 1, predict_method=lambda _: "predict"
            ),
            "treatment2": _ModelSpecifications(
                cardinality=lambda _: 1, predict_method=lambda _: "predict"
            ),
        }

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return True

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return False

    def _validate_models(self) -> None: ...

    def fit(self, X, y, w):
        for model_kind in self.__class__.nuisance_model_specifications():
            for model_ord in range(
                self.nuisance_model_specifications()[model_kind]["cardinality"](self)
            ):
                self._nuisance_models[model_kind][model_ord].fit(X, y)
        for model_kind in self.__class__.treatment_model_specifications():
            for model_ord in range(
                self.treatment_model_specifications()[model_kind]["cardinality"](self)
            ):
                self._treatment_models[model_kind][model_ord].fit(X, y)
        return self

    def predict(self, X, is_oos, oos_method=None):
        return np.zeros(len(X))

    def evaluate(self, X, y, w, is_oos, oos_method=None):
        return {}

    def predict_conditional_average_outcomes(self, X, is_oos, oos_method=None):
        return np.zeros((len(X), 1))


@pytest.mark.parametrize("nuisance_model_factory", [LGBMRegressor])
@pytest.mark.parametrize("treatment_model_factory", [LGBMRegressor])
@pytest.mark.parametrize("is_classification", [True, False])
@pytest.mark.parametrize("nuisance_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize("treatment_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize(
    "feature_set",
    [
        None,
        {
            "nuisance1": ["X1"],
            "nuisance2": ["X2"],
            "propensity_model": ["Xp"],
            "treatment1": ["X1"],
            "treatment2": ["X2"],
        },
    ],
)
@pytest.mark.parametrize("n_folds", [5])
@pytest.mark.parametrize("propensity_model_factory", [None, LGBMClassifier])
@pytest.mark.parametrize("propensity_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize("n_variants", [2, 5, 10])
def test_metalearner_init(
    nuisance_model_factory,
    treatment_model_factory,
    propensity_model_factory,
    is_classification,
    n_variants,
    nuisance_model_params,
    treatment_model_params,
    propensity_model_params,
    feature_set,
    n_folds,
):
    _TestMetaLearner(
        nuisance_model_factory=nuisance_model_factory,
        is_classification=is_classification,
        n_variants=n_variants,
        treatment_model_factory=treatment_model_factory,
        propensity_model_factory=propensity_model_factory,
        nuisance_model_params=nuisance_model_params,
        treatment_model_params=treatment_model_params,
        propensity_model_params=propensity_model_params,
        feature_set=feature_set,
        n_folds=n_folds,
    )


@pytest.mark.parametrize(
    "implementation", [_TestMetaLearner, TLearner, SLearner, XLearner, RLearner]
)
def test_metalearner_categorical(
    mixed_experiment_dataset_continuous_outcome_binary_treatment_linear_te,
    implementation,
):
    covariates, _, treatment, observed_outcomes, _, _ = (
        mixed_experiment_dataset_continuous_outcome_binary_treatment_linear_te
    )
    ml = implementation(
        nuisance_model_factory=LGBMRegressor,
        is_classification=False,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        nuisance_model_params={"n_estimators": 1},  # Just to make the test faster
        treatment_model_params={"n_estimators": 1},
        propensity_model_params={"n_estimators": 1},
    )
    ml.fit(X=covariates, y=observed_outcomes, w=treatment)
    categorical_columns = [
        covariates.columns.get_loc(col)
        for col in covariates.select_dtypes(include="category").columns
    ]
    if implementation == SLearner:
        # We need to add the treatment columns as LGBM supports categoricals
        categorical_columns.append(len(covariates.columns))
    for cf_estimator_list in chain(
        ml._nuisance_models.values(), ml._treatment_models.values()
    ):
        for cf_estimator in cf_estimator_list:
            assert (
                categorical_columns
                == cf_estimator._estimators[0]._Booster.params["categorical_column"]
            )
            if cf_estimator.enable_overall:
                assert (
                    categorical_columns
                    == cf_estimator._overall_estimator._Booster.params[
                        "categorical_column"
                    ]
                )


@pytest.mark.parametrize(
    "implementation", [_TestMetaLearner, TLearner, SLearner, XLearner, RLearner]
)
def test_metalearner_missing_data_smoke(
    mixed_experiment_dataset_continuous_outcome_binary_treatment_linear_te,
    implementation,
    rng,
):
    covariates, _, treatment, observed_outcomes, _, _ = (
        mixed_experiment_dataset_continuous_outcome_binary_treatment_linear_te
    )

    covariates_with_missing = insert_missing(
        covariates, missing_probability=0.25, rng=rng
    )
    ml = implementation(
        nuisance_model_factory=LGBMRegressor,
        is_classification=False,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        nuisance_model_params={"n_estimators": 1},  # Just to make the test faster
        treatment_model_params={"n_estimators": 1},
        propensity_model_params={"n_estimators": 1},
    )
    ml.fit(X=covariates_with_missing, y=observed_outcomes, w=treatment)


@pytest.mark.parametrize(
    "implementation", [_TestMetaLearner, TLearner, SLearner, XLearner, RLearner]
)
def test_metalearner_missing_data_error(
    numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te,
    implementation,
    rng,
):
    covariates, _, treatment, observed_outcomes, _, _ = (
        numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te
    )
    covariates_with_missing = insert_missing(
        covariates, missing_probability=0.25, rng=rng
    )
    ml = implementation(
        nuisance_model_factory=LinearRegression,
        is_classification=False,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        nuisance_model_params=None,
        treatment_model_params={"n_estimators": 1},  # Just to make the test faster
        propensity_model_params={"n_estimators": 1},
    )
    with pytest.raises(
        ValueError, match=r"LinearRegression does not accept missing values*"
    ):
        ml.fit(X=covariates_with_missing, y=observed_outcomes, w=treatment)


@pytest.mark.parametrize(
    "implementation", [_TestMetaLearner, TLearner, SLearner, XLearner, RLearner]
)
def test_metalearner_format_consistent(
    numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te,
    implementation,
):
    covariates, _, treatment, observed_outcomes, _, _ = (
        numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te
    )

    np_ml = implementation(
        nuisance_model_factory=LGBMRegressor,
        is_classification=False,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        nuisance_model_params={"n_estimators": 1},  # Just to make the test faster
        treatment_model_params={"n_estimators": 1},
        propensity_model_params={"n_estimators": 1},
    )
    pd_ml = implementation(
        nuisance_model_factory=LGBMRegressor,
        is_classification=False,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        nuisance_model_params={"n_estimators": 1},  # Just to make the test faster
        treatment_model_params={"n_estimators": 1},
        propensity_model_params={"n_estimators": 1},
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


@pytest.mark.parametrize(
    "implementation", [_TestMetaLearner, TLearner, SLearner, XLearner, RLearner]
)
def test_metalearner_model_names(implementation):
    set1 = set(implementation.nuisance_model_specifications().keys())
    set2 = set(implementation.treatment_model_specifications().keys())
    assert len(set1 | set2) == len(set1) + len(set2)


@pytest.mark.parametrize(
    "propensity_specs, nuisance_specs, nuisance_model_names, expected",
    [
        (
            None,
            LGBMRegressor,
            {"nuisance1", "nuisance2"},
            {"nuisance1": LGBMRegressor, "nuisance2": LGBMRegressor},
        ),
        (
            LGBMClassifier,
            LGBMRegressor,
            {"nuisance1", "nuisance2"},
            {"nuisance1": LGBMRegressor, "nuisance2": LGBMRegressor},
        ),
        (
            LGBMClassifier,
            LGBMRegressor,
            {"nuisance1", "nuisance2", "propensity_model"},
            {
                "nuisance1": LGBMRegressor,
                "nuisance2": LGBMRegressor,
                "propensity_model": LGBMClassifier,
            },
        ),
        (
            LGBMClassifier,
            {"nuisance1": LGBMRegressor, "nuisance2": LGBMClassifier},
            {"nuisance1", "nuisance2", "propensity_model"},
            {
                "nuisance1": LGBMRegressor,
                "nuisance2": LGBMClassifier,
                "propensity_model": LGBMClassifier,
            },
        ),
    ],
)
def test_combine_propensity_and_nuisance_specs(
    propensity_specs, nuisance_specs, nuisance_model_names, expected
):
    assert (
        _combine_propensity_and_nuisance_specs(
            propensity_specs, nuisance_specs, nuisance_model_names
        )
        == expected
    )
