# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping, Sequence
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.sparse import csr_matrix
from shap import TreeExplainer, summary_plot
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from metalearners._typing import _ScikitModel
from metalearners.cross_fit_estimator import CrossFitEstimator
from metalearners.data_generation import insert_missing
from metalearners.drlearner import DRLearner
from metalearners.metalearner import (
    NUISANCE,
    PROPENSITY_MODEL,
    TREATMENT,
    TREATMENT_MODEL,
    VARIANT_OUTCOME_MODEL,
    MetaLearner,
    _combine_propensity_and_nuisance_specs,
    _ModelSpecifications,
    _parse_fit_params,
    _validate_n_folds_synchronize,
)
from metalearners.rlearner import _SAMPLE_WEIGHT, OUTCOME_MODEL, RLearner
from metalearners.slearner import _BASE_MODEL, SLearner
from metalearners.tlearner import TLearner
from metalearners.xlearner import CONTROL_EFFECT_MODEL, TREATMENT_EFFECT_MODEL, XLearner

_SEED = 1337


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

    def fit_all_nuisance(
        self,
        X,
        y,
        w,
        n_jobs_cross_fitting: int | None = None,
        fit_params: dict | None = None,
        synchronize_cross_fitting: bool = True,
        n_jobs_base_learners: int | None = None,
    ):
        for model_kind in self.__class__.nuisance_model_specifications():
            for model_ord in range(
                self.nuisance_model_specifications()[model_kind]["cardinality"](self)
            ):
                self.fit_nuisance(X, y, model_kind, model_ord)
        return self

    def fit_all_treatment(
        self,
        X,
        y,
        w,
        n_jobs_cross_fitting: int | None = None,
        fit_params: dict | None = None,
        synchronize_cross_fitting: bool = True,
        n_jobs_base_learners: int | None = None,
    ):
        for model_kind in self.__class__.treatment_model_specifications():
            for model_ord in range(
                self.treatment_model_specifications()[model_kind]["cardinality"](self)
            ):
                self.fit_treatment(X, y, model_kind, model_ord)
        return self

    def predict(self, X, is_oos, oos_method=None):
        return np.zeros((len(X), self.n_variants - 1, 1))

    def evaluate(self, X, y, w, is_oos, oos_method=None, scoring=None):
        return {}

    def predict_conditional_average_outcomes(self, X, is_oos, oos_method=None):
        return np.zeros((len(X), 2, 1))

    def _build_onnx(self, models: Mapping[str, Sequence], output_name: str = "tau"): ...

    @classmethod
    def _necessary_onnx_models(cls) -> dict[str, list[_ScikitModel]]:
        return {}


@pytest.mark.parametrize("is_classification", [True, False])
@pytest.mark.parametrize("nuisance_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize("treatment_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize(
    "feature_set",
    [
        None,
        {
            VARIANT_OUTCOME_MODEL: ["X1"],
            CONTROL_EFFECT_MODEL: ["X2"],
            TREATMENT_EFFECT_MODEL: ["X1"],
            TREATMENT_MODEL: ["X2"],
            PROPENSITY_MODEL: ["Xp"],
            OUTCOME_MODEL: ["X1"],
            _BASE_MODEL: ["X2"],
        },
    ],
)
@pytest.mark.parametrize(
    "n_folds",
    [
        5,
        {
            VARIANT_OUTCOME_MODEL: 5,
            CONTROL_EFFECT_MODEL: 5,
            TREATMENT_EFFECT_MODEL: 5,
            TREATMENT_MODEL: 5,
            PROPENSITY_MODEL: 5,
            OUTCOME_MODEL: 5,
            _BASE_MODEL: 5,
        },
    ],
)
@pytest.mark.parametrize("propensity_model_params", [None, {}, {"n_estimators": 5}])
@pytest.mark.parametrize("n_variants", [2, 5, 10])
@pytest.mark.parametrize(
    "implementation",
    [TLearner, SLearner, XLearner, RLearner, DRLearner],
)
def test_metalearner_init(
    is_classification,
    n_variants,
    nuisance_model_params,
    treatment_model_params,
    propensity_model_params,
    feature_set,
    n_folds,
    implementation,
):
    propensity_model_factory = LGBMClassifier
    nuisance_model_factory = LGBMClassifier if is_classification else LGBMRegressor
    treatment_model_factory = LGBMRegressor
    model = implementation(
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
    all_base_models = set(model.nuisance_model_specifications().keys()) | set(
        model.treatment_model_specifications().keys()
    )
    assert set(model.n_folds.keys()) == all_base_models
    assert all(isinstance(n_fold, int) for n_fold in model.n_folds.values())
    assert set(model.feature_set.keys()) == all_base_models


@pytest.mark.parametrize(
    "implementation",
    [TLearner, SLearner, XLearner, RLearner, DRLearner],
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
    "implementation",
    [TLearner, SLearner, XLearner, RLearner, DRLearner],
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
    "implementation",
    [TLearner, SLearner, XLearner, RLearner, DRLearner],
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
    "implementation",
    [TLearner, SLearner, XLearner, RLearner, DRLearner],
)
def test_metalearner_format_consistent(
    numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te,
    implementation,
):
    covariates, _, treatment, observed_outcomes, _, _ = (
        numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te
    )
    metalearner_params = {
        "nuisance_model_factory": LGBMRegressor,
        "is_classification": False,
        "n_variants": len(np.unique(treatment)),
        "treatment_model_factory": LGBMRegressor,
        "propensity_model_factory": LGBMClassifier,
        "nuisance_model_params": {
            "n_estimators": 5,
            "seed": _SEED,
        },  # Just to make the test faster
        "treatment_model_params": {
            "n_estimators": 5,
            "seed": _SEED,
        },
        "propensity_model_params": {
            "n_estimators": 5,
            "seed": _SEED,
        },
        "random_state": _SEED,
    }

    np_ml = implementation(**metalearner_params)
    pd_ml = implementation(**metalearner_params)

    np_ml.fit(X=covariates, y=observed_outcomes, w=treatment)
    X_pd = pd.DataFrame(covariates)
    y_pd = pd.Series(observed_outcomes)
    w_pd = pd.Series(treatment)
    pd_ml.fit(X_pd, y_pd, w_pd)

    np_cate_estimates = np_ml.predict(covariates, is_oos=False)
    pd_cate_estimates = pd_ml.predict(X_pd, is_oos=False)
    np.testing.assert_allclose(np_cate_estimates, pd_cate_estimates)

    if implementation != RLearner and implementation != DRLearner:
        # For the reindexed we can only compare out-of-sample predictions with oos_method="overall"
        # as the folds are not the same.
        # Also RLearner and DRLearner can't be tested as in the _pseudo_outcome method
        # they use the folds for the in-sample predictions.
        pd_reindexed_ml = implementation(**metalearner_params)
        X_reindexed = X_pd.sample(frac=1)
        y_reindexed = y_pd[X_reindexed.index]
        w_reindexed = w_pd[X_reindexed.index]
        pd_reindexed_ml.fit(X=X_reindexed, y=y_reindexed, w=w_reindexed)

        X_test = X_pd.sample(frac=0.3)
        pd_reindexed_cate_estimates_test = pd_reindexed_ml.predict(
            X_test, is_oos=True, oos_method="overall"
        )
        pd_cate_estimates_test = pd_ml.predict(
            X_test, is_oos=True, oos_method="overall"
        )
        np.testing.assert_allclose(
            pd_reindexed_cate_estimates_test, pd_cate_estimates_test
        )


@pytest.mark.parametrize(
    "n_folds", [5, {"nuisance1": 1, "nuisance2": 1, "treatment1": 5, "treatment2": 10}]
)
def test_n_folds(n_folds):
    ml = _TestMetaLearner(
        nuisance_model_factory=LinearRegression,
        is_classification=False,
        n_variants=2,
        treatment_model_factory=LinearRegression,
        n_folds=n_folds,
    )
    for model_kind, cf_estimator_list in chain(
        ml._nuisance_models.items(), ml._treatment_models.items()
    ):
        expected_folds = n_folds[model_kind] if isinstance(n_folds, dict) else n_folds
        for cf_estimator in cf_estimator_list:
            assert cf_estimator.n_folds == expected_folds


@pytest.mark.parametrize(
    "implementation",
    [TLearner, SLearner, XLearner, RLearner, DRLearner],
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


@pytest.mark.parametrize(
    "feature_set, expected_n_features",
    [
        (
            None,
            {
                "nuisance1": None,
                "nuisance2": None,
                "treatment1": None,
                "treatment2": None,
            },
        ),
        (
            [0, 1, 2],
            {
                "nuisance1": 3,
                "nuisance2": 3,
                "treatment1": 3,
                "treatment2": 3,
            },
        ),
        (
            {
                "nuisance1": [],
                "nuisance2": None,
                "treatment1": [0, 1],
                "treatment2": [2, 3, 4],
            },
            {
                "nuisance1": 1,
                "nuisance2": None,
                "treatment1": 2,
                "treatment2": 3,
            },
        ),
    ],
)
@pytest.mark.parametrize("backend", ["np", "pd", "csr"])
def test_feature_set(feature_set, expected_n_features, backend, rng):
    ml = _TestMetaLearner(
        nuisance_model_factory=LGBMRegressor,
        is_classification=False,
        n_variants=2,
        treatment_model_factory=LGBMRegressor,
        feature_set=feature_set,
        n_folds=2,
    )
    sample_size = 100
    n_features = 10
    X = rng.standard_normal((sample_size, n_features))
    y = rng.standard_normal(sample_size)
    w = rng.integers(0, 2, sample_size)
    if backend == "pd":
        X = pd.DataFrame(X)
        y = pd.Series(y)
        w = pd.Series(w)
    elif backend == "csr":
        X = csr_matrix(X)
    ml.fit(X, y, w)

    for model_kind, model_kind_list in ml._nuisance_models.items():
        exp = (
            expected_n_features[model_kind]
            if expected_n_features[model_kind]
            else n_features
        )
        for m in model_kind_list:
            assert m._overall_estimator.n_features_ == exp  # type: ignore


def test_model_reusage_init():
    # TODO: Split up into several tests.
    prefitted_models = [CrossFitEstimator(10, LGBMRegressor)]
    ml = _TestMetaLearner(
        nuisance_model_factory=LinearRegression,
        is_classification=False,
        n_variants=2,
        treatment_model_factory=LGBMRegressor,
        fitted_nuisance_models={"nuisance1": prefitted_models},
    )
    assert ml._nuisance_models["nuisance1"][0].estimator_factory == LGBMRegressor
    assert ml._nuisance_models["nuisance2"][0].estimator_factory == LinearRegression
    with pytest.raises(ValueError, match="A model for the nuisance model nuisance2"):
        _TestMetaLearner(
            is_classification=False,
            n_variants=2,
            treatment_model_factory=LGBMRegressor,
            fitted_nuisance_models={"nuisance1": prefitted_models},
        )

    with pytest.raises(ValueError, match="The keys present"):
        _TestMetaLearner(
            nuisance_model_factory=LGBMRegressor,
            is_classification=False,
            n_variants=2,
            treatment_model_factory=LGBMRegressor,
            fitted_nuisance_models={"nuisance3": prefitted_models},
        )

    with pytest.raises(ValueError, match="The keys present"):
        RLearner(
            propensity_model_factory=LGBMClassifier,
            nuisance_model_factory=LGBMRegressor,
            is_classification=False,
            n_variants=2,
            treatment_model_factory=LGBMRegressor,
            fitted_nuisance_models={PROPENSITY_MODEL: prefitted_models},
        )


@pytest.mark.parametrize(
    "fit_params, nuisance_model_names, treatment_model_names, expected",
    [
        (
            None,
            {"n1", "n2"},
            {"t1"},
            {NUISANCE: {"n1": dict(), "n2": dict()}, TREATMENT: {"t1": dict()}},
        ),
        (
            dict(),
            {"n1", "n2"},
            {"t1"},
            {NUISANCE: {"n1": dict(), "n2": dict()}, TREATMENT: {"t1": dict()}},
        ),
        (
            {"arg": 0},
            {"n1", "n2"},
            {"t1"},
            {
                NUISANCE: {"n1": {"arg": 0}, "n2": {"arg": 0}},
                TREATMENT: {"t1": {"arg": 0}},
            },
        ),
        (
            {"arg": 0},
            {"n1", "n2"},
            set(),
            {NUISANCE: {"n1": {"arg": 0}, "n2": {"arg": 0}}, TREATMENT: dict()},
        ),
        (
            {NUISANCE: {"n2": {"arg": 0, "arg2": 1}}},
            {"n1", "n2"},
            {"t1"},
            {
                NUISANCE: {"n1": dict(), "n2": {"arg": 0, "arg2": 1}},
                TREATMENT: {"t1": dict()},
            },
        ),
        (
            {TREATMENT: {"t1": {"arg": 0, "arg2": 1}}},
            {"n1", "n2"},
            {"t1"},
            {
                NUISANCE: {"n1": dict(), "n2": dict()},
                TREATMENT: {"t1": {"arg": 0, "arg2": 1}},
            },
        ),
    ],
)
def test_parse_fit_params(
    fit_params, nuisance_model_names, treatment_model_names, expected
):
    actual = _parse_fit_params(fit_params, nuisance_model_names, treatment_model_names)
    assert actual == expected


class ParamEstimator(BaseEstimator):
    def __init__(self, expected_fit_params):
        self.expected_fit_params = set(expected_fit_params)
        self._estimator_type = "Neutral"

    def fit(self, X, y, **fit_params):
        if set(fit_params.keys()) - {"sample_weight"} != self.expected_fit_params:
            raise ValueError()
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


class ParamEstimatorFactory:
    def __init__(self, expected_fit_params):
        self.expected_fit_params = expected_fit_params
        self._estimator_type = "Neutral"

    def __call__(self):
        return ParamEstimator(self.expected_fit_params)

    def fit(self, X, y, sample_weight):
        return self


_PROPENSITY = "propensity"


@pytest.mark.parametrize(
    "metalearner_factory, fit_params, expected_keys",
    [
        (
            SLearner,
            {"arg": 0},
            {
                NUISANCE: {_BASE_MODEL: {"arg": 0}},
                _PROPENSITY: dict(),
                TREATMENT: dict(),
            },
        ),
        (
            SLearner,
            {"arg": 0},
            {
                NUISANCE: {_BASE_MODEL: {"arg": 0}},
                _PROPENSITY: dict(),
                TREATMENT: dict(),
            },
        ),
        (
            TLearner,
            {"arg": 0},
            {
                NUISANCE: {VARIANT_OUTCOME_MODEL: {"arg": 0}},
                _PROPENSITY: dict(),
                TREATMENT: dict(),
            },
        ),
        (
            XLearner,
            {"arg": 0},
            {
                NUISANCE: {VARIANT_OUTCOME_MODEL: {"arg": 0}},
                _PROPENSITY: {PROPENSITY_MODEL: {"arg": 0}},
                TREATMENT: {
                    CONTROL_EFFECT_MODEL: {"arg": 0},
                    TREATMENT_EFFECT_MODEL: {"arg": 0},
                },
            },
        ),
        (
            RLearner,
            {"arg": 0},
            {
                NUISANCE: {OUTCOME_MODEL: {"arg": 0}},
                _PROPENSITY: {PROPENSITY_MODEL: {"arg": 0}},
                TREATMENT: {TREATMENT_MODEL: {"arg": 0}},
            },
        ),
        (
            DRLearner,
            {"arg": 0},
            {
                NUISANCE: {VARIANT_OUTCOME_MODEL: {"arg": 0}},
                _PROPENSITY: {PROPENSITY_MODEL: {"arg": 0}},
                TREATMENT: {TREATMENT_MODEL: {"arg": 0}},
            },
        ),
    ],
)
def test_fit_params(metalearner_factory, fit_params, expected_keys, dummy_dataset):
    X, y, w = dummy_dataset

    propensity_model_factory = (
        ParamEstimatorFactory(
            expected_fit_params=expected_keys[_PROPENSITY][PROPENSITY_MODEL]
        )
        if _PROPENSITY in expected_keys
        and PROPENSITY_MODEL in expected_keys[_PROPENSITY]
        else None
    )
    metalearner = metalearner_factory(
        nuisance_model_factory={
            model_kind: ParamEstimatorFactory(expected_fit_params=params)
            for model_kind, params in expected_keys[NUISANCE].items()
        },
        propensity_model_factory=propensity_model_factory,
        treatment_model_factory={
            model_kind: ParamEstimatorFactory(expected_fit_params=params)
            for model_kind, params in expected_keys[TREATMENT].items()
        },
        n_variants=2,
        is_classification=False,
        n_folds=1,
    )
    # Using cross-fitting is not possible with a single fold.
    metalearner.fit(
        X=X, y=y, w=w, fit_params=fit_params, synchronize_cross_fitting=False
    )


def test_fit_params_rlearner_error(dummy_dataset):
    X, y, w = dummy_dataset

    rlearner = RLearner(
        nuisance_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        treatment_model_factory=LinearRegression,
        n_variants=2,
        is_classification=False,
    )
    with pytest.raises(ValueError, match=f"The parameter {_SAMPLE_WEIGHT}"):
        rlearner.fit(
            X=X,
            y=y,
            w=w,
            fit_params={_SAMPLE_WEIGHT: np.ones(len(X))},
        )


@pytest.mark.parametrize(
    "implementation, needs_estimates",
    [
        (TLearner, True),
        (SLearner, True),
        (XLearner, True),
        (RLearner, False),
        (DRLearner, False),
    ],
)
@pytest.mark.parametrize("n_variants", [2, 10])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("use_custom_feature_names", [False, True])
@pytest.mark.parametrize("use_explainer", [False, True])
@pytest.mark.parametrize("sort_values", [False, True])
def test_feature_importances_smoke(
    implementation,
    needs_estimates,
    normalize,
    n_variants,
    rng,
    use_custom_feature_names,
    use_explainer,
    sort_values,
):
    sample_size = 1000
    n_features = 10

    X = rng.standard_normal((sample_size, n_features))
    y = rng.standard_normal(sample_size)
    w = rng.integers(0, n_variants, sample_size)

    ml = implementation(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LogisticRegression,
        treatment_model_params={"n_estimators": 1},
    )

    ml.fit(X=X, y=y, w=w)
    cate_estimates = ml.predict(X=X, is_oos=False)

    if use_custom_feature_names:
        feature_names = [f"x_{i}" for i in range(n_features)]
        expected_feature_names = feature_names
    else:
        feature_names = None
        expected_feature_names = [f"Feature {i}" for i in range(n_features)]
    if use_explainer:
        explainer = ml.explainer(
            X=X,
            cate_estimates=cate_estimates,
            cate_model_factory=LGBMRegressor,
            cate_model_params={"n_estimators": 1},
        )
        feature_importances = ml.feature_importances(
            normalize=normalize,
            feature_names=feature_names,
            explainer=explainer,
            sort_values=sort_values,
        )
    else:
        feature_importances = ml.feature_importances(
            normalize=normalize,
            feature_names=feature_names,
            sort_values=sort_values,
            X=X,
            cate_estimates=cate_estimates,
            cate_model_factory=LGBMRegressor,
            cate_model_params={"n_estimators": 1},
        )
    assert len(feature_importances) == n_variants - 1
    for tv in range(n_variants - 1):
        assert len(feature_importances[tv]) == n_features
        # the nan check is for _TestMetaLearner which returns a 0 importance for all
        # and therefore when normalizing returns nan.
        if normalize and not pd.isna(feature_importances[tv]).all():
            assert np.sum(feature_importances[tv]) == pytest.approx(1)
            if sort_values:
                assert feature_importances[tv].is_monotonic_decreasing
                assert set(feature_importances[tv].index) == set(expected_feature_names)
            else:
                assert (feature_importances[tv].index == expected_feature_names).all()

    if not needs_estimates:
        if use_explainer:
            explainer = ml.explainer()
            feature_importances = ml.feature_importances(
                normalize=normalize,
                feature_names=feature_names,
                explainer=explainer,
                sort_values=sort_values,
            )
        else:
            feature_importances = ml.feature_importances(
                normalize=normalize,
                feature_names=feature_names,
                sort_values=sort_values,
            )
        assert len(feature_importances) == n_variants - 1
        for tv in range(n_variants - 1):
            assert len(feature_importances[tv]) == n_features
            if normalize and not pd.isna(feature_importances[tv]).all():
                assert np.sum(feature_importances[tv]) == pytest.approx(1)
            if sort_values:
                assert feature_importances[tv].is_monotonic_decreasing
                assert set(feature_importances[tv].index) == set(expected_feature_names)
            else:
                assert (feature_importances[tv].index == expected_feature_names).all()


@pytest.mark.parametrize(
    "implementation, needs_estimates",
    [
        (TLearner, True),
        (XLearner, True),
        # (RLearner, False),
        (DRLearner, False),
    ],
)
def test_feature_importances_known(
    implementation, needs_estimates, feature_importance_dataset
):
    """The SLearner can not represent properly this CATE with LinearRegression as there
    are no interactions.

    The RLearner does not learn as good CATEs as the other metalearners because there is
    only one outcome model. It may be interesting to test and see if other parameters
    can help on passing this test for the S and R learners.
    """
    X, y, w = feature_importance_dataset
    n_variants = len(np.unique(w))

    ml = implementation(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LogisticRegression,
    )
    ml.fit(X=X, y=y, w=w)
    cate_estimates = ml.predict(X=X, is_oos=False)
    explainer = ml.explainer(
        X=X,
        cate_estimates=cate_estimates,
        cate_model_factory=LGBMRegressor,
    )
    feature_importances = ml.feature_importances(
        feature_names=X.columns, explainer=explainer
    )

    assert feature_importances[0].idxmax() == "x1"
    assert feature_importances[1].idxmax() == "x2"

    if not needs_estimates:
        feature_importances = ml.feature_importances(
            feature_names=X.columns,
        )
        assert feature_importances[0].idxmax() == "x1"
        assert feature_importances[1].idxmax() == "x2"


@pytest.mark.parametrize(
    "implementation, needs_estimates",
    [
        (TLearner, True),
        (SLearner, True),
        (XLearner, True),
        (RLearner, False),
        (DRLearner, False),
    ],
)
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("use_explainer", [False, True])
def test_shap_values_smoke(
    implementation,
    needs_estimates,
    n_variants,
    use_explainer,
    rng,
):
    sample_size = 1000
    n_features = 10

    X = rng.standard_normal((sample_size, n_features))
    y = rng.standard_normal(sample_size)
    w = rng.integers(0, n_variants, sample_size)

    ml = implementation(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LogisticRegression,
        treatment_model_params={"n_estimators": 1},
    )

    ml.fit(X=X, y=y, w=w)
    cate_estimates = ml.predict(X=X, is_oos=False)

    if use_explainer:
        explainer = ml.explainer(
            X=X,
            cate_estimates=cate_estimates,
            cate_model_factory=LGBMRegressor,
            cate_model_params={"n_estimators": 1},
        )
        shap_values = ml.shap_values(X, TreeExplainer, explainer=explainer)
    else:
        shap_values = ml.shap_values(
            X,
            TreeExplainer,
            cate_estimates=cate_estimates,
            cate_model_factory=LGBMRegressor,
            cate_model_params={"n_estimators": 1},
        )
    assert len(shap_values) == n_variants - 1
    for tv in range(n_variants - 1):
        assert shap_values[tv].shape == (sample_size, n_features)
        summary_plot(shap_values[tv], show=False, features=X)
        plt.clf()

    if not needs_estimates:
        if use_explainer:
            explainer = ml.explainer()
            shap_values = ml.shap_values(X, TreeExplainer, explainer=explainer)
        else:
            shap_values = ml.shap_values(X, TreeExplainer)
        assert len(shap_values) == n_variants - 1
        for tv in range(n_variants - 1):
            assert shap_values[tv].shape == (sample_size, n_features)
            summary_plot(shap_values[tv], show=False, features=X)
            plt.clf()


@pytest.mark.parametrize(
    "implementation",
    [
        TLearner,
        SLearner,
        XLearner,
        RLearner,
        DRLearner,
    ],
)
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("synchronize_cross_fitting", [False, True])
def test_synchronization_smoke(
    implementation, n_variants, synchronize_cross_fitting, rng
):
    sample_size = 100
    n_features = 2

    X = rng.standard_normal((sample_size, n_features))
    y = rng.standard_normal(sample_size)
    w = rng.integers(0, n_variants, sample_size)

    ml = implementation(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
    )
    ml.fit(X=X, y=y, w=w, synchronize_cross_fitting=synchronize_cross_fitting)


@pytest.mark.parametrize(
    "n_folds,success",
    [
        ({"propensity": 5, "outcome": 2}, False),
        ({"propensity": 5, "outcome": 5}, True),
        ({"propensity": 1, "outcome": 2}, False),
        ({"propensity": 1, "outcome": 1}, False),
    ],
)
def test_validate_n_folds_synchronize(n_folds, success):
    if success:
        _validate_n_folds_synchronize(n_folds)
    else:
        with pytest.raises(ValueError, match="synchronization"):
            _validate_n_folds_synchronize(n_folds)


@pytest.mark.parametrize(
    "implementation",
    [TLearner, XLearner, RLearner, DRLearner],
)
def test_n_jobs_base_learners(implementation, rng):
    n_variants = 5
    X = rng.standard_normal((1000, 10))
    y = rng.standard_normal(1000)
    w = rng.integers(0, n_variants, 1000)

    ml = implementation(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        random_state=_SEED,
    )

    ml.fit(X, y, w, n_jobs_base_learners=None)

    ml_2 = implementation(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        random_state=_SEED,
    )

    ml_2.fit(X, y, w, n_jobs_base_learners=-1)

    np.testing.assert_allclose(ml.predict(X, False), ml_2.predict(X, False))
    np.testing.assert_allclose(ml.predict(X, True), ml_2.predict(X, True))


@pytest.mark.parametrize(
    "implementation",
    [TLearner, SLearner, XLearner, RLearner, DRLearner],
)
@pytest.mark.parametrize("backend", ["np", "pd", "csr"])
def test_validate_outcome_one_class(implementation, backend, rng):
    X = rng.standard_normal((10, 2))
    y = np.zeros(10)
    w = rng.integers(0, 2, 10)
    if backend == "pandas":
        X = pd.DataFrame(X)
        y = pd.Series(y)
        w = pd.Series(w)
    elif backend == "csr":
        X = csr_matrix(X)

    ml = implementation(
        True,
        2,
        LogisticRegression,
        LinearRegression,
        LogisticRegression,
    )
    with pytest.raises(
        ValueError,
        match="There is only one class present in the classification outcome",
    ):
        ml.fit(X, y, w)


@pytest.mark.parametrize(
    "implementation",
    [TLearner, SLearner, XLearner, RLearner, DRLearner],
)
@pytest.mark.parametrize("backend", ["np", "pd", "csr"])
def test_validate_outcome_different_classes(implementation, backend, rng):
    X = rng.standard_normal((4, 2))
    y = np.array([0, 1, 0, 0])
    w = np.array([0, 0, 1, 1])
    if backend == "pd":
        X = pd.DataFrame(X)
        y = pd.Series(y)
        w = pd.Series(w)
    elif backend == "csr":
        X = csr_matrix(X)

    ml = implementation(
        True,
        2,
        LogisticRegression,
        LinearRegression,
        LogisticRegression,
    )
    with pytest.raises(
        ValueError, match="have seen different sets of classification outcomes."
    ):
        ml.fit(X, y, w)


@pytest.mark.parametrize(
    "implementation",
    [TLearner, SLearner, XLearner, RLearner, DRLearner],
)
def test_init_args(implementation):
    ml = implementation(
        True,
        2,
        LogisticRegression,
        LinearRegression,
        LogisticRegression,
    )
    ml2 = implementation(**ml.init_args)

    assert set(ml.__dict__.keys()) == set(ml2.__dict__.keys())
    for key in ml.__dict__:
        assert ml.__dict__[key] == ml2.__dict__[key]
