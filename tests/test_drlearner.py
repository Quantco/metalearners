# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from itertools import repeat
from typing import TypedDict

import numpy as np
import onnxruntime as rt
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from onnx import ModelProto
from onnxmltools import convert_lightgbm, convert_xgboost
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import RadiusNeighborsRegressor
from xgboost import XGBRegressor

from metalearners import DRLearner
from metalearners._typing import OosMethod, Params
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import (
    PROPENSITY_MODEL,
    TREATMENT_MODEL,
    VARIANT_OUTCOME_MODEL,
)

from .conftest import all_sklearn_regressors


def _multi_variant_dataset(n_obs=120, n_variants=4):
    sample_index = np.arange(n_obs)
    X = np.column_stack(
        [
            sample_index / n_obs,
            (sample_index % 5) / 5,
            np.sin(sample_index / 7),
        ]
    )
    w = sample_index % n_variants
    y = X[:, 0] - 0.5 * X[:, 1] + 0.25 * w
    return X, y, w


class _NuisancePredictionCallArgs(TypedDict):
    is_oos: bool | None
    oos_method: OosMethod


def _spy_on_drlearner_nuisance_predictions(monkeypatch, learner):
    nuisance_estimate_calls = {
        "conditional_average_outcome_estimates": 0,
        PROPENSITY_MODEL: 0,
    }
    nuisance_estimate_call_args: dict[str, list[_NuisancePredictionCallArgs]] = {
        "conditional_average_outcome_estimates": [],
        PROPENSITY_MODEL: [],
    }
    original_predict_conditional_average_outcomes = (
        learner.predict_conditional_average_outcomes
    )
    original_predict_nuisance = learner.predict_nuisance

    def predict_conditional_average_outcomes_spy(*args, **kwargs):
        nuisance_estimate_calls["conditional_average_outcome_estimates"] += 1
        nuisance_estimate_call_args["conditional_average_outcome_estimates"].append(
            {
                "is_oos": kwargs.get("is_oos", args[1] if len(args) > 1 else None),
                "oos_method": kwargs.get(
                    "oos_method", args[2] if len(args) > 2 else OVERALL
                ),
            }
        )
        return original_predict_conditional_average_outcomes(*args, **kwargs)

    def predict_nuisance_spy(*args, **kwargs):
        model_kind = kwargs.get("model_kind", args[1] if len(args) > 1 else None)
        if model_kind == PROPENSITY_MODEL:
            nuisance_estimate_calls[PROPENSITY_MODEL] += 1
            nuisance_estimate_call_args[PROPENSITY_MODEL].append(
                {
                    "is_oos": kwargs.get("is_oos", args[3] if len(args) > 3 else None),
                    "oos_method": kwargs.get(
                        "oos_method", args[4] if len(args) > 4 else OVERALL
                    ),
                }
            )
        return original_predict_nuisance(*args, **kwargs)

    monkeypatch.setattr(
        learner,
        "predict_conditional_average_outcomes",
        predict_conditional_average_outcomes_spy,
    )
    monkeypatch.setattr(learner, "predict_nuisance", predict_nuisance_spy)
    return nuisance_estimate_calls, nuisance_estimate_call_args


def test_adaptive_clipping_smoke(dummy_dataset):
    X, y, w = dummy_dataset
    ml = DRLearner(
        False,
        2,
        LinearRegression,
        LinearRegression,
        LogisticRegression,
        n_folds=2,
        adaptive_clipping=True,
    )
    ml.fit(X, y, w)


def test_drlearner_reuses_nuisance_estimates_across_treatment_variants(monkeypatch):
    n_variants = 4
    X, y, w = _multi_variant_dataset(n_variants=n_variants)

    learner = DRLearner(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        propensity_model_params={"max_iter": 1000},
        n_folds=2,
        random_state=0,
    )
    learner.fit_all_nuisance(X, y, w)

    nuisance_estimate_calls, _ = _spy_on_drlearner_nuisance_predictions(
        monkeypatch, learner
    )

    learner.fit_all_treatment(X, y, w)

    assert nuisance_estimate_calls == {
        "conditional_average_outcome_estimates": 1,
        PROPENSITY_MODEL: 1,
    }


def test_drlearner_evaluate_reuses_nuisance_estimates_across_treatment_variants(
    monkeypatch,
):
    n_variants = 4
    X, y, w = _multi_variant_dataset(n_variants=n_variants)

    learner = DRLearner(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        propensity_model_params={"max_iter": 1000},
        n_folds=2,
        random_state=0,
    )
    learner.fit(X, y, w)

    nuisance_estimate_calls, nuisance_estimate_call_args = (
        _spy_on_drlearner_nuisance_predictions(monkeypatch, learner)
    )

    learner.evaluate(
        X,
        y,
        w,
        is_oos=True,
        oos_method="mean",
        scoring={
            PROPENSITY_MODEL: [],
            VARIANT_OUTCOME_MODEL: [],
            TREATMENT_MODEL: [],
        },
    )

    assert nuisance_estimate_calls == {
        "conditional_average_outcome_estimates": 1,
        PROPENSITY_MODEL: 1,
    }
    assert nuisance_estimate_call_args == {
        "conditional_average_outcome_estimates": [
            {"is_oos": True, "oos_method": "mean"}
        ],
        PROPENSITY_MODEL: [{"is_oos": True, "oos_method": "mean"}],
    }


def test_drlearner_average_treatment_effect_reuses_nuisance_estimates_across_treatment_variants(
    monkeypatch,
):
    n_variants = 4
    X, y, w = _multi_variant_dataset(n_variants=n_variants)

    learner = DRLearner(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        propensity_model_params={"max_iter": 1000},
        n_folds=2,
        random_state=0,
    )
    learner.fit_all_nuisance(X, y, w)

    nuisance_estimate_calls, nuisance_estimate_call_args = (
        _spy_on_drlearner_nuisance_predictions(monkeypatch, learner)
    )

    learner.average_treatment_effect(X, y, w, is_oos=True)

    assert nuisance_estimate_calls == {
        "conditional_average_outcome_estimates": 1,
        PROPENSITY_MODEL: 1,
    }
    assert nuisance_estimate_call_args == {
        "conditional_average_outcome_estimates": [
            {"is_oos": True, "oos_method": OVERALL}
        ],
        PROPENSITY_MODEL: [{"is_oos": True, "oos_method": OVERALL}],
    }


@pytest.mark.parametrize(
    "treatment_model_factory, onnx_converter",
    (
        list(
            zip(
                all_sklearn_regressors,
                repeat(convert_sklearn),
            )
        )
        + [
            (LGBMRegressor, convert_lightgbm),
            (XGBRegressor, convert_xgboost),
        ]
    ),
)
@pytest.mark.parametrize("is_classification", [True, False])
def test_drlearner_onnx(
    treatment_model_factory, onnx_converter, is_classification, onnx_dataset
):
    supports_categoricals = treatment_model_factory in [
        LGBMRegressor,
        # convert_sklearn does not support categoricals https://github.com/onnx/sklearn-onnx/issues/1051
        # HistGradientBoostingRegressor,
        # convert_xgboost does not support categoricals https://github.com/onnx/onnxmltools/issues/469#issuecomment-1993880910
        # XGBRegressor,
    ]
    treatment_model_params: Params | None = None
    if treatment_model_factory == RadiusNeighborsRegressor:
        treatment_model_params = {"radius": 10}

    X_numerical, X_with_categorical, y_class, y_reg, w = onnx_dataset
    n_numerical_features = X_numerical.shape[1]

    if supports_categoricals:
        X = X_with_categorical
        n_categorical_features = X.shape[1] - n_numerical_features
    else:
        X = X_numerical
        n_categorical_features = 0
    n_variants = len(np.unique(w))
    if is_classification:
        y = y_class
        nuisance_model_factory = LogisticRegression
    else:
        y = y_reg
        nuisance_model_factory = LinearRegression

    ml = DRLearner(
        is_classification,
        n_variants,
        nuisance_model_factory=nuisance_model_factory,
        propensity_model_factory=LGBMClassifier,
        treatment_model_factory=treatment_model_factory,
        propensity_model_params={"n_estimators": 1},
        treatment_model_params=treatment_model_params,
        n_folds=2,
    )
    ml.fit(X, y, w)

    necessary_models = ml._necessary_onnx_models()
    onnx_models: dict[str, list[ModelProto]] = {}

    for model_kind, models in necessary_models.items():
        onnx_models[model_kind] = []
        for model in models:
            onnx_models[model_kind].append(
                onnx_converter(
                    model,
                    initial_types=[
                        (
                            "X",
                            FloatTensorType(
                                [None, n_numerical_features + n_categorical_features]
                            ),
                        )
                    ],
                )
            )

    final = ml._build_onnx(onnx_models)
    sess = rt.InferenceSession(
        final.SerializeToString(), providers=rt.get_available_providers()
    )

    if supports_categoricals:
        onnx_X = X.to_numpy(np.float32)
        # This is needed for categoricals as LGBM uses the categorical codes, when
        # other implementations support categoricals this may need to be changed
        onnx_X[:, n_numerical_features] = X[n_numerical_features].cat.codes
        onnx_X[:, n_numerical_features + 1] = X[n_numerical_features + 1].cat.codes
    else:
        onnx_X = X.astype(np.float32)

    (pred_onnx,) = sess.run(
        ["tau"],
        {"X": onnx_X},
    )
    np.testing.assert_allclose(
        ml.predict(X, True, "overall"), pred_onnx, atol=5e-2, rtol=0.01
    )


def test_average_treatment_effect(
    numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te,
):
    X, _, W, Y, _, tau = (
        numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te
    )
    ml = DRLearner(
        False,
        2,
        LinearRegression,
        LinearRegression,
        LogisticRegression,
        n_folds=2,
    )
    ml.fit_all_nuisance(X, Y, W)
    ate_estimate, _ = ml.average_treatment_effect(X, Y, W, is_oos=False)
    np.testing.assert_almost_equal(ate_estimate, tau.mean(), decimal=1)
