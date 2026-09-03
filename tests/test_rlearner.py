# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from itertools import repeat
from typing import TypedDict

import numpy as np
import onnxruntime as rt
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from onnx import ModelProto
from onnxmltools import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as FloatTensorTypeOMLT
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as FloatTensorTypeSkl
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor

from metalearners._typing import OosMethod
from metalearners._utils import function_has_argument
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import PROPENSITY_MODEL, TREATMENT_MODEL
from metalearners.rlearner import OUTCOME_MODEL, RLearner, r_loss

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


def _spy_on_rlearner_nuisance_predictions(monkeypatch, learner):
    predict_nuisance_calls = {PROPENSITY_MODEL: 0, OUTCOME_MODEL: 0}
    predict_nuisance_call_args: dict[str, list[_NuisancePredictionCallArgs]] = {
        PROPENSITY_MODEL: [],
        OUTCOME_MODEL: [],
    }
    original_predict_nuisance = learner.predict_nuisance

    def predict_nuisance_spy(*args, **kwargs):
        model_kind = kwargs.get("model_kind", args[1] if len(args) > 1 else None)
        if model_kind in predict_nuisance_calls:
            predict_nuisance_calls[model_kind] += 1
            predict_nuisance_call_args[model_kind].append(
                {
                    "is_oos": kwargs.get("is_oos", args[3] if len(args) > 3 else None),
                    "oos_method": kwargs.get(
                        "oos_method", args[4] if len(args) > 4 else OVERALL
                    ),
                }
            )
        return original_predict_nuisance(*args, **kwargs)

    monkeypatch.setattr(learner, "predict_nuisance", predict_nuisance_spy)
    return predict_nuisance_calls, predict_nuisance_call_args


@pytest.mark.parametrize("use_pandas", [True, False])
def test_r_loss(use_pandas):
    factory = pd.Series if use_pandas else np.array
    cate_estimates = factory([2, 2])
    outcomes = factory([6.1, 6.1])
    outcome_estimates = factory([3.1, 3.1])
    treatments = factory([1, 1])
    propensity_scores = factory([0.5, 0.5])
    # (6.1 - 3.1) - 2(1 -.5)
    # = 3 - 1 = 2
    result = r_loss(
        cate_estimates=cate_estimates,
        outcomes=outcomes,
        outcome_estimates=outcome_estimates,
        treatments=treatments,
        propensity_scores=propensity_scores,
    )
    assert result == pytest.approx(2, abs=1e-4, rel=1e-4)


def test_rlearner_reuses_nuisance_estimates_across_treatment_variants(monkeypatch):
    n_variants = 4
    X, y, w = _multi_variant_dataset(n_variants=n_variants)

    learner = RLearner(
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

    predict_nuisance_calls, _ = _spy_on_rlearner_nuisance_predictions(
        monkeypatch, learner
    )

    learner.fit_all_treatment(X, y, w)

    assert predict_nuisance_calls == {PROPENSITY_MODEL: 1, OUTCOME_MODEL: 1}


def test_rlearner_evaluate_reuses_nuisance_estimates_across_treatment_variants(
    monkeypatch,
):
    n_variants = 4
    X, y, w = _multi_variant_dataset(n_variants=n_variants)

    learner = RLearner(
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

    predict_nuisance_calls, predict_nuisance_call_args = (
        _spy_on_rlearner_nuisance_predictions(monkeypatch, learner)
    )

    learner.evaluate(
        X,
        y,
        w,
        is_oos=True,
        oos_method="mean",
        scoring={PROPENSITY_MODEL: [], OUTCOME_MODEL: [], TREATMENT_MODEL: []},
    )

    assert predict_nuisance_calls == {PROPENSITY_MODEL: 1, OUTCOME_MODEL: 1}
    assert predict_nuisance_call_args == {
        PROPENSITY_MODEL: [{"is_oos": True, "oos_method": "mean"}],
        OUTCOME_MODEL: [{"is_oos": True, "oos_method": "mean"}],
    }


def test_rlearner_in_sample_cross_fitting_uses_full_x_nuisance_estimates_before_masking():
    class RecordingRegressor(RegressorMixin, BaseEstimator):
        _estimator_type = "regressor"
        fit_records = []

        def fit(self, X, y, sample_weight=None):
            self.constant_ = float(np.average(y, weights=sample_weight))
            self.fit_records.append(
                {
                    "X": np.asarray(X).copy(),
                    "y": np.asarray(y).copy(),
                    "sample_weight": np.asarray(sample_weight).copy(),
                }
            )
            return self

        def predict(self, X):
            return np.full(len(X), self.constant_)

    n_variants = 4
    X, y, w = _multi_variant_dataset(n_variants=n_variants)

    learner = RLearner(
        is_classification=False,
        n_variants=n_variants,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=RecordingRegressor,
        propensity_model_factory=LogisticRegression,
        propensity_model_params={"max_iter": 1000},
        n_folds=2,
        random_state=0,
    )
    learner.fit_all_nuisance(X, y, w)

    outcome_estimates = learner.predict_nuisance(
        X=X, model_kind=OUTCOME_MODEL, model_ord=0, is_oos=False
    )
    propensity_estimates = learner.predict_nuisance(
        X=X, model_kind=PROPENSITY_MODEL, model_ord=0, is_oos=False
    )

    learner.fit_all_treatment(X, y, w)

    mask_len = np.count_nonzero((w == 0) | (w == 1))
    overall_fit_records = [
        record
        for record in RecordingRegressor.fit_records
        if len(record["y"]) == mask_len
    ]
    assert len(overall_fit_records) == n_variants - 1

    matched_record_indices = set()
    for treatment_variant in range(1, n_variants):
        mask = (w == 0) | (w == treatment_variant)
        expected_y, expected_sample_weight = learner._pseudo_outcome_and_weights(
            y=y,
            w=w,
            treatment_variant=treatment_variant,
            outcome_estimates=outcome_estimates,
            propensity_estimates=propensity_estimates,
            mask=mask,
        )

        matches = []
        for record_index, record in enumerate(overall_fit_records):
            if record_index in matched_record_indices:
                continue
            if not np.array_equal(record["X"], X[mask]):
                continue
            try:
                np.testing.assert_allclose(record["y"], expected_y)
                np.testing.assert_allclose(
                    record["sample_weight"], expected_sample_weight
                )
            except AssertionError:
                continue
            matches.append(record_index)

        assert len(matches) == 1
        matched_record_indices.add(matches[0])


@pytest.mark.parametrize(
    "treatment_model_factory, onnx_converter, TensorType",
    (
        list(
            zip(
                all_sklearn_regressors,
                repeat(convert_sklearn),
                repeat(FloatTensorTypeSkl),
            )
        )
        + [
            (LGBMRegressor, convert_lightgbm, FloatTensorTypeOMLT),
            (XGBRegressor, convert_xgboost, FloatTensorTypeOMLT),
        ]
    ),
)
@pytest.mark.parametrize("is_classification", [True, False])
def test_rlearner_onnx(
    treatment_model_factory, onnx_converter, is_classification, onnx_dataset, TensorType
):
    if not function_has_argument(treatment_model_factory.fit, "sample_weight"):
        pytest.skip()

    supports_categoricals = treatment_model_factory in [
        LGBMRegressor,
        # convert_sklearn does not support categoricals https://github.com/onnx/sklearn-onnx/issues/1051
        # HistGradientBoostingRegressor,
        # convert_xgboost does not support categoricals https://github.com/onnx/onnxmltools/issues/469#issuecomment-1993880910
        # XGBRegressor,
    ]

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

    ml = RLearner(
        is_classification,
        n_variants,
        nuisance_model_factory=nuisance_model_factory,
        propensity_model_factory=LGBMClassifier,
        treatment_model_factory=treatment_model_factory,
        propensity_model_params={"n_estimators": 1},
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
                            TensorType(
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
