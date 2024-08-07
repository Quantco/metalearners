# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from itertools import repeat

import numpy as np
import onnxruntime as rt
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from onnx import ModelProto
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools import convert_lightgbm, convert_xgboost
from skl2onnx import convert_sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import RadiusNeighborsRegressor
from xgboost import XGBRegressor

from metalearners import DRLearner
from metalearners._typing import Params

from .conftest import all_sklearn_regressors


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
    np.testing.assert_allclose(ml.predict(X, True, "overall"), pred_onnx, atol=5e-4)


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
