# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import onnxruntime as rt
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from onnxmltools import convert_lightgbm, convert_xgboost
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor

from metalearners import DRLearner
from metalearners.metalearner import TREATMENT_MODEL

from .conftest import all_sklearn_regressors


@pytest.mark.parametrize(
    "treatment_model_factory, onnx_converter",
    (
        list(
            zip(
                all_sklearn_regressors,
                [convert_sklearn] * len(all_sklearn_regressors),
            )
        )
        + [
            (LGBMRegressor, convert_lightgbm),
            (XGBRegressor, convert_xgboost),
        ]
    ),
)
@pytest.mark.parametrize("is_classification", [True, False])
def test_rlearner_onnx(treatment_model_factory, onnx_converter, is_classification, rng):
    supports_categoricals = treatment_model_factory in [
        LGBMRegressor,
        # convert_sklearn does not support categoricals https://github.com/onnx/sklearn-onnx/issues/1051
        # HistGradientBoostingRegressor,
        # convert_xgboost does not support categoricals https://github.com/onnx/onnxmltools/issues/469#issuecomment-1993880910
        # XGBRegressor,
    ]

    # TODO: move this generation to a fixture
    n_samples = 300
    n_numerical_features = 5
    n_variants = 3

    X = rng.standard_normal((n_samples, n_numerical_features))
    if supports_categoricals:
        n_categorical_features = 2
        X = pd.DataFrame(X)
        X[n_numerical_features] = pd.Series(
            rng.integers(10, 13, n_samples), dtype="category"
        )  # not start at 0
        X[n_numerical_features + 1] = pd.Series(
            rng.choice([-5, 4, -10, -32], size=n_samples), dtype="category"
        )  # not consecutive
    else:
        n_categorical_features = 0

    if is_classification:
        n_classes = 2
        y = rng.integers(0, n_classes, size=n_samples)
        nuisance_model_factory = LogisticRegression
    else:
        y = rng.standard_normal(n_samples)
        nuisance_model_factory = LinearRegression
    w = rng.integers(0, n_variants, n_samples)

    ml = DRLearner(
        is_classification,
        n_variants,
        nuisance_model_factory=nuisance_model_factory,
        propensity_model_factory=LGBMClassifier,
        treatment_model_factory=treatment_model_factory,
        propensity_model_params={"n_estimators": 1},
        n_folds=2,
    )
    ml.fit(X, y, w)

    onnx_models = []
    for tv in range(n_variants - 1):
        model = ml._treatment_models[TREATMENT_MODEL][tv]._overall_estimator
        onnx_model = onnx_converter(
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
        onnx_models.append(onnx_model)

    final = ml.build_onnx({TREATMENT_MODEL: onnx_models})
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

    pred_onnx = sess.run(
        ["tau"],
        {"X": onnx_X},
    )
    np.testing.assert_allclose(ml.predict(X, True, "overall"), pred_onnx[0], atol=5e-4)
