# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from itertools import repeat

import numpy as np
import onnxruntime as rt
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from onnx import ModelProto
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools import convert_lightgbm, convert_xgboost
from skl2onnx import convert_sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor

from metalearners._utils import function_has_argument
from metalearners.rlearner import RLearner, r_loss

from .conftest import all_sklearn_regressors


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
def test_rlearner_onnx(
    treatment_model_factory, onnx_converter, is_classification, onnx_dataset
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
