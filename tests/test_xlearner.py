# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import random
from functools import partial

import numpy as np
import onnx
import onnxruntime as rt
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from onnxmltools import convert_lightgbm, convert_xgboost
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor

from metalearners import XLearner
from metalearners._typing import Params
from metalearners.metalearner import PROPENSITY_MODEL
from metalearners.xlearner import CONTROL_EFFECT_MODEL, TREATMENT_EFFECT_MODEL

from .conftest import all_sklearn_classifiers, all_sklearn_regressors


@pytest.mark.parametrize(
    "treatment_model_factory, treatment_onnx_converter",
    random.sample(  # With this we run a different set of tests every time as running all of them is too costly (there are ~50 regressors and classifiers)
        list(
            zip(
                all_sklearn_regressors,
                [convert_sklearn] * len(all_sklearn_regressors),
            )
        )
        + [
            (LGBMRegressor, convert_lightgbm),
            (XGBRegressor, convert_xgboost),
        ],
        4,
    ),
)
@pytest.mark.parametrize(
    "propensity_model_factory, propensity_onnx_converter",
    random.sample(
        list(
            zip(
                all_sklearn_classifiers,
                [partial(convert_sklearn, options={"zipmap": False})]
                * len(all_sklearn_classifiers),
            )
        )
        + [
            (LGBMClassifier, partial(convert_lightgbm, zipmap=False)),
            (XGBClassifier, convert_xgboost),
        ],
        4,
    ),
)
@pytest.mark.parametrize("is_classification", [True, False])
def test_xlearner_onnx(
    treatment_model_factory,
    propensity_model_factory,
    treatment_onnx_converter,
    propensity_onnx_converter,
    is_classification,
    onnx_dataset,
):
    treatment_model_params: Params | None
    if treatment_model_factory == RadiusNeighborsRegressor:
        treatment_model_params = {"radius": 10}
    else:
        treatment_model_params = None
    propensity_model_params: Params | None
    if propensity_model_factory == RadiusNeighborsClassifier:
        propensity_model_params = {"radius": 10}
    else:
        propensity_model_params = None

    X, _, y_class, y_reg, w = onnx_dataset
    n_numerical_features = X.shape[1]
    n_variants = len(np.unique(w))
    if is_classification:
        y = y_class
        nuisance_model_factory = LGBMClassifier
    else:
        y = y_reg
        nuisance_model_factory = LGBMRegressor

    nuisance_model_params = {"n_estimators": 1}

    ml = XLearner(
        is_classification,
        n_variants,
        nuisance_model_factory=nuisance_model_factory,
        nuisance_model_params=nuisance_model_params,
        propensity_model_factory=propensity_model_factory,
        propensity_model_params=propensity_model_params,
        treatment_model_factory=treatment_model_factory,
        treatment_model_params=treatment_model_params,
        n_folds=2,
    )
    ml.fit(X, y, w)

    onnx_models: dict[str, list[onnx.ModelProto]] = {
        CONTROL_EFFECT_MODEL: [],
        TREATMENT_EFFECT_MODEL: [],
        PROPENSITY_MODEL: [],
    }
    for tv in range(n_variants - 1):
        model = ml._treatment_models[CONTROL_EFFECT_MODEL][tv]._overall_estimator
        onnx_model = treatment_onnx_converter(
            model, initial_types=[("X", FloatTensorType([None, n_numerical_features]))]
        )
        onnx_models[CONTROL_EFFECT_MODEL].append(onnx_model)

        model = ml._treatment_models[TREATMENT_EFFECT_MODEL][tv]._overall_estimator
        onnx_model = treatment_onnx_converter(
            model, initial_types=[("X", FloatTensorType([None, n_numerical_features]))]
        )
        onnx_models[TREATMENT_EFFECT_MODEL].append(onnx_model)

    model = ml._nuisance_models[PROPENSITY_MODEL][0]._overall_estimator
    onnx_model = propensity_onnx_converter(
        model,
        initial_types=[("X", FloatTensorType([None, n_numerical_features]))],
    )
    onnx_models[PROPENSITY_MODEL].append(onnx_model)

    final = ml.build_onnx(onnx_models)

    sess = rt.InferenceSession(
        final.SerializeToString(), providers=rt.get_available_providers()
    )

    pred_onnx = sess.run(
        ["tau"],
        {"X": X.astype(np.float32)},
    )
    np.testing.assert_allclose(ml.predict(X, True, "overall"), pred_onnx[0], atol=5e-4)
