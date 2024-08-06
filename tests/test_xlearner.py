# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import random
from functools import partial
from itertools import repeat

import numpy as np
import onnxruntime as rt
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from onnx import ModelProto
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools import convert_lightgbm, convert_xgboost
from skl2onnx import convert_sklearn
from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor

from metalearners import XLearner
from metalearners._typing import Params
from metalearners.xlearner import CONTROL_EFFECT_MODEL, TREATMENT_EFFECT_MODEL

from .conftest import all_sklearn_classifiers, all_sklearn_regressors


@pytest.mark.parametrize(
    "treatment_model_factory, treatment_onnx_converter",
    random.sample(  # With this we run a different set of tests every time as running all of them is too costly (there are ~50 regressors and classifiers)
        list(
            zip(
                all_sklearn_regressors,
                repeat(convert_sklearn),
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

    necessary_models = ml._necessary_onnx_models()
    onnx_models: dict[str, list[ModelProto]] = {}

    for model_kind, models in necessary_models.items():
        onnx_models[model_kind] = []
        if model_kind in [CONTROL_EFFECT_MODEL, TREATMENT_EFFECT_MODEL]:
            onnx_converter = treatment_onnx_converter
        else:
            onnx_converter = propensity_onnx_converter
        for model in models:
            onnx_models[model_kind].append(
                onnx_converter(
                    model,
                    initial_types=[
                        (
                            "X",
                            FloatTensorType([None, n_numerical_features]),
                        )
                    ],
                )
            )

    final = ml._build_onnx(onnx_models)

    sess = rt.InferenceSession(
        final.SerializeToString(), providers=rt.get_available_providers()
    )

    (pred_onnx,) = sess.run(
        ["tau"],
        {"X": X.astype(np.float32)},
    )
    np.testing.assert_allclose(ml.predict(X, True, "overall"), pred_onnx, atol=5e-4)
