# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from itertools import repeat

import numpy as np
import onnxruntime as rt
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from onnx import ModelProto
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools import convert_lightgbm, convert_xgboost
from skl2onnx.convert import convert_sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import (
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from xgboost import XGBClassifier, XGBRegressor

from metalearners import TLearner
from metalearners._typing import Params

from .conftest import all_sklearn_classifiers, all_sklearn_regressors


@pytest.mark.parametrize(
    "nuisance_model_factory, onnx_converter, is_classification",
    (
        list(
            zip(
                all_sklearn_classifiers,
                repeat(partial(convert_sklearn, options={"zipmap": False})),
                repeat([True]),
            )
        )
        + list(zip(all_sklearn_regressors, repeat(convert_sklearn), repeat(False)))
        + [
            (LGBMClassifier, partial(convert_lightgbm, zipmap=False), True),
            (LGBMRegressor, convert_lightgbm, False),
            (XGBClassifier, convert_xgboost, True),
            (XGBRegressor, convert_xgboost, False),
        ]
    ),
)
def test_tlearner_onnx(
    nuisance_model_factory, onnx_converter, is_classification, onnx_dataset
):
    supports_categoricals = nuisance_model_factory in [
        LGBMClassifier,
        LGBMRegressor,
        # convert_sklearn does not support categoricals https://github.com/onnx/sklearn-onnx/issues/1051
        # HistGradientBoostingClassifier,
        # HistGradientBoostingRegressor,
        # convert_xgboost does not support categoricals https://github.com/onnx/onnxmltools/issues/469#issuecomment-1993880910
        # XGBClassifier,
        # XGBRegressor,
    ]

    nuisance_model_params: Params | None
    if nuisance_model_factory in [RadiusNeighborsClassifier, RadiusNeighborsRegressor]:
        nuisance_model_params = {"radius": 10}
    elif nuisance_model_factory in [
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    ]:
        # This is unnecessary for now but if at some point convert_sklearn supports categoricals this is needed
        nuisance_model_params = {"categorical_features": "from_dtype"}
    elif nuisance_model_factory in [XGBClassifier, XGBRegressor]:
        # This is unnecessary for now but if at some point convert_xgboost supports categoricals this is needed
        nuisance_model_params = {"enable_categorical": True}
    else:
        nuisance_model_params = None

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
    else:
        y = y_reg

    ml = TLearner(
        is_classification,
        n_variants,
        nuisance_model_factory=nuisance_model_factory,
        nuisance_model_params=nuisance_model_params,
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
        for i in range(n_categorical_features):
            onnx_X[:, n_numerical_features + i] = X[n_numerical_features + i].cat.codes
    else:
        onnx_X = X.astype(np.float32)

    (pred_onnx,) = sess.run(
        ["tau"],
        {"X": onnx_X},
    )
    np.testing.assert_allclose(ml.predict(X, True, "overall"), pred_onnx, atol=5e-4)
