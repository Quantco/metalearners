# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import numpy as np
import onnxruntime as rt
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from onnxmltools import convert_lightgbm, convert_xgboost
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.convert import convert_sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveRegressor,
    QuantileRegressor,
    RANSACRegressor,
    Ridge,
    RidgeCV,
    SGDRegressor,
    TheilSenRegressor,
    TweedieRegressor,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from xgboost import XGBClassifier, XGBRegressor

from metalearners import TLearner
from metalearners._typing import Params
from metalearners.metalearner import VARIANT_OUTCOME_MODEL

all_sklearn_classifiers = [
    AdaBoostClassifier,
    BaggingClassifier,
    CalibratedClassifierCV,
    DecisionTreeClassifier,
    ExtraTreeClassifier,
    ExtraTreesClassifier,
    # GaussianProcessClassifier, # This raises an error com.microsoft:Solve(-1) is not a registered function/op when inference on onnx. TODO: investigate it further
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    KNeighborsClassifier,
    LinearDiscriminantAnalysis,
    LogisticRegression,
    LogisticRegressionCV,
    MLPClassifier,
    QuadraticDiscriminantAnalysis,
    RadiusNeighborsClassifier,
    RandomForestClassifier,
]  # extracted from all_estimators("classifier"), models which have predict_proba and convert_sklearn supports them

all_sklearn_regressors = [
    ARDRegression,
    AdaBoostRegressor,
    BaggingRegressor,
    BayesianRidge,
    DecisionTreeRegressor,
    ElasticNet,
    ElasticNetCV,
    ExtraTreeRegressor,
    ExtraTreesRegressor,
    GaussianProcessRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    HuberRegressor,
    KNeighborsRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    LinearSVR,
    MLPRegressor,
    NuSVR,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    # PLSRegression, # The output shape of the onnx converted model is wrong
    PassiveAggressiveRegressor,
    QuantileRegressor,
    RANSACRegressor,
    RadiusNeighborsRegressor,
    RandomForestRegressor,
    Ridge,
    RidgeCV,
    SGDRegressor,
    SVR,
    TheilSenRegressor,
    TweedieRegressor,
]  # regressors which are supported by convert_sklearn and support regression in the reals


@pytest.mark.parametrize(
    "nuisance_model_factory, onnx_converter, is_classification",
    (
        list(
            zip(
                all_sklearn_classifiers,
                [partial(convert_sklearn, options={"zipmap": False})]
                * len(all_sklearn_classifiers),
                [True] * len(all_sklearn_classifiers),
            )
        )
        + list(
            zip(
                all_sklearn_regressors,
                [convert_sklearn] * len(all_sklearn_regressors),
                [False] * len(all_sklearn_regressors),
            )
        )
        + [
            (LGBMClassifier, partial(convert_lightgbm, zipmap=False), True),
            (LGBMRegressor, convert_lightgbm, False),
            (XGBClassifier, convert_xgboost, True),
            (XGBRegressor, convert_xgboost, False),
        ]
    ),
)
def test_tlearner_onnx(nuisance_model_factory, onnx_converter, is_classification, rng):
    if nuisance_model_factory == QuadraticDiscriminantAnalysis:
        # TODO: investigate the cause why the assertion fails
        pytest.skip()

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
    n_samples = 300
    n_numerical_features = 5
    n_variants = 3
    n_classes = (
        2 if nuisance_model_factory == GaussianProcessClassifier else 3
    )  # convert_sklearn only supports binary classification with GaussianProcessClassifier
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
        y = rng.integers(0, n_classes, size=n_samples)
    else:
        y = rng.standard_normal(n_samples)
    w = rng.integers(0, n_variants, n_samples)

    ml = TLearner(
        is_classification,
        n_variants,
        nuisance_model_factory=nuisance_model_factory,
        nuisance_model_params=nuisance_model_params,
        n_folds=2,
    )
    ml.fit(X, y, w)

    onnx_models = []
    for tv in range(n_variants):
        model = ml._nuisance_models[VARIANT_OUTCOME_MODEL][tv]._overall_estimator
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

    final = ml.build_onnx({VARIANT_OUTCOME_MODEL: onnx_models})
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

    pred_onx = sess.run(
        ["tau"],
        {"input": onnx_X},
    )
    np.testing.assert_allclose(ml.predict(X, True, "overall"), pred_onx[0], atol=1e-4)
