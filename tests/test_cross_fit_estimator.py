# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from functools import partial

import lightgbm as lgbm
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, log_loss

from metalearners.cross_fit_estimator import CrossFitEstimator


@pytest.mark.parametrize("use_clf", [False, True])
@pytest.mark.parametrize("predict_proba", [True, False])
@pytest.mark.parametrize("is_oos", [True, False])
@pytest.mark.parametrize("oos_method", ["overall", "mean", "median"])
@pytest.mark.parametrize("use_np", [True, False])
def test_crossfitestimator_oos_smoke(
    mindset_data, rng, use_clf, predict_proba, is_oos, oos_method, use_np
):
    if not use_clf and predict_proba:
        pytest.skip()
    if use_clf and not predict_proba and oos_method == "mean":
        pytest.skip()
    if not is_oos and oos_method not in ["mean", "median"]:
        pytest.skip()

    predict_method = "predict_proba" if predict_proba else "predict"

    estimator_factory = lgbm.LGBMClassifier if use_clf else lgbm.LGBMRegressor
    estimator_params = {"n_estimators": 5}

    df, outcome_column, _, feature_columns, _ = mindset_data
    X = df[feature_columns]
    y = df[outcome_column]
    if use_clf:
        # Arbitrary cut-off
        y = y > 0.8

    if use_np:
        X = X.to_numpy()
        y = y.to_numpy()

    cfe = CrossFitEstimator(
        n_folds=5,
        estimator_factory=estimator_factory,
        estimator_params=estimator_params,
        enable_overall=True,
    )
    cfe.fit(X=X, y=y)
    predictions = getattr(cfe, predict_method)(
        X=X, is_oos=is_oos, oos_method=oos_method
    )

    assert len(predictions) == len(y)
    assert predictions.ndim <= 2

    if predict_proba:
        assert predictions.shape[1] == 2
    else:
        assert predictions.ndim == 1 or predictions.shape[1] == 1

    if not use_clf:
        try:
            from sklearn.metrics import root_mean_squared_error

            rmse = root_mean_squared_error
        # TODO: Remove as soon as we can lower-bound sklearn to 1.4
        except ImportError:
            from sklearn.metrics import mean_squared_error

            rmse = partial(mean_squared_error, squared=False)
        assert rmse(y, predictions) < y.std()
    elif predict_proba:
        assert log_loss(y, predictions[:, 1]) < log_loss(y, 0.5 * np.ones(len(y)))
    else:
        assert (
            accuracy_score(y, predictions) >= (max(sum(y), sum(1 - y)) / len(y)) - 0.01
        )
