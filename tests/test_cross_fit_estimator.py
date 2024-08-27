# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.sparse import csr_matrix
from sklearn.base import is_classifier, is_regressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

from metalearners.cross_fit_estimator import (
    CrossFitEstimator,
    _PredictContext,
    _validate_data_match_prior_split,
)

_SEED = 1337


@pytest.mark.parametrize("use_clf", [False, True])
@pytest.mark.parametrize("predict_proba", [True, False])
@pytest.mark.parametrize("is_oos", [True, False])
@pytest.mark.parametrize("oos_method", ["overall", "mean", "median"])
@pytest.mark.parametrize("backend", ["np", "pd", "csr"])
@pytest.mark.parametrize("pass_cv", [True, False])
def test_crossfitestimator_oos_smoke(
    mindset_data, rng, use_clf, predict_proba, is_oos, oos_method, backend, pass_cv
):
    if not use_clf and predict_proba:
        pytest.skip()
    if use_clf and not predict_proba and oos_method == "mean":
        pytest.skip()
    if not is_oos and oos_method not in ["mean", "median"]:
        pytest.skip()
    if use_clf and predict_proba and oos_method == "median" and is_oos:
        pytest.skip()

    predict_method = "predict_proba" if predict_proba else "predict"

    estimator_factory = LGBMClassifier if use_clf else LGBMRegressor
    estimator_params = {"n_estimators": 5}

    df, outcome_column, _, feature_columns, _ = mindset_data
    X = df[feature_columns]
    y = df[outcome_column]
    if use_clf:
        # Arbitrary cut-off
        y = y > 0.8

    if backend == "np":
        X = X.to_numpy()
        y = y.to_numpy()
    elif backend == "csr":
        X = csr_matrix(df.values)
        y = y.to_numpy()

    cfe = CrossFitEstimator(
        n_folds=5,
        estimator_factory=estimator_factory,
        estimator_params=estimator_params,
        enable_overall=True,
        random_state=_SEED,
    )
    if pass_cv:
        cv = list(KFold(n_splits=5, shuffle=True, random_state=_SEED).split(X))
    else:
        cv = None

    cfe.fit(X=X, y=y, cv=cv)
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


@pytest.mark.parametrize("estimator_factory", [LGBMClassifier, LogisticRegression])
def test_n_classes(estimator_factory, twins_data):
    (
        df,
        outcome_column,
        _,
        feature_columns,
        categorical_feature_columns,
    ) = twins_data
    cfe = CrossFitEstimator(
        n_folds=2,
        estimator_factory=estimator_factory,
        enable_overall=False,
    )
    numerical_features = [
        column
        for column in feature_columns
        if column not in categorical_feature_columns
    ]
    missing_indices = df[numerical_features].isna().any(axis=1)
    X = df[numerical_features][~missing_indices]
    y = df[outcome_column][~missing_indices]
    cfe.fit(X=X, y=y)
    assert cfe._n_classes == 2


@pytest.mark.parametrize("estimator_factory", [LGBMClassifier, LogisticRegression])
def test_crossfitestimator_classes(estimator_factory, twins_data):
    (
        df,
        outcome_column,
        _,
        feature_columns,
        categorical_feature_columns,
    ) = twins_data
    cfe = CrossFitEstimator(
        n_folds=2,
        estimator_factory=estimator_factory,
        enable_overall=False,
    )
    numerical_features = [
        column
        for column in feature_columns
        if column not in categorical_feature_columns
    ]
    missing_indices = df[numerical_features].isna().any(axis=1)
    X = df[numerical_features][~missing_indices]
    y = df[outcome_column][~missing_indices]
    cfe.fit(X=X, y=y)
    assert cfe.classes_ is not None
    assert np.array_equal(cfe.classes_, np.array([0.0, 1.0]))


def test_fit_params(mindset_data):
    df, outcome_column, _, feature_columns, _ = mindset_data

    class NewEstimator(LGBMRegressor):
        def fit(self, *args, **kwargs):
            if "key" not in kwargs or kwargs["key"] != "val":
                raise AssertionError("fit_params were not forwarded.")
            del kwargs["key"]
            return super().fit(*args, **kwargs)

    cfe = CrossFitEstimator(
        n_folds=2,
        estimator_factory=NewEstimator,
        enable_overall=False,
    )
    cfe.fit(X=df[feature_columns], y=df[outcome_column], fit_params={"key": "val"})


def test_predict_context(rng):
    model = CrossFitEstimator(10, LogisticRegression)
    n_train_obs = 1000
    n_test_obs = 500
    X = rng.standard_normal((n_train_obs, 5))
    y = rng.integers(2, size=n_train_obs)
    X_test = rng.standard_normal((n_test_obs, 5))
    model.fit(X, y)
    in_sample_pred = model.predict(X, False)
    in_sample_pred_proba = model.predict_proba(X, False)
    with _PredictContext(model, False, None) as modified_model:
        assert np.all(in_sample_pred == modified_model.predict(X))
        assert np.all(in_sample_pred_proba == modified_model.predict_proba(X))

    oos_pred = model.predict(X_test, True, "overall")
    oos_pred_proba = model.predict_proba(X_test, True, "overall")
    with _PredictContext(model, True, "overall") as modified_model:
        assert np.all(oos_pred == modified_model.predict(X_test))
        assert np.all(oos_pred_proba == modified_model.predict_proba(X_test))

    # test that the original methods are restored
    with pytest.raises(
        TypeError,
        match="CrossFitEstimator.predict\\(\\) missing 1 required positional argument: 'is_oos'",
    ):
        model.predict(X)  # type: ignore
    with pytest.raises(
        TypeError,
        match="CrossFitEstimator.predict_proba\\(\\) missing 1 required positional argument: 'is_oos'",
    ):
        model.predict_proba(X)  # type: ignore


def test_error_n_folds():
    with pytest.raises(
        ValueError, match="CrossFitting is deactivated as 'n_folds' is set to 1,"
    ):
        CrossFitEstimator(1, LogisticRegression, enable_overall=False)


def test_crossfitestimator_n_folds_1(rng, sample_size):
    cfe = CrossFitEstimator(
        n_folds=1,
        estimator_factory=LinearRegression,
    )
    X = rng.standard_normal((sample_size, 10))
    y = rng.standard_normal(sample_size)

    cfe.fit(X, y)

    with pytest.warns(
        UserWarning, match="Cross-fitting is deactivated. Using overall model"
    ):
        in_sample_predictions = cfe.predict(X, False)
    np.testing.assert_allclose(cfe.predict(X, True, "overall"), in_sample_predictions)


@pytest.mark.parametrize(
    "n_observations,test_indices,success",
    [
        (5, None, True),
        (-1, None, False),
        (0, None, False),
        (5, (np.array([1, 2]), np.array([3, 4, 5])), True),
        (
            5,
            (
                np.array([1, 2]),
                np.array(
                    [
                        3,
                        4,
                    ]
                ),
            ),
            False,
        ),
        (5, (np.array([1, 2]), np.array([3, 4, 5, 6])), False),
    ],
)
def test_validate_data_match(n_observations, test_indices, success):
    if n_observations < 1:
        with pytest.raises(
            ValueError, match=r"was expected to be (strictly )?positive"
        ):
            _validate_data_match_prior_split(n_observations, test_indices)
        return
    if success:
        _validate_data_match_prior_split(n_observations, test_indices)
    else:
        with pytest.raises(
            ValueError, match="rely on different numbers of observations"
        ):
            _validate_data_match_prior_split(n_observations, test_indices)


@pytest.mark.parametrize(
    "estimator",
    [LGBMClassifier, LGBMRegressor],
)
def test_score_smoke(estimator, rng):
    n_samples = 1000
    X = rng.standard_normal((n_samples, 3))
    if is_classifier(estimator):
        y = rng.integers(0, 4, n_samples)
    elif is_regressor(estimator):
        y = rng.standard_normal(n_samples)

    cfe = CrossFitEstimator(5, estimator, {"n_estimators": 3})
    cfe.fit(X, y)
    cfe.score(X, y, False)
