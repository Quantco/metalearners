# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from dataclasses import dataclass, field
from functools import partial

import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from typing_extensions import Self

from metalearners._typing import OosMethod, PredictMethod
from metalearners._utils import Matrix, Vector, _ScikitModel, index_matrix

OVERALL: OosMethod = "overall"
MEDIAN: OosMethod = "median"
_MEAN: OosMethod = "mean"
_OOS_WHITELIST = [OVERALL, MEDIAN, _MEAN]


def _validate_oos_method(oos_method: OosMethod | None, enable_overall: bool) -> None:
    if oos_method not in _OOS_WHITELIST:
        raise ValueError(
            f"oos_method {oos_method} not supported. Supported values are "
            f"{_OOS_WHITELIST}."
        )
    if oos_method == OVERALL and not enable_overall:
        raise ValueError(
            "In order to use 'overall' prediction method, the estimator's "
            "enable_overall property has to be set to True."
        )


def _validate_n_folds(n_folds: int) -> None:
    if n_folds <= 0:
        raise ValueError(
            f"n_folds needs to be a strictly positive integer but is {n_folds}."
        )


@dataclass
class CrossFitEstimator:
    """Helper class for cross-fitting estimators on data.

    Conceptually, it allows for fitting ``n_folds`` or ``n_folds`` + 1 models on
    ``n_folds`` folds of the data.

    ``estimator_factory`` is a class implementing an estimator with a scikit-learn
    interface. Instantiation parameters can be passed to ``estimator_params``.
    An example argument for ``estimator_factory`` would be ``lightgbm.LGBMRegressor``.

    Importantly, the ``CrossFitEstimator`` can handle in-sample and out-of-sample
    ('oos') data for prediction. When doing in-sample prediction the single model will
    be used in which the respective data point has not been part of the training set.
    When doing oos prediction, different options exist. These options either rely on
    combining the ``n_folds`` models or using a model trained on all of the data
    (``enable_overall``).
    """

    n_folds: int
    estimator_factory: type[_ScikitModel]
    estimator_params: dict = field(default_factory=dict)
    enable_overall: bool = True
    random_state: int | None = None
    _estimators: list[_ScikitModel] = field(init=False)
    _estimator_type: str = field(init=False)
    _overall_estimator: _ScikitModel | None = field(init=False)
    _test_indices: tuple[np.ndarray] | None = field(init=False)
    _n_classes: int | None = field(init=False)

    def __post_init__(self):
        _validate_n_folds(self.n_folds)
        self._estimators: list[_ScikitModel] = []
        self._estimator_type: str = self.estimator_factory._estimator_type
        self._overall_estimator: _ScikitModel | None = None
        self._test_indices: tuple[np.ndarray] | None = None
        self._n_classes: int | None = None

    def _train_overall_estimator(
        self, X: Matrix, y: Matrix | Vector, fit_params: dict | None = None
    ) -> _ScikitModel:
        fit_params = fit_params or dict()
        model = self.estimator_factory(**self.estimator_params)
        return model.fit(X, y, **fit_params)

    def fit(
        self,
        X: Matrix,
        y: Vector | Matrix,
        fit_params: dict | None = None,
        **kwargs,
    ) -> Self:
        """Fit the underlying estimators.

        One estimator is trained per ``n_folds``.

        If ``enable_overall`` is set, an additional estimator is trained on all data.
        """
        if fit_params is None:
            fit_params = dict()
        if is_classifier(self):
            cv = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            cv = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        cv_result = cross_validate(
            self.estimator_factory(**self.estimator_params),
            X,
            y,
            cv=cv,
            return_estimator=True,
            return_indices=True,
            params=fit_params,
        )
        self._estimators = cv_result["estimator"]
        self._test_indices = cv_result["indices"]["test"]
        if self.enable_overall:
            self._overall_estimator = self._train_overall_estimator(X, y, fit_params)

        if is_classifier(self):
            self._n_classes = len(np.unique(y))

        return self

    def _initialize_prediction_tensor(
        self, n_observations: int, n_outputs: int, n_folds: int
    ) -> np.ndarray:
        return np.zeros((n_observations, n_outputs, n_folds))

    def _n_outputs(self, method: PredictMethod) -> int:
        if method == "predict_proba" and self._n_classes:
            return self._n_classes
        return 1

    def _predict_all(self, X: Matrix, method: PredictMethod) -> np.ndarray:
        n_outputs = self._n_outputs(method)
        predictions = self._initialize_prediction_tensor(
            n_observations=len(X),
            n_outputs=n_outputs,
            n_folds=self.n_folds,
        )
        for i, estimator in enumerate(self._estimators):
            predictions[:, :, i] = np.reshape(
                getattr(estimator, method)(X), (-1, n_outputs)
            )
        if n_outputs == 1:
            return predictions[:, 0, :]
        return predictions

    def _predict_mean(self, X: Matrix, method: PredictMethod) -> np.ndarray:
        all_predictions = self._predict_all(X=X, method=method)
        return np.mean(all_predictions, axis=-1)

    def _predict_median(self, X: Matrix, method: PredictMethod) -> np.ndarray:
        all_predictions = self._predict_all(X=X, method=method)
        return np.median(all_predictions, axis=-1)

    def _predict_in_sample(
        self,
        X: Matrix,
        method: PredictMethod,
    ) -> np.ndarray:
        if not self._test_indices:
            raise ValueError()
        if len(X) != sum(len(fold) for fold in self._test_indices):
            raise ValueError(
                "Trying to predict in-sample on data that is unlike data encountered in training."
                f"Training data included {sum(len(fold) for fold in self._test_indices)} "
                f"observations while prediction data includes {len(X)} observations."
            )
        n_outputs = self._n_outputs(method)
        predictions = self._initialize_prediction_tensor(
            n_observations=len(X),
            n_outputs=n_outputs,
            n_folds=1,
        )
        for estimator, indices in zip(self._estimators, self._test_indices):
            fold_predictions = np.reshape(
                getattr(estimator, method)(index_matrix(X, indices)), (-1, n_outputs, 1)
            )
            predictions[indices] = fold_predictions
        if n_outputs == 1:
            return predictions[:, 0, 0]
        return predictions[:, :, 0]

    def _predict(
        self,
        X: Matrix,
        is_oos: bool,
        method: PredictMethod,
        oos_method: OosMethod | None = None,
        w: Vector | Matrix | None = None,
    ) -> np.ndarray:
        if is_oos:
            _validate_oos_method(oos_method, self.enable_overall)
            if oos_method == OVERALL:
                return getattr(self._overall_estimator, method)(X)
            if oos_method == _MEAN:
                if method != "predict_proba" and any(
                    is_classifier(est) for est in self._estimators
                ):
                    raise ValueError(
                        "Cannot create a mean of classes. Please use a different oos_method."
                    )
                return self._predict_mean(X, method=method)
            if method == "predict_proba":
                raise ValueError(
                    "Cannot create median of class probabilities. Please use a different oos_method."
                )
            return self._predict_median(X, method=method)
        return self._predict_in_sample(X, method=method)

    def predict(
        self,
        X: Matrix,
        is_oos: bool,
        oos_method: OosMethod | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict from ``X``.

        If ``is_oos``, the ``oos_method`` will be used to generate predictions
        on 'out of sample' data. 'Out of sample' refers to this data not having been
        used in the ``fit`` method. The ``oos_method`` ``'overall'`` can only be used
        if the ``CrossFitEstimator`` has been initialized with
        ``enable_overall=True``.
        """
        return self._predict(
            X=X,
            is_oos=is_oos,
            method="predict",
            oos_method=oos_method,
        )

    def predict_proba(
        self,
        X: Matrix,
        is_oos: bool,
        oos_method: OosMethod | None = None,
    ) -> np.ndarray:
        """Predict probability from ``X``.

        If ``is_oos``, the ``oos_method`` will be used to generate predictions
        on 'out of sample' data. 'Out of sample' refers to this data not having been
        used in the ``fit`` method. The ``oos_method`` ``'overall'`` can only be used
        if the ``CrossFitEstimator`` has been initialized with
        ``enable_overall=True``.
        """
        return self._predict(
            X=X,
            is_oos=is_oos,
            method="predict_proba",
            oos_method=oos_method,
        )

    def score(self, X, y, sample_weight=None, **kwargs):
        raise NotImplementedError()

    def set_params(self, **params):
        raise NotImplementedError()


class _PredictContext:
    def __init__(
        self,
        model: CrossFitEstimator,
        is_oos: bool,
        oos_method: OosMethod | None = None,
    ):
        if is_oos and oos_method is None:
            raise ValueError(
                "Can not use _PredictContext with is_oos set to True and oos_method ",
                "not defined.",
            )
        self.model = model
        self.is_oos = is_oos
        self.oos_method = oos_method
        self.original_predict = model.predict
        self.original_predict_proba = model.predict_proba

    def __enter__(self):
        self.model.predict = partial(  # type: ignore
            self.model.predict, is_oos=self.is_oos, oos_method=self.oos_method
        )
        self.model.predict_proba = partial(  # type: ignore
            self.model.predict_proba, is_oos=self.is_oos, oos_method=self.oos_method
        )
        return self.model

    def __exit__(self, *args):
        self.model.predict = self.original_predict  # type: ignore
        self.model.predict_proba = self.original_predict_proba  # type: ignore
