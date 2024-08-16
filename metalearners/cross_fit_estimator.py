# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
)
from typing_extensions import Self

from metalearners._typing import Matrix, OosMethod, PredictMethod, SplitIndices, Vector
from metalearners._utils import (
    _ScikitModel,
    index_matrix,
    safe_len,
    validate_number_positive,
)

OVERALL: OosMethod = "overall"
MEDIAN: OosMethod = "median"
_MEAN: OosMethod = "mean"
_OOS_WHITELIST = [OVERALL, MEDIAN, _MEAN]


def _validate_oos_method(
    oos_method: OosMethod | None, enable_overall: bool, n_folds: int
) -> None:
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
    if n_folds == 1 and oos_method != OVERALL:
        raise ValueError(
            'Cross-fitting is deactivated and therefore only the ``"overall"`` prediction method '
            "can be used."
        )


def _validate_n_folds(n_folds: int) -> None:
    # TODO: Use _utils.validate_number_positive instead.
    if n_folds <= 0:
        raise ValueError(
            f"n_folds needs to be a strictly positive integer but is {n_folds}."
        )


def _validate_data_match_prior_split(
    n_observations: int, test_indices: tuple[np.ndarray] | None
) -> None:
    """Validate whether the previous test_indices and the passed data are based on the
    same number of observations."""
    validate_number_positive(n_observations, "n_observations", strict=True)
    if test_indices is None:
        return
    expected_n_observations = sum(len(x) for x in test_indices)
    if expected_n_observations != n_observations:
        raise ValueError(
            "CrossFitEstimator is given data to fit X and splits cv "
            "which rely on different numbers of observations."
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

    ``n_folds`` can be set to 1 if the user desires to deactivate cross-fitting. In
    that case, the ``CrossFitEstimator`` would only fit one overall model which would be
    the one used for either in sample or out of sample predictions. Note that this is
    not recommended since it can lead to data leakage when doing in-sample predictions.
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
    classes_: np.ndarray | None = field(init=False)

    def __post_init__(self):
        _validate_n_folds(self.n_folds)
        if self.n_folds == 1 and not self.enable_overall:
            raise ValueError(
                "CrossFitting is deactivated as 'n_folds' is set to 1, but 'enable_overall' "
                "is set to 'False'. If you wish to deactivate CrossFitting, please ensure "
                " 'enable_overall' is set to 'True'."
            )
        self._estimators: list[_ScikitModel] = []
        self._estimator_type: str = self.estimator_factory._estimator_type
        self._overall_estimator: _ScikitModel | None = None
        self._test_indices: tuple[np.ndarray] | None = None
        self._n_classes: int | None = None
        self.classes_: np.ndarray | None = None

    def _train_overall_estimator(
        self, X: Matrix, y: Matrix | Vector, fit_params: dict | None = None
    ) -> _ScikitModel:
        fit_params = fit_params or dict()
        model = self.estimator_factory(**self.estimator_params)
        return model.fit(X, y, **fit_params)

    def clone(self) -> "CrossFitEstimator":
        r"""Construct a new unfitted CrossFitEstimator with the same init parameters."""
        return CrossFitEstimator(
            n_folds=self.n_folds,
            estimator_factory=self.estimator_factory,
            estimator_params=self.estimator_params,
            enable_overall=self.enable_overall,
            random_state=self.random_state,
        )

    def fit(
        self,
        X: Matrix,
        y: Vector | Matrix,
        fit_params: dict | None = None,
        n_jobs_cross_fitting: int | None = None,
        cv: SplitIndices | None = None,
    ) -> Self:
        """Fit the underlying estimators.

        One estimator is trained per ``n_folds``.

        If ``enable_overall`` is set, an additional estimator is trained on all data.

        ``n_jobs_cross_fitting`` can be used to specify the number of jobs for cross-fitting.
        For more information see the `sklearn glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_.

        ``cv`` can optionally be passed. If passed, it is expected to be a list of
        (train_indices, test_indices) tuples indicating how to split the data at hand
        into train and test/estimation sets for different folds.
        """
        _validate_data_match_prior_split(safe_len(X), self._test_indices)

        if fit_params is None:
            fit_params = dict()
        if self.n_folds > 1:
            if cv is None:
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
                n_jobs=n_jobs_cross_fitting,
            )
            self._estimators = cv_result["estimator"]
            self._test_indices = cv_result["indices"]["test"]
        if self.enable_overall:
            self._overall_estimator = self._train_overall_estimator(X, y, fit_params)

        if is_classifier(self):
            self._n_classes = len(np.unique(y))
            self.classes_ = np.unique(y)
            for e in self._estimators:
                if set(e.classes_) != set(self.classes_):  # type: ignore
                    raise ValueError(
                        "Some folds in cross-fitting had fewer classes than "
                        "the overall dataset. Please check the cv parameter. If you are "
                        "synchronizing the folds in a MetaLearner consider not doing it."
                    )
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
            n_observations=safe_len(X),
            n_outputs=n_outputs,
            n_folds=self.n_folds,
        )
        for i, estimator in enumerate(self._estimators):
            predictions[:, :, i] = np.reshape(
                getattr(estimator, method)(X), (safe_len(X), n_outputs)
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
        if safe_len(X) != sum(len(fold) for fold in self._test_indices):
            raise ValueError(
                "Trying to predict in-sample on data that is unlike data encountered in training. "
                f"Training data included {sum(len(fold) for fold in self._test_indices)} "
                f"observations while prediction data includes {safe_len(X)} observations."
            )
        n_outputs = self._n_outputs(method)
        predictions = self._initialize_prediction_tensor(
            n_observations=safe_len(X),
            n_outputs=n_outputs,
            n_folds=1,
        )
        for estimator, indices in zip(self._estimators, self._test_indices):
            fold_predictions = np.reshape(
                getattr(estimator, method)(index_matrix(X, indices)),
                (len(indices), n_outputs, 1),
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
    ) -> np.ndarray:
        if is_oos:
            _validate_oos_method(oos_method, self.enable_overall, self.n_folds)
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
        if self.n_folds == 1:
            warnings.warn(
                "Cross-fitting is deactivated. Using overall model for in sample predictions."
            )
            return getattr(self._overall_estimator, method)(X)
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

    def score(
        self,
        X: Matrix,
        y: Vector,
        is_oos: bool,
        oos_method: OosMethod | None = None,
        sample_weight: Vector | None = None,
    ) -> float:
        """Return the coefficient of determination of the prediction if the estimator is
        a regressor or the mean accuracy if it is a classifier."""
        if is_classifier(self):
            return accuracy_score(
                y, self.predict(X, is_oos, oos_method), sample_weight=sample_weight
            )
        elif is_regressor(self):
            return r2_score(
                y, self.predict(X, is_oos, oos_method), sample_weight=sample_weight
            )
        else:
            raise NotImplementedError(
                "score is not implemented for this type of estimator."
            )

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
        new_predict = partial(
            self.model.predict, is_oos=self.is_oos, oos_method=self.oos_method
        )
        new_predict.__name__ = "predict"  # type: ignore
        self.model.predict = new_predict  # type: ignore

        new_predict_proba = partial(
            self.model.predict_proba, is_oos=self.is_oos, oos_method=self.oos_method
        )
        new_predict_proba.__name__ = "predict_proba"  # type: ignore
        self.model.predict_proba = new_predict_proba  # type: ignore
        return self.model

    def __exit__(self, *args):
        self.model.predict = self.original_predict  # type: ignore
        self.model.predict_proba = self.original_predict_proba  # type: ignore
