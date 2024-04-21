# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Optional, Union

import numpy as np
from typing_extensions import Self

from metalearners._utils import Matrix, Vector, _ScikitModel, validate_number_positive
from metalearners.cross_fit_estimator import CrossFitEstimator, OosMethod

Params = dict[str, Union[int, float, str]]
Features = Union[Collection[str], Collection[int]]
ModelFactory = Union[type[_ScikitModel], dict[str, type[_ScikitModel]]]


def _initialize_model_dict(argument, expected_names: Collection[str]) -> dict:
    if isinstance(argument, dict) and set(argument.keys()) == set(expected_names):
        return argument
    return {name: argument for name in expected_names}


class MetaLearner(ABC):

    @classmethod
    @abstractmethod
    def nuisance_model_names(cls) -> list[str]: ...

    @classmethod
    @abstractmethod
    def treatment_model_names(cls) -> list[str]: ...

    def __init__(
        self,
        nuisance_model_factory: ModelFactory,
        treatment_model_factory: ModelFactory,
        nuisance_model_params: Optional[Union[Params, dict[str, Params]]] = None,
        treatment_model_params: Optional[Union[Params, dict[str, Params]]] = None,
        feature_set: Optional[Union[Features, dict[str, Features]]] = None,
        # TODO: Consider implementing selection of number of folds for various estimators.
        n_folds: int = 10,
        random_state: Optional[int] = None,
    ):
        """Initialize a MetaLearner.

        All of
        * ``nuisance_model_factory``
        * ``treatment_model_factory``
        * ``nuisance_model_params``
        * ``treatment_model_params``
        * ``feature_set``

        can either

        * contain a single value, such that the value will be used for all relevant models
        of the respective MetaLearner or
        * a dictionary mapping from the relevant models (``model_kind``, a ``str``) to the
        respective value
        """
        nuisance_model_names = self.__class__.nuisance_model_names()
        treatment_model_names = self.__class__.treatment_model_names()

        self.nuisance_model_factory = _initialize_model_dict(
            nuisance_model_factory, nuisance_model_names
        )
        if nuisance_model_params is None:
            self.nuisance_model_params = _initialize_model_dict(
                {}, nuisance_model_names
            )
        else:
            self.nuisance_model_params = _initialize_model_dict(
                nuisance_model_params, nuisance_model_names
            )
        self.treatment_model_factory = _initialize_model_dict(
            treatment_model_factory, treatment_model_names
        )
        if treatment_model_params is None:
            self.treatment_model_params = _initialize_model_dict(
                {}, treatment_model_names
            )
        else:
            self.treatment_model_params = _initialize_model_dict(
                treatment_model_params, treatment_model_names
            )

        validate_number_positive(n_folds, "n_folds")
        self.n_folds = n_folds
        self.random_state = random_state

        if feature_set is None:
            self.feature_set = None
        else:
            self.feature_set = _initialize_model_dict(
                feature_set, nuisance_model_names + treatment_model_names
            )

        self._nuisance_models: dict[str, _ScikitModel] = {
            name: CrossFitEstimator(
                n_folds=self.n_folds,
                estimator_factory=self.nuisance_model_factory[name],
                estimator_params=self.nuisance_model_params[name],
                random_state=self.random_state,
            )
            for name in nuisance_model_names
        }
        self._treatment_models: dict[str, _ScikitModel] = {
            name: CrossFitEstimator(
                n_folds=self.n_folds,
                estimator_factory=self.treatment_model_factory[name],
                estimator_params=self.treatment_model_params[name],
                random_state=self.random_state,
            )
            for name in treatment_model_names
        }

    def fit_nuisance(self, X: Matrix, y: Vector, model_kind: str) -> Self:
        """Fit a given nuisance model of a MetaLearner.

        ``y`` represents the objective of the given nuisance model, not necessarily the outcome of the experiment.
        """
        X_filtered = X[self.feature_set[model_kind]] if self.feature_set else X
        self._nuisance_models[model_kind].fit(X_filtered, y)
        return self

    def fit_treatment(self, X: Matrix, y: Vector, model_kind: str) -> Self:
        """Fit the tratment model of a MetaLearner.

        ``y`` represents the objective of the given treatment model, not necessarily the outcome of the experiment.
        """
        X_filtered = X[self.feature_set[model_kind]] if self.feature_set else X
        self._treatment_models[model_kind].fit(X_filtered, y)
        return self

    @abstractmethod
    def fit(self, X: Matrix, y: Vector, w: Vector) -> Self:
        """Fit all models of a MetaLearner."""
        ...

    def predict_nuisance(
        self,
        X: Matrix,
        model_kind: str,
        is_oos: bool,
        oos_method: Optional[OosMethod] = None,
    ) -> np.ndarray:
        """Estimate based on a given nuisance model.

        Importantly, this method needs to implement the subselection of ``X`` based on
        the ``feature_set`` field of ``MetaLearner``.
        """
        X_filtered = X[self.feature_set[model_kind]] if self.feature_set else X
        return self._nuisance_models[model_kind].predict(X_filtered, is_oos, oos_method)

    def predict_treatment(
        self,
        X: Matrix,
        model_kind: str,
        is_oos: bool,
        oos_method: Optional[OosMethod] = None,
    ) -> np.ndarray:
        """Estimate based on a given treatment model.

        Importantly, this method needs to implement the subselection of ``X`` based on
        the ``feature_set`` field of ``MetaLearner``.
        """
        X_filtered = X[self.feature_set[model_kind]] if self.feature_set else X
        return self._treatment_models[model_kind].predict(
            X_filtered, is_oos, oos_method
        )

    @abstractmethod
    def predict(
        self, X: Matrix, is_oos: bool, oos_method: Optional[OosMethod] = None
    ) -> np.ndarray:
        """Estimate the Conditional Average Treatment Effect.

        This method can be identical to predict_treatment but doesn't need to.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        is_regression: bool,
        is_oos: bool,
        oos_method: Optional[OosMethod] = None,
    ) -> dict[str, Union[float, int]]:
        """Evaluate all models contained in a MetaLearner."""
        ...

    @abstractmethod
    def predict_potential_outcomes(
        self, X: Matrix, is_oos: bool, oos_method: Optional[OosMethod] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the vectors of potential outcomes."""
        ...

    @abstractmethod
    def _pseudo_outcome(self, *args, **kwargs) -> Vector:
        """Compute the vector of pseudo outcomes of the respective MetaLearner ."""
        ...
