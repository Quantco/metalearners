# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from abc import ABC, abstractmethod
from collections.abc import Collection

import numpy as np
from typing_extensions import Self

from metalearners._utils import Matrix, Vector, _ScikitModel, validate_number_positive
from metalearners.cross_fit_estimator import (
    OVERALL,
    CrossFitEstimator,
    OosMethod,
    PredictMethod,
)

Params = dict[str, int | float | str]
Features = Collection[str] | Collection[int]
ModelFactory = type[_ScikitModel] | dict[str, type[_ScikitModel]]
PROPENSITY_MODEL = "propensity_model"


def _initialize_model_dict(argument, expected_names: Collection[str]) -> dict:
    if isinstance(argument, dict) and set(argument.keys()) == set(expected_names):
        return argument
    return {name: argument for name in expected_names}


def _validate_nuisance_predict_methods(factual: set[str], expected: set[str]) -> None:
    if len(excess := (factual - expected)) > 0:
        raise ValueError(
            "Mapping from nuisance model kind to predict method contains excess keys:",
            str(excess),
        )
    if len(lack := (expected - factual)) > 0:
        raise ValueError(
            "Mapping from nuisance model kind to predict method lacks keys:",
            str(lack),
        )


def _combine_propensity_and_nuisance_specs(
    propensity_specs, nuisance_specs, nuisance_model_names: set[str]
) -> dict:
    if PROPENSITY_MODEL in nuisance_model_names:
        non_propensity_nuisance_model_names = nuisance_model_names - {PROPENSITY_MODEL}
        non_propensity_model_dict = _initialize_model_dict(
            nuisance_specs, non_propensity_nuisance_model_names
        )
        return non_propensity_model_dict | {PROPENSITY_MODEL: propensity_specs}

    return _initialize_model_dict(nuisance_specs, nuisance_model_names)


class MetaLearner(ABC):

    @classmethod
    @abstractmethod
    def nuisance_model_names(cls) -> set[str]: ...

    @classmethod
    @abstractmethod
    def treatment_model_names(cls) -> set[str]: ...

    def _validate_params(self, **kwargs): ...

    @classmethod
    @abstractmethod
    def _supports_multi_treatment(cls) -> bool: ...

    @classmethod
    @abstractmethod
    def _supports_multi_class(cls) -> bool: ...

    @classmethod
    def _check_treatment(cls, w: Vector) -> None:
        if (
            (n_variants := len(np.unique(w))) > 2
        ) and not cls._supports_multi_treatment():
            raise NotImplementedError(
                f"Current implementation of {cls.__name__} only supports binary "
                f"treatment variants. Yet, we found {n_variants} different "
                "variants."
            )
        # TODO: add support for different encoding of treatment variants (str, not consecutive ints...)
        if set(np.unique(w)) != set(range(n_variants)):
            raise ValueError(
                "Treatment variant should be encoded with values "
                f"{{0...{n_variants -1}}} and all variants should be present. "
                f"Yet we found the values {set(np.unique(w))}."
            )

    def _check_outcome(self, y: Vector) -> None:
        if (
            self.is_classification
            and not self._supports_multi_class()
            and len(np.unique(y)) > 2
        ):
            raise ValueError(
                f"{self.__class__.__name__} does not support multiclass classification."
                f" Yet we found {len(np.unique(y))} classes."
            )

    @abstractmethod
    def _validate_models(self) -> None:
        """Validate that the models are of the correct type (classifier or regressor)"""
        ...

    def __init__(
        self,
        nuisance_model_factory: ModelFactory,
        is_classification: bool,
        # TODO: Consider whether we can make this not a state of the MetaLearner
        # but rather just a parameter of a predict call.
        treatment_model_factory: ModelFactory | None = None,
        propensity_model_factory: type[_ScikitModel] | None = None,
        nuisance_model_params: Params | dict[str, Params] | None = None,
        treatment_model_params: Params | dict[str, Params] | None = None,
        propensity_model_params: Params | None = None,
        feature_set: Features | dict[str, Features] | None = None,
        # TODO: Consider implementing selection of number of folds for various estimators.
        n_folds: int = 10,
        random_state: int | None = None,
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
        self._validate_params(
            nuisance_model_factory=nuisance_model_factory,
            treatment_model_factory=treatment_model_factory,
            propensity_model_factory=propensity_model_factory,
            is_classification=is_classification,
            nuisance_model_params=nuisance_model_params,
            treatment_model_params=treatment_model_params,
            propensity_model_params=propensity_model_params,
            feature_set=feature_set,
            n_folds=n_folds,
            random_state=random_state,
        )

        nuisance_model_names = self.__class__.nuisance_model_names()
        treatment_model_names = self.__class__.treatment_model_names()

        if PROPENSITY_MODEL in treatment_model_names:
            raise ValueError(
                f"{PROPENSITY_MODEL} can't be used as a treatment model name"
            )
        if (
            isinstance(nuisance_model_factory, dict)
            and PROPENSITY_MODEL in nuisance_model_factory.keys()
        ):
            raise ValueError(
                "Propensity model factory should be defined using propensity_model_factory "
                "and not nuisance_model_factory."
            )
        if (
            isinstance(nuisance_model_params, dict)
            and PROPENSITY_MODEL in nuisance_model_params.keys()
        ):
            raise ValueError(
                "Propensity model params should be defined using propensity_model_params "
                "and not nuisance_model_params."
            )
        if (
            PROPENSITY_MODEL in nuisance_model_names
            and propensity_model_factory is None
        ):
            raise ValueError(
                f"propensity_model_factory needs to be defined as the {self.__class__.__name__}"
                " has a propensity model."
            )

        self.is_classification = is_classification

        self.nuisance_model_factory = _combine_propensity_and_nuisance_specs(
            propensity_model_factory, nuisance_model_factory, nuisance_model_names
        )
        if nuisance_model_params is None:
            nuisance_model_params = {}  # type: ignore
        if propensity_model_params is None:
            propensity_model_params = {}
        self.nuisance_model_params = _combine_propensity_and_nuisance_specs(
            propensity_model_params, nuisance_model_params, nuisance_model_names
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
                feature_set,
                nuisance_model_names | treatment_model_names,
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

        _validate_nuisance_predict_methods(
            factual=set(self._nuisance_predict_methods.keys()),
            expected=set(self.__class__.nuisance_model_names()),
        )

        self._validate_models()

    @property
    @abstractmethod
    def _nuisance_predict_methods(self) -> dict[str, PredictMethod]:
        """Map each nuisance ``model_kind`` to a ``PredictMethod``."""
        ...

    def _nuisance_tensors(self, n_obs: int) -> dict[str, np.ndarray]:
        # TODO: This type ignoring hints at a design flaw.
        # We should fix this.
        def dimension(n_obs, model_kind, predict_method):
            if (
                n_outputs := self._nuisance_models[model_kind]._n_outputs(  # type: ignore
                    predict_method
                )
            ) > 1:
                return (n_obs, n_outputs)
            return (n_obs,)

        return {
            model_kind: np.zeros(dimension(n_obs, model_kind, predict_method))
            for (model_kind, predict_method) in self._nuisance_predict_methods.items()
        }

    def fit_nuisance(
        self, X: Matrix, y: Vector, model_kind: str, fit_params: dict | None = None
    ) -> Self:
        """Fit a given nuisance model of a MetaLearner.

        ``y`` represents the objective of the given nuisance model, not necessarily the outcome of the experiment.
        """
        X_filtered = X[self.feature_set[model_kind]] if self.feature_set else X
        self._nuisance_models[model_kind].fit(X_filtered, y, fit_params=fit_params)
        return self

    def fit_treatment(
        self, X: Matrix, y: Vector, model_kind: str, fit_params: dict | None = None
    ) -> Self:
        """Fit the treatment model of a MetaLearner.

        ``y`` represents the objective of the given treatment model, not necessarily the outcome of the experiment.
        """
        X_filtered = X[self.feature_set[model_kind]] if self.feature_set else X
        self._treatment_models[model_kind].fit(X_filtered, y, fit_params=fit_params)
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
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        """Estimate based on a given nuisance model.

        Importantly, this method needs to implement the subselection of ``X`` based on
        the ``feature_set`` field of ``MetaLearner``.
        """
        X_filtered = X[self.feature_set[model_kind]] if self.feature_set else X
        predict_method_name = self._nuisance_predict_methods[model_kind]
        predict_method = getattr(self._nuisance_models[model_kind], predict_method_name)
        return predict_method(X_filtered, is_oos, oos_method)

    def predict_treatment(
        self,
        X: Matrix,
        model_kind: str,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
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
        self,
        X: Matrix,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
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
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> dict[str, float | int]:
        """Evaluate all models contained in a MetaLearner."""
        ...
