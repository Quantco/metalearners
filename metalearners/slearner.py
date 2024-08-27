# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from typing_extensions import Self

from metalearners._typing import (
    Features,
    Matrix,
    ModelFactory,
    OosMethod,
    Params,
    Scoring,
    Vector,
    _ScikitModel,
)
from metalearners._utils import (
    convert_treatment,
    get_one,
    safe_len,
    supports_categoricals,
)
from metalearners.cross_fit_estimator import OVERALL, CrossFitEstimator
from metalearners.metalearner import (
    NUISANCE,
    MetaLearner,
    _evaluate_model_kind,
    _ModelSpecifications,
)

_BASE_MODEL = "base_model"


def _append_treatment_to_covariates(
    X: Matrix, w: Vector, supports_categoricals: bool, n_variants: int
) -> Matrix:
    """Appends treatment columns to covariates and one-hot-encode if necessary."""
    w = convert_treatment(w)
    w = pd.Series(w, dtype="category").cat.set_categories(list(range(n_variants)))

    if isinstance(X, pd.DataFrame) and "treatment" in X.columns:
        raise ValueError('"treatment" cannot be a column name in X.')

    if isinstance(X, pd.DataFrame):
        # This is needed in case the index is not 0 based
        X = X.reset_index(drop=True)

    if not supports_categoricals:
        w_dummies = pd.get_dummies(w, prefix="treatment", dtype=int, drop_first=True)
        if isinstance(X, pd.DataFrame):
            X_with_w = pd.concat([X, w_dummies], axis=1)
            # This is because some models (LinearRegression) raise an error if some column
            # names are integers and some strings.
            X_with_w.columns = X_with_w.columns.astype(str)
            return X_with_w
        elif isinstance(X, csr_matrix):
            return hstack((X, w_dummies), format="csr")
        else:
            return np.concatenate([X, w_dummies], axis=1)

    # This is necessary as each model works differently with categoricals,
    # in some you need to specify them on instantiation while some others on
    # fitting. This solutions converts it to a pd.DataFrame as most of the models
    # have some "automatic" detection of categorical features based on pandas
    # dtypes. Theoretically it would be possible to get around this conversion
    # but it would require loads of model specific code.
    if isinstance(X, np.ndarray):
        warnings.warn(
            "Converting the input covariates X from np.ndarray to a "
            f"pd.DataFrame as the {_BASE_MODEL} supports categorical variables."
        )
        X = pd.DataFrame(X, copy=True)
    elif isinstance(X, csr_matrix):
        warnings.warn(
            "Converting the input covariates X from a scipy csr_matrix to a "
            f"pd.DataFrame as the {_BASE_MODEL} supports categorical variables."
        )
        X = pd.DataFrame.sparse.from_spmatrix(X)

    X_with_w = pd.concat([X, pd.Series(w, dtype="category", name="treatment")], axis=1)
    X_with_w.columns = X_with_w.columns.astype(str)

    return X_with_w


class SLearner(MetaLearner):
    """S-Learner for CATE estimation as described by `Kuenzel et al (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_."""

    @classmethod
    def nuisance_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            _BASE_MODEL: _ModelSpecifications(
                cardinality=get_one,
                predict_method=MetaLearner._outcome_predict_method,
            )
        }

    @classmethod
    def treatment_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return dict()

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return True

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return True

    def __init__(
        self,
        is_classification: bool,
        n_variants: int,
        nuisance_model_factory: ModelFactory | None = None,
        treatment_model_factory: ModelFactory | None = None,
        propensity_model_factory: type[_ScikitModel] | None = None,
        nuisance_model_params: Params | dict[str, Params] | None = None,
        treatment_model_params: Params | dict[str, Params] | None = None,
        propensity_model_params: Params | None = None,
        fitted_nuisance_models: dict[str, list[CrossFitEstimator]] | None = None,
        fitted_propensity_model: CrossFitEstimator | None = None,
        feature_set: Features | dict[str, Features] | None = None,
        n_folds: int | dict[str, int] = 10,
        random_state: int | None = None,
    ):
        if feature_set is not None:
            # For SLearner it does not make sense to allow feature set as we only have one model
            # and having it would bring problems when using fit_nuisance and predict_nuisance
            # as we need to add the treatment column.
            warnings.warn(
                "Base-model specific feature_sets were provided to S-Learner. "
                "These will be ignored and all available features will be used instead."
            )
        super().__init__(
            is_classification=is_classification,
            n_variants=n_variants,
            nuisance_model_factory=nuisance_model_factory,
            treatment_model_factory=treatment_model_factory,
            propensity_model_factory=propensity_model_factory,
            nuisance_model_params=nuisance_model_params,
            treatment_model_params=treatment_model_params,
            propensity_model_params=propensity_model_params,
            fitted_nuisance_models=fitted_nuisance_models,
            fitted_propensity_model=fitted_propensity_model,
            feature_set=None,
            n_folds=n_folds,
            random_state=random_state,
        )

    def fit_all_nuisance(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        n_jobs_cross_fitting: int | None = None,
        fit_params: dict | None = None,
        synchronize_cross_fitting: bool = True,
        n_jobs_base_learners: int | None = None,
    ) -> Self:
        self._validate_treatment(w)
        self._validate_outcome(y, w)
        self._fitted_treatments = convert_treatment(w)

        mock_model = self.nuisance_model_factory[_BASE_MODEL](
            **self.nuisance_model_params[_BASE_MODEL]
        )
        self._supports_categoricals = supports_categoricals(mock_model)
        X_with_w = _append_treatment_to_covariates(
            X, w, self._supports_categoricals, self.n_variants
        )

        qualified_fit_params = self._qualified_fit_params(fit_params)

        self.fit_nuisance(
            X=X_with_w,
            y=y,
            model_kind=_BASE_MODEL,
            model_ord=0,
            n_jobs_cross_fitting=n_jobs_cross_fitting,
            fit_params=qualified_fit_params[NUISANCE][_BASE_MODEL],
        )
        return self

    def fit_all_treatment(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        n_jobs_cross_fitting: int | None = None,
        fit_params: dict | None = None,
        synchronize_cross_fitting: bool = True,
        n_jobs_base_learners: int | None = None,
    ) -> Self:
        return self

    def predict(
        self,
        X: Matrix,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        conditional_average_outcomes = self.predict_conditional_average_outcomes(
            X=X, is_oos=is_oos, oos_method=oos_method
        )

        return conditional_average_outcomes[:, 1:] - (
            conditional_average_outcomes[:, [0]]
        )

    def evaluate(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
        scoring: Scoring | None = None,
    ) -> dict[str, float]:
        safe_scoring = self._scoring(scoring)

        X_with_w = _append_treatment_to_covariates(
            X, w, self._supports_categoricals, self.n_variants
        )
        return _evaluate_model_kind(
            cfes=self._nuisance_models[_BASE_MODEL],
            Xs=[X_with_w],
            ys=[y],
            scorers=safe_scoring[_BASE_MODEL],
            model_kind=_BASE_MODEL,
            is_oos=is_oos,
            oos_method=oos_method,
            is_treatment_model=False,
            feature_set=self.feature_set[_BASE_MODEL],
        )

    def predict_conditional_average_outcomes(
        self, X: Matrix, is_oos: bool, oos_method: OosMethod = OVERALL
    ) -> np.ndarray:
        n_obs = safe_len(X)
        conditional_average_outcomes_list = []

        for treatment_variant in range(self.n_variants):
            w = np.array([treatment_variant] * n_obs)
            X_with_w = _append_treatment_to_covariates(
                X, w, self._supports_categoricals, self.n_variants
            )
            variant_predictions = self.predict_nuisance(
                X=X_with_w,
                model_kind=_BASE_MODEL,
                model_ord=0,
                is_oos=is_oos,
                oos_method=oos_method,
            )

            conditional_average_outcomes_list.append(variant_predictions)

        return np.stack(conditional_average_outcomes_list, axis=1).reshape(
            n_obs, self.n_variants, -1
        )

    @classmethod
    def _necessary_onnx_models(cls) -> dict[str, list[_ScikitModel]]:
        raise ValueError(
            "The SLearner does not implement this method. Please refer to comment in the tutorial."
        )

    def _build_onnx(self, models: Mapping[str, Sequence], output_name: str = "tau"):
        raise ValueError(
            "The SLearner does not implement this method. Please refer to comment in the tutorial."
        )
