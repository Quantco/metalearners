# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
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
        else:
            return np.concatenate([X, w_dummies], axis=1)

    if isinstance(X, np.ndarray):
        # This is necessary as each model works differently with categoricals,
        # in some you need to specify them on instantiation while some others on
        # fitting. This solutions converts it to a pd.DataFrame as most of the models
        # have some "automatic" detection of categorical features based on pandas
        # dtypes. Theoretically it would be possible to get around this conversion
        # but it would require loads of model specific code.
        warnings.warn(
            "Converting the input covariates X from np.ndarray to a "
            f"pd.DataFrame as the {_BASE_MODEL} supports categorical variables."
        )
        X = pd.DataFrame(X, copy=True)

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
        r"""Predict the vectors of conditional average outcomes.

        These are defined as :math:`\mathbb{E}[Y_i(w) | X]` for each treatment variant
        :math:`w`.

        If ``is_oos``, an acronym for 'is out of sample' is ``False``,
        the estimates will stem from cross-fitting. Otherwise,
        various approaches exist, specified via ``oos_method``.

        The returned ndarray is of shape:

        * :math:`(n_{obs}, n_{variants}, 1)` if the outcome is a scalar, i.e. in case
          of a regression problem.

        * :math:`(n_{obs}, n_{variants}, n_{classes})` if the outcome is a class,
          i.e. in case of a classification problem.
        """
        n_obs = len(X)
        conditional_average_outcomes_list = []

        # The idea behind using is_oos = True for in sample predictions is the following:
        # Assuming observation i has received variant v then the model has been trained
        # on row (X_i, v), therefore when predicting the conditional average outcome for
        # variant v we have to use cross fitting to avoid prediciting on an identical row
        # which the model has been trained on. (This happens either with overall, mean
        # or median as some of the models would be trained with this row). On the other
        # hand, when predicting the conditional average outcome for variant v' != v,
        # the model has never seen the row (X_i, v'), so we can use it as it was out of
        # sample.
        # This can bring some issues where the cross fitted predictions are based on models
        # which have been trained with a smaller dataset (K-1 folds) than the overall
        # model and this may produce some different distributions in the outputs, for this
        # it may make sense to restrict the oos_method to mean or median when is_oos = False,
        # although further investigation is needed.
        if not is_oos:
            X_with_w = _append_treatment_to_covariates(
                X,
                self._fitted_treatments,
                self._supports_categoricals,
                self.n_variants,
            )
            in_sample_pred = self.predict_nuisance(
                X=X_with_w, model_kind=_BASE_MODEL, model_ord=0, is_oos=False
            )

        for v in range(self.n_variants):
            w = np.array([v] * n_obs)
            X_with_w = _append_treatment_to_covariates(
                X, w, self._supports_categoricals, self.n_variants
            )
            variant_predictions = self.predict_nuisance(
                X=X_with_w,
                model_kind=_BASE_MODEL,
                model_ord=0,
                is_oos=True,
                oos_method=oos_method,
            )
            if not is_oos:
                variant_predictions[self._fitted_treatments == v] = in_sample_pred[
                    self._fitted_treatments == v
                ]

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
