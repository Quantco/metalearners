# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, root_mean_squared_error
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import Matrix, Vector, convert_treatment, supports_categoricals
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import MetaLearner, _ModelSpecifications

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
                cardinality=lambda _: 1,
                predict_method=lambda ml: (
                    "predict_proba" if ml.is_classification else "predict"
                ),
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

    def _validate_params(self, feature_set, **kwargs):
        if feature_set is not None:
            # For SLearner it does not make sense to allow feature set as we only have one model
            # and having it would bring problems when using fit_nuisance and predict_nuisance
            # as we need to add the treatment column.
            raise ValueError("SLearner does not support feature set definition.")

    def fit(self, X: Matrix, y: Vector, w: Vector) -> Self:
        self._validate_treatment(w)
        self._validate_outcome(y)
        self._fitted_treatments = convert_treatment(w)

        mock_model = self.nuisance_model_factory[_BASE_MODEL](
            **self.nuisance_model_params[_BASE_MODEL]
        )
        self._supports_categoricals = supports_categoricals(mock_model)
        X_with_w = _append_treatment_to_covariates(
            X, w, self._supports_categoricals, self.n_variants
        )

        self.fit_nuisance(
            X=X_with_w,
            y=y,
            model_kind=_BASE_MODEL,
            model_ord=0,
        )
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

        if self.n_variants == 2:
            return (
                conditional_average_outcomes[:, 1] - conditional_average_outcomes[:, 0]
            )
        else:
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
    ) -> dict[str, float | int]:
        # TODO: Parameterize evaluation approaches.
        X_with_w = _append_treatment_to_covariates(
            X, w, self._supports_categoricals, self.n_variants
        )
        y_pred = self.predict_nuisance(
            X=X_with_w, model_kind=_BASE_MODEL, model_ord=0, is_oos=is_oos
        )
        if self.is_classification:
            return {"cross_entropy": log_loss(y, y_pred)}
        return {"rmse": root_mean_squared_error(y, y_pred)}

    def predict_conditional_average_outcomes(
        self, X: Matrix, is_oos: bool, oos_method: OosMethod = OVERALL
    ) -> np.ndarray:
        r"""Predict the vectors of conditional average outcomes.

        These are defined as :math:`\mathbb{E}[Y_i(w) | X]` for each treatment variant
        :math:`w`.

        The returned matrix is of shape :math:`(n_{obs}, n_{variants})` if
        there's only one output, i.e. a regression problem, or :math:`(n_{obs},
        n_{variants}, n_{classes})` if it's a classification problem.

        If ``is_oos``, an acronym for 'is out of sample' is ``False``,
        the estimates will stem from cross-fitting. Otherwise,
        various approaches exist, specified via ``oos_method``.
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

        return np.stack(conditional_average_outcomes_list, axis=1)
