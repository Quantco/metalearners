# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Mapping, Sequence

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from typing_extensions import Self

from metalearners._narwhals_utils import (
    infer_native_namespace,
    nw_to_dummies,
    stringify_column_names,
    vector_to_nw,
)
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
    adapt_treatment_dtypes,
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

_TREATMENT = "treatment"


def _np_to_dummies(
    x: np.ndarray, categories: Sequence, drop_first: bool = True
) -> np.ndarray:
    """Turn a vector into a matrix with dummies.

    This operation is also referred to as one-hot-encoding.

    The resulting columns will correspond to the order of values in ``categories``.
    """

    if x.ndim != 1:
        raise ValueError("Can only convert 1-d array to dummies.")

    if len(categories) < 2:
        raise ValueError(
            "categories to be used for nw_to_dummies must have at least two "
            "distinct values."
        )

    if set(categories) < set(np.unique(x)):
        raise ValueError("We observed a value which isn't part of the categories.")

    dummy_matrix = np.eye(len(categories), dtype="int8")[x]
    if drop_first:
        return dummy_matrix[:, 1:]
    return dummy_matrix


def _append_treatment_to_covariates_with_one_hot_encoding(
    X: Matrix,
    w: Vector,
    categories: Sequence,
) -> Matrix:
    if nw.dependencies.is_into_dataframe(X):
        X_nw = nw.from_native(X, eager_only=True)
        # Some models (e.g. sklearn's LinearRegression) raise an error if some column
        # names are integers and some strings.
        X_nw = stringify_column_names(X_nw)

        # Note that it could be the case that w is a np.ndarray object
        # even if X is a dataframe. Therefore we have a conversion
        # with a case distinction.
        w_nw = vector_to_nw(w, native_namespace=infer_native_namespace(X_nw)).rename(
            _TREATMENT
        )
        w_dummies_nw = nw_to_dummies(
            w_nw, categories, column_name=_TREATMENT, drop_first=True
        )

        X_with_w_nw = nw.concat([X_nw, w_dummies_nw], how="horizontal")

        return X_with_w_nw.to_native()

    if isinstance(X, csr_matrix):
        if not isinstance(w, np.ndarray):
            raise TypeError(
                "When covariates X are provided as a scipy csr_matrix, treatments "
                f"should be provided as a np.ndarray, not {type(w)}."
            )
        w_dummies = _np_to_dummies(w, categories)
        return hstack((X, w_dummies), format="csr")

    if isinstance(X, np.ndarray):
        if not isinstance(w, np.ndarray):
            raise TypeError(
                "When covariates X are provided as a numpy.ndarray object, treatments "
                f"should be provided as a np.ndarray, not {type(w)}."
            )
        w_dummies = _np_to_dummies(w, categories)
        return np.concatenate([X, w_dummies], axis=1)

    raise TypeError(
        "Cannot append treatments to covariates X if covariates are of type"
        + str(type(X))
    )


def _append_treatment_to_covariates_with_categorical(
    X: Matrix,
    w: Vector,
    categories: Sequence,
) -> Matrix:
    """Append treatment column as categorical to covariates.

    This is useful for models which automatically detect categoricals based on dtypes of
    the columns.

    For reference, see e.g.
    https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support
    """

    def convert_matrix_to_nw(X: Matrix) -> nw.DataFrame:
        if isinstance(X, np.ndarray):
            warnings.warn(
                "Converting the input covariates X from np.ndarray to a "
                f"DataFrame as {_BASE_MODEL} supports categorical variables."
            )
            # By default, the columns are named column_i by narwhals.
            # The pandas behavior we are used to relies on naming them i.
            # We imitate the pandas behavior here.
            column_names = [str(i) for i in range(X.shape[1])]
            return nw.from_numpy(X, native_namespace=pd, schema=column_names)

        if isinstance(X, csr_matrix):
            warnings.warn(
                "Converting the input covariates X from a scipy csr_matrix to a "
                f"pd.DataFrame as {_BASE_MODEL} supports categorical variables."
            )
            return nw.from_native(pd.DataFrame.sparse.from_spmatrix(X), eager_only=True)
        return nw.from_native(X, eager_only=True)

    X_nw = convert_matrix_to_nw(X)
    X_nw = stringify_column_names(X_nw)

    w_nw = vector_to_nw(w, native_namespace=infer_native_namespace(X_nw)).rename(
        _TREATMENT
    )
    # narwhal's concat expects two DataFrames -- in contrast to mixing DataFrames
    # and Series.
    w_nw_categorical = (
        w_nw.cast(nw.String).cast(nw.Categorical).rename(_TREATMENT).to_frame()
    )

    X_with_w_nw = nw.concat([X_nw, w_nw_categorical], how="horizontal")  # type: ignore
    X_result = X_with_w_nw.to_native()

    # Ideally, we would like to do the analogous operation for polars, too. See
    # https://github.com/pola-rs/polars/issues/21337
    if isinstance(X_result, pd.DataFrame):
        X_result[_TREATMENT] = X_result[_TREATMENT].cat.set_categories(categories)

    return X_result


def _append_treatment_to_covariates(
    X: Matrix, w: Vector, supports_categoricals: bool, n_variants: int
) -> Matrix:
    """Append treatment column to covariates.

    If `support_categoricals` is `False`:

    * the returned result will be of the same type as ``X``
    * the treatment is appended with one-hot-encoding, where the
      treatment value column is omitted

    If `support_categoricals` is `True`:

    * the returned result will be a `pandas.DataFrame`, except
      if `X` was a `polars.DataFrame`; in the latter case
      the returned result will be a `polars.DataFrame`, too
    * the treatment is appended with a categorical column
    """
    w = adapt_treatment_dtypes(w)

    if hasattr(X, "columns") and _TREATMENT in X.columns:
        raise ValueError(f"{_TREATMENT} cannot be a column name in X.")

    if isinstance(X, pd.DataFrame):
        # This is needed in case the index is not 0-based.
        X = X.reset_index(drop=True)

    if isinstance(w, pd.Series):
        w = w.reset_index(drop=True)

    categories = list(range(n_variants))

    if not supports_categoricals:
        return _append_treatment_to_covariates_with_one_hot_encoding(
            X=X,
            w=w,
            categories=categories,
        )
    return _append_treatment_to_covariates_with_categorical(
        X=X,
        w=w,
        categories=categories,
    )


class SLearner(MetaLearner):
    """S-Learner for CATE estimation as described by [Kuenzel et al (2019)](https://arxiv.org/pdf/1706.03461.pdf)."""

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
        self._fitted_treatments = adapt_treatment_dtypes(w)

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

        return (
            conditional_average_outcomes[:, 1:] - (conditional_average_outcomes[:, [0]])
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
