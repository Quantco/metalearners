# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Collection

import numpy as np
import pandas as pd
import shap

from metalearners._typing import Matrix, _ScikitModel
from metalearners._utils import safe_len, simplify_output_2d
from metalearners.metalearner import Params


def _build_feature_importance_dict(
    feature_importance: np.ndarray, feature_names: Collection[str] | None = None
) -> pd.Series:
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
    return pd.Series(feature_importance, index=feature_names)


# TODO: implement permutation importance?
class Explainer:
    r"""Responsible class for managing all functions related to feature explanation and
    interpretation.

    The ``cate_models`` parameter should be a list of length :math:`n_{variants} -1` containing
    a model for each treatment variant which estimates :math:`\tau_k`. The models should not be a
    :class:`~metalearners.cross_fit_estimator.CrossFitEstimator` rather just a plain ``sklearn``
    ``BaseEstimator``. A suggested option in the case of a :class:`~metalearners.cross_fit_estimator.CrossFitEstimator`
    would be to use their ``_overall_estimator``. These models should already be fitted
    on the data.
    """

    def __init__(
        self,
        cate_models: list[_ScikitModel],
    ):
        self.n_variants = len(cate_models) + 1
        self.cate_models = cate_models

    # The quotes on Explainer are necessary for type hinting as it's not yet defined
    # here. Check https://stackoverflow.com/questions/55320236/does-python-evaluate-type-hinting-of-a-forward-reference
    # At some point evaluation at runtime will be the default and then this won't be needed.
    @classmethod
    def from_estimates(
        cls,
        X: Matrix,
        cate_estimates: np.ndarray,
        cate_model_factory: type[_ScikitModel],
        cate_model_params: Params | None = None,
    ) -> "Explainer":
        r"""Create an ``Explainer`` object from CATE estimates.

        This function will fit a model for each treatment variant with ``X`` as its input
        and the corresponding CATE estimates as its output.

        The ``cate_estimates`` should be the raw outcome of a MetaLearner with 3 dimensions
        and should not be simplified.
        """
        if safe_len(X) != len(cate_estimates) or safe_len(X) == 0:
            raise ValueError(
                "X and cate_estimates should contain the same number of observations "
                "and not be empty."
            )
        if np.any(np.isnan(cate_estimates)) or np.any(np.isinf(cate_estimates)):
            raise ValueError("cate_estimates can not contain any NaN or inf.")

        cate_estimates = simplify_output_2d(
            cate_estimates
        )  # TODO: This does not work for multiclass, do we want to consider it?
        if cate_model_params is None:
            cate_model_params = {}

        n_variants = cate_estimates.shape[1] + 1
        cate_models = [
            cate_model_factory(**cate_model_params).fit(X, cate_estimates[:, tv])
            for tv in range(n_variants - 1)
        ]
        return cls(cate_models)

    def feature_importances(
        self,
        normalize: bool = False,
        feature_names: Collection[str] | None = None,
        sort_values: bool = False,
    ) -> list[pd.Series]:
        r"""Calculates the feature importance for each treatment group.

        If ``normalization = True``, for each treatment variant the feature importances
        are normalized so that they sum to 1.

        ``feature_names`` is optional but in the case it's not passed the names of the
        features will default to ``f"Feature {i}"`` where ``i`` is the corresponding
        feature index.
        """
        feature_importances: list[pd.Series] = []
        for tv in range(self.n_variants - 1):
            if not hasattr(self.cate_models[tv], "feature_importances_"):
                raise ValueError(
                    f"Model used for treatment variant {tv + 1} has no attribute feature_importances_. "
                    "You need to use a model which computes them, e.g. LGBMRegressor."
                )

            variant_feature_importance = self.cate_models[tv].feature_importances_  # type: ignore
            if normalize:
                variant_feature_importance = variant_feature_importance / np.sum(
                    variant_feature_importance
                )
            variant_feature_importance = _build_feature_importance_dict(
                variant_feature_importance, feature_names
            )
            if sort_values:
                variant_feature_importance = variant_feature_importance.sort_values(
                    ascending=False
                )
            feature_importances.append(variant_feature_importance)

        return feature_importances

    def shap_values(
        self,
        X: Matrix,
        shap_explainer_factory: type[shap.Explainer],
        shap_explainer_params: dict | None = None,
    ) -> list[np.ndarray]:
        """Calculates the shap values for each treatment group.

        The parameter ``shap_explainer_factory`` can be used to specify the type of shap
        explainer, for the different options see
        `here <https://shap.readthedocs.io/en/latest/api.html#explainers>`_.
        """
        if shap_explainer_params is None:
            shap_explainer_params = {}
        shap_values = []
        for tv in range(self.n_variants - 1):
            shap_explainer = shap_explainer_factory(
                model=self.cate_models[tv],
                **shap_explainer_params,
            )
            variant_shap_values = shap_explainer.shap_values(X)
            shap_values.append(variant_shap_values)
        return shap_values
