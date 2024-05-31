# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
from sklearn.metrics import log_loss, root_mean_squared_error
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import Matrix, Vector, index_matrix
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import (
    NUISANCE,
    VARIANT_OUTCOME_MODEL,
    MetaLearner,
    _ConditionalAverageOutcomeMetaLearner,
    _ModelSpecifications,
)


class TLearner(_ConditionalAverageOutcomeMetaLearner):
    """T-Learner for CATE estimation as described by `Kuenzel et al (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_."""

    # TODO: Parametrize instantiation of the TLearner as to add an optional
    # second-stage model regularizing the treatment effects, rather than
    # just the respective outcomes individually.

    @classmethod
    def nuisance_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            VARIANT_OUTCOME_MODEL: _ModelSpecifications(
                cardinality=MetaLearner._get_n_variants,
                predict_method=MetaLearner._outcome_predict_method,
            ),
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

    def fit(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        n_jobs_cross_fitting: int | None = None,
        fit_params: dict | None = None,
    ) -> Self:
        self._validate_treatment(w)
        self._validate_outcome(y)

        self._treatment_variants_indices = []

        for v in range(self.n_variants):
            self._treatment_variants_indices.append(w == v)

        qualified_fit_params = self._qualified_fit_params(fit_params)

        # TODO: Consider multiprocessing
        for treatment_variant in range(self.n_variants):
            self.fit_nuisance(
                X=index_matrix(X, self._treatment_variants_indices[treatment_variant]),
                y=y[self._treatment_variants_indices[treatment_variant]],
                model_kind=VARIANT_OUTCOME_MODEL,
                model_ord=treatment_variant,
                n_jobs_cross_fitting=n_jobs_cross_fitting,
                fit_params=qualified_fit_params[NUISANCE][VARIANT_OUTCOME_MODEL],
            )

        return self

    def predict(
        self,
        X,
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
    ) -> dict[str, float | int]:
        # TODO: Parametrize evaluation approaches.
        conditional_average_outcomes = self.predict_conditional_average_outcomes(
            X=X, is_oos=is_oos, oos_method=oos_method
        )
        evaluation_metrics = {}
        for treatment_variant in range(self.n_variants):
            prefix = f"variant_{treatment_variant}"
            variant_outcomes = conditional_average_outcomes[:, treatment_variant]
            if self.is_classification:
                evaluation_metrics[f"{prefix}_cross_entropy"] = log_loss(
                    y[w == treatment_variant], variant_outcomes[w == treatment_variant]
                )
            else:
                evaluation_metrics[f"{prefix}_rmse"] = root_mean_squared_error(
                    y[w == treatment_variant], variant_outcomes[w == treatment_variant]
                )
        return evaluation_metrics
