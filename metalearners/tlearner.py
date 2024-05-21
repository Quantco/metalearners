# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
from sklearn.metrics import log_loss, root_mean_squared_error
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import Matrix, Vector, index_matrix
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import (
    CONTROL_OUTCOME_MODEL,
    TREATMENT_OUTCOME_MODEL,
    ConditionalAverageOutcomeMetaLearner,
    _ModelSpecifications,
)


class TLearner(ConditionalAverageOutcomeMetaLearner):
    """T-Learner for CATE estimation as described by `Kuenzel et al (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_.

    Importantly, this implementation currently only supports binary treatment variants.
    """

    # TODO: Parametrize instantiation of the TLearner as to add an optional
    # second-stage model regularizing the treatment effects, rather than
    # just the respective outcomes individually.

    @classmethod
    def nuisance_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        """Return the names of all first-stage models."""

        return {
            TREATMENT_OUTCOME_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda ml: (
                    "predict_proba" if ml.is_classification else "predict"
                ),
            ),
            CONTROL_OUTCOME_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda ml: (
                    "predict_proba" if ml.is_classification else "predict"
                ),
            ),
        }

    @classmethod
    def treatment_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        """Return the names of all second-stage models."""
        return dict()

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return False

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return True

    def fit(self, X: Matrix, y: Vector, w: Vector) -> Self:
        """Fit all models of the T-Learner."""
        self._validate_treatment(w)
        self._validate_outcome(y)
        self._treatment_indices = w == 1
        self._control_indices = w == 0
        # TODO: Consider multiprocessing
        self.fit_nuisance(
            X=index_matrix(X, self._treatment_indices),
            y=y[self._treatment_indices],
            model_kind=TREATMENT_OUTCOME_MODEL,
            model_ord=0,
        )
        self.fit_nuisance(
            X=index_matrix(X, self._control_indices),
            y=y[self._control_indices],
            model_kind=CONTROL_OUTCOME_MODEL,
            model_ord=0,
        )
        return self

    def predict(
        self,
        X,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        """Estimate the Conditional Average Treatment Effect.

        If ``is_oos``, an acronym for 'is out of sample' is ``False``,
        the estimates will stem from cross-fitting. Otherwise,
        various approaches exist, specified via ``oos_method``.
        """
        conditional_average_outcomes = self.predict_conditional_average_outcomes(
            X=X, is_oos=is_oos, oos_method=oos_method
        )
        return conditional_average_outcomes[:, 1] - conditional_average_outcomes[:, 0]

    def evaluate(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> dict[str, float | int]:
        """Evaluate all models contained in the T-Learner."""
        # TODO: Parametrize evaluation approaches.
        conditional_average_outcomes = self.predict_conditional_average_outcomes(
            X=X, is_oos=is_oos, oos_method=oos_method
        )
        control_outcomes = conditional_average_outcomes[:, 0]
        treatment_outcomes = conditional_average_outcomes[:, 1]
        if not self.is_classification:
            return {
                "treatment_rmse": root_mean_squared_error(
                    y[w == 1], treatment_outcomes[w == 1]
                ),
                "control_rmse": root_mean_squared_error(
                    y[w == 0], control_outcomes[w == 0]
                ),
            }
        return {
            "treatment_cross_entropy": log_loss(
                y[w == 1], treatment_outcomes[w == 1][:, 1]
            ),
            "control_cross_entropy": log_loss(
                y[w == 0], control_outcomes[w == 0][:, 1]
            ),
        }
