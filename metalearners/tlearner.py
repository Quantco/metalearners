# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
from sklearn.metrics import log_loss, root_mean_squared_error
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import Matrix, Vector, index_matrix
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import (
    MetaLearner,
    _ModelSpecifications,
)

_TREATMENT_MODEL = "treatment_model"
_CONTROL_MODEL = "control_model"


class TLearner(MetaLearner):
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
            _TREATMENT_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda ml: (
                    "predict_proba" if ml.is_classification else "predict"
                ),
            ),
            _CONTROL_MODEL: _ModelSpecifications(
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
        self._check_treatment(w)
        self._check_outcome(y)
        self._treatment_indices = w == 1
        self._control_indices = w == 0
        # TODO: Consider multiprocessing
        self.fit_nuisance(
            X=index_matrix(X, self._treatment_indices),
            y=y[self._treatment_indices],
            model_kind=_TREATMENT_MODEL,
            model_ord=0,
        )
        self.fit_nuisance(
            X=index_matrix(X, self._control_indices),
            y=y[self._control_indices],
            model_kind=_CONTROL_MODEL,
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

    def predict_conditional_average_outcomes(
        self, X: Matrix, is_oos: bool, oos_method: OosMethod = OVERALL
    ) -> np.ndarray:
        """Predict the vectors of conditional average outcomes.

        The returned matrix should be of shape :math:`(n_{obs}, n_{variants})` if
        there's only one output, i.e. a regression problem, or :math:`(n_{obs},
        n_{variants}, n_{classes})` if it's a classification problem.

        If ``is_oos``, an acronym for 'is out of sample' is ``False``,
        the estimates will stem from cross-fitting. Otherwise,
        various approaches exist, specified via ``oos_method``.
        """
        # TODO: Consider multiprocessing
        if is_oos:
            treatment_outcomes = self.predict_nuisance(
                X=X,
                model_kind=_TREATMENT_MODEL,
                model_ord=0,
                is_oos=is_oos,
                oos_method=oos_method,
            )
            control_outcomes = self.predict_nuisance(
                X=X,
                model_kind=_CONTROL_MODEL,
                model_ord=0,
                is_oos=is_oos,
                oos_method=oos_method,
            )
        else:
            treatment_outcomes_treated = self.predict_nuisance(
                X=X[self._treatment_indices],
                model_kind=_TREATMENT_MODEL,
                model_ord=0,
                is_oos=False,
            )
            control_outcomes_treated = self.predict_nuisance(
                X=X[self._treatment_indices],
                model_kind=_CONTROL_MODEL,
                model_ord=0,
                is_oos=True,
                oos_method=oos_method,
            )

            treatment_outcomes_control = self.predict_nuisance(
                X=X[self._control_indices],
                model_kind=_TREATMENT_MODEL,
                model_ord=0,
                is_oos=True,
                oos_method=oos_method,
            )
            control_outcomes_control = self.predict_nuisance(
                X=X[self._control_indices],
                model_kind=_CONTROL_MODEL,
                model_ord=0,
                is_oos=False,
            )

            nuisance_tensors = self._nuisance_tensors(len(X))

            treatment_outcomes = nuisance_tensors[_TREATMENT_MODEL][0]
            control_outcomes = nuisance_tensors[_CONTROL_MODEL][0]

            treatment_outcomes[self._control_indices] = treatment_outcomes_control
            treatment_outcomes[self._treatment_indices] = treatment_outcomes_treated
            control_outcomes[self._control_indices] = control_outcomes_control
            control_outcomes[self._treatment_indices] = control_outcomes_treated

        return np.stack([control_outcomes, treatment_outcomes], axis=1)

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
        effect_outcomes = conditional_average_outcomes[:, 0]
        treatment_outcomes = conditional_average_outcomes[:, 1]
        if not self.is_classification:
            return {
                "treatment_rmse": root_mean_squared_error(
                    y[w == 1], treatment_outcomes[w == 1]
                ),
                "effect_rmse": root_mean_squared_error(
                    y[w == 0], effect_outcomes[w == 0]
                ),
            }
        return {
            "treatment_cross_entropy": log_loss(
                y[w == 1], treatment_outcomes[w == 1][:, 1]
            ),
            "effect_cross_entropy": log_loss(y[w == 0], effect_outcomes[w == 0][:, 1]),
        }
