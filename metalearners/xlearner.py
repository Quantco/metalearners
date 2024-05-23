# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import Matrix, Vector, index_matrix
from metalearners.cross_fit_estimator import MEDIAN, OVERALL
from metalearners.metalearner import (
    CONTROL_OUTCOME_MODEL,
    PROPENSITY_MODEL,
    TREATMENT_OUTCOME_MODEL,
    _ConditionalAverageOutcomeMetaLearner,
    _ModelSpecifications,
)

CONTROL_EFFECT_MODEL = "control_effect_model"
TREATMENT_EFFECT_MODEL = "treatment_effect_model"


class XLearner(_ConditionalAverageOutcomeMetaLearner):
    """X-Learner for CATE estimation as described by `Kuenzel et al (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_.

    Importantly, the current X-Learner implementation only supports:

        * binary treatment variants
        * binary classes in case of a classification outcome
    """

    @classmethod
    def nuisance_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            CONTROL_OUTCOME_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda ml: (
                    "predict_proba" if ml.is_classification else "predict"
                ),
            ),
            TREATMENT_OUTCOME_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda ml: (
                    "predict_proba" if ml.is_classification else "predict"
                ),
            ),
            PROPENSITY_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda _: "predict_proba",
            ),
        }

    @classmethod
    def treatment_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            CONTROL_EFFECT_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda _: "predict",
            ),
            TREATMENT_EFFECT_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda _: "predict",
            ),
        }

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return False

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return False

    def fit(self, X: Matrix, y: Vector, w: Vector) -> Self:
        self._validate_treatment(w)
        self._validate_outcome(y)

        for v in range(self.n_variants):
            self._treatment_variants_indices.append(w == v)

        # TODO: Consider multiprocessing
        self.fit_nuisance(
            X=index_matrix(X, self._treatment_variants_indices[1]),
            y=y[self._treatment_variants_indices[1]],
            model_kind=TREATMENT_OUTCOME_MODEL,
            model_ord=0,
        )
        self.fit_nuisance(
            X=index_matrix(X, self._treatment_variants_indices[0]),
            y=y[self._treatment_variants_indices[0]],
            model_kind=CONTROL_OUTCOME_MODEL,
            model_ord=0,
        )
        self.fit_nuisance(
            X=X,
            y=w,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
        )

        pseudo_outcome = self._pseudo_outcome(X, y, w)

        self.fit_treatment(
            X=index_matrix(X, self._treatment_variants_indices[1]),
            y=pseudo_outcome[self._treatment_variants_indices[1]],
            model_kind=TREATMENT_EFFECT_MODEL,
            model_ord=0,
        )
        self.fit_treatment(
            X=index_matrix(X, self._treatment_variants_indices[0]),
            y=pseudo_outcome[self._treatment_variants_indices[0]],
            model_kind=CONTROL_EFFECT_MODEL,
            model_ord=0,
        )

        return self

    def predict(
        self,
        X: Matrix,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        if is_oos:
            tau_hat_treatment = self.predict_treatment(
                X=X,
                model_kind=TREATMENT_EFFECT_MODEL,
                model_ord=0,
                is_oos=is_oos,
                oos_method=oos_method,
            )
            tau_hat_control = self.predict_treatment(
                X=X,
                model_kind=CONTROL_EFFECT_MODEL,
                model_ord=0,
                is_oos=is_oos,
                oos_method=oos_method,
            )
        else:
            treatment_effect_treated = self.predict_treatment(
                X=index_matrix(X, self._treatment_variants_indices[1]),
                model_kind=TREATMENT_EFFECT_MODEL,
                model_ord=0,
                is_oos=False,
            )
            control_effect_treated = self.predict_treatment(
                X=index_matrix(X, self._treatment_variants_indices[1]),
                model_kind=CONTROL_EFFECT_MODEL,
                model_ord=0,
                is_oos=True,
                oos_method=oos_method,
            )

            treatment_effect_control = self.predict_treatment(
                X=index_matrix(X, self._treatment_variants_indices[0]),
                model_kind=TREATMENT_EFFECT_MODEL,
                model_ord=0,
                is_oos=True,
                oos_method=oos_method,
            )
            control_effect_control = self.predict_treatment(
                X=index_matrix(X, self._treatment_variants_indices[0]),
                model_kind=CONTROL_EFFECT_MODEL,
                model_ord=0,
                is_oos=False,
            )

            tau_hat_treatment = np.zeros(len(X))
            tau_hat_control = np.zeros(len(X))

            tau_hat_treatment[self._treatment_variants_indices[0]] = (
                treatment_effect_control
            )
            tau_hat_treatment[self._treatment_variants_indices[1]] = (
                treatment_effect_treated
            )
            tau_hat_control[self._treatment_variants_indices[0]] = (
                control_effect_control
            )
            tau_hat_control[self._treatment_variants_indices[1]] = (
                control_effect_treated
            )

        # Propensity score model is always a classifier so we can't use MEDIAN
        propensity_score_oos = OVERALL if oos_method == MEDIAN else oos_method
        propensity_score = self.predict_nuisance(
            X=X,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
            is_oos=is_oos,
            oos_method=propensity_score_oos,
        )[:, 1]

        tau_hat = (
            propensity_score * tau_hat_control
            + (1 - propensity_score) * tau_hat_treatment
        )
        if self.is_classification:
            # This is to be consistent with other MetaLearners (e.g. S and T) that automatically
            # work with multiclass outcomes and return the CATE estimate for each class. As the X-Learner only
            # works with binary classes (the pseudo outcome formula does not make sense with
            # multiple classes unless some adaptation is done) we can manually infer the
            # CATE estimate for the complementary class  -- returning a matrix of shape (N, 2).
            return np.stack([-tau_hat, tau_hat], axis=1)
        return tau_hat

    def evaluate(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> dict[str, float | int]:
        raise NotImplementedError(
            "This feature is not yet implemented for the X-Learner."
        )

    def _pseudo_outcome(self, X: Matrix, y: Vector, w: Vector) -> np.ndarray:
        pseudo_outcome = np.zeros(len(y))
        treatment_indices = w == 1
        control_indices = w == 0

        # This is always oos because the CONTROL_OUTCOME_MODEL is used to predict the
        # control outcomes of the treated observations and vice versa.
        control_outcome = self.predict_nuisance(
            X=index_matrix(X, treatment_indices),
            model_kind=CONTROL_OUTCOME_MODEL,
            model_ord=0,
            is_oos=True,
            oos_method=OVERALL,
        )
        treatment_outcome = self.predict_nuisance(
            X=index_matrix(X, control_indices),
            model_kind=TREATMENT_OUTCOME_MODEL,
            model_ord=0,
            is_oos=True,
            oos_method=OVERALL,
        )
        if self.is_classification:
            # Get the probability of positive class, multiclass is currently not supported.
            control_outcome = control_outcome[:, 1]
            treatment_outcome = treatment_outcome[:, 1]

        imputed_te_treatment = y[treatment_indices] - control_outcome
        imputed_te_control = treatment_outcome - y[control_indices]

        pseudo_outcome[treatment_indices] = imputed_te_treatment
        pseudo_outcome[control_indices] = imputed_te_control
        return pseudo_outcome
