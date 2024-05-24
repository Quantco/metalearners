# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import numpy as np
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import (
    Matrix,
    Vector,
    clip_element_absolute_value_to_epsilon,
    index_matrix,
)
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import (
    PROPENSITY_MODEL,
    TREATMENT_MODEL,
    VARIANT_OUTCOME_MODEL,
    _ConditionalAverageOutcomeMetaLearner,
    _ModelSpecifications,
)

_EPSILON = 1e-09


class DRLearner(_ConditionalAverageOutcomeMetaLearner):
    r"""DR-Learner for CATE estimation as described by `Kennedy (2020) <https://arxiv.org/pdf/2004.14497>`_.

    Importantly, the current DR-Learner implementation only supports:

        * binary treatment variants
        * binary classes in case of a classification outcome

    The DR-Learner contains three nuisance models

        * a ``"propensity_model"`` estimating :math:`\Pr[W=1|X]`
        * two ``"variant_outcome_model"`` estimating :math:`\mathbb{E}[Y|X, W=0]` and
          :math:`\mathbb{E}[Y|X, W=1]`

    and one treatment model

        * ``"treatment_model"`` which estimates :math:`\mathbb{E}[Y(1) - Y(0) | X]`

    """

    @classmethod
    def nuisance_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            PROPENSITY_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1, predict_method=lambda _: "predict_proba"
            ),
            VARIANT_OUTCOME_MODEL: _ModelSpecifications(
                cardinality=lambda _: 2,
                predict_method=lambda ml: (
                    "predict_proba" if ml.is_classification else "predict"
                ),
            ),
        }

    @classmethod
    def treatment_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            TREATMENT_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda _: "predict",
            )
        }

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return False

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return False

    def fit(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
    ) -> Self:
        self._validate_treatment(w)
        self._validate_outcome(y)

        for treatment_variant in range(self.n_variants):
            self._treatment_variants_indices.append(w == treatment_variant)

        self.fit_nuisance(
            X=X,
            y=w,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
        )

        self.fit_nuisance(
            X=index_matrix(X, self._treatment_variants_indices[1]),
            y=y[self._treatment_variants_indices[1]],
            model_kind=VARIANT_OUTCOME_MODEL,
            model_ord=1,
        )
        self.fit_nuisance(
            X=index_matrix(X, self._treatment_variants_indices[0]),
            y=y[self._treatment_variants_indices[0]],
            model_kind=VARIANT_OUTCOME_MODEL,
            model_ord=0,
        )

        propensity_estimates = self.predict_nuisance(
            X=X,
            is_oos=False,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
        )[:, 1]

        conditional_average_outcome_estimates = (
            self.predict_conditional_average_outcomes(
                X=X,
                is_oos=False,
            )
        )

        pseudo_outcomes = self._pseudo_outcome(
            X=X,
            w=w,
            y=y,
            propensity_estimates=propensity_estimates,
            conditional_average_outcome_estimates=conditional_average_outcome_estimates,
        )
        self.fit_treatment(
            X=X,
            y=pseudo_outcomes,
            model_kind=TREATMENT_MODEL,
            model_ord=0,
        )
        return self

    def predict(
        self,
        X,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        estimates = self.predict_treatment(
            X,
            is_oos=is_oos,
            oos_method=oos_method,
            model_kind=TREATMENT_MODEL,
            model_ord=0,
        )
        if self.is_classification:
            # This is to be consistent with other MetaLearners (e.g. S and T) that automatically
            # work with multiclass outcomes and return the CATE estimate for each class. As the DR-Learner only
            # works with binary classes (the pseudo outcome formula does not make sense with
            # multiple classes unless some adaptation is done) we can manually infer the
            # CATE estimate for the complementary class  -- returning a matrix of shape (N, 2).
            return np.stack([-estimates, estimates], axis=-1)
        return estimates

    def evaluate(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> dict[str, float | int]:
        raise NotImplementedError(
            "This feature is not yet implemented for the DR-Learner."
        )

    def _pseudo_outcome(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        propensity_estimates: Vector,
        conditional_average_outcome_estimates: Matrix,
        epsilon: float = _EPSILON,
    ) -> np.ndarray:
        y0_estimate = conditional_average_outcome_estimates[:, 0]
        y1_estimate = conditional_average_outcome_estimates[:, 1]

        if self.is_classification:
            y0_estimate = y0_estimate[:, 1]
            y1_estimate = y1_estimate[:, 1]

        numerator = w - propensity_estimates
        denominator = propensity_estimates * (1 - propensity_estimates)
        denominator_padded = clip_element_absolute_value_to_epsilon(
            denominator, epsilon
        )

        conditional_outcome_estimate = np.where(w, y1_estimate, y0_estimate)
        return (
            numerator / (denominator_padded) * (y - conditional_outcome_estimate)
            + y1_estimate
            - y0_estimate
        )
