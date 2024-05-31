# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import numpy as np
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import (
    Matrix,
    Vector,
    clip_element_absolute_value_to_epsilon,
    get_one,
    get_predict,
    get_predict_proba,
    index_matrix,
    validate_valid_treatment_variant_not_control,
)
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import (
    NUISANCE,
    PROPENSITY_MODEL,
    TREATMENT,
    TREATMENT_MODEL,
    VARIANT_OUTCOME_MODEL,
    MetaLearner,
    _ConditionalAverageOutcomeMetaLearner,
    _ModelSpecifications,
)

_EPSILON = 1e-09


class DRLearner(_ConditionalAverageOutcomeMetaLearner):
    r"""DR-Learner for CATE estimation as described by `Kennedy (2020) <https://arxiv.org/pdf/2004.14497>`_.

    Importantly, the current DR-Learner implementation only supports:

        * binary classes in case of a classification outcome

    The DR-Learner contains the following nuisance models:

        * a ``"propensity_model"`` estimating :math:`\Pr[W=k|X]`
        * one ``"variant_outcome_model"`` for each treatment variant (including control)
          estimating :math:`\mathbb{E}[Y|X, W=k]`

    and one treatment model for each treatment variant (without control):

        * ``"treatment_model"`` which estimates :math:`\mathbb{E}[Y(k) - Y(0) | X]`

    """

    @classmethod
    def nuisance_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            PROPENSITY_MODEL: _ModelSpecifications(
                cardinality=get_one,
                predict_method=get_predict_proba,
            ),
            VARIANT_OUTCOME_MODEL: _ModelSpecifications(
                cardinality=MetaLearner._get_n_variants,
                predict_method=MetaLearner._outcome_predict_method,
            ),
        }

    @classmethod
    def treatment_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            TREATMENT_MODEL: _ModelSpecifications(
                cardinality=MetaLearner._get_n_variants_minus_one,
                predict_method=get_predict,
            )
        }

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return True

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return False

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

        qualified_fit_params = self._qualified_fit_params(fit_params)

        for treatment_variant in range(self.n_variants):
            self._treatment_variants_indices.append(w == treatment_variant)

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

        self.fit_nuisance(
            X=X,
            y=w,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
            n_jobs_cross_fitting=n_jobs_cross_fitting,
            fit_params=qualified_fit_params[NUISANCE][PROPENSITY_MODEL],
        )

        for treatment_variant in range(1, self.n_variants):
            pseudo_outcomes = self._pseudo_outcome(
                X=X,
                w=w,
                y=y,
                treatment_variant=treatment_variant,
            )

            self.fit_treatment(
                X=X,
                y=pseudo_outcomes,
                model_kind=TREATMENT_MODEL,
                model_ord=treatment_variant - 1,
                n_jobs_cross_fitting=n_jobs_cross_fitting,
                fit_params=qualified_fit_params[TREATMENT][TREATMENT_MODEL],
            )
        return self

    def predict(
        self,
        X,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        n_outputs = 2 if self.is_classification else 1
        estimates = np.zeros((len(X), self.n_variants - 1, n_outputs))
        for treatment_variant in range(1, self.n_variants):
            estimates_variant = self.predict_treatment(
                X,
                is_oos=is_oos,
                oos_method=oos_method,
                model_kind=TREATMENT_MODEL,
                model_ord=treatment_variant - 1,
            )
            if self.is_classification:
                # This is to be consistent with other MetaLearners (e.g. S and T) that automatically
                # work with multiclass outcomes and return the CATE estimate for each class. As the DR-Learner only
                # works with binary classes (the pseudo outcome formula does not make sense with
                # multiple classes unless some adaptation is done) we can manually infer the
                # CATE estimate for the complementary class  -- returning a matrix of shape (N, 2).
                estimates_variant = np.stack(
                    [-estimates_variant, estimates_variant], axis=1
                )
            else:
                estimates_variant = np.expand_dims(estimates_variant, 1)

            estimates[:, treatment_variant - 1] = estimates_variant
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
        treatment_variant: int,
        epsilon: float = _EPSILON,
    ) -> np.ndarray:
        """Compute the DR-Learner pseudo outcome.

        Importantly, this method assumes to be applied on in-sample data.
        In other words, ``is_oos`` will always be set to ``False`` when calling
        ``predict_nuisance``.
        """
        validate_valid_treatment_variant_not_control(treatment_variant, self.n_variants)

        conditional_average_outcome_estimates = (
            self.predict_conditional_average_outcomes(
                X=X,
                is_oos=False,
            )
        )

        propensity_estimates = self.predict_nuisance(
            X=X,
            is_oos=False,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
        )

        y0_estimate = conditional_average_outcome_estimates[:, 0]
        y1_estimate = conditional_average_outcome_estimates[:, treatment_variant]

        if self.is_classification:
            y0_estimate = y0_estimate[:, 1]
            y1_estimate = y1_estimate[:, 1]
        else:
            y0_estimate = y0_estimate[:, 0]
            y1_estimate = y1_estimate[:, 0]

        pseudo_outcome = (
            (
                (y - y1_estimate)
                / clip_element_absolute_value_to_epsilon(
                    propensity_estimates[:, treatment_variant], epsilon
                )
            )
            * (w == treatment_variant)
            + y1_estimate
            - (
                (y - y0_estimate)
                / clip_element_absolute_value_to_epsilon(
                    propensity_estimates[:, 0], epsilon
                )
            )
            * (w == 0)
            - y0_estimate
        )

        return pseudo_outcome
