# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import Matrix, Vector, index_matrix
from metalearners.cross_fit_estimator import MEDIAN, OVERALL
from metalearners.metalearner import (
    PROPENSITY_MODEL,
    VARIANT_OUTCOME_MODEL,
    _ConditionalAverageOutcomeMetaLearner,
    _ModelSpecifications,
)

CONTROL_EFFECT_MODEL = "control_effect_model"
TREATMENT_EFFECT_MODEL = "treatment_effect_model"


class XLearner(_ConditionalAverageOutcomeMetaLearner):
    """X-Learner for CATE estimation as described by `Kuenzel et al (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_.

    Importantly, the current X-Learner implementation only supports:

        * binary classes in case of a classification outcome
    """

    @classmethod
    def nuisance_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            VARIANT_OUTCOME_MODEL: _ModelSpecifications(
                cardinality=lambda ml: ml.n_variants,
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
                cardinality=lambda ml: ml.n_variants - 1,
                predict_method=lambda _: "predict",
            ),
            TREATMENT_EFFECT_MODEL: _ModelSpecifications(
                cardinality=lambda ml: ml.n_variants - 1,
                predict_method=lambda _: "predict",
            ),
        }

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return True

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return False

    def fit(
        self, X: Matrix, y: Vector, w: Vector, n_jobs_cross_fitting: int | None = None
    ) -> Self:
        self._validate_treatment(w)
        self._validate_outcome(y)

        for v in range(self.n_variants):
            self._treatment_variants_indices.append(w == v)

        # TODO: Consider multiprocessing
        for treatment_variant in range(self.n_variants):
            self.fit_nuisance(
                X=index_matrix(X, self._treatment_variants_indices[treatment_variant]),
                y=y[self._treatment_variants_indices[treatment_variant]],
                model_kind=VARIANT_OUTCOME_MODEL,
                model_ord=treatment_variant,
                n_jobs_cross_fitting=n_jobs_cross_fitting,
            )

        self.fit_nuisance(
            X=X,
            y=w,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
            n_jobs_cross_fitting=n_jobs_cross_fitting,
        )

        for treatment_variant in range(1, self.n_variants):
            imputed_te_control, imputed_te_treatment = self._pseudo_outcome(
                X, y, w, treatment_variant
            )

            self.fit_treatment(
                X=index_matrix(X, self._treatment_variants_indices[treatment_variant]),
                y=imputed_te_treatment,
                model_kind=TREATMENT_EFFECT_MODEL,
                model_ord=treatment_variant - 1,
                n_jobs_cross_fitting=n_jobs_cross_fitting,
            )
            self.fit_treatment(
                X=index_matrix(X, self._treatment_variants_indices[0]),
                y=imputed_te_control,
                model_kind=CONTROL_EFFECT_MODEL,
                model_ord=treatment_variant - 1,
                n_jobs_cross_fitting=n_jobs_cross_fitting,
            )

        return self

    def predict(
        self,
        X: Matrix,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        n_outputs = 2 if self.is_classification else 1
        tau_hat = np.zeros((len(X), self.n_variants - 1, n_outputs))
        # Propensity score model is always a classifier so we can't use MEDIAN
        propensity_score_oos = OVERALL if oos_method == MEDIAN else oos_method
        propensity_score = self.predict_nuisance(
            X=X,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
            is_oos=is_oos,
            oos_method=propensity_score_oos,
        )

        control_indices = self._treatment_variants_indices[0]
        non_control_indices = ~control_indices

        for treatment_variant in range(1, self.n_variants):
            treatment_variant_indices = self._treatment_variants_indices[
                treatment_variant
            ]
            non_treatment_variant_indices = ~treatment_variant_indices
            if is_oos:
                tau_hat_treatment = self.predict_treatment(
                    X=X,
                    model_kind=TREATMENT_EFFECT_MODEL,
                    model_ord=treatment_variant - 1,
                    is_oos=is_oos,
                    oos_method=oos_method,
                )
                tau_hat_control = self.predict_treatment(
                    X=X,
                    model_kind=CONTROL_EFFECT_MODEL,
                    model_ord=treatment_variant - 1,
                    is_oos=is_oos,
                    oos_method=oos_method,
                )
            else:
                tau_hat_treatment = np.zeros(len(X))
                tau_hat_control = np.zeros(len(X))

                tau_hat_treatment[non_treatment_variant_indices] = (
                    self.predict_treatment(
                        X=index_matrix(X, non_treatment_variant_indices),
                        model_kind=TREATMENT_EFFECT_MODEL,
                        model_ord=treatment_variant - 1,
                        is_oos=True,
                        oos_method=oos_method,
                    )
                )
                tau_hat_treatment[treatment_variant_indices] = self.predict_treatment(
                    X=index_matrix(X, treatment_variant_indices),
                    model_kind=TREATMENT_EFFECT_MODEL,
                    model_ord=treatment_variant - 1,
                    is_oos=False,
                )
                tau_hat_control[control_indices] = self.predict_treatment(
                    X=index_matrix(X, control_indices),
                    model_kind=CONTROL_EFFECT_MODEL,
                    model_ord=treatment_variant - 1,
                    is_oos=False,
                )
                tau_hat_control[non_control_indices] = self.predict_treatment(
                    X=index_matrix(X, non_control_indices),
                    model_kind=CONTROL_EFFECT_MODEL,
                    model_ord=treatment_variant - 1,
                    is_oos=True,
                    oos_method=oos_method,
                )

            propensity_score_treatment = propensity_score[:, treatment_variant] / (
                propensity_score[:, 0] + propensity_score[:, treatment_variant]
            )

            tau_hat_treatment_variant = (
                propensity_score_treatment * tau_hat_control
                + (1 - propensity_score_treatment) * tau_hat_treatment
            )

            if self.is_classification:
                # This is to be consistent with other MetaLearners (e.g. S and T) that automatically
                # work with multiclass outcomes and return the CATE estimate for each class. As the X-Learner only
                # works with binary classes (the pseudo outcome formula does not make sense with
                # multiple classes unless some adaptation is done) we can manually infer the
                # CATE estimate for the complementary class  -- returning a matrix of shape (N, 2).
                tau_hat_treatment_variant = np.stack(
                    [-tau_hat_treatment_variant, tau_hat_treatment_variant], axis=1
                )
            else:
                tau_hat_treatment_variant = np.expand_dims(tau_hat_treatment_variant, 1)

            tau_hat[:, treatment_variant - 1] = tau_hat_treatment_variant

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

    def _pseudo_outcome(
        self, X: Matrix, y: Vector, w: Vector, treatment_variant: int
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = (w == treatment_variant) | (w == 0)

        X_filt = X[mask]
        y_filt = y[mask]
        w_filt = (w[mask] == treatment_variant).astype(int)

        treatment_indices = w_filt == 1
        control_indices = w_filt == 0

        # This is always oos because the VARIANT_OUTCOME_MODEL[0] is used to predict the
        # control outcomes of the treated observations and vice versa.
        control_outcome = self.predict_nuisance(
            X=index_matrix(X_filt, treatment_indices),
            model_kind=VARIANT_OUTCOME_MODEL,
            model_ord=0,
            is_oos=True,
            oos_method=OVERALL,
        )
        treatment_outcome = self.predict_nuisance(
            X=index_matrix(X_filt, control_indices),
            model_kind=VARIANT_OUTCOME_MODEL,
            model_ord=treatment_variant,
            is_oos=True,
            oos_method=OVERALL,
        )
        if self.is_classification:
            # Get the probability of positive class, multiclass is currently not supported.
            control_outcome = control_outcome[:, 1]
            treatment_outcome = treatment_outcome[:, 1]

        imputed_te_treatment = y_filt[treatment_indices] - control_outcome
        imputed_te_control = treatment_outcome - y_filt[control_indices]

        return imputed_te_control, imputed_te_treatment
