# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
from sklearn.base import is_classifier, is_regressor
from typing_extensions import Self

from metalearners._utils import Matrix, Vector, index_matrix
from metalearners.cross_fit_estimator import MEDIAN, OVERALL, OosMethod, PredictMethod
from metalearners.metalearner import PROPENSITY_MODEL, MetaLearner

CONTROL_OUTCOME_MODEL = "control_outcome_model"
TREATMENT_OUTCOME_MODEL = "treatment_outcome_model"
CONTROL_EFFECT_MODEL = "control_effect_model"
TREATMENT_EFFECT_MODEL = "treatment_effect_model"


class XLearner(MetaLearner):
    """X-Learner for CATE estimation as described by `Kuenzel et al (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_.

    Importantly, the current X-Learner implementation only supports:

        * binary treatment variants
        * binary classes in case of a classification outcome
    """

    @classmethod
    def nuisance_model_names(cls) -> set[str]:
        return {CONTROL_OUTCOME_MODEL, TREATMENT_OUTCOME_MODEL, PROPENSITY_MODEL}

    @classmethod
    def treatment_model_names(cls) -> set[str]:
        return {CONTROL_EFFECT_MODEL, TREATMENT_EFFECT_MODEL}

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return False

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return False

    def _validate_models(self) -> None:
        if self.is_classification and not is_classifier(
            self.nuisance_model_factory[CONTROL_OUTCOME_MODEL]
        ):
            raise ValueError(
                f"is_classification is set to True but the {CONTROL_OUTCOME_MODEL} "
                "is not a classifier."
            )
        if self.is_classification and not is_classifier(
            self.nuisance_model_factory[TREATMENT_OUTCOME_MODEL]
        ):
            raise ValueError(
                f"is_classification is set to True but the {TREATMENT_OUTCOME_MODEL} "
                "is not a classifier."
            )
        if not self.is_classification and not is_regressor(
            self.nuisance_model_factory[CONTROL_OUTCOME_MODEL]
        ):
            raise ValueError(
                f"is_classification is set to False but the {CONTROL_OUTCOME_MODEL} "
                "is not a regressor."
            )
        if not self.is_classification and not is_regressor(
            self.nuisance_model_factory[TREATMENT_OUTCOME_MODEL]
        ):
            raise ValueError(
                f"is_classification is set to False but the {TREATMENT_OUTCOME_MODEL} "
                "is not a regressor."
            )

        if not is_classifier(self.nuisance_model_factory[PROPENSITY_MODEL]):
            raise ValueError(f"{PROPENSITY_MODEL} is not a classifier.")
        if not is_regressor(self.treatment_model_factory[CONTROL_EFFECT_MODEL]):
            raise ValueError(f"{CONTROL_EFFECT_MODEL} is not a regressor.")
        if not is_regressor(self.treatment_model_factory[TREATMENT_EFFECT_MODEL]):
            raise ValueError(f"{TREATMENT_EFFECT_MODEL} is not a regressor.")

    @property
    def _nuisance_predict_methods(
        self,
    ) -> dict[str, PredictMethod]:
        predict_method: PredictMethod = (
            "predict_proba" if self.is_classification else "predict"
        )

        return {
            CONTROL_OUTCOME_MODEL: predict_method,
            TREATMENT_OUTCOME_MODEL: predict_method,
            PROPENSITY_MODEL: "predict_proba",
        }

    def fit(self, X: Matrix, y: Vector, w: Vector) -> Self:
        self._check_treatment(w)
        self._check_outcome(y)
        self._treatment_indices = w == 1
        self._control_indices = w == 0

        # TODO: Consider multiprocessing
        self.fit_nuisance(
            X=index_matrix(X, self._treatment_indices),
            y=y[self._treatment_indices],
            model_kind=TREATMENT_OUTCOME_MODEL,
        )
        self.fit_nuisance(
            X=index_matrix(X, self._control_indices),
            y=y[self._control_indices],
            model_kind=CONTROL_OUTCOME_MODEL,
        )
        self.fit_nuisance(
            X=X,
            y=w,
            model_kind=PROPENSITY_MODEL,
        )

        pseudo_outcome = self._pseudo_outcome(X, y, w)

        self.fit_treatment(
            X=index_matrix(X, self._treatment_indices),
            y=pseudo_outcome[self._treatment_indices],
            model_kind=TREATMENT_EFFECT_MODEL,
        )
        self.fit_treatment(
            X=index_matrix(X, self._control_indices),
            y=pseudo_outcome[self._control_indices],
            model_kind=CONTROL_EFFECT_MODEL,
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
                is_oos=is_oos,
                oos_method=oos_method,
            )
            tau_hat_control = self.predict_treatment(
                X=X,
                model_kind=CONTROL_EFFECT_MODEL,
                is_oos=is_oos,
                oos_method=oos_method,
            )
        else:
            treatment_effect_treated = self.predict_treatment(
                X=index_matrix(X, self._treatment_indices),
                model_kind=TREATMENT_EFFECT_MODEL,
                is_oos=False,
            )
            control_effect_treated = self.predict_treatment(
                X=index_matrix(X, self._treatment_indices),
                model_kind=CONTROL_EFFECT_MODEL,
                is_oos=True,
                oos_method=oos_method,
            )

            treatment_effect_control = self.predict_treatment(
                X=index_matrix(X, self._control_indices),
                model_kind=TREATMENT_EFFECT_MODEL,
                is_oos=True,
                oos_method=oos_method,
            )
            control_effect_control = self.predict_treatment(
                X=index_matrix(X, self._control_indices),
                model_kind=CONTROL_EFFECT_MODEL,
                is_oos=False,
            )

            tau_hat_treatment = np.zeros(len(X))
            tau_hat_control = np.zeros(len(X))

            tau_hat_treatment[self._control_indices] = treatment_effect_control
            tau_hat_treatment[self._treatment_indices] = treatment_effect_treated
            tau_hat_control[self._control_indices] = control_effect_control
            tau_hat_control[self._treatment_indices] = control_effect_treated

        # Propensity score model is always a classifier so we can't use MEDIAN
        propensity_score_oos = OVERALL if oos_method == MEDIAN else oos_method
        propensity_score = self.predict_nuisance(
            X=X,
            model_kind=PROPENSITY_MODEL,
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

    def predict_conditional_average_outcomes(
        self, X: Matrix, is_oos: bool, oos_method: OosMethod = OVERALL
    ) -> np.ndarray:
        # TODO: Consider multiprocessing
        if is_oos:
            treatment_outcomes = self.predict_nuisance(
                X=X,
                model_kind=TREATMENT_OUTCOME_MODEL,
                is_oos=is_oos,
                oos_method=oos_method,
            )
            control_outcomes = self.predict_nuisance(
                X=X,
                model_kind=CONTROL_OUTCOME_MODEL,
                is_oos=is_oos,
                oos_method=oos_method,
            )
        else:
            treatment_outcomes_treated = self.predict_nuisance(
                X=index_matrix(X, self._treatment_indices),
                model_kind=TREATMENT_OUTCOME_MODEL,
                is_oos=False,
            )
            control_outcomes_treated = self.predict_nuisance(
                X=index_matrix(X, self._treatment_indices),
                model_kind=CONTROL_OUTCOME_MODEL,
                is_oos=True,
                oos_method=oos_method,
            )

            treatment_outcomes_control = self.predict_nuisance(
                X=index_matrix(X, self._control_indices),
                model_kind=TREATMENT_OUTCOME_MODEL,
                is_oos=True,
                oos_method=oos_method,
            )
            control_outcomes_control = self.predict_nuisance(
                X=index_matrix(X, self._control_indices),
                model_kind=CONTROL_OUTCOME_MODEL,
                is_oos=False,
            )

            nuisance_tensors = self._nuisance_tensors(len(X))

            treatment_outcomes = nuisance_tensors[TREATMENT_OUTCOME_MODEL]
            control_outcomes = nuisance_tensors[CONTROL_OUTCOME_MODEL]

            treatment_outcomes[self._control_indices] = treatment_outcomes_control
            treatment_outcomes[self._treatment_indices] = treatment_outcomes_treated
            control_outcomes[self._control_indices] = control_outcomes_control
            control_outcomes[self._treatment_indices] = control_outcomes_treated

        return np.stack([control_outcomes, treatment_outcomes], axis=1)

    def _pseudo_outcome(self, X: Matrix, y: Vector, w: Vector) -> np.ndarray:
        pseudo_outcome = np.zeros(len(y))
        treatment_indices = w == 1
        control_indices = w == 0

        # This is always oos because the CONTROL_OUTCOME_MODEL is used to predict the
        # control outcomes of the treated observations and vice versa.
        control_outcome = self.predict_nuisance(
            X=index_matrix(X, treatment_indices),
            model_kind=CONTROL_OUTCOME_MODEL,
            is_oos=True,
            oos_method=OVERALL,
        )
        treatment_outcome = self.predict_nuisance(
            X=index_matrix(X, control_indices),
            model_kind=TREATMENT_OUTCOME_MODEL,
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