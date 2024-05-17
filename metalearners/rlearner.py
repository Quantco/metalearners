# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
from sklearn.metrics import log_loss, root_mean_squared_error
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import (
    Matrix,
    Vector,
    function_has_argument,
    validate_all_vectors_same_index,
)
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import MetaLearner, _ModelSpecifications

PROPENSITY_MODEL = "propensity_model"
OUTCOME_MODEL = "outcome_model"
TREATMENT_MODEL = "treatment_model"

_EPSILON = 1e-09

_SAMPLE_WEIGHT = "sample_weight"


def r_loss(
    cate_estimates: Vector,
    outcome_estimates: Vector,
    propensity_scores: Vector,
    outcomes: Vector,
    treatments: Vector,
) -> float:
    r"""Compute the square-root of the R-loss as introduced by Nie et al.

    This function computes:

    .. math::
        \sqrt{\frac{1}{N}\sum_{i=1}^N ((y_i - \mu(X_i)) - \hat{\tau}(X_i)
        (w_i - e(X_i)))^2}

    The R-Learner proposed in `Nie et al. (2017) <https://arxiv.org/pdf/1712.04912.pdf>`_
    relies on a loss function which can be used in combination with empirical risk
    minimization to learn a CATE model.

    Independently of the R-Learner, one can use the R-loss for evaluating CATE estimates
    in general.
    """
    inputs = [
        cate_estimates,
        outcome_estimates,
        propensity_scores,
        outcomes,
        treatments,
    ]
    validate_all_vectors_same_index(inputs)

    residualised_outcomes = outcomes - outcome_estimates
    residualised_treatments = treatments - propensity_scores
    return root_mean_squared_error(
        residualised_outcomes, cate_estimates * residualised_treatments
    )


class RLearner(MetaLearner):
    r"""R-Learner for CATE estimation as described by `Nie et al. (2017) <https://arxiv.org/pdf/1712.04912.pdf>`_.

    Importantly, the current R-Learner implementation only supports:

        * binary treatment variants
        * binary classes in case of a classification outcome

    The R-Learner contains two nuisance models

        * a "propensity_model" estimating :math:`\Pr[W=1|X]`
        * an "outcome_model" estimating :math:`\mathbb{E}[Y|X]`

    and one treatment model

        * "treatment_model" which estimates :math:`\mathbb{E}[Y(1) - Y(0) | X]`

    The ``treatment_model_factory`` provided needs to support the argument
    ``sample_weight`` in its ``fit`` method.
    """

    def _validate_models(self) -> None:
        """Validate that the base models are appropriate.

        In particular, it is validated that a base model to be used with ``"predict"`` is
        recognized by ``scikit-learn`` as a regressor via ``sklearn.base.is_regressor`` and
        a model to be used with ``"predict_proba"`` is recognized by ``scikit-learn` as
        a classifier via ``sklearn.base.is_classifier``.

        Additionally, this method ensures that the treatment model "treatment_model" supports
        the ``"sample_weight"`` argument in its ``fit`` method.
        """
        if not function_has_argument(
            self.treatment_model_factory[TREATMENT_MODEL].fit, _SAMPLE_WEIGHT
        ):
            raise ValueError(
                f"{TREATMENT_MODEL}'s fit method does not support 'sample_weight' argument."
            )
        super()._validate_models()

    @classmethod
    def nuisance_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        """Return the names of all first-stage models."""

        return {
            PROPENSITY_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1, predict_method=lambda _: "predict_proba"
            ),
            OUTCOME_MODEL: _ModelSpecifications(
                cardinality=lambda _: 1,
                predict_method=lambda ml: (
                    "predict_proba" if ml.is_classification else "predict"
                ),
            ),
        }

    @classmethod
    def treatment_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        """Return the names of all second-stage models."""
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
        epsilon: float = _EPSILON,
    ) -> Self:
        self._check_treatment(w)
        self._check_outcome(y)

        self.fit_nuisance(
            X=X,
            y=w,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
        )
        self.fit_nuisance(
            X=X,
            y=y,
            model_kind=OUTCOME_MODEL,
            model_ord=0,
        )

        pseudo_outcomes, weights = self._pseudo_outcome_and_weights(
            X=X, w=w, y=y, epsilon=epsilon
        )

        self.fit_treatment(
            X=X,
            y=pseudo_outcomes,
            model_kind=TREATMENT_MODEL,
            model_ord=0,
            fit_params={_SAMPLE_WEIGHT: weights},
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
            # work with multiclass outcomes and return the CATE estimate for each class. As the R-Learner only
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
        w_hat = self.predict_nuisance(
            X=X,
            is_oos=is_oos,
            oos_method=oos_method,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
        )[:, 1]
        y_hat = self.predict_nuisance(
            X=X,
            is_oos=is_oos,
            oos_method=oos_method,
            model_kind=OUTCOME_MODEL,
            model_ord=0,
        )
        if self.is_classification:
            y_hat = y_hat[:, 1]
        tau_hat = self.predict_treatment(
            X=X,
            is_oos=is_oos,
            oos_method=oos_method,
            model_kind=TREATMENT_MODEL,
            model_ord=0,
        )

        outcome_evaluation = (
            {"outcome_log_loss": log_loss(y, y_hat)}
            if self.is_classification
            else {"outcome_rmse": root_mean_squared_error(y, y_hat)}
        )

        return outcome_evaluation | {
            "propensity_cross_entropy": log_loss(w, w_hat),
            "r_loss": r_loss(
                cate_estimates=tau_hat,
                outcome_estimates=y_hat,
                propensity_scores=w_hat,
                outcomes=y,
                treatments=w,
            ),
        }

    def _pseudo_outcome_and_weights(
        self, X: Matrix, y: Vector, w: Vector, epsilon: float = _EPSILON
    ) -> tuple[Vector, Vector]:
        """Compute the R-Learner pseudo outcome and corresponding weights.

        Importantly, this method assumes to be applied on in-sample data.
        In other words, ``is_oos`` will always be set to ``False`` when calling
        ``predict_nuisance``.

        Since the pseudo outcome is a fraction of residuals, we add a small
        constant ``epsilon`` to the denominator in order to avoid numerical problems.
        """
        y_estimates = self.predict_nuisance(
            X=X,
            is_oos=False,
            model_kind=OUTCOME_MODEL,
            model_ord=0,
        )
        if self.is_classification:
            y_estimates = y_estimates[:, 1]
        y_residuals = y - y_estimates

        w_residuals = (
            w
            - self.predict_nuisance(
                X=X, is_oos=False, model_kind=PROPENSITY_MODEL, model_ord=0
            )[:, 1]
        )

        # We want to avoid a case in which adding epsilon actually causes numerical
        # harm, e.g. if w_residuals is approximately -epsilon. Therefore we add
        # epsilon in the existing direction pointing away from 0.
        epsilons = np.where(w_residuals < 0, -1, 1) * epsilon

        pseudo_outcomes = y_residuals / (w_residuals + epsilons)
        weights = np.square(w_residuals)

        return pseudo_outcomes, weights
