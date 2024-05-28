# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
from sklearn.metrics import log_loss, root_mean_squared_error
from typing_extensions import Self

from metalearners._typing import OosMethod
from metalearners._utils import (
    Matrix,
    Vector,
    clip_element_absolute_value_to_epsilon,
    function_has_argument,
    index_matrix,
    validate_all_vectors_same_index,
    validate_valid_treatment_variant_not_control,
)
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import (
    PROPENSITY_MODEL,
    TREATMENT_MODEL,
    MetaLearner,
    _ModelSpecifications,
)

OUTCOME_MODEL = "outcome_model"

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

        * binary classes in case of a classification outcome

    The R-Learner contains two nuisance models

        * a ``"propensity_model"`` estimating :math:`\Pr[W=1|X]`
        * an ``"outcome_model"`` estimating :math:`\mathbb{E}[Y|X]`

    and one treatment model per treatment variant which isn't control

        * ``"treatment_model"`` which estimates :math:`\mathbb{E}[Y(k) - Y(0) | X]`

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
        return {
            TREATMENT_MODEL: _ModelSpecifications(
                cardinality=lambda ml: ml.n_variants - 1,
                predict_method=lambda _: "predict",
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
        epsilon: float = _EPSILON,
    ) -> Self:
        self._validate_treatment(w)
        self._validate_outcome(y)

        self._variants_indices = []

        self.fit_nuisance(
            X=X,
            y=w,
            model_kind=PROPENSITY_MODEL,
            model_ord=0,
            n_jobs_cross_fitting=n_jobs_cross_fitting,
        )
        self.fit_nuisance(
            X=X,
            y=y,
            model_kind=OUTCOME_MODEL,
            model_ord=0,
            n_jobs_cross_fitting=n_jobs_cross_fitting,
        )

        for treatment_variant in range(1, self.n_variants):

            is_treatment = w == treatment_variant
            is_control = w == 0
            mask = is_treatment | is_control

            self._variants_indices.append(mask)

            pseudo_outcomes, weights = self._pseudo_outcome_and_weights(
                X=X,
                w=w,
                y=y,
                treatment_variant=treatment_variant,
                mask=mask,
                epsilon=epsilon,
            )

            X_filtered = index_matrix(X, mask)

            self.fit_treatment(
                X=X_filtered,
                y=pseudo_outcomes,
                model_kind=TREATMENT_MODEL,
                model_ord=treatment_variant - 1,
                fit_params={_SAMPLE_WEIGHT: weights},
                n_jobs_cross_fitting=n_jobs_cross_fitting,
            )
        return self

    def predict(
        self,
        X,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        n_outputs = 2 if self.is_classification else 1
        tau_hat = np.zeros((len(X), self.n_variants - 1, n_outputs))

        if is_oos:

            for treatment_variant in range(1, self.n_variants):
                variant_estimates = self.predict_treatment(
                    X,
                    is_oos=is_oos,
                    oos_method=oos_method,
                    model_kind=TREATMENT_MODEL,
                    model_ord=treatment_variant - 1,
                )
                if self.is_classification:
                    # This is to be consistent with other MetaLearners (e.g. S and T) that automatically
                    # work with multiclass outcomes and return the CATE estimate for each class. As the R-Learner only
                    # works with binary classes (the pseudo outcome formula does not make sense with
                    # multiple classes unless some adaptation is done) we can manually infer the
                    # CATE estimate for the complementary class  -- returning a matrix of shape (N, 2).
                    variant_estimates = np.stack(
                        [-variant_estimates, variant_estimates], axis=-1
                    )
                variant_estimates = variant_estimates.reshape(len(X), n_outputs)
                tau_hat[:, treatment_variant - 1, :] = variant_estimates

            return tau_hat

        for treatment_variant in range(1, self.n_variants):
            variant_indices = self._variants_indices[treatment_variant - 1]

            variant_estimates = self.predict_treatment(
                index_matrix(X, variant_indices),
                is_oos=False,
                model_kind=TREATMENT_MODEL,
                model_ord=treatment_variant - 1,
            )
            if sum(~variant_indices) > 0:
                non_variant_estimates = self.predict_treatment(
                    index_matrix(X, ~variant_indices),
                    is_oos=True,
                    oos_method=oos_method,
                    model_kind=TREATMENT_MODEL,
                    model_ord=treatment_variant - 1,
                )
            if self.is_classification:
                # This is to be consistent with other MetaLearners (e.g. S and T) that automatically
                # work with multiclass outcomes and return the CATE estimate for each class. As the R-Learner only
                # works with binary classes (the pseudo outcome formula does not make sense with
                # multiple classes unless some adaptation is done) we can manually infer the
                # CATE estimate for the complementary class  -- returning a matrix of shape (N, 2).
                variant_estimates = np.stack(
                    [-variant_estimates, variant_estimates], axis=-1
                )
                if sum(~variant_indices) > 0:
                    non_variant_estimates = np.stack(
                        [-non_variant_estimates, non_variant_estimates], axis=-1
                    )
            variant_estimates = variant_estimates.reshape(
                (sum(variant_indices), n_outputs)
            )
            if sum(~variant_indices) > 0:
                non_variant_estimates = non_variant_estimates.reshape(
                    (sum(~variant_indices), n_outputs)
                )
                tau_hat[~variant_indices, treatment_variant - 1] = non_variant_estimates

            tau_hat[variant_indices, treatment_variant - 1] = variant_estimates
        return tau_hat

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
        )
        propensity_evaluation = {"propensity_cross_entropy": log_loss(w, w_hat)}

        y_hat = self.predict_nuisance(
            X=X,
            is_oos=is_oos,
            oos_method=oos_method,
            model_kind=OUTCOME_MODEL,
            model_ord=0,
        )
        if self.is_classification:
            y_hat = y_hat[:, 1]

        outcome_evaluation = (
            {"outcome_log_loss": log_loss(y, y_hat)}
            if self.is_classification
            else {"outcome_rmse": root_mean_squared_error(y, y_hat)}
        )

        treatment_evaluation = {}
        tau_hat = self.predict(X=X, is_oos=is_oos, oos_method=oos_method)
        for treatment_variant in range(1, self.n_variants):
            propensity_estimates = w_hat[:, treatment_variant] / (
                w_hat[:, 0] + w_hat[:, treatment_variant]
            )
            cate_estimates = (
                tau_hat[:, treatment_variant - 1, 1]
                if self.is_classification
                else tau_hat[:, treatment_variant - 1, 0]
            )
            treatment_evaluation[f"r_loss_{treatment_variant}_vs_0"] = r_loss(
                cate_estimates=cate_estimates,
                outcome_estimates=y_hat,
                propensity_scores=propensity_estimates,
                outcomes=y,
                treatments=w,
            )

        return propensity_evaluation | outcome_evaluation | treatment_evaluation

    def _pseudo_outcome_and_weights(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        treatment_variant: int,
        mask: Vector | None = None,
        epsilon: float = _EPSILON,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the R-Learner pseudo outcome and corresponding weights.

        Importantly, this method assumes to be applied on in-sample data.
        In other words, ``is_oos`` will always be set to ``False`` when calling
        ``predict_nuisance``.

        If ``mask`` is provided, the retuned pseudo outcomes and weights are only
        with respect the observations that the mask selects.

        Since the pseudo outcome is a fraction of residuals, we add a small
        constant ``epsilon`` to the denominator in order to avoid numerical problems.
        """
        if mask is None:
            mask = np.ones(len(X), dtype=bool)

        validate_valid_treatment_variant_not_control(treatment_variant, self.n_variants)

        # Note that if we already applied the mask as an input to this call, we wouldn't
        # be able to match original observations with their corresponding folds.
        y_estimates = self.predict_nuisance(
            X=X,
            is_oos=False,
            model_kind=OUTCOME_MODEL,
            model_ord=0,
        )[mask]
        w_estimates = self.predict_nuisance(
            X=X, is_oos=False, model_kind=PROPENSITY_MODEL, model_ord=0
        )[mask]
        w_estimates_binarized = w_estimates[:, treatment_variant] / (
            w_estimates[:, 0] + w_estimates[:, treatment_variant]
        )

        if self.is_classification:
            y_estimates = y_estimates[:, 1]

        y_residuals = y[mask] - y_estimates

        w_binarized = w[mask] == treatment_variant
        w_residuals = w_binarized - w_estimates_binarized
        w_residuals_padded = clip_element_absolute_value_to_epsilon(
            w_residuals, epsilon
        )

        pseudo_outcomes = y_residuals / w_residuals_padded
        weights = np.square(w_residuals)

        return pseudo_outcomes, weights
