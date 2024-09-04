# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed
from typing_extensions import Any, Self

from metalearners._typing import (
    Features,
    Matrix,
    ModelFactory,
    OosMethod,
    Params,
    Scoring,
    SplitIndices,
    Vector,
    _ScikitModel,
)
from metalearners._utils import (
    check_spox_installed,
    clip_element_absolute_value_to_epsilon,
    copydoc,
    get_one,
    get_predict,
    get_predict_proba,
    index_matrix,
    infer_input_dict,
    safe_len,
    validate_valid_treatment_variant_not_control,
    warning_experimental_feature,
)
from metalearners.cross_fit_estimator import OVERALL, CrossFitEstimator
from metalearners.metalearner import (
    NUISANCE,
    PROPENSITY_MODEL,
    TREATMENT,
    TREATMENT_MODEL,
    VARIANT_OUTCOME_MODEL,
    MetaLearner,
    _ConditionalAverageOutcomeMetaLearner,
    _evaluate_model_kind,
    _fit_cross_fit_estimator_joblib,
    _ModelSpecifications,
    _ParallelJoblibSpecification,
    get_overall_estimators,
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

    If ``adaptive_clipping`` is set to ``True``, then the pseudo outcomes are computed using
    adaptive propensity clipping described in section 4.1, equation *DR-Switch* of
    `Mahajan et al. (2024) <https://arxiv.org/pdf/2211.01939>`_.
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

    def __init__(
        self,
        is_classification: bool,
        n_variants: int,
        nuisance_model_factory: ModelFactory | None = None,
        treatment_model_factory: ModelFactory | None = None,
        propensity_model_factory: type[_ScikitModel] | None = None,
        nuisance_model_params: Params | dict[str, Params] | None = None,
        treatment_model_params: Params | dict[str, Params] | None = None,
        propensity_model_params: Params | None = None,
        fitted_nuisance_models: dict[str, list[CrossFitEstimator]] | None = None,
        fitted_propensity_model: CrossFitEstimator | None = None,
        feature_set: Features | dict[str, Features] | None = None,
        n_folds: int | dict[str, int] = 10,
        random_state: int | None = None,
        adaptive_clipping: bool = False,
    ):
        super().__init__(
            nuisance_model_factory=nuisance_model_factory,
            is_classification=is_classification,
            n_variants=n_variants,
            treatment_model_factory=treatment_model_factory,
            propensity_model_factory=propensity_model_factory,
            nuisance_model_params=nuisance_model_params,
            treatment_model_params=treatment_model_params,
            propensity_model_params=propensity_model_params,
            fitted_nuisance_models=fitted_nuisance_models,
            fitted_propensity_model=fitted_propensity_model,
            feature_set=feature_set,
            n_folds=n_folds,
            random_state=random_state,
        )
        self.adaptive_clipping = adaptive_clipping

    def fit_all_nuisance(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        n_jobs_cross_fitting: int | None = None,
        fit_params: dict | None = None,
        synchronize_cross_fitting: bool = True,
        n_jobs_base_learners: int | None = None,
    ) -> Self:
        self._validate_treatment(w)
        self._validate_outcome(y, w)

        self._treatment_variants_mask = []

        qualified_fit_params = self._qualified_fit_params(fit_params)

        for treatment_variant in range(self.n_variants):
            self._treatment_variants_mask.append(w == treatment_variant)

        self._cv_split_indices: SplitIndices | None

        if synchronize_cross_fitting:
            self._cv_split_indices = self._split(X)
        else:
            self._cv_split_indices = None

        nuisance_jobs: list[_ParallelJoblibSpecification | None] = []
        for treatment_variant in range(self.n_variants):
            nuisance_jobs.append(
                self._nuisance_joblib_specifications(
                    X=index_matrix(X, self._treatment_variants_mask[treatment_variant]),
                    y=y[self._treatment_variants_mask[treatment_variant]],
                    model_kind=VARIANT_OUTCOME_MODEL,
                    model_ord=treatment_variant,
                    n_jobs_cross_fitting=n_jobs_cross_fitting,
                    fit_params=qualified_fit_params[NUISANCE][VARIANT_OUTCOME_MODEL],
                )
            )

        nuisance_jobs.append(
            self._nuisance_joblib_specifications(
                X=X,
                y=w,
                model_kind=PROPENSITY_MODEL,
                model_ord=0,
                n_jobs_cross_fitting=n_jobs_cross_fitting,
                fit_params=qualified_fit_params[NUISANCE][PROPENSITY_MODEL],
                cv=self._cv_split_indices,
            )
        )

        parallel = Parallel(n_jobs=n_jobs_base_learners)
        results = parallel(
            delayed(_fit_cross_fit_estimator_joblib)(job)
            for job in nuisance_jobs
            if job is not None
        )

        self._assign_joblib_nuisance_results(results)
        self._nuisance_models_fit = True
        return self

    def fit_all_treatment(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        n_jobs_cross_fitting: int | None = None,
        fit_params: dict | None = None,
        synchronize_cross_fitting: bool = True,
        n_jobs_base_learners: int | None = None,
    ) -> Self:
        if not hasattr(self, "_cv_split_indices"):
            raise ValueError(
                "The nuisance models need to be fitted before fitting the treatment models."
                "In particular, the MetaLearner's attribute _cv_split_indices, "
                "typically set during nuisance fitting, does not exist."
            )
        qualified_fit_params = self._qualified_fit_params(fit_params)
        treatment_jobs: list[_ParallelJoblibSpecification] = []
        for treatment_variant in range(1, self.n_variants):
            pseudo_outcomes = self._pseudo_outcome(
                X=X,
                w=w,
                y=y,
                treatment_variant=treatment_variant,
                is_oos=False,
            )

            treatment_jobs.append(
                self._treatment_joblib_specifications(
                    X=X,
                    y=pseudo_outcomes,
                    model_kind=TREATMENT_MODEL,
                    model_ord=treatment_variant - 1,
                    n_jobs_cross_fitting=n_jobs_cross_fitting,
                    fit_params=qualified_fit_params[TREATMENT][TREATMENT_MODEL],
                    cv=self._cv_split_indices,
                )
            )
        parallel = Parallel(n_jobs=n_jobs_base_learners)
        results = parallel(
            delayed(_fit_cross_fit_estimator_joblib)(job) for job in treatment_jobs
        )
        self._assign_joblib_treatment_results(results)
        return self

    def predict(
        self,
        X,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        n_outputs = 2 if self.is_classification else 1
        estimates = np.zeros((safe_len(X), self.n_variants - 1, n_outputs))
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
        scoring: Scoring | None = None,
    ) -> dict[str, float]:
        safe_scoring = self._scoring(scoring)

        variant_outcome_evaluation = _evaluate_model_kind(
            cfes=self._nuisance_models[VARIANT_OUTCOME_MODEL],
            Xs=[X[w == tv] for tv in range(self.n_variants)],
            ys=[y[w == tv] for tv in range(self.n_variants)],
            scorers=safe_scoring[VARIANT_OUTCOME_MODEL],
            model_kind=VARIANT_OUTCOME_MODEL,
            is_oos=is_oos,
            oos_method=oos_method,
            is_treatment_model=False,
            feature_set=self.feature_set[VARIANT_OUTCOME_MODEL],
        )

        propensity_evaluation = _evaluate_model_kind(
            cfes=self._nuisance_models[PROPENSITY_MODEL],
            Xs=[X],
            ys=[w],
            scorers=safe_scoring[PROPENSITY_MODEL],
            model_kind=PROPENSITY_MODEL,
            is_oos=is_oos,
            oos_method=oos_method,
            is_treatment_model=False,
            feature_set=self.feature_set[PROPENSITY_MODEL],
        )

        pseudo_outcome: list[np.ndarray] = []
        for treatment_variant in range(1, self.n_variants):
            tv_pseudo_outcome = self._pseudo_outcome(
                X=X,
                y=y,
                w=w,
                treatment_variant=treatment_variant,
                is_oos=is_oos,
                oos_method=oos_method,
            )
            pseudo_outcome.append(tv_pseudo_outcome)

        treatment_evaluation = _evaluate_model_kind(
            self._treatment_models[TREATMENT_MODEL],
            Xs=[X for _ in range(1, self.n_variants)],
            ys=pseudo_outcome,
            scorers=safe_scoring[TREATMENT_MODEL],
            model_kind=TREATMENT_MODEL,
            is_oos=is_oos,
            oos_method=oos_method,
            is_treatment_model=True,
            feature_set=self.feature_set[TREATMENT_MODEL],
        )

        return variant_outcome_evaluation | propensity_evaluation | treatment_evaluation

    def average_treatment_effect(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        is_oos: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Average Treatment Effect (ATE) for each treatment variant using the
        Augmented IPW estimator (Robins et al 1994). Does not require fitting a second-
        stage treatment model: it uses the pseudo-outcome alone and computes the point
        estimate and standard error. Can be used following the
        :meth:`~metalearners.drlearner.DRLearner.fit_all_nuisance` method.

        Args:
            X (Matrix): Covariate matrix
            y (Vector): Outcome vector
            w (Vector): Treatment vector
            is_oos (bool): indicator whether data is out of sample

        Returns:
            np.ndarray: Treatment effect for each treatment variant.
            np.ndarray: Standard error for each treatment variant.
        """
        if not self._nuisance_models_fit:
            raise ValueError(
                "The nuisance models need to be fitted before computing the treatment effect."
            )
        gamma_matrix = np.zeros((safe_len(X), self.n_variants - 1))
        for treatment_variant in range(1, self.n_variants):
            gamma_matrix[:, treatment_variant - 1] = self._pseudo_outcome(
                X=X,
                w=w,
                y=y,
                treatment_variant=treatment_variant,
                is_oos=is_oos,
            )
        treatment_effect = gamma_matrix.mean(axis=0)
        standard_error = gamma_matrix.std(axis=0) / np.sqrt(safe_len(X))
        return treatment_effect, standard_error

    def _pseudo_outcome(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        treatment_variant: int,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
        epsilon: float = _EPSILON,
    ) -> np.ndarray:
        """Compute the DR-Learner pseudo outcome."""
        validate_valid_treatment_variant_not_control(treatment_variant, self.n_variants)

        conditional_average_outcome_estimates = (
            self.predict_conditional_average_outcomes(
                X=X,
                is_oos=is_oos,
                oos_method=oos_method,
            )
        )

        propensity_estimates = self.predict_nuisance(
            X=X,
            is_oos=is_oos,
            oos_method=oos_method,
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

        if self.adaptive_clipping:
            t_pseudo_outcome = y1_estimate - y0_estimate
            pseudo_outcome = np.where(
                propensity_estimates.min(axis=1) < epsilon,
                t_pseudo_outcome,
                pseudo_outcome,
            )

        return pseudo_outcome

    @property
    def init_args(self) -> dict[str, Any]:
        return super().init_args | {"adaptive_clipping": self.adaptive_clipping}

    def _necessary_onnx_models(self) -> dict[str, list[_ScikitModel]]:
        return {
            TREATMENT_MODEL: get_overall_estimators(
                self._treatment_models[TREATMENT_MODEL]
            )
        }

    @copydoc(MetaLearner._build_onnx, sep="")
    def _build_onnx(self, models: Mapping[str, Sequence], output_name: str = "tau"):
        """In the DRLearner case, the necessary models are:

        * ``"treatment_model"``
        """
        warning_experimental_feature("_build_onnx")
        check_spox_installed()
        import spox.opset.ai.onnx.v21 as op
        from onnx.checker import check_model
        from spox import Var, build, inline

        self._validate_feature_set_none()
        self._validate_onnx_models(models, set(self._necessary_onnx_models().keys()))

        input_dict = infer_input_dict(models[TREATMENT_MODEL][0])

        treatment_output_name = models[TREATMENT_MODEL][0].graph.output[0].name

        tau_hat: list[Var] = []
        for m in models[TREATMENT_MODEL]:
            tau_hat_tv = inline(m)(**input_dict)[treatment_output_name]
            tau_hat_tv = op.unsqueeze(tau_hat_tv, axes=op.constant(value_int=2))
            if self.is_classification:
                if self._supports_multi_class():
                    raise ValueError(
                        "ONNX conversion is not implemented for a multi-class output."
                    )
                tau_hat_tv = op.concat([op.neg(tau_hat_tv), tau_hat_tv], axis=-1)
            tau_hat.append(tau_hat_tv)

        cate = op.concat(tau_hat, axis=1)
        final_model = build(input_dict, {output_name: cate})
        check_model(final_model, full_check=True)
        return final_model
