# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed
from typing_extensions import Self

from metalearners._typing import Matrix, OosMethod, Scoring, Vector, _ScikitModel
from metalearners._utils import (
    check_spox_installed,
    copydoc,
    get_one,
    get_predict,
    get_predict_proba,
    index_matrix,
    infer_input_dict,
    infer_probabilities_output,
    validate_valid_treatment_variant_not_control,
    warning_experimental_feature,
)
from metalearners.cross_fit_estimator import MEDIAN, OVERALL
from metalearners.metalearner import (
    NUISANCE,
    PROPENSITY_MODEL,
    TREATMENT,
    VARIANT_OUTCOME_MODEL,
    MetaLearner,
    _ConditionalAverageOutcomeMetaLearner,
    _evaluate_model_kind,
    _fit_cross_fit_estimator_joblib,
    _ModelSpecifications,
    _ParallelJoblibSpecification,
    get_overall_estimators,
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
                cardinality=MetaLearner._get_n_variants,
                predict_method=MetaLearner._outcome_predict_method,
            ),
            PROPENSITY_MODEL: _ModelSpecifications(
                cardinality=get_one,
                predict_method=get_predict_proba,
            ),
        }

    @classmethod
    def treatment_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            CONTROL_EFFECT_MODEL: _ModelSpecifications(
                cardinality=MetaLearner._get_n_variants_minus_one,
                predict_method=get_predict,
            ),
            TREATMENT_EFFECT_MODEL: _ModelSpecifications(
                cardinality=MetaLearner._get_n_variants_minus_one,
                predict_method=get_predict,
            ),
        }

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return True

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return False

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

        self._treatment_variants_indices = []

        qualified_fit_params = self._qualified_fit_params(fit_params)

        # TODO: Move this to object initialization.
        if not synchronize_cross_fitting:
            raise ValueError(
                "The X-Learner does not support synchronize_cross_fitting=False."
            )

        self._cv_split_indices = self._split(X)
        self._treatment_cv_split_indices = {}

        for treatment_variant in range(self.n_variants):
            self._treatment_variants_indices.append(w == treatment_variant)
            treatment_indices = np.where(
                self._treatment_variants_indices[treatment_variant]
            )[0]
            self._treatment_cv_split_indices[treatment_variant] = [
                (
                    np.intersect1d(train_indices, treatment_indices),
                    np.intersect1d(test_indices, treatment_indices),
                )
                for train_indices, test_indices in self._cv_split_indices
            ]

        nuisance_jobs: list[_ParallelJoblibSpecification | None] = []
        for treatment_variant in range(self.n_variants):
            nuisance_jobs.append(
                self._nuisance_joblib_specifications(
                    X=X,
                    y=y,
                    model_kind=VARIANT_OUTCOME_MODEL,
                    model_ord=treatment_variant,
                    n_jobs_cross_fitting=n_jobs_cross_fitting,
                    fit_params=qualified_fit_params[NUISANCE][VARIANT_OUTCOME_MODEL],
                    cv=self._treatment_cv_split_indices[treatment_variant],
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
            )
        )

        parallel = Parallel(n_jobs=n_jobs_base_learners)
        results = parallel(
            delayed(_fit_cross_fit_estimator_joblib)(job)
            for job in nuisance_jobs
            if job is not None
        )
        self._assign_joblib_nuisance_results(results)

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
        if self._treatment_variants_indices is None:
            raise ValueError(
                "The nuisance models need to be fitted before fitting the treatment models. "
                "In particular, the MetaLearner's attribute _treatment_variant_indices, "
                "typically set during nuisance fitting, is None."
            )
        if not hasattr(self, "_treatment_cv_split_indices"):
            raise ValueError(
                "The nuisance models need to be fitted before fitting the treatment models. "
                "In particular, the MetaLearner's attribute _treatment_cv_split_indices, "
                "typically set during nuisance fitting, does not exist."
            )
        qualified_fit_params = self._qualified_fit_params(fit_params)

        treatment_jobs: list[_ParallelJoblibSpecification] = []

        conditional_average_outcome_estimates = (
            self.predict_conditional_average_outcomes(
                X=X,
                is_oos=False,
            )
        )
        for treatment_variant in range(1, self.n_variants):
            imputed_te_control, imputed_te_treatment = self._pseudo_outcome(
                y, w, treatment_variant, conditional_average_outcome_estimates
            )

            treatment_jobs.append(
                self._treatment_joblib_specifications(
                    X=X,
                    y=imputed_te_treatment,
                    model_kind=TREATMENT_EFFECT_MODEL,
                    model_ord=treatment_variant - 1,
                    n_jobs_cross_fitting=n_jobs_cross_fitting,
                    fit_params=qualified_fit_params[TREATMENT][TREATMENT_EFFECT_MODEL],
                    cv=self._treatment_cv_split_indices[treatment_variant],
                )
            )

            treatment_jobs.append(
                self._treatment_joblib_specifications(
                    X=X,
                    y=imputed_te_control,
                    model_kind=CONTROL_EFFECT_MODEL,
                    model_ord=treatment_variant - 1,
                    n_jobs_cross_fitting=n_jobs_cross_fitting,
                    fit_params=qualified_fit_params[TREATMENT][CONTROL_EFFECT_MODEL],
                    cv=self._treatment_cv_split_indices[0],
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
        X: Matrix,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        if self._treatment_variants_indices is None:
            raise ValueError(
                "The MetaLearner needs to be fitted before predicting. "
                "In particular, the X-Learner's attribute _treatment_variant_indices, "
                "typically set during fitting, is None."
            )
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
                    X=X,
                    model_kind=TREATMENT_EFFECT_MODEL,
                    model_ord=treatment_variant - 1,
                    is_oos=False,
                )[treatment_variant_indices]
                tau_hat_control[control_indices] = self.predict_treatment(
                    X=X,
                    model_kind=CONTROL_EFFECT_MODEL,
                    model_ord=treatment_variant - 1,
                    is_oos=False,
                )[control_indices]
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
        scoring: Scoring | None = None,
    ) -> dict[str, float]:
        safe_scoring = self._scoring(scoring)

        variant_outcome_evaluation = _evaluate_model_kind(
            cfes=self._nuisance_models[VARIANT_OUTCOME_MODEL],
            Xs=[X] * self.n_variants,
            ys=[y] * self.n_variants,
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

        conditional_average_outcome_estimates = (
            self.predict_conditional_average_outcomes(
                X=X,
                is_oos=is_oos,
                oos_method=oos_method,
            )
        )

        imputed_te_control: list[np.ndarray] = []
        imputed_te_treatment: list[np.ndarray] = []
        for treatment_variant in range(1, self.n_variants):
            tv_imputed_te_control, tv_imputed_te_treatment = self._pseudo_outcome(
                y, w, treatment_variant, conditional_average_outcome_estimates
            )
            imputed_te_control.append(tv_imputed_te_control)
            imputed_te_treatment.append(tv_imputed_te_treatment)

        te_treatment_evaluation = _evaluate_model_kind(
            self._treatment_models[TREATMENT_EFFECT_MODEL],
            Xs=[X] * self.n_variants,
            ys=imputed_te_treatment,
            scorers=safe_scoring[TREATMENT_EFFECT_MODEL],
            model_kind=TREATMENT_EFFECT_MODEL,
            is_oos=is_oos,
            oos_method=oos_method,
            is_treatment_model=True,
            feature_set=self.feature_set[TREATMENT_EFFECT_MODEL],
        )

        te_control_evaluation = _evaluate_model_kind(
            self._treatment_models[CONTROL_EFFECT_MODEL],
            Xs=[X] * self.n_variants,
            ys=imputed_te_control,
            scorers=safe_scoring[CONTROL_EFFECT_MODEL],
            model_kind=CONTROL_EFFECT_MODEL,
            is_oos=is_oos,
            oos_method=oos_method,
            is_treatment_model=True,
            feature_set=self.feature_set[CONTROL_EFFECT_MODEL],
        )

        return (
            variant_outcome_evaluation
            | propensity_evaluation
            | te_treatment_evaluation
            | te_control_evaluation
        )

    def _pseudo_outcome(
        self,
        y: Vector,
        w: Vector,
        treatment_variant: int,
        conditional_average_outcome_estimates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the X-Learner pseudo outcome.

        Importantly this method only returns the imputed treatment effect using the control
        outcome model for the observatons treated with variant ``treatment_variant`` and
        the imputed treatment effect using the treatment variant outcome model for the
        observations in the control group.

        This function can be used with both in-sample or out-of-sample data.
        """
        validate_valid_treatment_variant_not_control(treatment_variant, self.n_variants)
        treatment_outcome = conditional_average_outcome_estimates[:, treatment_variant]
        control_outcome = conditional_average_outcome_estimates[:, 0]

        if self.is_classification:
            # Get the probability of positive class, multiclass is currently not supported.
            control_outcome = control_outcome[:, 1]
            treatment_outcome = treatment_outcome[:, 1]
        else:
            control_outcome = control_outcome[:, 0]
            treatment_outcome = treatment_outcome[:, 0]

        imputed_te_treatment = y - control_outcome
        imputed_te_control = treatment_outcome - y

        return imputed_te_control, imputed_te_treatment

    def _necessary_onnx_models(self) -> dict[str, list[_ScikitModel]]:
        return {
            PROPENSITY_MODEL: get_overall_estimators(
                self._nuisance_models[PROPENSITY_MODEL]
            ),
            CONTROL_EFFECT_MODEL: get_overall_estimators(
                self._treatment_models[CONTROL_EFFECT_MODEL]
            ),
            TREATMENT_EFFECT_MODEL: get_overall_estimators(
                self._treatment_models[TREATMENT_EFFECT_MODEL]
            ),
        }

    @copydoc(MetaLearner._build_onnx, sep="")
    def _build_onnx(self, models: Mapping[str, Sequence], output_name: str = "tau"):
        """In the XLearner case, the necessary models are:

        * ``"propensity_model"``
        * ``"control_effect_model"``
        * ``"treatment_effect_model"``
        """
        warning_experimental_feature("_build_onnx")
        check_spox_installed()
        import spox.opset.ai.onnx.v21 as op
        from onnx.checker import check_model
        from spox import Var, build, inline

        self._validate_feature_set_none()
        self._validate_onnx_models(models, set(self._necessary_onnx_models().keys()))

        input_dict = infer_input_dict(models[PROPENSITY_MODEL][0])

        treatment_output_name = models[CONTROL_EFFECT_MODEL][0].graph.output[0].name

        tau_hat_control: list[Var] = []
        for m in models[CONTROL_EFFECT_MODEL]:
            tau_hat_control.append(inline(m)(**input_dict)[treatment_output_name])
        tau_hat_effect: list[Var] = []
        for m in models[TREATMENT_EFFECT_MODEL]:
            tau_hat_effect.append(inline(m)(**input_dict)[treatment_output_name])

        _, propensity_output_name = infer_probabilities_output(
            models[PROPENSITY_MODEL][0]
        )

        propensity_scores = inline(models[PROPENSITY_MODEL][0])(**input_dict)[
            propensity_output_name
        ]

        slice_0 = op.slice(
            propensity_scores,
            starts=op.const(np.array([0])),
            ends=op.const(np.array([1])),
            axes=op.const(np.array([1])),
        )

        tau_hat = []
        for tv in range(self.n_variants - 1):
            slice_tv = op.slice(
                propensity_scores,
                starts=op.const(np.array([tv + 1])),
                ends=op.const(np.array([tv + 2])),
                axes=op.const(np.array([1])),
            )
            denominator = op.add(slice_0, slice_tv)
            scaled_propensity = op.div(slice_tv, denominator)
            tau_hat_tv = op.add(
                op.mul(scaled_propensity, tau_hat_control[tv]),
                op.mul(
                    op.sub(op.constant(value_float=1), scaled_propensity),
                    tau_hat_effect[tv],
                ),
            )
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

    def predict_conditional_average_outcomes(
        self, X: Matrix, is_oos: bool, oos_method: OosMethod = OVERALL
    ) -> np.ndarray:
        if self._treatment_variants_indices is None:
            raise ValueError(
                "The metalearner needs to be fitted before predicting."
                "In particular, the MetaLearner's attribute _treatment_variant_indices, "
                "typically set during fitting, is None."
            )
        # TODO: Consider multiprocessing
        n_obs = len(X)
        cao_tensor = self._nuisance_tensors(n_obs)[VARIANT_OUTCOME_MODEL][0]
        predict_method_name = self.nuisance_model_specifications()[
            VARIANT_OUTCOME_MODEL
        ]["predict_method"](self)
        conditional_average_outcomes_list = []

        for tv in range(self.n_variants):
            if is_oos:
                conditional_average_outcomes_list.append(
                    self.predict_nuisance(
                        X=X,
                        model_kind=VARIANT_OUTCOME_MODEL,
                        model_ord=tv,
                        is_oos=is_oos,
                        oos_method=oos_method,
                    )
                )
            else:
                # TODO: Consider moving this logic to CrossFitEstimator.predict.
                cfe = self._nuisance_models[VARIANT_OUTCOME_MODEL][tv]
                conditional_average_outcome_estimates = cao_tensor.copy()

                for fold_index, (train_indices, prediction_indices) in enumerate(
                    self._cv_split_indices
                ):
                    fold_model = cfe._estimators[fold_index]
                    predict_method = getattr(fold_model, predict_method_name)
                    fold_estimates = predict_method(index_matrix(X, prediction_indices))
                    conditional_average_outcome_estimates[prediction_indices] = (
                        fold_estimates
                    )

                conditional_average_outcomes_list.append(
                    conditional_average_outcome_estimates
                )

        return np.stack(conditional_average_outcomes_list, axis=1).reshape(
            n_obs, self.n_variants, -1
        )
