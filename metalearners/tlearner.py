# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed
from typing_extensions import Self

from metalearners._typing import Matrix, OosMethod, Scoring, Vector
from metalearners._utils import (
    check_onnx_installed,
    check_spox_installed,
    index_matrix,
    infer_dtype_and_shape_onnx,
    infer_probabilities_output,
)
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import (
    NUISANCE,
    VARIANT_OUTCOME_MODEL,
    MetaLearner,
    _ConditionalAverageOutcomeMetaLearner,
    _evaluate_model_kind,
    _fit_cross_fit_estimator_joblib,
    _ModelSpecifications,
    _ParallelJoblibSpecification,
)


class TLearner(_ConditionalAverageOutcomeMetaLearner):
    """T-Learner for CATE estimation as described by `Kuenzel et al (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_."""

    # TODO: Parametrize instantiation of the TLearner as to add an optional
    # second-stage model regularizing the treatment effects, rather than
    # just the respective outcomes individually.

    @classmethod
    def nuisance_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return {
            VARIANT_OUTCOME_MODEL: _ModelSpecifications(
                cardinality=MetaLearner._get_n_variants,
                predict_method=MetaLearner._outcome_predict_method,
            ),
        }

    @classmethod
    def treatment_model_specifications(cls) -> dict[str, _ModelSpecifications]:
        return dict()

    @classmethod
    def _supports_multi_treatment(cls) -> bool:
        return True

    @classmethod
    def _supports_multi_class(cls) -> bool:
        return True

    def fit(
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
        self._validate_outcome(y)

        self._treatment_variants_indices = []

        for v in range(self.n_variants):
            self._treatment_variants_indices.append(w == v)

        qualified_fit_params = self._qualified_fit_params(fit_params)

        nuisance_jobs: list[_ParallelJoblibSpecification | None] = []
        for treatment_variant in range(self.n_variants):
            nuisance_jobs.append(
                self._nuisance_joblib_specifications(
                    X=index_matrix(
                        X, self._treatment_variants_indices[treatment_variant]
                    ),
                    y=y[self._treatment_variants_indices[treatment_variant]],
                    model_kind=VARIANT_OUTCOME_MODEL,
                    model_ord=treatment_variant,
                    n_jobs_cross_fitting=n_jobs_cross_fitting,
                    fit_params=qualified_fit_params[NUISANCE][VARIANT_OUTCOME_MODEL],
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

    def predict(
        self,
        X,
        is_oos: bool,
        oos_method: OosMethod = OVERALL,
    ) -> np.ndarray:
        conditional_average_outcomes = self.predict_conditional_average_outcomes(
            X=X, is_oos=is_oos, oos_method=oos_method
        )

        return conditional_average_outcomes[:, 1:] - (
            conditional_average_outcomes[:, [0]]
        )

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

        return _evaluate_model_kind(
            cfes=self._nuisance_models[VARIANT_OUTCOME_MODEL],
            Xs=[X[w == tv] for tv in range(self.n_variants)],
            ys=[y[w == tv] for tv in range(self.n_variants)],
            scorers=safe_scoring[VARIANT_OUTCOME_MODEL],
            model_kind=VARIANT_OUTCOME_MODEL,
            is_oos=is_oos,
            oos_method=oos_method,
            is_treatment_model=False,
        )

    # TODO: Fix typing without importing onnx
    def build_onnx(self, models: Mapping[str, Sequence]):
        check_onnx_installed()
        check_spox_installed()
        import spox.opset.ai.onnx.v21 as op
        from onnx.checker import check_model
        from spox import Tensor, argument, build, inline

        self._validate_onnx_models(models, {VARIANT_OUTCOME_MODEL})

        input_dtype, input_shape = infer_dtype_and_shape_onnx(
            models[VARIANT_OUTCOME_MODEL][0].graph.input[0]
        )

        if not self.is_classification:
            output_index = 0
            output_name = models[VARIANT_OUTCOME_MODEL][0].graph.output[0].name
        else:
            output_index, output_name = infer_probabilities_output(
                models[VARIANT_OUTCOME_MODEL][0]
            )

        output_dtype, output_shape = infer_dtype_and_shape_onnx(
            models[VARIANT_OUTCOME_MODEL][0].graph.output[output_index]
        )

        input_tensor = argument(Tensor(input_dtype, input_shape))

        a = argument(Tensor(output_dtype, output_shape))
        b = argument(Tensor(output_dtype, output_shape))
        subtraction = op.sub(a, b)
        sub_model = build({"a": a, "b": b}, {"subtraction": subtraction})

        output_0 = inline(models[VARIANT_OUTCOME_MODEL][0])(input_tensor)

        variant_cates = []
        for m in models[VARIANT_OUTCOME_MODEL][1:]:
            variant_output = inline(m)(input_tensor)
            variant_cates.append(
                op.unsqueeze(
                    inline(sub_model)(
                        variant_output[output_name],
                        output_0[output_name],
                    )["subtraction"],
                    axes=op.constant(value_int=1),
                )
            )
        cate = op.concat(variant_cates, axis=1)
        final_model = build({"input": input_tensor}, {"tau": cate})
        check_model(final_model, full_check=True)
        return final_model
