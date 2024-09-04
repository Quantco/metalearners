# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import train_test_split

from metalearners.cross_fit_estimator import _OOS_WHITELIST
from metalearners.drlearner import DRLearner
from metalearners.metalearner import (
    PROPENSITY_MODEL,
    TREATMENT_MODEL,
    VARIANT_OUTCOME_MODEL,
    MetaLearner,
    Params,
)
from metalearners.rlearner import OUTCOME_MODEL, RLearner
from metalearners.tlearner import TLearner
from metalearners.utils import metalearner_factory, simplify_output
from metalearners.xlearner import CONTROL_EFFECT_MODEL, TREATMENT_EFFECT_MODEL, XLearner

# Chosen arbitrarily.
_OOS_REFERENCE_VALUE_TOLERANCE = 0.05
_IN_SAMPLE_REFERENCE_VALUE_TOLERANCE = 0.2
_SEED = 1337
_TEST_FRACTION = 0.2
_LOG_REG_MAX_ITER = 500


def _is_classification(outcome_kind: str) -> bool:
    return outcome_kind == "binary"


def _linear_base_learner(is_classification: bool):
    if is_classification:
        return LogisticRegression
    return LinearRegression


def _tree_base_learner(is_classification: bool):
    if is_classification:
        return LGBMClassifier
    return LGBMRegressor


def _linear_base_learner_params(
    is_classification: bool,
) -> dict[str, int | float | str]:
    if is_classification:
        # Using the default value for max_iter sometimes
        # didn't lead to convergence.
        return {"random_state": _SEED, "max_iter": _LOG_REG_MAX_ITER}
    return {}


@pytest.mark.parametrize(
    "metalearner, outcome_kind, in_sample_reference_value, oos_reference_value, treatment_kind, te_kind",
    [
        ("T", "binary", 0.0212, 0.0215, "binary", "linear"),
        ("T", "continuous", 0.0459, 0.0456, "binary", "linear"),
        ("T", "continuous", 0.0615, 0.0617, "multi", "linear"),
        ("T", "continuous", 0.0753, 0.0753, "multi", "constant"),
        ("S", "binary", 0.2291, 0.2286, "binary", "linear"),
        ("S", "continuous", 14.5706, 14.6248, "binary", "linear"),
        ("S", "continuous", 14.147, 14.185, "multi", "linear"),
        ("S", "continuous", 0.0111, 0.0111, "multi", "constant"),
        ("X", "binary", 0.3046, 0.3019, "binary", "linear"),
        ("X", "continuous", 0.0459, 0.0456, "binary", "linear"),
        ("X", "continuous", 0.0615, 0.0617, "multi", "linear"),
        ("X", "continuous", 0.0753, 0.0753, "multi", "constant"),
        ("R", "binary", 0.3046, 0.3018, "binary", "linear"),
        ("R", "continuous", 0.0455, 0.0460, "binary", "linear"),
        # The multi-variant R-Learner runs lack a baseline.
        ("R", "continuous", 0.288, 0.28, "multi", "linear"),
        ("R", "continuous", 0.085, 0.08, "multi", "constant"),
        ("DR", "binary", 0.3046, 0.3019, "binary", "linear"),
        ("DR", "continuous", 0.0465, 0.0454, "binary", "linear"),
        ("DR", "continuous", 0.0649, 0.0649, "multi", "linear"),
        ("DR", "continuous", 0.0754, 0.0761, "multi", "constant"),
    ],
)
def test_learner_synthetic(
    metalearner,
    outcome_kind,
    in_sample_reference_value,
    oos_reference_value,
    treatment_kind,
    te_kind,
    request,
):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_{treatment_kind}_treatment_{te_kind}_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)
    classifier_learner_factory = _linear_base_learner(True)
    regressor_learner_factory = _linear_base_learner(False)
    classifier_learner_params = _linear_base_learner_params(True)
    regressor_learner_params = _linear_base_learner_params(False)
    nuisance_learner_factory = (
        classifier_learner_factory if is_classification else regressor_learner_factory
    )
    nuisance_learner_params = (
        classifier_learner_params if is_classification else regressor_learner_params
    )

    factory = metalearner_factory(metalearner)
    learner = factory(
        nuisance_model_factory=nuisance_learner_factory,
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=regressor_learner_factory,
        propensity_model_factory=classifier_learner_factory,
        nuisance_model_params=nuisance_learner_params,
        treatment_model_params=regressor_learner_params,
        propensity_model_params=classifier_learner_params,
        random_state=_SEED,
    )
    (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    ) = train_test_split(
        covariates,
        observed_outcomes,
        treatment,
        true_cate,
        test_size=_TEST_FRACTION,
        random_state=_SEED,
    )
    learner.fit(
        covariates_train,
        observed_outcomes_train,
        treatment_train,
        synchronize_cross_fitting=True,
        n_jobs_base_learners=-1,
    )

    # In sample CATEs
    cate_estimates_in_sample = simplify_output(
        learner.predict(covariates_train, is_oos=False)
    )

    rmse = root_mean_squared_error(true_cate_train, cate_estimates_in_sample)
    assert rmse < in_sample_reference_value * (1 + _IN_SAMPLE_REFERENCE_VALUE_TOLERANCE)
    if metalearner == "T":
        np.testing.assert_allclose(
            cate_estimates_in_sample.reshape(len(cate_estimates_in_sample), -1),
            true_cate_train,
            atol=0.4,
            rtol=0.3,
        )

    # oos CATEs
    for oos_method in _OOS_WHITELIST:
        if outcome_kind == "binary" and oos_method == "median":
            continue

        cate_estimates_oos = simplify_output(
            learner.predict(covariates_test, is_oos=True, oos_method=oos_method)
        )

        rmse = root_mean_squared_error(true_cate_test, cate_estimates_oos)
        # See the benchmarking directory for the original reference values.
        assert rmse < oos_reference_value * (1 + _OOS_REFERENCE_VALUE_TOLERANCE)
        if metalearner == "T":
            np.testing.assert_allclose(
                cate_estimates_oos.reshape(len(cate_estimates_oos), -1),
                true_cate_test,
                atol=0.4,
                rtol=0.3,
            )


@pytest.mark.parametrize(
    "metalearner, treatment_kind",
    [
        ("T", "binary"),
        ("T", "multi"),
        ("S", "binary"),
        ("S", "multi"),
        ("X", "binary"),
        ("X", "multi"),
        ("R", "binary"),
        ("R", "multi"),
        ("DR", "binary"),
        ("DR", "multi"),
    ],
)
def test_learner_synthetic_oos_ate(metalearner, treatment_kind, request):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_continuous_outcome_{treatment_kind}_treatment_linear_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset
    is_classification = False
    learner: MetaLearner
    base_learner = _linear_base_learner(is_classification)
    base_learner_params = _linear_base_learner_params(is_classification)
    factory = metalearner_factory(metalearner)
    learner = factory(
        nuisance_model_factory=base_learner,
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        nuisance_model_params=base_learner_params,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        propensity_model_params=_linear_base_learner_params(True),
        random_state=_SEED,
    )
    (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    ) = train_test_split(
        covariates,
        observed_outcomes,
        treatment,
        true_cate,
        test_size=_TEST_FRACTION,
        random_state=_SEED,
    )
    learner.fit(
        covariates_train,
        observed_outcomes_train,
        treatment_train,
        synchronize_cross_fitting=True,
        n_jobs_base_learners=-1,
    )
    for oos_method in _OOS_WHITELIST:
        cate_estimates = learner.predict(
            covariates_test, is_oos=True, oos_method=oos_method
        )
        actual_ate_estimate = np.squeeze(np.mean(cate_estimates, axis=0), axis=-1)
        target_ate_estimate = np.mean(true_cate_test, axis=0)
        np.testing.assert_allclose(
            actual_ate_estimate, target_ate_estimate, atol=1e-2, rtol=5e-2
        )


@pytest.mark.parametrize(
    "metalearner, reference_value",
    # Since we don't have a reference implementation for the DR-Learner,
    # we reuse the reference value of the R-Learner plus some tolerance.
    [("T", 0.3456), ("S", 0.3186), ("X", 0.3353), ("R", 0.3474), ("DR", 0.3474 * 1.05)],
)
def test_learner_twins(metalearner, reference_value, twins_data, rng):
    chosen_df, outcome_column, treatment_column, feature_columns, _ = twins_data

    covariates = chosen_df[feature_columns]
    observed_outcomes = chosen_df[outcome_column]
    treatment = chosen_df[treatment_column]
    true_cate = chosen_df["mu_1"] - chosen_df["mu_0"]

    (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    ) = train_test_split(
        covariates,
        observed_outcomes,
        treatment,
        true_cate,
        test_size=_TEST_FRACTION,
        random_state=_SEED,
    )

    factory = metalearner_factory(metalearner)
    learner = factory(
        nuisance_model_factory=LGBMClassifier,
        is_classification=True,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        nuisance_model_params={"random_state": rng},
        treatment_model_params={"random_state": rng},
        propensity_model_params={"random_state": rng},
        random_state=_SEED,
    )
    learner.fit(
        covariates_train,
        observed_outcomes_train,
        treatment_train,
        synchronize_cross_fitting=True,
    )
    for oos_method in ["overall", "mean"]:
        cate_estimates = simplify_output(
            learner.predict(covariates_test, is_oos=True, oos_method=oos_method)  # type: ignore
        )

        rmse = root_mean_squared_error(true_cate_test, cate_estimates)
        # See the benchmarking directory for reference values.
        assert rmse < reference_value * (1 + _OOS_REFERENCE_VALUE_TOLERANCE)


@pytest.mark.parametrize("metalearner", ["S", "T", "X", "R", "DR"])
@pytest.mark.parametrize("n_classes", [2, 5, 10])
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("is_classification", [True, False])
@pytest.mark.parametrize("is_oos", [True, False])
def test_learner_evaluate(
    metalearner, is_classification, rng, n_classes, n_variants, is_oos
):
    sample_size = 1000
    factory = metalearner_factory(metalearner)
    if n_variants > 2 and not factory._supports_multi_treatment():
        pytest.skip()
    if n_classes != 2 and not is_classification:
        pytest.skip()  # skip repeated tests
    if is_classification and n_classes > 2 and not factory._supports_multi_class():
        pytest.skip()
    test_size = 250
    X = rng.standard_normal((sample_size, 10))
    X_test = rng.standard_normal((test_size, 10)) if is_oos else X
    w = rng.integers(0, n_variants, size=sample_size)
    w_test = rng.integers(0, n_variants, test_size) if is_oos else w
    if is_classification:
        y = rng.integers(0, n_classes, size=sample_size)
        y_test = rng.integers(0, n_classes, test_size) if is_oos else y
    else:
        y = rng.standard_normal(sample_size)
        y_test = rng.standard_normal(test_size) if is_oos else y

    base_learner = _linear_base_learner(is_classification)

    learner = factory(
        nuisance_model_factory=base_learner,
        is_classification=is_classification,
        n_variants=n_variants,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        n_folds=2,
    )
    learner.fit(X=X, y=y, w=w)
    evaluation = learner.evaluate(X=X_test, y=y_test, w=w_test, is_oos=is_oos)
    if is_classification:
        if metalearner == "S":
            assert set(evaluation.keys()) == {"base_model_neg_log_loss"}
        elif metalearner in ["T", "X", "DR"]:
            for v in range(n_variants):
                assert f"variant_outcome_model_{v}_neg_log_loss" in evaluation
        elif metalearner == "R":
            assert "outcome_model_neg_log_loss" in evaluation
    else:
        if metalearner == "S":
            assert set(evaluation.keys()) == {"base_model_neg_root_mean_squared_error"}
        elif metalearner in ["T", "X", "DR"]:
            for v in range(n_variants):
                assert (
                    f"variant_outcome_model_{v}_neg_root_mean_squared_error"
                    in evaluation
                )
        elif metalearner == "R":
            assert "outcome_model_neg_root_mean_squared_error" in evaluation
    if metalearner == "R":
        assert (
            {f"r_loss_{i}_vs_0" for i in range(1, n_variants)}
            | {"propensity_model_neg_log_loss"}
            | {
                f"treatment_model_{i}_vs_0_neg_root_mean_squared_error"
                for i in range(1, n_variants)
            }
        ) <= set(evaluation.keys())
    elif metalearner == "X":
        assert "propensity_model_neg_log_loss" in evaluation
        for v in range(1, n_variants):
            assert (
                f"treatment_effect_model_{v}_vs_0_neg_root_mean_squared_error"
                in evaluation
            )
            assert (
                f"control_effect_model_{v}_vs_0_neg_root_mean_squared_error"
                in evaluation
            )
    elif metalearner == "DR":
        assert "propensity_model_neg_log_loss" in evaluation
        for v in range(1, n_variants):
            assert f"treatment_model_{v}_vs_0_neg_root_mean_squared_error" in evaluation


def new_score(estimator, X, y):
    # This score doesn't make sense.
    return np.mean(y - estimator.predict(X))


def new_score_2(y, y_pred):
    # This score doesn't make sense.
    return np.mean(y - y_pred)


@pytest.mark.parametrize(
    "metalearner, is_classification, scoring, expected_keys",
    [
        ("S", True, {"base_model": ["accuracy"]}, {"base_model_accuracy"}),
        ("S", False, {"base_model": ["max_error"]}, {"base_model_max_error"}),
        (
            "T",
            False,
            {
                "variant_outcome_model": [new_score, make_scorer(new_score_2)],
                "to_ignore": [],
            },
            {
                "variant_outcome_model_0_custom_scorer_0",
                "variant_outcome_model_0_custom_scorer_1",
                "variant_outcome_model_1_custom_scorer_0",
                "variant_outcome_model_1_custom_scorer_1",
                "variant_outcome_model_2_custom_scorer_0",
                "variant_outcome_model_2_custom_scorer_1",
            },
        ),
        (
            "X",
            True,
            {
                "variant_outcome_model": ["f1"],
                "propensity_model": [],
                "control_effect_model": [],
                "treatment_effect_model": ["r2", new_score],
            },
            {
                "variant_outcome_model_0_f1",
                "variant_outcome_model_1_f1",
                "variant_outcome_model_2_f1",
                "treatment_effect_model_1_vs_0_r2",
                "treatment_effect_model_1_vs_0_custom_scorer_1",
                "treatment_effect_model_2_vs_0_r2",
                "treatment_effect_model_2_vs_0_custom_scorer_1",
            },
        ),
        (
            "R",
            False,
            {
                "outcome_model": [make_scorer(new_score_2)],
                "propensity_model": [],
                "treatment_model": ["neg_mean_absolute_error"],
            },
            {
                "outcome_model_custom_scorer_0",
                "r_loss_1_vs_0",
                "r_loss_2_vs_0",
                "treatment_model_1_vs_0_neg_mean_absolute_error",
                "treatment_model_2_vs_0_neg_mean_absolute_error",
            },
        ),
        (
            "DR",
            True,
            {
                "variant_outcome_model": ["f1"],
                "propensity_model": [],
                "treatment_model": ["r2", new_score],
            },
            {
                "variant_outcome_model_0_f1",
                "variant_outcome_model_1_f1",
                "variant_outcome_model_2_f1",
                "treatment_model_1_vs_0_r2",
                "treatment_model_1_vs_0_custom_scorer_1",
                "treatment_model_2_vs_0_r2",
                "treatment_model_2_vs_0_custom_scorer_1",
            },
        ),
    ],
)
@pytest.mark.parametrize("is_oos", [True, False])
def test_learner_evaluate_scoring(
    metalearner, is_classification, scoring, expected_keys, is_oos, rng
):
    factory = metalearner_factory(metalearner)
    nuisance_model_factory = _linear_base_learner(is_classification)
    nuisance_model_params = _linear_base_learner_params(is_classification)

    n_variants = 3
    sample_size = 1000
    test_size = 250
    X = rng.standard_normal((sample_size, 10))
    X_test = rng.standard_normal((test_size, 10)) if is_oos else X
    w = rng.integers(0, n_variants, size=sample_size)
    w_test = rng.integers(0, n_variants, test_size) if is_oos else w
    if is_classification:
        y = rng.integers(0, 2, size=sample_size)
        y_test = rng.integers(0, 2, test_size) if is_oos else y
    else:
        y = rng.standard_normal(sample_size)
        y_test = rng.standard_normal(test_size) if is_oos else y

    ml = factory(
        is_classification=is_classification,
        n_variants=n_variants,
        nuisance_model_factory=nuisance_model_factory,
        propensity_model_factory=LGBMClassifier,
        treatment_model_factory=LinearRegression,
        nuisance_model_params=nuisance_model_params,
        propensity_model_params={"n_estimators": 1},
        n_folds=2,
    )
    ml.fit(X, y, w)
    evaluation = ml.evaluate(X_test, y_test, w_test, is_oos, scoring=scoring)
    assert set(evaluation.keys()) == expected_keys


@pytest.mark.parametrize("outcome_kind", ["binary", "continuous"])
@pytest.mark.parametrize("is_oos", [True, False])
def test_x_t_conditional_average_outcomes(outcome_kind, is_oos, request):
    """This test is to check that the conditional average outcomes predictions are the
    same for the TLearner and the XLearner as the nuisance models of the XLearner should
    be the same as the TLearner (except the propensity model)"""
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_binary_treatment_linear_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset
    (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    ) = train_test_split(
        covariates,
        observed_outcomes,
        treatment,
        true_cate,
        test_size=_TEST_FRACTION,
        random_state=_SEED,
    )

    is_classification = _is_classification(outcome_kind)
    classifier_learner_factory = _linear_base_learner(True)
    regressor_learner_factory = _linear_base_learner(False)
    classifier_learner_params = _linear_base_learner_params(True)
    regressor_learner_params = _linear_base_learner_params(False)
    nuisance_learner_factory = (
        classifier_learner_factory if is_classification else regressor_learner_factory
    )
    nuisance_learner_params = (
        classifier_learner_params if is_classification else regressor_learner_params
    )

    tlearner = TLearner(
        nuisance_model_factory=nuisance_learner_factory,
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        nuisance_model_params=nuisance_learner_params,
        random_state=_SEED,
    )
    xlearner = XLearner(
        nuisance_model_factory=nuisance_learner_factory,
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=regressor_learner_factory,
        propensity_model_factory=classifier_learner_factory,
        nuisance_model_params=nuisance_learner_params,
        treatment_model_params=regressor_learner_params,
        propensity_model_params=classifier_learner_params,
        random_state=_SEED,
    )
    tlearner.fit(
        covariates_train,
        observed_outcomes_train,
        treatment_train,
        synchronize_cross_fitting=False,
        n_jobs_base_learners=-1,
    )
    xlearner.fit(
        covariates_train,
        observed_outcomes_train,
        treatment_train,
        synchronize_cross_fitting=False,
        n_jobs_base_learners=-1,
    )

    if not is_oos:
        covariates_test = covariates_train

    tlearner_cond_avg_outcomes = tlearner.predict_conditional_average_outcomes(
        covariates_test, is_oos=is_oos
    )
    xlearner_cond_avg_outcomes = xlearner.predict_conditional_average_outcomes(
        covariates_test, is_oos=is_oos
    )
    np.testing.assert_allclose(xlearner_cond_avg_outcomes, tlearner_cond_avg_outcomes)


@pytest.mark.parametrize(
    "metalearner_prefix,success",
    [
        ("S", True),
        ("T", True),
        ("X", True),
        ("R", True),
        ("DR", True),
    ],
)
def test_validate_n_variants_error_multi(metalearner_prefix, success):
    factory = metalearner_factory(metalearner_prefix)
    n_variants = 10
    if success:
        _ = factory(
            nuisance_model_factory=LinearRegression,
            is_classification=False,
            n_variants=n_variants,
            treatment_model_factory=LinearRegression,
            propensity_model_factory=LogisticRegression,
            n_folds=2,
        )
    else:
        with pytest.raises(NotImplementedError, match="Current implementation of"):
            _ = factory(
                nuisance_model_factory=LinearRegression,
                is_classification=False,
                n_variants=n_variants,
                treatment_model_factory=LinearRegression,
                propensity_model_factory=LogisticRegression,
                n_folds=2,
            )


@pytest.mark.parametrize("n_variants", [2.0, 1])
@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "X", "R", "DR"])
def test_validate_n_variants_error_format(metalearner_prefix, n_variants):
    factory = metalearner_factory(metalearner_prefix)
    with pytest.raises(
        ValueError, match="n_variants needs to be an integer strictly greater than 1."
    ):
        _ = factory(
            nuisance_model_factory=LinearRegression,
            is_classification=False,
            n_variants=n_variants,
            treatment_model_factory=LinearRegression,
            propensity_model_factory=LogisticRegression,
            n_folds=2,
        )


@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "X", "R", "DR"])
def test_validate_treatment_error_encoding(metalearner_prefix):
    covariates = np.zeros((10, 1))
    w = np.array([1, 2] * 5)
    y = np.zeros(10)

    factory = metalearner_factory(metalearner_prefix)
    learner = factory(
        nuisance_model_factory=LinearRegression,
        is_classification=False,
        n_variants=len(np.unique(w)),
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        n_folds=2,
    )

    with pytest.raises(ValueError, match="Treatment variant should be encoded"):
        learner.fit(covariates, y, w)


@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "X", "R", "DR"])
def test_validate_treatment_error_different_instantiation(metalearner_prefix):
    covariates = np.zeros((10, 1))
    w = np.array(range(10))
    y = np.zeros(10)

    factory = metalearner_factory(metalearner_prefix)
    learner = factory(
        nuisance_model_factory=LinearRegression,
        is_classification=False,
        n_variants=2,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        n_folds=2,
    )
    with pytest.raises(
        ValueError, match="Number of variants present in the treatment are different"
    ):
        learner.fit(covariates, y, w)


@pytest.mark.parametrize(
    "metalearner_prefix,success",
    [
        ("S", True),
        ("T", True),
        ("X", False),
        ("R", False),
        ("DR", False),
    ],
)
def test_validate_outcome_multi_class(metalearner_prefix, success):
    covariates = np.zeros((20, 1))
    w = np.array([0] * 10 + [1] * 10)
    y = np.array([0, 1, 2, 3, 4] * 4)

    factory = metalearner_factory(metalearner_prefix)
    learner = factory(
        nuisance_model_factory=LogisticRegression,
        is_classification=True,
        n_variants=len(np.unique(w)),
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        n_folds=2,
    )

    if success:
        learner.fit(covariates, y, w)
    else:
        with pytest.raises(
            ValueError, match="does not support multiclass classification."
        ):
            learner.fit(covariates, y, w)


@pytest.mark.parametrize("is_classification", [True, False])
@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "R", "X", "DR"])
def test_conditional_average_outcomes_smoke(
    metalearner_prefix, is_classification, request
):
    (
        df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
    ) = request.getfixturevalue("twins_data" if is_classification else "mindset_data")
    factory = metalearner_factory(metalearner_prefix)
    learner = factory(
        nuisance_model_factory=_tree_base_learner(is_classification),
        nuisance_model_params={"n_estimators": 1},
        is_classification=is_classification,
        n_variants=len(np.unique(df[treatment_column])),
        treatment_model_factory=LGBMRegressor,
        treatment_model_params={"n_estimators": 1},
        propensity_model_factory=LGBMClassifier,
        propensity_model_params={"n_estimators": 1},
        n_folds=2,
    )
    learner.fit(df[feature_columns], df[outcome_column], df[treatment_column])
    result = learner.predict_conditional_average_outcomes(  # type: ignore
        df[feature_columns], is_oos=False
    )
    if is_classification:
        assert result.shape == (
            len(df),
            df[treatment_column].nunique(),
            df[outcome_column].nunique(),
        )
    else:
        assert result.shape == (len(df), df[treatment_column].nunique(), 1)


@pytest.mark.parametrize(
    "metalearner_prefix", ["S", "T"]
)  # the ones that support multiclass
@pytest.mark.parametrize("n_classes", [5, 10])
@pytest.mark.parametrize("n_variants", [2, 5])
def test_conditional_average_outcomes_smoke_multi_class(
    metalearner_prefix, rng, n_classes, n_variants
):
    sample_size = 1000
    factory = metalearner_factory(metalearner_prefix)

    X = rng.standard_normal((sample_size, 10))
    w = rng.integers(0, n_variants, size=sample_size)
    y = rng.integers(0, n_classes, size=sample_size)
    learner = factory(
        nuisance_model_factory=_tree_base_learner(True),
        nuisance_model_params={"n_estimators": 1},
        n_variants=n_variants,
        is_classification=True,
        n_folds=2,
    )
    learner.fit(X, y, w)
    result = learner.predict_conditional_average_outcomes(  # type: ignore
        X, is_oos=False
    )
    assert result.shape == (
        len(X),
        len(np.unique(w)),
        len(np.unique(y)),
    )
    np.testing.assert_allclose(result.sum(axis=-1), 1)


@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "X", "R", "DR"])
@pytest.mark.parametrize("n_classes", [2, 5, 10])
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("is_classification", [True, False])
def test_predict_smoke(
    metalearner_prefix, is_classification, rng, n_classes, n_variants
):
    sample_size = 1000
    factory = metalearner_factory(metalearner_prefix)
    if n_variants > 2 and not factory._supports_multi_treatment():
        pytest.skip()
    if n_classes != 2 and not is_classification:
        pytest.skip()  # skip repeated tests
    if is_classification and n_classes > 2 and not factory._supports_multi_class():
        pytest.skip()
    X = rng.standard_normal((sample_size, 10))
    w = rng.integers(0, n_variants, size=sample_size)
    if is_classification:
        y = rng.integers(0, n_classes, size=sample_size)
    else:
        y = rng.standard_normal(sample_size)
    learner = factory(
        nuisance_model_factory=_tree_base_learner(is_classification),
        nuisance_model_params={"n_estimators": 1},
        n_variants=n_variants,
        is_classification=is_classification,
        treatment_model_factory=LGBMRegressor,
        treatment_model_params={"n_estimators": 1},
        propensity_model_factory=LGBMClassifier,
        propensity_model_params={"n_estimators": 1},
        n_folds=2,
    )
    learner.fit(X, y, w)
    result = learner.predict(X, is_oos=False)
    if is_classification:
        assert result.shape == (len(X), n_variants - 1, n_classes)
        np.testing.assert_allclose(result.sum(axis=-1), 0, atol=1e-10)
    else:
        result.shape == (len(X), n_variants - 1, 1)


@pytest.mark.parametrize("outcome_kind", ["binary", "continuous"])
def test_model_reusage(outcome_kind, request):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_binary_treatment_linear_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset
    is_classification = _is_classification(outcome_kind)
    classifier_learner_factory = _tree_base_learner(True)
    regressor_learner_factory = _tree_base_learner(False)
    classifier_learner_params: Params = {"n_estimators": 5}
    regressor_learner_params: Params = {"n_estimators": 5}
    nuisance_learner_factory = (
        classifier_learner_factory if is_classification else regressor_learner_factory
    )
    nuisance_learner_params = (
        classifier_learner_params if is_classification else regressor_learner_params
    )

    tlearner = TLearner(
        nuisance_model_factory=nuisance_learner_factory,
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        nuisance_model_params=nuisance_learner_params,
    )
    tlearner.fit(covariates, observed_outcomes, treatment, n_jobs_base_learners=-1)
    xlearner = XLearner(
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=regressor_learner_factory,
        treatment_model_params=regressor_learner_params,
        propensity_model_factory=classifier_learner_factory,
        propensity_model_params=classifier_learner_params,
        fitted_nuisance_models={
            VARIANT_OUTCOME_MODEL: tlearner._nuisance_models[VARIANT_OUTCOME_MODEL]
        },
    )
    # We need to manually copy _treatment_variants_mask for the xlearner as it's needed
    # for predict, the user should not have to do this as they should call fit before predict.
    # This is just for testing.
    xlearner._treatment_variants_mask = tlearner._treatment_variants_mask
    np.testing.assert_allclose(
        tlearner.predict_conditional_average_outcomes(covariates, False),
        xlearner.predict_conditional_average_outcomes(covariates, False),
    )
    assert xlearner._prefitted_nuisance_models == {VARIANT_OUTCOME_MODEL}
    tlearner_pred_before_refitting = tlearner.predict_conditional_average_outcomes(
        covariates, False
    )
    xlearner.fit(covariates, observed_outcomes, treatment, n_jobs_base_learners=-1)
    np.testing.assert_allclose(
        tlearner.predict_conditional_average_outcomes(covariates, False),
        tlearner_pred_before_refitting,
    )
    drlearner = DRLearner(
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=regressor_learner_factory,
        treatment_model_params=regressor_learner_params,
        fitted_nuisance_models={
            VARIANT_OUTCOME_MODEL: tlearner._nuisance_models[VARIANT_OUTCOME_MODEL]
        },
        fitted_propensity_model=xlearner._nuisance_models[PROPENSITY_MODEL][0],
    )
    assert drlearner._prefitted_nuisance_models == {
        VARIANT_OUTCOME_MODEL,
        PROPENSITY_MODEL,
    }
    np.testing.assert_allclose(
        xlearner.predict_nuisance(covariates, PROPENSITY_MODEL, 0, False),
        drlearner.predict_nuisance(covariates, PROPENSITY_MODEL, 0, False),
    )


@pytest.mark.parametrize(
    "metalearner_factory, feature_set",
    [
        (TLearner, {VARIANT_OUTCOME_MODEL: [0, 1]}),
        (
            XLearner,
            {
                VARIANT_OUTCOME_MODEL: [0],
                PROPENSITY_MODEL: [2, 3],
                TREATMENT_EFFECT_MODEL: [4],
                CONTROL_EFFECT_MODEL: None,
            },
        ),
        (
            RLearner,
            {OUTCOME_MODEL: None, PROPENSITY_MODEL: [4], TREATMENT_MODEL: [3]},
        ),
        (
            DRLearner,
            {VARIANT_OUTCOME_MODEL: [], PROPENSITY_MODEL: None, TREATMENT_MODEL: [0]},
        ),
    ],
)
@pytest.mark.parametrize("backend", ["np", "pd", "csr"])
def test_evaluate_feature_set_smoke(metalearner_factory, feature_set, rng, backend):
    n_samples = 100
    X = rng.standard_normal((n_samples, 5))
    y = rng.standard_normal(n_samples)
    w = rng.integers(0, 2, n_samples)
    if backend == "pd":
        X = pd.DataFrame(X)
        y = pd.Series(y)
        w = pd.Series(w)
    elif backend == "csr":
        X = csr_matrix(X)

    ml = metalearner_factory(
        n_variants=2,
        is_classification=False,
        nuisance_model_factory=LinearRegression,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        feature_set=feature_set,
        n_folds=2,
    )
    ml.fit(X, y, w)
    ml.evaluate(X, y, w, False)
