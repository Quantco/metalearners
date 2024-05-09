# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from metalearners.metalearner import MetaLearner
from metalearners.slearner import SLearner
from metalearners.tlearner import TLearner

# Chosen arbitrarily.
_REFERENCE_VALUE_TOLERANCE = 0.05
_SEED = 1337
_TEST_FRACTION = 0.2
_LOG_REG_MAX_ITER = 500


def _is_classification(outcome_kind: str) -> bool:
    return outcome_kind == "binary"


def _linear_base_learner(is_classification: bool):
    if is_classification:
        return LogisticRegression
    return LinearRegression


def _linear_base_learner_params(
    is_classification: bool,
) -> dict[str, int | float | str]:
    if is_classification:
        # Using the default value for max_iter sometimes
        # didn't lead to convergence.
        return {"random_state": _SEED, "max_iter": _LOG_REG_MAX_ITER}
    return {}


@pytest.mark.parametrize(
    "metalearner, outcome_kind, reference_value, treatment_kind, te_kind",
    [
        ("T", "binary", 0.0149, "binary", "linear"),
        ("T", "continuous", 0.0121, "binary", "linear"),
        ("S", "binary", 0.2568, "binary", "linear"),
        ("S", "continuous", 11.6777, "binary", "linear"),
    ],
)
def test_learner_synthetic_in_sample(
    metalearner, outcome_kind, reference_value, treatment_kind, te_kind, request
):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_{treatment_kind}_treatment_{te_kind}_te"
    )

    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)
    base_learner = _linear_base_learner(is_classification)
    base_learner_params = _linear_base_learner_params(is_classification)
    if metalearner == "S":
        learner = SLearner(
            base_learner,
            is_classification,
            nuisance_model_params=base_learner_params,
            random_state=_SEED,
        )
    elif metalearner == "T":
        learner = TLearner(  # type: ignore
            base_learner,
            is_classification,
            nuisance_model_params=base_learner_params,
            random_state=_SEED,
        )

    learner.fit(covariates, observed_outcomes, treatment)
    cate_estimates = learner.predict(covariates, is_oos=False)
    if is_classification:
        cate_estimates = cate_estimates[:, 1]

    rmse = root_mean_squared_error(true_cate, cate_estimates)
    assert rmse < reference_value * (1 + _REFERENCE_VALUE_TOLERANCE)
    if metalearner == "T":
        np.testing.assert_allclose(cate_estimates, true_cate.reshape(-1), atol=0.15)


@pytest.mark.parametrize(
    "metalearner, outcome_kind, reference_value, treatment_kind, te_kind",
    [
        ("T", "binary", 0.0149, "binary", "linear"),
        ("T", "continuous", 0.0121, "binary", "linear"),
        ("S", "binary", 0.2563, "binary", "linear"),
        ("S", "continuous", 11.6584, "binary", "linear"),
        ("S", "continuous", 11.5957, "multi", "linear"),
        ("S", "continuous", 0.004997, "multi", "constant"),
    ],
)
@pytest.mark.parametrize("oos_method", ["overall", "mean", "median"])
def test_learner_synthetic_oos(
    metalearner,
    outcome_kind,
    reference_value,
    treatment_kind,
    te_kind,
    oos_method,
    request,
):
    if outcome_kind == "binary" and oos_method == "median":
        pytest.skip()

    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_{treatment_kind}_treatment_{te_kind}_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)
    base_learner = _linear_base_learner(is_classification)
    base_learner_params = _linear_base_learner_params(is_classification)
    if metalearner == "S":
        learner = SLearner(
            base_learner,
            is_classification,
            nuisance_model_params=base_learner_params,
            random_state=_SEED,
        )
    elif metalearner == "T":
        learner = TLearner(  # type: ignore
            base_learner,
            is_classification,
            nuisance_model_params=base_learner_params,
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
    learner.fit(covariates_train, observed_outcomes_train, treatment_train)
    cate_estimates = learner.predict(
        covariates_test, is_oos=True, oos_method=oos_method
    )
    if is_classification:
        cate_estimates = cate_estimates[:, 1]
    rmse = root_mean_squared_error(true_cate_test, cate_estimates)
    # See the benchmarking directory for the original reference values.
    assert rmse < reference_value * (1 + _REFERENCE_VALUE_TOLERANCE)
    if metalearner == "T":
        np.testing.assert_allclose(
            cate_estimates, true_cate_test.reshape(-1), atol=0.15
        )


@pytest.mark.parametrize(
    "metalearner, treatment_kind",
    [
        ("T", "binary"),
        ("S", "binary"),
        ("S", "multi"),
    ],
)
@pytest.mark.parametrize("oos_method", ["overall", "mean", "median"])
def test_learner_synthetic_oos_ate(metalearner, treatment_kind, oos_method, request):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_continuous_outcome_{treatment_kind}_treatment_linear_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset
    is_classification = False
    learner: MetaLearner
    base_learner = _linear_base_learner(is_classification)
    base_learner_params = _linear_base_learner_params(is_classification)
    if metalearner == "S":
        learner = SLearner(
            base_learner,
            is_classification,
            nuisance_model_params=base_learner_params,
            random_state=_SEED,
        )
    elif metalearner == "T":
        learner = TLearner(
            base_learner,
            is_classification,
            nuisance_model_params=base_learner_params,
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
    learner.fit(covariates_train, observed_outcomes_train, treatment_train)
    cate_estimates = learner.predict(
        covariates_test, is_oos=True, oos_method=oos_method
    )
    actual_ate_estimate = np.mean(cate_estimates)
    target_ate_estimate = np.mean(true_cate_test)
    assert actual_ate_estimate == pytest.approx(target_ate_estimate, abs=1e-2, rel=1e-2)


@pytest.mark.parametrize("metalearner, reference_value", [("T", 0.3623), ("S", 0.3186)])
@pytest.mark.parametrize("oos_method", ["overall", "mean"])
def test_learner_twins(metalearner, reference_value, twins_data, oos_method, rng):
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

    base_learner = LGBMClassifier
    base_learner_params = {"random_state": rng}

    if metalearner == "S":
        learner = SLearner(
            base_learner,
            True,
            nuisance_model_params=base_learner_params,
            random_state=_SEED,
        )
    elif metalearner == "T":
        learner = TLearner(  # type: ignore
            base_learner,
            True,
            nuisance_model_params=base_learner_params,
            random_state=_SEED,
        )
    learner.fit(covariates_train, observed_outcomes_train, treatment_train)
    cate_estimates = learner.predict(
        covariates_test, is_oos=True, oos_method=oos_method
    )[:, 1]
    rmse = root_mean_squared_error(true_cate_test, cate_estimates)
    # See the benchmarking directory for reference values.
    assert rmse < reference_value * (1 + _REFERENCE_VALUE_TOLERANCE)


@pytest.mark.parametrize("metalearner", ["S", "T"])
@pytest.mark.parametrize("outcome_kind", ["binary", "continuous"])
def test_learner_evaluate(metalearner, outcome_kind, request):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_binary_treatment_linear_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)
    base_learner = _linear_base_learner(is_classification)
    if metalearner == "S":
        learner = SLearner(base_learner, is_classification)
    elif metalearner == "T":
        learner = TLearner(base_learner, is_classification)  # type: ignore
    learner.fit(covariates, observed_outcomes, treatment)
    evaluation = learner.evaluate(
        X=covariates, y=observed_outcomes, w=treatment, is_oos=False
    )
    if is_classification:
        if metalearner == "S":
            assert "cross_entropy" in evaluation
        elif metalearner == "T":
            assert "treatment_cross_entropy" in evaluation
            assert "effect_cross_entropy" in evaluation
    else:
        if metalearner == "S":
            assert "rmse" in evaluation
        elif metalearner == "T":
            assert "treatment_rmse" in evaluation
            assert "effect_rmse" in evaluation
