# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from metalearners.tlearner import TLearner

# Chosen arbitrarily.
_REFERENCE_VALUE_TOLERANCE = 0.2
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
    "outcome_kind, reference_value", [("binary", 0.0149), ("continuous", 0.0121)]
)
def test_tlearner_synthetic_in_sample(outcome_kind, reference_value, request):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome"
    )

    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)
    base_learner = _linear_base_learner(is_classification)
    base_learner_params = _linear_base_learner_params(is_classification)
    tlearner = TLearner(
        base_learner,
        is_classification,
        nuisance_model_params=base_learner_params,
        random_state=_SEED,
    )

    tlearner.fit(covariates, observed_outcomes, treatment)
    cate_estimates = tlearner.predict(covariates, is_oos=False)
    if is_classification:
        cate_estimates = cate_estimates[:, 1]

    rmse = root_mean_squared_error(true_cate, cate_estimates)
    assert rmse < reference_value * (1 + _REFERENCE_VALUE_TOLERANCE)
    np.testing.assert_allclose(cate_estimates, true_cate.reshape(-1), atol=0.15)


@pytest.mark.parametrize(
    "outcome_kind, reference_value", [("binary", 0.0149), ("continuous", 0.0121)]
)
@pytest.mark.parametrize("oos_method", ["overall", "mean", "median"])
def test_tlearner_synthetic_oos(outcome_kind, reference_value, oos_method, request):
    if outcome_kind == "binary" and oos_method == "median":
        pytest.skip()

    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)
    base_learner = _linear_base_learner(is_classification)
    base_learner_params = _linear_base_learner_params(is_classification)
    tlearner = TLearner(
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
    tlearner.fit(covariates_train, observed_outcomes_train, treatment_train)
    cate_estimates = tlearner.predict(
        covariates_test, is_oos=True, oos_method=oos_method
    )
    if is_classification:
        cate_estimates = cate_estimates[:, 1]
    rmse = root_mean_squared_error(true_cate_test, cate_estimates)
    # See the benchmarking directory for the original reference values.
    assert rmse < reference_value * (1 + _REFERENCE_VALUE_TOLERANCE)
    np.testing.assert_allclose(cate_estimates, true_cate_test.reshape(-1), atol=0.15)


@pytest.mark.parametrize("oos_method", ["overall", "mean"])
def test_tlearner_twins(twins_data, oos_method, rng):
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
    tlearner = TLearner(
        base_learner,
        True,
        nuisance_model_params=base_learner_params,
        random_state=_SEED,
    )
    tlearner.fit(covariates_train, observed_outcomes_train, treatment_train)
    cate_estimates = tlearner.predict(
        covariates_test, is_oos=True, oos_method=oos_method
    )[:, 1]
    rmse = root_mean_squared_error(true_cate_test, cate_estimates)
    # See the benchmarking directory for reference values.
    assert rmse < 0.3623 * (1 + _REFERENCE_VALUE_TOLERANCE)


@pytest.mark.parametrize("outcome_kind", ["binary", "continuous"])
def test_tlearner_evaluate(outcome_kind, request):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)
    base_learner = _linear_base_learner(is_classification)

    tlearner = TLearner(base_learner, is_classification)
    tlearner.fit(covariates, observed_outcomes, treatment)
    evaluation = tlearner.evaluate(
        X=covariates, y=observed_outcomes, w=treatment, is_oos=False
    )
    if is_classification:
        assert "treatment_cross_entropy" in evaluation
        assert "effect_cross_entropy" in evaluation
    else:
        assert "treatment_rmse" in evaluation
        assert "effect_rmse" in evaluation


def test_validate_models():
    with pytest.raises(
        ValueError,
        match="is_classification is set to True but the treatment_model is not a classifier.",
    ):
        TLearner(LGBMRegressor, True)
    with pytest.raises(
        ValueError,
        match="is_classification is set to False but the treatment_model is not a regressor.",
    ):
        TLearner(LGBMClassifier, False)
