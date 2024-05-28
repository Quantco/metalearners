# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
import pytest

from metalearners._utils import get_linear_dimension, load_mindset_data, load_twins_data
from metalearners.data_generation import (
    compute_experiment_outputs,
    generate_covariates,
    generate_treatment,
)
from metalearners.outcome_functions import (
    constant_treatment_effect,
    linear_treatment_effect,
)

_SEED = 1337
_SIGMA_TAU = 0.5


def _generate_rct_experiment_data(
    covariates,
    is_classification: bool,
    rng,
    propensity_score: float | list[float] = 0.3,
    tau=None,
):
    if isinstance(propensity_score, list):
        n_variants = len(propensity_score)
        propensity_scores = np.array(propensity_score) * np.ones(
            (covariates.shape[0], n_variants)
        )
    elif isinstance(propensity_score, float):
        n_variants = 2
        propensity_scores = propensity_score * np.ones(covariates.shape[0])

    treatment = generate_treatment(propensity_scores, rng=rng)
    dim = get_linear_dimension(covariates)
    if tau is None:
        outcome_function = linear_treatment_effect(dim, n_variants=n_variants, rng=rng)
    else:
        outcome_function = constant_treatment_effect(dim, tau=tau, rng=rng)
    potential_outcomes = outcome_function(covariates)

    observed_outcomes, true_cate = compute_experiment_outputs(
        potential_outcomes,
        treatment,
        sigma_tau=_SIGMA_TAU,
        n_variants=n_variants,
        is_classification=is_classification,
        return_probability_cate=True,
        rng=rng,
    )

    return (
        covariates,
        propensity_scores,
        treatment,
        observed_outcomes,
        potential_outcomes,
        true_cate,
    )


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng(_SEED)


@pytest.fixture(scope="session")
def mindset_data():
    return load_mindset_data()


@pytest.fixture(scope="function")
def twins_data(rng):
    (
        chosen_df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
        _,
    ) = load_twins_data(rng)
    return (
        chosen_df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
    )


@pytest.fixture(scope="module")
def n_numericals():
    return 25


@pytest.fixture(scope="module")
def n_categoricals():
    return 5


@pytest.fixture(scope="module")
def sample_size():
    return 100_000


@pytest.fixture(scope="function")
def numerical_covariates(sample_size, n_numericals, rng):
    return generate_covariates(sample_size, n_numericals, format="numpy", rng=rng)


@pytest.fixture(scope="function")
def mixed_covariates(sample_size, n_numericals, n_categoricals, rng):
    return generate_covariates(
        sample_size,
        n_numericals + n_categoricals,
        n_categoricals=n_categoricals,
        format="pandas",
        rng=rng,
    )


@pytest.fixture(scope="function")
def numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te(
    numerical_covariates, rng
):
    covariates, _, _ = numerical_covariates
    return _generate_rct_experiment_data(covariates, False, rng, 0.3, None)


@pytest.fixture(scope="function")
def numerical_experiment_dataset_binary_outcome_binary_treatment_linear_te(
    numerical_covariates, rng
):
    covariates, _, _ = numerical_covariates
    return _generate_rct_experiment_data(covariates, True, rng, 0.3, None)


@pytest.fixture(scope="function")
def mixed_experiment_dataset_continuous_outcome_binary_treatment_linear_te(
    mixed_covariates, rng
):
    covariates, _, _ = mixed_covariates
    return _generate_rct_experiment_data(covariates, False, rng, 0.3, None)


@pytest.fixture(scope="function")
def numerical_experiment_dataset_continuous_outcome_multi_treatment_linear_te(
    numerical_covariates, rng
):
    covariates, _, _ = numerical_covariates
    return _generate_rct_experiment_data(
        covariates, False, rng, [0.2, 0.1, 0.3, 0.15, 0.25], None
    )


@pytest.fixture(scope="function")
def numerical_experiment_dataset_continuous_outcome_multi_treatment_constant_te(
    numerical_covariates, rng
):
    covariates, _, _ = numerical_covariates
    return _generate_rct_experiment_data(
        covariates, False, rng, [0.2, 0.1, 0.3, 0.15, 0.25], np.array([-2, 5, 0, 3])
    )
