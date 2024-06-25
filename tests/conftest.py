# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
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


@pytest.fixture(scope="session")
def twins_data():
    rng = np.random.default_rng(_SEED)
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


@pytest.fixture(scope="session")
def n_numericals():
    return 25


@pytest.fixture(scope="session")
def n_categoricals():
    return 5


@pytest.fixture(scope="session")
def sample_size():
    return 100_000


@pytest.fixture(scope="session")
def numerical_covariates(sample_size, n_numericals):
    rng = np.random.default_rng(_SEED)
    return generate_covariates(sample_size, n_numericals, format="numpy", rng=rng)


@pytest.fixture(scope="session")
def mixed_covariates(sample_size, n_numericals, n_categoricals):
    rng = np.random.default_rng(_SEED)
    return generate_covariates(
        sample_size,
        n_numericals + n_categoricals,
        n_categoricals=n_categoricals,
        format="pandas",
        rng=rng,
    )


@pytest.fixture(scope="session")
def numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te(
    sample_size, n_numericals
):
    rng = np.random.default_rng(_SEED)
    covariates, _, _ = generate_covariates(
        sample_size, n_numericals, format="numpy", rng=rng
    )
    return _generate_rct_experiment_data(covariates, False, rng, 0.3, None)


@pytest.fixture(scope="session")
def numerical_experiment_dataset_binary_outcome_binary_treatment_linear_te(
    sample_size, n_numericals
):
    rng = np.random.default_rng(_SEED)
    covariates, _, _ = generate_covariates(
        sample_size, n_numericals, format="numpy", rng=rng
    )
    return _generate_rct_experiment_data(covariates, True, rng, 0.3, None)


@pytest.fixture(scope="session")
def mixed_experiment_dataset_continuous_outcome_binary_treatment_linear_te(
    sample_size, n_numericals, n_categoricals
):
    rng = np.random.default_rng(_SEED)
    covariates, _, _ = generate_covariates(
        sample_size,
        n_numericals + n_categoricals,
        n_categoricals=n_categoricals,
        format="pandas",
        rng=rng,
    )
    return _generate_rct_experiment_data(covariates, False, rng, 0.3, None)


@pytest.fixture(scope="session")
def numerical_experiment_dataset_continuous_outcome_multi_treatment_linear_te(
    sample_size, n_numericals
):
    rng = np.random.default_rng(_SEED)
    covariates, _, _ = generate_covariates(
        sample_size, n_numericals, format="numpy", rng=rng
    )
    return _generate_rct_experiment_data(
        covariates, False, rng, [0.2, 0.1, 0.3, 0.15, 0.25], None
    )


@pytest.fixture(scope="session")
def numerical_experiment_dataset_continuous_outcome_multi_treatment_constant_te(
    sample_size, n_numericals
):
    rng = np.random.default_rng(_SEED)
    covariates, _, _ = generate_covariates(
        sample_size, n_numericals, format="numpy", rng=rng
    )
    return _generate_rct_experiment_data(
        covariates, False, rng, [0.2, 0.1, 0.3, 0.15, 0.25], np.array([-2, 5, 0, 3])
    )


@pytest.fixture(scope="session")
def dummy_dataset():
    rng = np.random.default_rng(_SEED)
    sample_size = 100
    n_features = 10
    X = rng.standard_normal((sample_size, n_features))
    y = rng.standard_normal(sample_size)
    w = rng.integers(0, 2, sample_size)
    return X, y, w


@pytest.fixture(scope="session")
def feature_importance_dataset():
    rng = np.random.default_rng(_SEED)
    n_samples = 10000
    x0 = rng.normal(10, 1, n_samples)
    x1 = rng.normal(2, 1, n_samples)
    x2 = rng.normal(-5, 1, n_samples)
    w = rng.integers(0, 3, n_samples)

    noise = rng.normal(0, 0.05, n_samples)

    y = np.zeros(n_samples)
    y[w == 0] = x0[w == 0] + noise[w == 0]
    y[w == 1] = x0[w == 1] + x1[w == 1] + noise[w == 1]
    y[w == 2] = x0[w == 2] + x2[w == 2] + noise[w == 2]
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    y = pd.Series(y)
    w = pd.Series(w)

    return X, y, w
