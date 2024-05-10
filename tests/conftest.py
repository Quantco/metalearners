# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
import pandas as pd
import pytest
from git_root import git_root

from metalearners._utils import get_linear_dimension
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
    df = pd.read_csv(git_root("data/learning_mindset.zip"))
    outcome_column = "achievement_score"
    treatment_column = "intervention"
    feature_columns = [
        column
        for column in df.columns
        if column not in [outcome_column, treatment_column]
    ]
    categorical_feature_columns = [
        "ethnicity",
        "gender",
        "frst_in_family",  # spellchecker:disable-line
        "school_urbanicity",
        "schoolid",
    ]
    # Note that explicitly setting the dtype of these features to category
    # allows both lightgbm as well as shap plots to
    # 1. Operate on features which are not of type int, bool or float
    # 2. Correctly interpret categoricals with int values to be
    #    interpreted as categoricals, as compared to ordinals/numericals.
    for categorical_feature_column in categorical_feature_columns:
        df[categorical_feature_column] = df[categorical_feature_column].astype(
            "category"
        )
    return (
        df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
    )


@pytest.fixture(scope="function")
def twins_data(rng):
    df = pd.read_csv(git_root("data/twins.zip"))
    drop_columns = [
        "bord",
        "brstate_reg",
        "stoccfipb_reg",
        "mplbir_reg",
        "infant_id",
        "wt",
    ]
    # We remove wt (weight) and bord (birth order) as they are different for each twin.
    # We remove _reg variables as they are already represented by the corresponding
    # variable without _reg and this new only groups them in bigger regions.
    # We remove infant_id as it's a unique identifier for each infant.
    df = df.drop(drop_columns, axis=1)
    outcome_column = "outcome"
    treatment_column = "treatment"
    feature_columns = [
        column
        for column in df.columns
        if column not in [outcome_column, treatment_column]
    ]
    assert len(feature_columns) == 45

    ordinary_feature_columns = [
        "dlivord_min",
        "dtotord_min",
    ]
    categorical_feature_columns = [
        column for column in feature_columns if column not in ordinary_feature_columns
    ]
    for categorical_feature_column in categorical_feature_columns:
        df[categorical_feature_column] = df[categorical_feature_column].astype(
            "category"
        )

    n_twins_pairs = df.shape[0] // 2
    chosen_twin = rng.binomial(n=1, p=0.3, size=n_twins_pairs)

    selected_rows = []
    for i in range(0, len(df), 2):
        pair_idx = i // 2
        selected_row_idx = i + chosen_twin[pair_idx]
        selected_rows.append(selected_row_idx)

    chosen_df = df.iloc[selected_rows].reset_index(drop=True)

    mu_0 = df[df[treatment_column] == 0][outcome_column].reset_index(drop=True)
    mu_1 = df[df[treatment_column] == 1][outcome_column].reset_index(drop=True)
    chosen_df["mu_0"] = mu_0
    chosen_df["mu_1"] = mu_1

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
