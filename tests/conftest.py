# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import numpy as np
import pandas as pd
import pytest
from git_root import git_root

from metalearners.data_generation import generate_covariates


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


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
        "frst_in_family",
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


@pytest.fixture(scope="session")
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
    chosen_twin = rng.binomial(n=1, p=0.5, size=n_twins_pairs)

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
    return 1_000_000


@pytest.fixture(scope="module")
def numerical_dataset(sample_size, n_numericals, rng):
    return generate_covariates(sample_size, n_numericals, format="numpy", rng=rng)


@pytest.fixture(scope="module")
def simulated_dataset(sample_size, n_numericals, n_categoricals, rng):
    return generate_covariates(
        sample_size,
        n_numericals + n_categoricals,
        n_categoricals=n_categoricals,
        format="pandas",
        rng=rng,
    )
