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
