# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from git_root import git_root
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveRegressor,
    QuantileRegressor,
    RANSACRegressor,
    Ridge,
    RidgeCV,
    SGDRegressor,
    TheilSenRegressor,
    TweedieRegressor,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

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

all_sklearn_classifiers = [
    # AdaBoostClassifier, # The output probabilities are wrong when there are only two classes, see https://github.com/onnx/sklearn-onnx/issues/1117
    BaggingClassifier,
    CalibratedClassifierCV,
    DecisionTreeClassifier,
    ExtraTreeClassifier,
    ExtraTreesClassifier,
    # GaussianProcessClassifier, # This raises an error com.microsoft:Solve(-1) is not a registered function/op when inference on onnx. TODO: investigate it further
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    KNeighborsClassifier,
    # LinearDiscriminantAnalysis, # The output probabilities are wrong when there are only two classes, see https://github.com/onnx/sklearn-onnx/issues/1116
    LogisticRegression,
    LogisticRegressionCV,
    MLPClassifier,
    QuadraticDiscriminantAnalysis,
    RadiusNeighborsClassifier,
    RandomForestClassifier,
]  # extracted from all_estimators("classifier"), models which have predict_proba and convert_sklearn supports them

all_sklearn_regressors = [
    ARDRegression,
    AdaBoostRegressor,
    BaggingRegressor,
    BayesianRidge,
    DecisionTreeRegressor,
    ElasticNet,
    ElasticNetCV,
    ExtraTreeRegressor,
    ExtraTreesRegressor,
    GaussianProcessRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    HuberRegressor,
    KNeighborsRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    LinearSVR,
    MLPRegressor,
    NuSVR,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    # PLSRegression, # The output shape of the onnx converted model is wrong
    PassiveAggressiveRegressor,
    QuantileRegressor,
    RANSACRegressor,
    RadiusNeighborsRegressor,
    RandomForestRegressor,
    Ridge,
    RidgeCV,
    SGDRegressor,
    SVR,
    TheilSenRegressor,
    TweedieRegressor,
]  # regressors which are supported by convert_sklearn and support regression in the reals


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
    return load_mindset_data(Path(git_root()) / "data" / "learning_mindset.zip")


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
    ) = load_twins_data(Path(git_root()) / "data" / "twins.zip", rng)
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


@pytest.fixture(scope="session")
def grid_search_data():
    rng = np.random.default_rng(_SEED)
    n_samples = 250
    n_test_samples = 100
    n_features = 3
    n_variants = 4
    X = rng.standard_normal((n_samples, n_features))
    X_test = rng.standard_normal((n_test_samples, n_features))

    y_class = rng.integers(0, 2, n_samples)
    y_test_class = rng.integers(0, 2, n_test_samples)

    y_reg = rng.standard_normal(n_samples)
    y_test_reg = rng.standard_normal(n_test_samples)

    w = rng.integers(0, n_variants, n_samples)
    w_test = rng.integers(0, n_variants, n_test_samples)

    return X, y_class, y_reg, w, X_test, y_test_class, y_test_reg, w_test


@pytest.fixture(scope="session")
def onnx_dataset():
    rng = np.random.default_rng(_SEED)
    n_samples = 300
    n_numerical_features = 5

    X_numerical = rng.standard_normal((n_samples, n_numerical_features))

    X_with_categorical = pd.DataFrame(X_numerical)
    X_with_categorical[n_numerical_features] = pd.Series(
        rng.integers(10, 13, n_samples), dtype="category"
    )  # not start at 0
    X_with_categorical[n_numerical_features + 1] = pd.Series(
        rng.choice([-5, 4, -10, -32], size=n_samples), dtype="category"
    )  # not consecutive

    y_class = rng.integers(0, 2, size=n_samples)
    y_reg = rng.standard_normal(n_samples)

    w = rng.integers(0, 3, n_samples)

    return X_numerical, X_with_categorical, y_class, y_reg, w
