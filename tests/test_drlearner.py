# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from metalearners.drlearner import DRLearner


def test_adaptive_clipping_smoke(dummy_dataset):
    X, y, w = dummy_dataset
    ml = DRLearner(
        False,
        2,
        LinearRegression,
        LinearRegression,
        LogisticRegression,
        n_folds=2,
        adaptive_clipping=True,
    )
    ml.fit(X, y, w)


def test_treatment_effect(
    numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te,
):
    X, _, W, Y, _, tau = (
        numerical_experiment_dataset_continuous_outcome_binary_treatment_linear_te
    )
    ml = DRLearner(
        False,
        2,
        LinearRegression,
        LinearRegression,
        LogisticRegression,
        n_folds=2,
    )
    ml.fit_all_nuisance(X, Y, W)
    est = ml.treatment_effect(X, Y, W)
    np.testing.assert_almost_equal(est[:, 0], tau.mean(), decimal=1)
