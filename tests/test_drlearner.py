# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

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
