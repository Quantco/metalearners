# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause


import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from metalearners.drlearner import DRLearner
from metalearners.grid_search import MetaLearnerGridSearch
from metalearners.metalearner import VARIANT_OUTCOME_MODEL
from metalearners.rlearner import RLearner
from metalearners.slearner import SLearner
from metalearners.tlearner import TLearner
from metalearners.xlearner import XLearner


@pytest.mark.parametrize(
    "metalearner_factory, is_classification, base_learner_grid, param_grid, expected_n_configs",
    [
        (
            SLearner,
            False,
            {"base_model": [LinearRegression, LGBMRegressor]},
            {"base_model": {"LGBMRegressor": {"n_estimators": [1, 2]}}},
            3,
        ),
        (
            SLearner,
            True,
            {"base_model": [LogisticRegression, LGBMClassifier]},
            {"base_model": {"LGBMClassifier": {"n_estimators": [1, 2]}}},
            3,
        ),
        (
            TLearner,
            False,
            {"variant_outcome_model": [LinearRegression, LGBMRegressor]},
            {"variant_outcome_model": {"LGBMRegressor": {"n_estimators": [1, 2, 3]}}},
            4,
        ),
        (
            XLearner,
            False,
            {
                "variant_outcome_model": [LinearRegression],
                "propensity_model": [LGBMClassifier],
                "control_effect_model": [LGBMRegressor],
                "treatment_effect_model": [LGBMRegressor],
            },
            {
                "propensity_model": {"LGBMClassifier": {"n_estimators": [1, 2, 3]}},
                "control_effect_model": {"LGBMRegressor": {"n_estimators": [1, 2]}},
                "treatment_effect_model": {"LGBMRegressor": {"n_estimators": [1]}},
            },
            6,
        ),
        (
            RLearner,
            False,
            {
                "outcome_model": [LinearRegression],
                "propensity_model": [LGBMClassifier],
                "treatment_model": [LGBMRegressor],
            },
            {
                "propensity_model": {"LGBMClassifier": {"n_estimators": [1, 2, 3]}},
                "treatment_model": {"LGBMRegressor": {"n_estimators": [1, 2, 3]}},
            },
            9,
        ),
        (
            DRLearner,
            False,
            {
                "variant_outcome_model": [LinearRegression],
                "propensity_model": [LGBMClassifier],
                "treatment_model": [LinearRegression],
            },
            {
                "propensity_model": {"LGBMClassifier": {"n_estimators": [1, 2, 3, 4]}},
            },
            4,
        ),
    ],
)
@pytest.mark.parametrize("n_variants", [2, 5])
def test_metalearnergridsearch_smoke(
    metalearner_factory,
    is_classification,
    n_variants,
    base_learner_grid,
    param_grid,
    rng,
    expected_n_configs,
):
    metalearner_params = {
        "is_classification": is_classification,
        "n_variants": n_variants,
        "n_folds": 2,
    }
    gs = MetaLearnerGridSearch(
        metalearner_factory=metalearner_factory,
        metalearner_params=metalearner_params,
        base_learner_grid=base_learner_grid,
        param_grid=param_grid,
    )
    n_samples = 250
    n_test_samples = 100
    X = rng.standard_normal((n_samples, 3))
    X_test = rng.standard_normal((n_test_samples, 3))
    if is_classification:
        y = rng.integers(0, 2, n_samples)
        y_test = rng.integers(0, 2, n_test_samples)
    else:
        y = rng.standard_normal(n_samples)
        y_test = rng.standard_normal(n_test_samples)
    w = rng.integers(0, n_variants, n_samples)
    w_test = rng.integers(0, n_variants, n_test_samples)

    gs.fit(X, y, w, X_test, y_test, w_test)
    assert gs.results_ is not None
    assert gs.results_.shape[0] == expected_n_configs

    train_scores_cols = set(
        c[6:] for c in list(gs.results_.columns) if c.startswith("train_")
    )
    test_scores_cols = set(
        c[5:] for c in list(gs.results_.columns) if c.startswith("test_")
    )
    assert train_scores_cols == test_scores_cols


def test_metalearnergridsearch_reuse_smoke(rng):
    n_variants = 3
    n_samples = 250
    n_test_samples = 100

    X = rng.standard_normal((n_samples, 3))
    X_test = rng.standard_normal((n_test_samples, 3))
    y = rng.standard_normal(n_samples)
    y_test = rng.standard_normal(n_test_samples)
    w = rng.integers(0, n_variants, n_samples)
    w_test = rng.integers(0, n_variants, n_test_samples)

    tl = TLearner(
        False,
        n_variants,
        LGBMRegressor,
        nuisance_model_params={"verbose": -1, "n_estimators": 1},
        n_folds=2,
    )
    tl.fit(X, y, w)

    gs = MetaLearnerGridSearch(
        DRLearner,
        {
            "is_classification": False,
            "n_variants": n_variants,
            "n_folds": 5,  # To test with different n_folds than the pretrained
            "fitted_nuisance_models": {
                VARIANT_OUTCOME_MODEL: tl._nuisance_models[VARIANT_OUTCOME_MODEL]
            },
        },
        {
            "propensity_model": [LGBMClassifier, LogisticRegression],
            "treatment_model": [LGBMRegressor],
        },
        {
            "treatment_model": {
                "LGBMRegressor": {"n_estimators": [1, 2], "verbose": [-1]}
            },
            "propensity_model": {
                "LGBMClassifier": {
                    "n_estimators": [1, 2, 3],
                    "verbose": [-1],
                }
            },
        },
        verbose=3,
        random_state=1,
    )
    gs.fit(X, y, w, X_test, y_test, w_test)
    assert gs.raw_results_ is not None
    for raw_result in gs.raw_results_:
        assert raw_result.metalearner._prefitted_nuisance_models == {
            VARIANT_OUTCOME_MODEL
        }
    assert gs.results_ is not None
    assert gs.results_.shape[0] == 8
