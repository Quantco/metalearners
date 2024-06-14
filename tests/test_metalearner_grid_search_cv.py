# # Copyright (c) QuantCo 2024-2024
# # SPDX-License-Identifier: BSD-3-Clause


import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from metalearners.drlearner import DRLearner
from metalearners.metalearner_grid_search_cv import MetaLearnerGridSearchCV
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
            {"LGBMRegressor": {"n_estimators": [1, 2]}},
            3,
        ),
        (
            SLearner,
            True,
            {"base_model": [LogisticRegression, LGBMClassifier]},
            {"LGBMClassifier": {"n_estimators": [1, 2]}},
            3,
        ),
        (
            TLearner,
            False,
            {"variant_outcome_model": [LinearRegression, LGBMRegressor]},
            {"LGBMRegressor": {"n_estimators": [1, 2, 3]}},
            4,
        ),
        (
            XLearner,
            False,
            {
                "variant_outcome_model": [LinearRegression],
                "propensity_model": [LGBMClassifier],
                "control_effect_model": [LinearRegression],
                "treatment_effect_model": [LinearRegression],
            },
            {"LGBMClassifier": {"n_estimators": [1, 2, 3]}},
            3,
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
                "LGBMClassifier": {"n_estimators": [1, 2, 3]},
                "LGBMRegressor": {"n_estimators": [1, 2, 3]},
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
            {"LGBMClassifier": {"n_estimators": [1, 2, 3, 4]}},
            4,
        ),
    ],
)
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("cv", [2, 3])
def test_metalearnergridsearchcv_smoke(
    metalearner_factory,
    is_classification,
    n_variants,
    base_learner_grid,
    param_grid,
    cv,
    rng,
    expected_n_configs,
):
    metalearner_params = {
        "is_classification": is_classification,
        "n_variants": n_variants,
        "n_folds": 2,
    }
    gs = MetaLearnerGridSearchCV(
        metalearner_factory=metalearner_factory,
        metalearner_params=metalearner_params,
        base_learner_grid=base_learner_grid,
        param_grid=param_grid,
        cv=cv,
    )
    n_samples = 250
    X = rng.standard_normal((n_samples, 3))
    if is_classification:
        y = rng.integers(0, 2, n_samples)
    else:
        y = rng.standard_normal(n_samples)
    w = rng.integers(0, n_variants, n_samples)

    gs.fit(X, y, w)
    assert gs.cv_results_ is not None
    assert gs.cv_results_.shape[0] == expected_n_configs * cv
