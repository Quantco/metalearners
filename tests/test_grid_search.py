# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause


from types import GeneratorType

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from metalearners.drlearner import DRLearner
from metalearners.grid_search import MetaLearnerGridSearch
from metalearners.metalearner import PROPENSITY_MODEL, VARIANT_OUTCOME_MODEL
from metalearners.rlearner import RLearner
from metalearners.slearner import SLearner
from metalearners.tlearner import TLearner
from metalearners.xlearner import XLearner


@pytest.mark.parametrize(
    "metalearner_factory, is_classification, base_learner_grid, param_grid, expected_n_configs, expected_index_cols",
    [
        (
            SLearner,
            False,
            {"base_model": [LinearRegression, LGBMRegressor]},
            {"base_model": {"LGBMRegressor": {"n_estimators": [1, 2]}}},
            3,
            ["metalearner", "base_model", "base_model_n_estimators"],
        ),
        (
            SLearner,
            True,
            {"base_model": [LogisticRegression, LGBMClassifier]},
            {"base_model": {"LGBMClassifier": {"n_estimators": [1, 2]}}},
            3,
            ["metalearner", "base_model", "base_model_n_estimators"],
        ),
        (
            TLearner,
            False,
            {"variant_outcome_model": [LinearRegression, LGBMRegressor]},
            {"variant_outcome_model": {"LGBMRegressor": {"n_estimators": [1, 2, 3]}}},
            4,
            [
                "metalearner",
                "variant_outcome_model",
                "variant_outcome_model_n_estimators",
            ],
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
            [
                "metalearner",
                "propensity_model",
                "propensity_model_n_estimators",
                "variant_outcome_model",
                "control_effect_model",
                "control_effect_model_n_estimators",
                "treatment_effect_model",
                "treatment_effect_model_n_estimators",
            ],
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
                "treatment_model": {
                    "LGBMRegressor": {"n_estimators": [1, 2, 3], "learning_rate": [0.4]}
                },
            },
            9,
            [
                "metalearner",
                "outcome_model",
                "propensity_model",
                "propensity_model_n_estimators",
                "treatment_model",
                "treatment_model_learning_rate",
                "treatment_model_n_estimators",
            ],
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
            [
                "metalearner",
                "propensity_model",
                "propensity_model_n_estimators",
                "variant_outcome_model",
                "treatment_model",
            ],
        ),
    ],
)
def test_metalearnergridsearch_smoke(
    metalearner_factory,
    is_classification,
    base_learner_grid,
    param_grid,
    expected_n_configs,
    expected_index_cols,
    grid_search_data,
):
    X, y_class, y_reg, w, X_test, y_test_class, y_test_reg, w_test = grid_search_data
    if is_classification:
        y = y_class
        y_test = y_test_class
    else:
        y = y_reg
        y_test = y_test_reg
    n_variants = len(np.unique(w))
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

    gs.fit(X, y, w, X_test, y_test, w_test)
    assert gs.results_ is not None
    assert gs.results_.shape[0] == expected_n_configs
    assert gs.results_.index.names == expected_index_cols
    assert gs.grid_size_ == expected_n_configs

    train_scores_cols = set(
        c[6:] for c in list(gs.results_.columns) if c.startswith("train_")
    )
    test_scores_cols = set(
        c[5:] for c in list(gs.results_.columns) if c.startswith("test_")
    )
    assert train_scores_cols == test_scores_cols


def test_metalearnergridsearch_reuse_nuisance_smoke(grid_search_data):
    X, _, y, w, X_test, _, y_test, w_test = grid_search_data
    n_variants = len(np.unique(w))

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
    assert len(gs.results_.index.names) == 7


def test_metalearnergridsearch_reuse_propensity_smoke(grid_search_data):
    X, _, y, w, X_test, _, y_test, w_test = grid_search_data
    n_variants = len(np.unique(w))

    rl = RLearner(
        False,
        n_variants,
        LGBMRegressor,
        LGBMRegressor,
        LGBMClassifier,
        nuisance_model_params={"verbose": -1, "n_estimators": 1},
        treatment_model_params={"verbose": -1, "n_estimators": 1},
        propensity_model_params={"verbose": -1, "n_estimators": 1},
        n_folds=2,
    )
    rl.fit(X, y, w)

    gs = MetaLearnerGridSearch(
        DRLearner,
        {
            "is_classification": False,
            "n_variants": n_variants,
            "n_folds": 5,  # To test with different n_folds than the pretrained
            "fitted_propensity_model": rl._nuisance_models[PROPENSITY_MODEL][0],
        },
        {
            "treatment_model": [LGBMRegressor],
            "variant_outcome_model": [LinearRegression],
        },
        {
            "treatment_model": {
                "LGBMRegressor": {"n_estimators": [1, 2], "verbose": [-1]}
            },
        },
        verbose=3,
        random_state=1,
    )
    gs.fit(X, y, w, X_test, y_test, w_test)
    assert gs.raw_results_ is not None
    for raw_result in gs.raw_results_:
        assert raw_result.metalearner._prefitted_nuisance_models == {PROPENSITY_MODEL}
    assert gs.results_ is not None
    assert gs.results_.shape[0] == 2
    assert len(gs.results_.index.names) == 5


@pytest.mark.parametrize(
    "store_raw_results, store_results, expected_type_raw_results, expected_type_results",
    [
        (True, True, list, pd.DataFrame),
        (True, False, list, type(None)),
        (False, True, type(None), pd.DataFrame),
        (False, False, GeneratorType, type(None)),
    ],
)
def test_metalearnergridsearch_store(
    store_raw_results,
    store_results,
    expected_type_raw_results,
    expected_type_results,
    grid_search_data,
):
    X, _, y, w, X_test, _, y_test, w_test = grid_search_data
    n_variants = len(np.unique(w))

    metalearner_params = {
        "is_classification": False,
        "n_variants": n_variants,
        "n_folds": 2,
    }

    gs = MetaLearnerGridSearch(
        metalearner_factory=SLearner,
        metalearner_params=metalearner_params,
        base_learner_grid={"base_model": [LinearRegression, LGBMRegressor]},
        param_grid={"base_model": {"LGBMRegressor": {"n_estimators": [1, 2]}}},
        store_raw_results=store_raw_results,
        store_results=store_results,
    )

    gs.fit(X, y, w, X_test, y_test, w_test)
    assert isinstance(gs.raw_results_, expected_type_raw_results)
    assert isinstance(gs.results_, expected_type_results)


def test_metalearnergridsearch_error(grid_search_data):
    X, _, y, w, X_test, _, y_test, w_test = grid_search_data
    n_variants = len(np.unique(w))

    metalearner_params = {
        "is_classification": False,
        "n_variants": n_variants,
        "n_folds": 2,
        "random_state": 1,
    }

    gs = MetaLearnerGridSearch(
        metalearner_factory=SLearner,
        metalearner_params=metalearner_params,
        base_learner_grid={"base_model": [LinearRegression, LGBMRegressor]},
        param_grid={"base_model": {"LGBMRegressor": {"n_estimators": [1, 2]}}},
    )
    with pytest.raises(
        ValueError, match="should not be specified in metalearner_params"
    ):
        gs.fit(X, y, w, X_test, y_test, w_test)
