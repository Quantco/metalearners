# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from metalearners.metalearner import MetaLearner
from metalearners.slearner import SLearner
from metalearners.tlearner import TLearner
from metalearners.utils import metalearner_factory
from metalearners.xlearner import XLearner

# Chosen arbitrarily.
_REFERENCE_VALUE_TOLERANCE = 0.05
_SEED = 1337
_TEST_FRACTION = 0.2
_LOG_REG_MAX_ITER = 500


def _is_classification(outcome_kind: str) -> bool:
    return outcome_kind == "binary"


def _linear_base_learner(is_classification: bool):
    if is_classification:
        return LogisticRegression
    return LinearRegression


def _tree_base_learner(is_classification: bool):
    if is_classification:
        return LGBMClassifier
    return LGBMRegressor


def _linear_base_learner_params(
    is_classification: bool,
) -> dict[str, int | float | str]:
    if is_classification:
        # Using the default value for max_iter sometimes
        # didn't lead to convergence.
        return {"random_state": _SEED, "max_iter": _LOG_REG_MAX_ITER}
    return {}


@pytest.mark.parametrize(
    "metalearner, outcome_kind, reference_value, treatment_kind, te_kind",
    [
        ("T", "binary", 0.0212, "binary", "linear"),
        ("T", "continuous", 0.0456, "binary", "linear"),
        ("S", "binary", 0.2290, "binary", "linear"),
        ("S", "continuous", 14.5706, "binary", "linear"),
        ("X", "binary", 0.3046, "binary", "linear"),
        ("X", "continuous", 0.0459, "binary", "linear"),
        ("R", "binary", 0.3046, "binary", "linear"),
        ("R", "continuous", 0.0470, "binary", "linear"),
    ],
)
def test_learner_synthetic_in_sample(
    metalearner, outcome_kind, reference_value, treatment_kind, te_kind, request
):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_{treatment_kind}_treatment_{te_kind}_te"
    )

    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)

    classifier_learner_factory = _linear_base_learner(True)
    regressor_learner_factory = _linear_base_learner(False)
    classifier_learner_params = _linear_base_learner_params(True)
    regressor_learner_params = _linear_base_learner_params(False)
    nuisance_learner_factory = (
        classifier_learner_factory if is_classification else regressor_learner_factory
    )
    nuisance_learner_params = (
        classifier_learner_params if is_classification else regressor_learner_params
    )

    factory = metalearner_factory(metalearner)
    learner = factory(
        nuisance_model_factory=nuisance_learner_factory,
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=regressor_learner_factory,
        propensity_model_factory=classifier_learner_factory,
        nuisance_model_params=nuisance_learner_params,
        treatment_model_params=regressor_learner_params,
        propensity_model_params=classifier_learner_params,
        random_state=_SEED,
    )

    learner.fit(covariates, observed_outcomes, treatment)
    cate_estimates = learner.predict(covariates, is_oos=False)
    if is_classification:
        cate_estimates = cate_estimates[:, 1]

    rmse = root_mean_squared_error(true_cate, cate_estimates)
    assert rmse < reference_value * (1 + _REFERENCE_VALUE_TOLERANCE)
    if metalearner == "T":
        np.testing.assert_allclose(
            cate_estimates, true_cate.reshape(-1), atol=0.3, rtol=0.3
        )


@pytest.mark.parametrize(
    "metalearner, outcome_kind, reference_value, treatment_kind, te_kind",
    [
        ("T", "binary", 0.0215, "binary", "linear"),
        ("T", "continuous", 0.0456, "binary", "linear"),
        ("S", "binary", 0.2286, "binary", "linear"),
        ("S", "continuous", 14.6248, "binary", "linear"),
        ("S", "continuous", 14.185, "multi", "linear"),
        ("S", "continuous", 0.0111, "multi", "constant"),
        ("X", "binary", 0.3019, "binary", "linear"),
        ("X", "continuous", 0.0456, "binary", "linear"),
        ("R", "binary", 0.3018, "binary", "linear"),
        ("R", "continuous", 0.0463, "binary", "linear"),
    ],
)
@pytest.mark.parametrize("oos_method", ["overall", "mean", "median"])
def test_learner_synthetic_oos(
    metalearner,
    outcome_kind,
    reference_value,
    treatment_kind,
    te_kind,
    oos_method,
    request,
):
    if outcome_kind == "binary" and oos_method == "median":
        pytest.skip()

    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_{treatment_kind}_treatment_{te_kind}_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)
    classifier_learner_factory = _linear_base_learner(True)
    regressor_learner_factory = _linear_base_learner(False)
    classifier_learner_params = _linear_base_learner_params(True)
    regressor_learner_params = _linear_base_learner_params(False)
    nuisance_learner_factory = (
        classifier_learner_factory if is_classification else regressor_learner_factory
    )
    nuisance_learner_params = (
        classifier_learner_params if is_classification else regressor_learner_params
    )

    factory = metalearner_factory(metalearner)
    learner = factory(
        nuisance_model_factory=nuisance_learner_factory,
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=regressor_learner_factory,
        propensity_model_factory=classifier_learner_factory,
        nuisance_model_params=nuisance_learner_params,
        treatment_model_params=regressor_learner_params,
        propensity_model_params=classifier_learner_params,
        random_state=_SEED,
    )
    (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    ) = train_test_split(
        covariates,
        observed_outcomes,
        treatment,
        true_cate,
        test_size=_TEST_FRACTION,
        random_state=_SEED,
    )
    learner.fit(covariates_train, observed_outcomes_train, treatment_train)
    cate_estimates = learner.predict(
        covariates_test, is_oos=True, oos_method=oos_method
    )

    if is_classification:
        cate_estimates = cate_estimates[:, 1]
    rmse = root_mean_squared_error(true_cate_test, cate_estimates)
    # See the benchmarking directory for the original reference values.
    assert rmse < reference_value * (1 + _REFERENCE_VALUE_TOLERANCE)
    if metalearner == "T":
        np.testing.assert_allclose(
            cate_estimates, true_cate_test.reshape(-1), atol=0.3, rtol=0.3
        )


@pytest.mark.parametrize(
    "metalearner, treatment_kind",
    [
        ("T", "binary"),
        ("S", "binary"),
        ("S", "multi"),
    ],
)
@pytest.mark.parametrize("oos_method", ["overall", "mean", "median"])
def test_learner_synthetic_oos_ate(metalearner, treatment_kind, oos_method, request):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_continuous_outcome_{treatment_kind}_treatment_linear_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset
    is_classification = False
    learner: MetaLearner
    base_learner = _linear_base_learner(is_classification)
    base_learner_params = _linear_base_learner_params(is_classification)
    if metalearner == "S":
        learner = SLearner(
            base_learner,
            is_classification,
            len(np.unique(treatment)),
            nuisance_model_params=base_learner_params,
            random_state=_SEED,
        )
    elif metalearner == "T":
        learner = TLearner(
            base_learner,
            is_classification,
            len(np.unique(treatment)),
            nuisance_model_params=base_learner_params,
            random_state=_SEED,
        )
    (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    ) = train_test_split(
        covariates,
        observed_outcomes,
        treatment,
        true_cate,
        test_size=_TEST_FRACTION,
        random_state=_SEED,
    )
    learner.fit(covariates_train, observed_outcomes_train, treatment_train)
    cate_estimates = learner.predict(
        covariates_test, is_oos=True, oos_method=oos_method
    )
    actual_ate_estimate = np.mean(cate_estimates)
    target_ate_estimate = np.mean(true_cate_test)
    assert actual_ate_estimate == pytest.approx(target_ate_estimate, abs=1e-2, rel=1e-1)


@pytest.mark.parametrize(
    "metalearner, reference_value",
    [("T", 0.3456), ("S", 0.3186), ("X", 0.3353), ("R", 0.3444)],
)
@pytest.mark.parametrize("oos_method", ["overall", "mean"])
def test_learner_twins(metalearner, reference_value, twins_data, oos_method, rng):
    chosen_df, outcome_column, treatment_column, feature_columns, _ = twins_data

    covariates = chosen_df[feature_columns]
    observed_outcomes = chosen_df[outcome_column]
    treatment = chosen_df[treatment_column]
    true_cate = chosen_df["mu_1"] - chosen_df["mu_0"]

    (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    ) = train_test_split(
        covariates,
        observed_outcomes,
        treatment,
        true_cate,
        test_size=_TEST_FRACTION,
        random_state=_SEED,
    )

    factory = metalearner_factory(metalearner)
    learner = factory(
        nuisance_model_factory=LGBMClassifier,
        is_classification=True,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=LGBMRegressor,
        propensity_model_factory=LGBMClassifier,
        nuisance_model_params={"random_state": rng},
        treatment_model_params={"random_state": rng},
        propensity_model_params={"random_state": rng},
        random_state=_SEED,
    )
    learner.fit(covariates_train, observed_outcomes_train, treatment_train)
    cate_estimates = learner.predict(
        covariates_test, is_oos=True, oos_method=oos_method
    )[:, 1]

    rmse = root_mean_squared_error(true_cate_test, cate_estimates)
    # See the benchmarking directory for reference values.
    assert rmse < reference_value * (1 + _REFERENCE_VALUE_TOLERANCE)


@pytest.mark.parametrize("metalearner", ["S", "T", "R"])
@pytest.mark.parametrize("outcome_kind", ["binary", "continuous"])
def test_learner_evaluate(metalearner, outcome_kind, request):
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_binary_treatment_linear_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset

    is_classification = _is_classification(outcome_kind)
    base_learner = _linear_base_learner(is_classification)

    factory = metalearner_factory(metalearner)
    learner = factory(
        nuisance_model_factory=base_learner,
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        n_folds=2,
    )
    learner.fit(X=covariates, y=observed_outcomes, w=treatment)
    evaluation = learner.evaluate(
        X=covariates, y=observed_outcomes, w=treatment, is_oos=False
    )
    if is_classification:
        if metalearner == "S":
            assert "cross_entropy" in evaluation
        elif metalearner == "T":
            assert "treatment_cross_entropy" in evaluation
            assert "effect_cross_entropy" in evaluation
        elif metalearner == "R":
            assert "outcome_log_loss" in evaluation
    else:
        if metalearner == "S":
            assert "rmse" in evaluation
        elif metalearner == "T":
            assert "treatment_rmse" in evaluation
            assert "effect_rmse" in evaluation
        elif metalearner == "R":
            assert "outcome_rmse" in evaluation
    if metalearner == "R":
        assert {"r_loss", "propensity_cross_entropy"} <= set(evaluation.keys())


@pytest.mark.parametrize("outcome_kind", ["binary", "continuous"])
@pytest.mark.parametrize("is_oos", [True, False])
def test_x_t_conditional_average_outcomes(outcome_kind, is_oos, request):
    """This test is to check that the conditional average outcomes predictions are the
    same for the TLearner and the XLearner as the nuisance models of the XLearner should
    be the same as the TLearner (except the propensity model)"""
    dataset = request.getfixturevalue(
        f"numerical_experiment_dataset_{outcome_kind}_outcome_binary_treatment_linear_te"
    )
    covariates, _, treatment, observed_outcomes, potential_outcomes, true_cate = dataset
    (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    ) = train_test_split(
        covariates,
        observed_outcomes,
        treatment,
        true_cate,
        test_size=_TEST_FRACTION,
        random_state=_SEED,
    )

    is_classification = _is_classification(outcome_kind)
    classifier_learner_factory = _linear_base_learner(True)
    regressor_learner_factory = _linear_base_learner(False)
    classifier_learner_params = _linear_base_learner_params(True)
    regressor_learner_params = _linear_base_learner_params(False)
    nuisance_learner_factory = (
        classifier_learner_factory if is_classification else regressor_learner_factory
    )
    nuisance_learner_params = (
        classifier_learner_params if is_classification else regressor_learner_params
    )

    tlearner = TLearner(
        nuisance_learner_factory,
        is_classification,
        n_variants=len(np.unique(treatment)),
        nuisance_model_params=nuisance_learner_params,
        random_state=_SEED,
    )
    xlearner = XLearner(
        nuisance_model_factory=nuisance_learner_factory,
        is_classification=is_classification,
        n_variants=len(np.unique(treatment)),
        treatment_model_factory=regressor_learner_factory,
        propensity_model_factory=classifier_learner_factory,
        nuisance_model_params=nuisance_learner_params,
        treatment_model_params=regressor_learner_params,
        propensity_model_params=classifier_learner_params,
        random_state=_SEED,
    )
    tlearner.fit(covariates_train, observed_outcomes_train, treatment_train)
    xlearner.fit(covariates_train, observed_outcomes_train, treatment_train)

    if not is_oos:
        covariates_test = covariates_train

    tlearner_cond_avg_outcomes = tlearner.predict_conditional_average_outcomes(
        covariates_test, is_oos=is_oos
    )
    xlearner_cond_avg_outcomes = xlearner.predict_conditional_average_outcomes(
        covariates_test, is_oos=is_oos
    )
    np.testing.assert_allclose(xlearner_cond_avg_outcomes, tlearner_cond_avg_outcomes)


@pytest.mark.parametrize(
    "metalearner_prefix,success",
    [
        ("S", True),
        ("T", False),
        ("X", False),
        ("R", False),
    ],
)
def test_check_n_variants_error_multi(metalearner_prefix, success):
    factory = metalearner_factory(metalearner_prefix)
    n_variants = 10
    if success:
        _ = factory(
            nuisance_model_factory=LinearRegression,
            is_classification=False,
            n_variants=n_variants,
            treatment_model_factory=LinearRegression,
            propensity_model_factory=LogisticRegression,
            n_folds=2,
        )
    else:
        with pytest.raises(NotImplementedError, match="Current implementation of"):
            _ = factory(
                nuisance_model_factory=LinearRegression,
                is_classification=False,
                n_variants=n_variants,
                treatment_model_factory=LinearRegression,
                propensity_model_factory=LogisticRegression,
                n_folds=2,
            )


@pytest.mark.parametrize("n_variants", [2.0, 1])
@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "X", "R"])
def test_check_n_variants_error_format(metalearner_prefix, n_variants):
    factory = metalearner_factory(metalearner_prefix)
    with pytest.raises(
        ValueError, match="n_variants needs to be an integer strictly greater than 1."
    ):
        _ = factory(
            nuisance_model_factory=LinearRegression,
            is_classification=False,
            n_variants=n_variants,
            treatment_model_factory=LinearRegression,
            propensity_model_factory=LogisticRegression,
            n_folds=2,
        )


@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "X", "R"])
def test_check_treatment_error_encoding(metalearner_prefix):
    covariates = np.zeros((10, 1))
    w = np.array([1, 2] * 5)
    y = np.zeros(10)

    factory = metalearner_factory(metalearner_prefix)
    learner = factory(
        nuisance_model_factory=LinearRegression,
        is_classification=False,
        n_variants=len(np.unique(w)),
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        n_folds=2,
    )

    with pytest.raises(ValueError, match="Treatment variant should be encoded"):
        learner.fit(covariates, y, w)


@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "X", "R"])
def test_check_treatment_error_different_instantiation(metalearner_prefix):
    covariates = np.zeros((10, 1))
    w = np.array(range(10))
    y = np.zeros(10)

    factory = metalearner_factory(metalearner_prefix)
    learner = factory(
        nuisance_model_factory=LinearRegression,
        is_classification=False,
        n_variants=2,
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        n_folds=2,
    )
    with pytest.raises(
        ValueError, match="Number of variants present in the treatment are different"
    ):
        learner.fit(covariates, y, w)


@pytest.mark.parametrize(
    "metalearner_prefix,success",
    [
        ("S", True),
        ("T", True),
        ("X", False),
        ("R", False),
    ],
)
def test_check_multi_class(metalearner_prefix, success):
    covariates = np.zeros((20, 1))
    w = np.array([0, 1] * 10)
    y = np.array([0, 1] * 8 + [2] * 4)

    factory = metalearner_factory(metalearner_prefix)
    learner = factory(
        nuisance_model_factory=LogisticRegression,
        is_classification=True,
        n_variants=len(np.unique(w)),
        treatment_model_factory=LinearRegression,
        propensity_model_factory=LogisticRegression,
        n_folds=2,
    )

    if success:
        learner.fit(covariates, y, w)
    else:
        with pytest.raises(
            ValueError, match="does not support multiclass classification."
        ):
            learner.fit(covariates, y, w)


@pytest.mark.parametrize("is_classification", [True, False])
@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "X"])
def test_conditional_average_outcomes_smoke(
    metalearner_prefix, is_classification, request
):
    (
        df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
    ) = request.getfixturevalue("twins_data" if is_classification else "mindset_data")
    factory = metalearner_factory(metalearner_prefix)
    learner = factory(
        nuisance_model_factory=_tree_base_learner(is_classification),
        nuisance_model_params={"n_estimators": 1},  # type: ignore
        is_classification=is_classification,
        n_variants=len(np.unique(df[treatment_column])),
        treatment_model_factory=LGBMRegressor,
        treatment_model_params={"n_estimators": 1},  # type: ignore
        propensity_model_factory=LGBMClassifier,
        propensity_model_params={"n_estimators": 1},  # type: ignore
        n_folds=2,
    )
    learner.fit(df[feature_columns], df[outcome_column], df[treatment_column])
    result = learner.predict_conditional_average_outcomes(  # type: ignore
        df[feature_columns], is_oos=False
    )
    if is_classification:
        assert result.shape == (
            len(df),
            df[treatment_column].nunique(),
            df[outcome_column].nunique(),
        )
    else:
        assert result.shape == (len(df), df[treatment_column].nunique())


@pytest.mark.parametrize(
    "metalearner_prefix", ["S", "T"]
)  # the ones that support multiclass
@pytest.mark.parametrize("n_classes", [5, 10])
@pytest.mark.parametrize("n_variants", [2, 5])
def test_conditional_average_outcomes_smoke_multi_class(
    metalearner_prefix, rng, sample_size, n_classes, n_variants
):
    factory = metalearner_factory(metalearner_prefix)
    if n_variants > 2 and not factory._supports_multi_treatment():
        pytest.skip()

    X = rng.standard_normal((sample_size, 10))
    w = rng.integers(0, n_variants, size=sample_size)
    y = rng.integers(0, n_classes, size=sample_size)
    learner = factory(
        nuisance_model_factory=_tree_base_learner(True),
        nuisance_model_params={"n_estimators": 1},  # type: ignore
        n_variants=n_variants,
        is_classification=True,
        n_folds=2,
    )
    learner.fit(X, y, w)
    result = learner.predict_conditional_average_outcomes(  # type: ignore
        X, is_oos=False
    )
    assert result.shape == (
        len(X),
        len(np.unique(w)),
        len(np.unique(y)),
    )
    np.testing.assert_allclose(result.sum(axis=-1), 1)


@pytest.mark.parametrize("metalearner_prefix", ["S", "T", "X", "R"])
@pytest.mark.parametrize("n_classes", [2, 5, 10])
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("is_classification", [True, False])
def test_predict_smoke(
    metalearner_prefix, is_classification, rng, sample_size, n_classes, n_variants
):
    factory = metalearner_factory(metalearner_prefix)
    if n_variants > 2 and not factory._supports_multi_treatment():
        pytest.skip()
    if n_classes != 2 and not is_classification:
        pytest.skip()  # skip repeated tests
    if is_classification and n_classes > 2 and not factory._supports_multi_class():
        pytest.skip()
    X = rng.standard_normal((sample_size, 10))
    w = rng.integers(0, n_variants, size=sample_size)
    if is_classification:
        y = rng.integers(0, n_classes, size=sample_size)
    else:
        y = rng.standard_normal(sample_size)
    learner = factory(
        nuisance_model_factory=_tree_base_learner(is_classification),
        nuisance_model_params={"n_estimators": 1},  # type: ignore
        n_variants=n_variants,
        is_classification=is_classification,
        treatment_model_factory=LGBMRegressor,
        treatment_model_params={"n_estimators": 1},  # type: ignore
        propensity_model_factory=LGBMClassifier,
        propensity_model_params={"n_estimators": 1},  # type: ignore
        n_folds=2,
    )
    learner.fit(X, y, w)
    result = learner.predict(X, is_oos=False)
    if is_classification:
        if n_variants > 2:
            assert result.shape == (len(X), n_variants - 1, n_classes)
        else:
            assert result.shape == (len(X), n_classes)
        np.testing.assert_allclose(result.sum(axis=-1), 0, atol=1e-10)
    else:
        if n_variants > 2:
            assert result.shape == (len(X), n_variants - 1)
        else:
            assert result.shape == (len(X),)
