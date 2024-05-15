# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import json
from pathlib import Path

import numpy as np
import pandas as pd
from causalml.inference.meta import (
    BaseSClassifier,
    BaseSRegressor,
    BaseTClassifier,
    BaseTRegressor,
    BaseXClassifier,
    BaseXRegressor,
)
from econml.metalearners import SLearner, TLearner, XLearner
from git_root import git_root
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

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
from metalearners.utils import metalearner_factory

_SEED = 1337


def _synthetic_data(
    is_classification,
    rng,
    sample_size=100_000,
    n_numericals=25,
    test_fraction=0.2,
    propensity_score=0.3,
    tau=None,
):
    covariates, _, _ = generate_covariates(
        sample_size, n_numericals, format="numpy", rng=rng
    )

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
        n_variants=n_variants,
        is_classification=is_classification,
        return_probability_cate=True,
        rng=rng,
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
        test_size=test_fraction,
        random_state=_SEED,
    )
    return (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    )


def _twins_data(rng, use_numpy=False, test_fraction=0.2):
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

    covariates = chosen_df[feature_columns]
    observed_outcomes = chosen_df[outcome_column]
    treatment = chosen_df[treatment_column]
    true_cate = mu_1 - mu_0

    if use_numpy:
        covariates = covariates.to_numpy()
        observed_outcomes = observed_outcomes.to_numpy()
        treatment = treatment.to_numpy()
        true_cate = true_cate.to_numpy()

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
        test_size=test_fraction,
        random_state=_SEED,
    )

    return (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    )


def causalml_estimates(
    metalearner,
    observed_outcomes_train,
    treatment_train,
    covariates_train,
    covariates_test,
    regressor_learner_factory,
    classifier_learner_factory,
    is_classification,
    n_variants,
    *args,
    regressor_learner_params=None,
    classifier_learner_params=None,
    **kwargs,
):
    if n_variants > 2:
        raise ValueError(
            "Causalml does a different model for each variant and "
            "hence it's not comparable"
        )
    regressor_learner_params = regressor_learner_params or {}
    classifier_learner_params = classifier_learner_params or {}
    if metalearner == "T":
        if is_classification:
            learner = BaseTClassifier(
                classifier_learner_factory(**classifier_learner_params)
            )
            if hasattr(covariates_test, "to_numpy"):
                # https://github.com/uber/causalml/blob/a0315660d9b14f5d943aa688d8242eb621d2ba76/causalml/inference/meta/tlearner.py#L375
                # In causalml there's a bug where they don't convert pandas to numpy when calling
                # predict in BaseTClassifier, also lightgbm does not behave the same way
                # with pandas than with numpy (not sure why) and therefore this gave different
                # predictions
                covariates_test = covariates_test.to_numpy()
        else:
            learner = BaseTRegressor(
                regressor_learner_factory(**regressor_learner_params)
            )
    elif metalearner == "S":
        if is_classification:
            learner = BaseSClassifier(
                classifier_learner_factory(**classifier_learner_params)
            )
        else:
            learner = BaseSRegressor(
                regressor_learner_factory(**regressor_learner_params)
            )
    elif metalearner == "X":
        if is_classification:
            learner = BaseXClassifier(
                outcome_learner=classifier_learner_factory(**classifier_learner_params),
                effect_learner=regressor_learner_factory(**regressor_learner_params),
            )
        else:
            learner = BaseXRegressor(
                learner=regressor_learner_factory(**regressor_learner_params),
            )
        # The default model does CV so it's not comparable
        learner.model_p = classifier_learner_factory(**classifier_learner_params)
    else:
        raise NotImplementedError

    learner.fit(covariates_train, treatment_train, observed_outcomes_train)
    return learner.predict(covariates_test)


def econml_estimates(
    metalearner,
    observed_outcomes_train,
    treatment_train,
    covariates_train,
    covariates_test,
    regressor_learner_factory,
    classifier_learner_factory,
    is_classification,
    n_variants,
    *args,
    regressor_learner_params=None,
    classifier_learner_params=None,
    **kwargs,
):
    regressor_learner_params = regressor_learner_params or {}
    if metalearner == "T":
        est = TLearner(
            models=regressor_learner_factory(**regressor_learner_params),
            allow_missing=True,
        )
    elif metalearner == "S":
        est = SLearner(
            overall_model=regressor_learner_factory(**regressor_learner_params),
            allow_missing=True,
        )
    elif metalearner == "X":
        # econml does not support classification so we always use regressor for
        # outcomes and effect models
        est = XLearner(
            models=regressor_learner_factory(**regressor_learner_params),
            allow_missing=True,
            propensity_model=classifier_learner_factory(**classifier_learner_params),
        )
    est.fit(observed_outcomes_train, treatment_train, X=covariates_train)
    if n_variants > 2:
        estimates = []
        for v in range(1, n_variants):
            estimates.append(est.effect(covariates_test, T0=0, T1=v))
        return np.stack(estimates, axis=1)
    else:
        return est.effect(covariates_test)


def metalearner_estimates(
    metalearner,
    observed_outcomes_train,
    treatment_train,
    covariates_train,
    covariates_test,
    regressor_learner_factory,
    classifier_learner_factory,
    is_classification,
    n_variants,
    is_oos,
    regressor_learner_params=None,
    classifier_learner_params=None,
):
    factory = metalearner_factory(metalearner)

    nuisance_learner_factory = (
        classifier_learner_factory if is_classification else regressor_learner_factory
    )
    nuisance_learner_params = (
        classifier_learner_params if is_classification else regressor_learner_params
    )
    learner = factory(
        nuisance_model_factory=nuisance_learner_factory,
        is_classification=is_classification,
        treatment_model_factory=regressor_learner_factory,
        propensity_model_factory=classifier_learner_factory,
        nuisance_model_params=nuisance_learner_params,
        treatment_model_params=regressor_learner_params,
        propensity_model_params=classifier_learner_params,
        random_state=_SEED,
    )
    learner.fit(covariates_train, observed_outcomes_train, treatment_train)
    estimates = learner.predict(covariates_test, is_oos=is_oos, oos_method="overall")
    if is_classification:
        return estimates[:, 1]
    return estimates


def eval(
    data,
    is_classification,
    metalearner,
    regressor_factory,
    classifier_factory,
    n_variants,
    regressor_learner_params=None,
    classifier_learner_params=None,
):
    (
        covariates_train,
        covariates_test,
        observed_outcomes_train,
        observed_outcomes_test,
        treatment_train,
        treatment_test,
        true_cate_train,
        true_cate_test,
    ) = data

    losses = {}

    for library, func in (
        ("causalml", causalml_estimates),
        ("econml", econml_estimates),
        ("metalearners", metalearner_estimates),
    ):
        if is_classification and library == "econml":
            # Econml's TLearner doesn't seem to support calling
            # predict_proba under the hood.
            continue
        if n_variants > 2 and library == "causalml":
            # Causalml does a different model for each variant and hence it's
            # not comparable
            continue
        print(f"{library}...")
        for eval_kind in ["in_sample", "oos"]:
            is_oos = eval_kind == "oos"
            covariates_eval = covariates_test if is_oos else covariates_train
            estimates = func(  # type: ignore
                metalearner,
                observed_outcomes_train,
                treatment_train,
                covariates_train,
                covariates_eval,
                regressor_learner_factory=regressor_factory,
                classifier_learner_factory=classifier_factory,
                is_classification=is_classification,
                n_variants=n_variants,
                is_oos=is_oos,
                regressor_learner_params=regressor_learner_params,
                classifier_learner_params=classifier_learner_params,
            )
            ground_truth = true_cate_test if is_oos else true_cate_train
            losses[library + "_" + eval_kind] = root_mean_squared_error(
                ground_truth, estimates
            )

    return losses


def losses_synthetic_data(
    is_classification, metalearner, propensity_score=0.3, tau=None
):
    rng = np.random.default_rng(_SEED)
    data = _synthetic_data(
        is_classification, rng, propensity_score=propensity_score, tau=tau
    )
    regressor_factory = LinearRegression
    classifier_factory = LogisticRegression
    regressor_learner_params: dict = {}
    classifier_learner_params = {"random_state": _SEED, "max_iter": 500}
    n_variants = len(propensity_score) if isinstance(propensity_score, list) else 2
    return eval(
        data,
        is_classification,
        metalearner,
        regressor_factory,
        classifier_factory,
        n_variants,
        regressor_learner_params=regressor_learner_params,
        classifier_learner_params=classifier_learner_params,
    )


def losses_twins_data(metalearner, use_numpy=False):
    rng = np.random.default_rng(_SEED)
    data = _twins_data(rng, use_numpy=use_numpy)
    classifier_factory = LGBMClassifier
    classifier_learner_params = {"verbose": -1, "random_state": rng}
    regressor_factory = LGBMRegressor
    regressor_learner_params: dict = {"verbose": -1, "random_state": rng}
    return eval(
        data,
        True,
        metalearner,
        regressor_factory,
        classifier_factory,
        2,
        regressor_learner_params=regressor_learner_params,
        classifier_learner_params=classifier_learner_params,
    )


def print_separator(n_dashes=15):
    print("".join(["-"] * n_dashes))


def dict_to_json_file(d, filename="comparison.json"):
    path = Path(git_root()) / "benchmarks" / filename
    with open(path, "w") as filehandle:
        json.dump(losses, filehandle, indent=4)
    print(f"Dumped results to {path}.")


def dict_to_markdown_file(d, filename="comparison.md"):
    text = ""
    for metalearner, data in d.items():
        df = pd.DataFrame(data).T
        df.index.name = metalearner
        text += df.to_markdown() + "\n\n"

    path = Path(git_root()) / "benchmarks" / filename
    with open(path, "w") as filehandle:
        filehandle.write(text)
    print(f"Dumped results as markdown table to {path}.")


if __name__ == "__main__":
    losses: dict[str, dict] = {}

    for metalearner in ["T", "S", "X"]:
        print(f"{metalearner}-learner...")
        print_separator()
        print_separator()

        losses[f"{metalearner}-learner"] = {}
        print("Start comparing libraries on synthetic data with continuous outcomes.")
        losses[f"{metalearner}-learner"][
            "synthetic_data_continuous_outcome_binary_treatment_linear_te"
        ] = losses_synthetic_data(is_classification=False, metalearner=metalearner)
        print_separator()

        print("Start comparing libraries on synthetic data with binary outcomes.")
        losses[f"{metalearner}-learner"][
            "synthetic_data_binary_outcome_binary_treatment_linear_te"
        ] = losses_synthetic_data(is_classification=True, metalearner=metalearner)
        print_separator()

        print("Start comparing libraries on real-world data.")
        losses[f"{metalearner}-learner"]["twins_pandas"] = losses_twins_data(
            metalearner=metalearner, use_numpy=False
        )
        losses[f"{metalearner}-learner"]["twins_numpy"] = losses_twins_data(
            metalearner=metalearner, use_numpy=True
        )
        print_separator()

        if metalearner in {"S"}:  # implemented multivariant support
            print(
                "Start comparing libraries on synthetic data with continuous "
                "outcomes, multiple treatments and linear treatment effect."
            )
            losses[f"{metalearner}-learner"][
                "synthetic_data_continuous_outcome_multi_treatment_linear_te"
            ] = losses_synthetic_data(
                is_classification=False,
                metalearner=metalearner,
                propensity_score=[0.2, 0.1, 0.3, 0.15, 0.25],
            )
            print_separator()

            print(
                "Start comparing libraries on synthetic data with continuous "
                "outcomes, multiple treatments and constant treatment effect."
            )
            losses[f"{metalearner}-learner"][
                "synthetic_data_continuous_outcome_multi_treatment_constant_te"
            ] = losses_synthetic_data(
                is_classification=False,
                metalearner=metalearner,
                propensity_score=[0.2, 0.1, 0.3, 0.15, 0.25],
                tau=np.array([-2, 5, 0, 3]),
            )
            print_separator()
            # TODO: Add benchmarking with classification outcomes and multiple treatments,
            # when data_generation allows for it.

    dict_to_json_file(losses)
    dict_to_markdown_file(losses)