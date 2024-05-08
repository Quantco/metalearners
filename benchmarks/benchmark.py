# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import json
from pathlib import Path

import numpy as np
import pandas as pd
from causalml.inference.meta import BaseTClassifier, BaseTRegressor
from econml.metalearners import TLearner
from git_root import git_root
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from metalearners import tlearner as tl
from metalearners._utils import get_linear_dimension
from metalearners.data_generation import (
    compute_experiment_outputs,
    generate_covariates,
    generate_treatment,
)
from metalearners.outcome_functions import linear_treatment_effect

_SEED = 1337


def _synthetic_data(
    is_classification, rng, sample_size=1_000_000, n_numericals=25, test_fraction=0.2
):
    covariates, _, _ = generate_covariates(
        sample_size, n_numericals, format="numpy", rng=rng
    )
    propensity_scores = 0.3 * np.ones(sample_size)
    treatment = generate_treatment(propensity_scores, rng=rng)
    dim = get_linear_dimension(covariates)
    outcome_function = linear_treatment_effect(dim, rng=rng)
    potential_outcomes = outcome_function(covariates)
    observed_outcomes, true_cate = compute_experiment_outputs(
        potential_outcomes,
        treatment,
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

    if use_numpy:
        covariates_train = covariates_train.to_numpy()
        covariates_test = covariates_test.to_numpy()

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
    observed_outcomes_train,
    treatment_train,
    covariates_train,
    covariates_test,
    base_learner_factory,
    is_classification,
    *args,
    base_learner_params=None,
    **kwargs,
):
    causal_factory = BaseTClassifier if is_classification else BaseTRegressor
    base_learner_params = base_learner_params or {}
    tlearner = causal_factory(base_learner_factory(**base_learner_params))
    tlearner.fit(covariates_train, treatment_train, observed_outcomes_train)
    return tlearner.predict(covariates_test)


def econml_estimates(
    observed_outcomes_train,
    treatment_train,
    covariates_train,
    covariates_test,
    base_learner_factory,
    is_classification,
    *args,
    base_learner_params=None,
    **kwargs,
):
    base_learner_params = base_learner_params or {}
    est = TLearner(
        models=base_learner_factory(**base_learner_params), allow_missing=True
    )
    est.fit(observed_outcomes_train, treatment_train, X=covariates_train)
    return est.effect(covariates_test)


def metalearner_estimates(
    observed_outcomes_train,
    treatment_train,
    covariates_train,
    covariates_test,
    base_learner_factory,
    is_classification,
    is_oos,
    base_learner_params=None,
):
    tlearner = tl.TLearner(
        base_learner_factory,
        is_classification,
        nuisance_model_params=base_learner_params,
        random_state=_SEED,
    )
    tlearner.fit(covariates_train, observed_outcomes_train, treatment_train)
    estimates = tlearner.predict(covariates_test, is_oos=is_oos, oos_method="overall")
    if is_classification:
        return estimates[:, 1]
    return estimates


def eval(data, is_classification, factory, base_learner_params=None):
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
        print(f"{library}...")
        for eval_kind in ["in_sample", "oos"]:
            is_oos = eval_kind == "oos"
            covariates_eval = covariates_test if is_oos else covariates_train
            estimates = func(  # type: ignore
                observed_outcomes_train,
                treatment_train,
                covariates_train,
                covariates_eval,
                base_learner_factory=factory,
                is_classification=is_classification,
                is_oos=is_oos,
                base_learner_params=base_learner_params,
            )
            ground_truth = true_cate_test if is_oos else true_cate_train
            losses[library + "_" + eval_kind] = root_mean_squared_error(
                ground_truth, estimates
            )

    return losses


def losses_synthetic_data(is_classification):
    rng = np.random.default_rng(_SEED)
    data = _synthetic_data(is_classification, rng)
    factory = LogisticRegression if is_classification else LinearRegression
    base_learner_params = (
        {"random_state": _SEED, "max_iter": 500} if is_classification else {}
    )
    return eval(
        data, is_classification, factory, base_learner_params=base_learner_params
    )


def losses_twins_data(use_numpy=False):
    rng = np.random.default_rng(_SEED)
    data = _twins_data(rng, use_numpy=use_numpy)
    base_learner = LGBMClassifier
    base_learner_params = {"verbose": -1, "random_state": rng}
    return eval(data, True, base_learner, base_learner_params)


def print_separator(n_dashes=15):
    print("".join(["-"] * n_dashes))


def dict_to_json_file(d, filename="comparison.json"):
    path = Path(git_root()) / "benchmarks" / filename
    with open(path, "w") as filehandle:
        json.dump(losses, filehandle, indent=4)
    print(f"Dumped results to {path}.")


def dict_to_markdown_file(d, filename="comparison.md"):
    df = pd.DataFrame(d).T
    text = df.to_markdown()

    path = Path(git_root()) / "benchmarks" / filename
    with open(path, "w") as filehandle:
        filehandle.write(text)
    print(f"Dumped results as markdown table to {path}.")


if __name__ == "__main__":
    losses = {}

    print("Start comparing libraries on synthetic data with continuous outcomes.")
    losses["synthetic_data_continuous_outcome"] = losses_synthetic_data(
        is_classification=False,
    )
    print_separator()

    print("Start comparing libraries on synthetic data with binary outcomes.")
    losses["synthetic_data_binary_outcome"] = losses_synthetic_data(
        is_classification=True
    )
    print_separator()

    print("Start comparing libraries on real-world data.")
    losses["twins_pandas"] = losses_twins_data(use_numpy=False)
    losses["twins_numpy"] = losses_twins_data(use_numpy=True)
    print_separator()

    dict_to_json_file(losses)
    dict_to_markdown_file(losses)
