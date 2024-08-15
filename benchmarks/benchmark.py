# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path

import numpy as np
import pandas as pd
from causalml.inference.meta import (
    BaseDRRegressor,
    BaseRClassifier,
    BaseRRegressor,
    BaseSClassifier,
    BaseSRegressor,
    BaseTClassifier,
    BaseTRegressor,
    BaseXClassifier,
    BaseXRegressor,
)
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from git_root import git_root
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from metalearners._utils import get_linear_dimension, load_twins_data
from metalearners.data_generation import (
    compute_experiment_outputs,
    generate_covariates,
    generate_treatment,
)
from metalearners.outcome_functions import (
    constant_treatment_effect,
    linear_treatment_effect,
)
from metalearners.utils import metalearner_factory, simplify_output

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
    (
        chosen_df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
        true_cate_column,
    ) = load_twins_data(Path(git_root()) / "data" / "twins.zip", rng)

    covariates = chosen_df[feature_columns]
    observed_outcomes = chosen_df[outcome_column]
    treatment = chosen_df[treatment_column]
    true_cate = chosen_df[true_cate_column]

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
    if n_variants > 2 and metalearner == "S":
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
    elif metalearner == "R":
        if is_classification:
            learner = BaseRClassifier(
                outcome_learner=classifier_learner_factory(**classifier_learner_params),
                effect_learner=regressor_learner_factory(**regressor_learner_params),
            )
        else:
            learner = BaseRRegressor(
                learner=regressor_learner_factory(**regressor_learner_params),
            )
        # The default model does CV so it's not comparable
        learner.model_p = classifier_learner_factory(**classifier_learner_params)
    elif metalearner == "DR":
        if is_classification:
            raise ValueError("causalml has no classifier version of the DRLearner.")
        else:
            # TODO: Unlike other MetaLearners, the causalml DR-Learner doesn't estimate
            # propensities via the model_p field. Rather, it relies on an independent
            # function called compute_propensity_score:
            # https://github.com/uber/causalml/blob/a0315660d9b14f5d943aa688d8242eb621d2ba76/causalml/inference/meta/drlearner.py#L162
            # Hence, the current setup uses a causalml internal approach of estimating
            # propensities, including CV.
            learner = BaseDRRegressor(
                learner=regressor_learner_factory(**regressor_learner_params),
            )
    else:
        raise NotImplementedError()

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
    elif metalearner == "DR":
        if is_classification:
            est = DRLearner(
                model_propensity=classifier_learner_factory(
                    **classifier_learner_params
                ),
                model_regression=classifier_learner_factory(
                    **classifier_learner_params
                ),
                model_final=regressor_learner_factory(**regressor_learner_params),
                allow_missing=True,
                discrete_outcome=True,
            )
        else:
            est = DRLearner(
                model_propensity=classifier_learner_factory(
                    **classifier_learner_params
                ),
                model_regression=regressor_learner_factory(**regressor_learner_params),
                model_final=regressor_learner_factory(**regressor_learner_params),
                allow_missing=True,
                discrete_outcome=False,
            )
    else:
        raise ValueError(f"{metalearner}-Learner not supported for econml.")
    est.fit(
        observed_outcomes_train.ravel(), treatment_train.ravel(), X=covariates_train
    )
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
        n_variants=n_variants,
        treatment_model_factory=regressor_learner_factory,
        propensity_model_factory=classifier_learner_factory,
        nuisance_model_params=nuisance_learner_params,
        treatment_model_params=regressor_learner_params,
        propensity_model_params=classifier_learner_params,
        random_state=_SEED,
    )

    learner.fit(
        covariates_train,
        observed_outcomes_train,
        treatment_train,
        synchronize_cross_fitting=True,
    )
    estimates = learner.predict(covariates_test, is_oos=is_oos, oos_method="overall")

    return simplify_output(estimates)


def evaluate(
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
        if (
            is_classification
            and library == "econml"
            and (metalearner != "DR" or pd.isna(covariates_test).any().any())
        ):
            # Most of econml's MetaLearners don't seem to support calling
            # predict_proba under the hood.
            # Moreover, the DRLearner classifier fails for missing values, even with
            # allow_missing=True due to this line:
            # https://github.com/py-why/EconML/blob/ea46d0d2816f2b70e67f5e6699157502038c8bf1/econml/_cate_estimator.py#L857
            continue
        if n_variants > 2 and library == "causalml" and metalearner == "S":
            # Causalml does a different model for each variant and hence it's
            # not comparable
            continue
        if metalearner == "DR" and is_classification and library == "causalml":
            # Causalml doesn't have a classifier version of the DR-Learner.
            continue
        if metalearner == "R" and library == "econml":
            # Econml has a private R-Learner implementation, see
            # https://github.com/py-why/EconML/blob/ea46d0d2816f2b70e67f5e6699157502038c8bf1/econml/dml/_rlearner.py#L115
            # Yet, this class has abstract methods and would need to be completed
            # and overwritten by an end-user.
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
    return evaluate(
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
    return evaluate(
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


# TODO: For some reason econml's DRLearner performs much worse
# than causalml and the metalearners library on synthetic data
# with a continuous outcome.

if __name__ == "__main__":
    losses: dict[str, dict] = {}

    for metalearner in ["T", "S", "X", "R", "DR"]:
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
