# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from metalearners._typing import Matrix, OosMethod, Scoring, Vector, _ScikitModel
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import PROPENSITY_MODEL, MetaLearner


@dataclass(frozen=True)
class _FitAndScoreJob:
    metalearner: MetaLearner
    X_train: Matrix
    y_train: Vector
    w_train: Vector
    X_test: Matrix | None
    y_test: Vector | None
    w_test: Vector | None
    oos_method: OosMethod
    scoring: Scoring | None
    # These are the params which are passed through kwargs in MetaLearnerGridSearch.fit
    # which should be unpacked and passed to MetaLearner.fit
    metalerner_fit_params: dict[str, Any]


@dataclass(frozen=True)
class _GSResult:
    r"""Result from a single grid search evaluation."""

    metalearner: MetaLearner
    train_scores: dict
    test_scores: dict | None
    fit_time: float
    score_time: float


def _fit_and_score(job: _FitAndScoreJob) -> _GSResult:
    start_time = time.time()
    job.metalearner.fit(
        job.X_train, job.y_train, job.w_train, **job.metalerner_fit_params
    )
    fit_time = time.time() - start_time

    train_scores = job.metalearner.evaluate(
        X=job.X_train,
        y=job.y_train,
        w=job.w_train,
        is_oos=False,
        scoring=job.scoring,
    )
    if job.X_test is not None and job.y_test is not None and job.w_test is not None:
        test_scores = job.metalearner.evaluate(
            X=job.X_test,
            y=job.y_test,
            w=job.w_test,
            is_oos=True,
            oos_method=job.oos_method,
            scoring=job.scoring,
        )
    else:
        test_scores = None
    score_time = time.time() - fit_time
    return _GSResult(
        metalearner=job.metalearner,
        fit_time=fit_time,
        score_time=score_time,
        train_scores=train_scores,
        test_scores=test_scores,
    )


def _format_results(results: Sequence[_GSResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row: dict[str, str | int | float] = {}
        row["metalearner"] = result.metalearner.__class__.__name__
        nuisance_models = (
            set(result.metalearner.nuisance_model_specifications().keys())
            - result.metalearner._prefitted_nuisance_models
        )
        treatment_models = set(
            result.metalearner.treatment_model_specifications().keys()
        )
        for model_kind in nuisance_models:
            row[model_kind] = result.metalearner.nuisance_model_factory[
                model_kind
            ].__name__
            for param, value in result.metalearner.nuisance_model_params[
                model_kind
            ].items():
                row[f"{model_kind}_{param}"] = value
        for model_kind in treatment_models:
            row[model_kind] = result.metalearner.treatment_model_factory[
                model_kind
            ].__name__
            for param, value in result.metalearner.treatment_model_params[
                model_kind
            ].items():
                row[f"{model_kind}_{param}"] = value
        row["fit_time"] = result.fit_time
        row["score_time"] = result.score_time
        for name, value in result.train_scores.items():
            row[f"train_{name}"] = value
        if result.test_scores is not None:
            for name, value in result.test_scores.items():
                row[f"test_{name}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    index_columns = [
        c
        for c in df.columns
        if not c.endswith("_time")
        and not c.startswith("train_")
        and not c.startswith("test_")
    ]
    df = df.set_index(index_columns)
    return df


class MetaLearnerGridSearch:
    """Exhaustive search over specified parameter values for a MetaLearner.

    ``metalearner_params`` should contain the necessary params for the MetaLearner initialization
    such as ``n_variants`` and ``is_classification``. If one wants to pass optional parameters
    to the ``MetaLearner`` initialization, such as ``n_folds`` or ``feature_set``, this should
    be done by this way, too.
    Importantly, ``random_state`` must be passed through the ``random_state`` parameter
    and not through ``metalearner_params``.

    ``base_learner_grid`` keys should be the names of the needed base models contained in the
    :class:`~metalearners.metalearners.MetaLearner` defined by ``metalearner_factory``, for
    information about this names check
    :meth:`~metalearners.metalearner.MetaLearner.nuisance_model_specifications` and
    :meth:`~metalearners.metalearner.MetaLearner.treatment_model_specifications`. The
    values should be sequences of model factories.

    If base models are meant to be reused, they should be passed through ``metalearner_params``
    and the corresponding keys should not be passed to ``base_learner_grid``.

    ``param_grid`` should contain the parameters grid for each type of model used by the
    base learners defined in ``base_learner_grid``. The keys should be strings with the
    model class name. An example for optimizing over the :class:`metalearners.DRLearner`
    would be:

    .. code-block:: python

        base_learner_grid = {
            "propensity_model": (LGBMClassifier, LogisticRegression),
            "variant_outcome_model": (LGBMRegressor, LinearRegression),
            "treatment_model": (LGBMRegressor)
        }

        param_grid = {
            "propensity_model": {
                "LGBMClassifier": {"n_estimators": [1, 2, 3], "verbose": [-1]}
            },
            "variant_outcome_model": {
                "LGBMRegressor": {"n_estimators": [1, 2], "verbose": [-1]},
            },
            "treatment_model": {
                "LGBMRegressor": {"n_estimators": [5, 10], "verbose": [-1]},
            },
        }

    If some model is not present in ``param_grid``, the default parameters will be used.

    For information on how to define ``scoring`` see :meth:`~metalearners.metalearner.MetaLearner.evaluate`.

    ``verbose`` will be passed to `joblib.Parallel <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`_.

    After fitting a dataframe with the results will be available in `results_`.
    """

    # TODO: Add a reference to a docs example once it is written.

    def __init__(
        self,
        metalearner_factory: type[MetaLearner],
        metalearner_params: Mapping[str, Any],
        base_learner_grid: Mapping[str, Sequence[type[_ScikitModel]]],
        param_grid: Mapping[str, Mapping[str, Mapping[str, Sequence]]],
        scoring: Scoring | None = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.metalearner_factory = metalearner_factory
        self.metalearner_params = metalearner_params
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.raw_results_: Sequence[_GSResult] | None = None
        self.results_: pd.DataFrame | None = None

        all_base_models = set(
            metalearner_factory.nuisance_model_specifications().keys()
        ) | set(metalearner_factory.treatment_model_specifications().keys())

        self.fitted_models = set(
            metalearner_params.get("fitted_nuisance_models", {}).keys()
        )
        if metalearner_params.get("fitted_propensity_model", None) is not None:
            self.fitted_models |= {PROPENSITY_MODEL}

        self.models_to_fit = all_base_models - self.fitted_models

        if set(base_learner_grid.keys()) != self.models_to_fit:
            raise ValueError(
                "base_learner_grid keys don't match the expected model names. base_learner_grid "
                f"keys were expected to be {self.models_to_fit}."
            )
        self.base_learner_grid = base_learner_grid
        self.param_grid = param_grid

    def fit(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        X_test: Matrix | None = None,
        y_test: Vector | None = None,
        w_test: Vector | None = None,
        oos_method: OosMethod = OVERALL,
        **kwargs,
    ):
        """Run fit with all sets of parameters.

        ``X_test``, ``y_test`` and ``w_test`` are optional, in case they are passed all the
        fitted metalearners will be evaluated on it.

        ``kwargs`` will be passed through to the :meth:`~metalearners.metalearner.MetaLearner.fit`
        call of each individual MetaLearner.
        """
        nuisance_models_wo_propensity = (
            set(self.metalearner_factory.nuisance_model_specifications().keys())
            - {PROPENSITY_MODEL}
        ) & self.models_to_fit

        # We don't need to intersect as treatment models can't be reused
        treatment_models = set(
            self.metalearner_factory.treatment_model_specifications().keys()
        )

        jobs: list[_FitAndScoreJob] = []

        for base_learners in ParameterGrid(self.base_learner_grid):
            nuisance_model_factory = {
                model_kind: base_learners[model_kind]
                for model_kind in nuisance_models_wo_propensity
            }
            treatment_model_factory = {
                model_kind: base_learners[model_kind] for model_kind in treatment_models
            }
            propensity_model_factory = base_learners.get(PROPENSITY_MODEL, None)
            base_learner_param_grids = {
                model_kind: list(
                    ParameterGrid(
                        self.param_grid.get(model_kind, {}).get(
                            base_learners[model_kind].__name__, {}
                        )
                    )
                )
                for model_kind in self.models_to_fit
            }
            for params in ParameterGrid(base_learner_param_grids):
                nuisance_model_params = {
                    model_kind: params[model_kind]
                    for model_kind in nuisance_models_wo_propensity
                }
                treatment_model_params = {
                    model_kind: params[model_kind] for model_kind in treatment_models
                }
                propensity_model_params = params.get(PROPENSITY_MODEL, None)

                ml = self.metalearner_factory(
                    **self.metalearner_params,
                    nuisance_model_factory=nuisance_model_factory,
                    treatment_model_factory=treatment_model_factory,
                    propensity_model_factory=propensity_model_factory,
                    nuisance_model_params=nuisance_model_params,
                    treatment_model_params=treatment_model_params,
                    propensity_model_params=propensity_model_params,
                    random_state=self.random_state,
                )

                jobs.append(
                    _FitAndScoreJob(
                        metalearner=ml,
                        X_train=X,
                        y_train=y,
                        w_train=w,
                        X_test=X_test,
                        y_test=y_test,
                        w_test=w_test,
                        oos_method=oos_method,
                        scoring=self.scoring,
                        metalerner_fit_params=kwargs,
                    )
                )

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        raw_results = parallel(delayed(_fit_and_score)(job) for job in jobs)
        self.raw_results_ = raw_results
        self.results_ = _format_results(results=raw_results)
