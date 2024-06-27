# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from operator import add

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, ParameterGrid

from metalearners._typing import Matrix, OosMethod, Scoring, Vector, _ScikitModel
from metalearners._utils import index_matrix, index_vector
from metalearners.cross_fit_estimator import OVERALL
from metalearners.metalearner import PROPENSITY_MODEL, MetaLearner


@dataclass(frozen=True)
class _FitAndScoreJob:
    metalearner: MetaLearner
    X_train: Matrix
    y_train: Vector
    w_train: Vector
    X_test: Matrix
    y_test: Vector
    w_test: Vector
    oos_method: OosMethod
    scoring: Scoring | None
    kwargs: dict
    cv_index: int


@dataclass(frozen=True)
class _CVResult:
    r"""Cross Validation Result."""

    metalearner: MetaLearner
    train_scores: dict
    test_scores: dict
    fit_time: float
    score_time: float
    cv_index: int


def _fit_and_score(job: _FitAndScoreJob) -> _CVResult:
    start_time = time.time()
    job.metalearner.fit(job.X_train, job.y_train, job.w_train, **job.kwargs)
    fit_time = time.time() - start_time

    train_scores = job.metalearner.evaluate(
        X=job.X_train,
        y=job.y_train,
        w=job.w_train,
        is_oos=False,
        scoring=job.scoring,
    )
    test_scores = job.metalearner.evaluate(
        X=job.X_test,
        y=job.y_test,
        w=job.w_test,
        is_oos=True,
        oos_method=job.oos_method,
        scoring=job.scoring,
    )
    score_time = time.time() - fit_time
    return _CVResult(
        metalearner=job.metalearner,
        fit_time=fit_time,
        score_time=score_time,
        train_scores=train_scores,
        test_scores=test_scores,
        cv_index=job.cv_index,
    )


def _format_results(results: Sequence[_CVResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row: dict[str, str | int | float] = {}
        row["metalearner"] = result.metalearner.__class__.__name__
        nuisance_models = set(result.metalearner.nuisance_model_specifications().keys())
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
        row["cv_index"] = result.cv_index
        row["fit_time"] = result.fit_time
        row["score_time"] = result.score_time
        for name, value in result.train_scores.items():
            row[f"train_{name}"] = value
        for name, value in result.test_scores.items():
            row[f"test_{name}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


class MetaLearnerGridSearchCV:
    """Exhaustive search over specified parameter values for a MetaLearner.

    ``metalearner_params`` should contain the necessary params for the MetaLearner initialization
    such as ``n_variants`` and ``is_classification``. It can also contain optional parameters
    that all MetaLearners should be initialized with such as ``n_folds`` or ``feature_set``.
    Importantly, ``random_state`` must be passed through the ``random_state`` parameter
    and not through ``metalearner_params``.

    ``base_learner_grid`` keys should be the names of all the base models contained in the :class:`~metalearners.metalearners.MetaLearner`
    defined by ``metalearner_factory``, for information about this names check
    :meth:`~metalearners.metalearner.MetaLearner.nuisance_model_specifications` and
    :meth:`~metalearners.metalearner.MetaLearner.treatment_model_specifications`. The
    values should be sequences of model factories.

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
            "LGBMRegressor": {"n_estimators": [1, 2], "verbose": [-1]},
            "LGBMClassifier": {
                "n_estimators": [1, 2, 3],
                "verbose": [-1],
            },
        }

    If some model is not present in ``param_grid``, the default parameters will be used.

    For how to define ``scoring`` check :meth:`~metalearners.metalearner.MetaLearner.evaluate`.

    ``verbose`` will be passed to `joblib.Parallel <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`_.
    """

    # TODO: Add a reference to a docs example once it is written.

    def __init__(
        self,
        metalearner_factory: type[MetaLearner],
        metalearner_params: Mapping,
        base_learner_grid: Mapping[str, Sequence[type[_ScikitModel]]],
        param_grid: Mapping[str, Mapping[str, Sequence]],
        scoring: Scoring | None = None,
        cv: int = 5,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.metalearner_factory = metalearner_factory
        self.metalearner_params = metalearner_params
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.raw_results_: Sequence[_CVResult] | None = None
        self.cv_results_: pd.DataFrame | None = None

        expected_base_models = set(
            metalearner_factory.nuisance_model_specifications().keys()
        ) | set(metalearner_factory.treatment_model_specifications().keys())

        if set(base_learner_grid.keys()) != expected_base_models:
            raise ValueError

        all_base_learners = set(reduce(add, base_learner_grid.values()))
        param_grid_empty: Mapping[str, Mapping[str, Sequence]] = {
            k.__name__: {} for k in all_base_learners if k.__name__ not in param_grid
        }
        self.base_learner_grid = list(ParameterGrid(base_learner_grid))

        # Mapping does not have union "|" operator, see
        # https://peps.python.org/pep-0584/#what-about-mapping-and-mutablemapping
        full_param_grid = {**param_grid_empty, **param_grid}
        self.base_learner_param_grids = {
            base_learner: list(ParameterGrid(base_learner_param_grid))
            for base_learner, base_learner_param_grid in full_param_grid.items()
        }

    def fit(
        self,
        X: Matrix,
        y: Vector,
        w: Vector,
        oos_method: OosMethod = OVERALL,
        **kwargs,
    ):
        """Run fit with all sets of parameters.

        ``kwargs`` will be passed through to the :meth:`~metalearners.metalearner.MetaLearner.fit`
        call of each individual MetaLearner.
        """
        cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        nuisance_models_no_propensity = set(
            self.metalearner_factory.nuisance_model_specifications().keys()
        ) - {PROPENSITY_MODEL}
        treatment_models = set(
            self.metalearner_factory.treatment_model_specifications().keys()
        )

        all_models = set(
            self.metalearner_factory.nuisance_model_specifications().keys()
        ) | set(self.metalearner_factory.treatment_model_specifications().keys())

        jobs: list[_FitAndScoreJob] = []
        for cv_index, (train_indices, test_indices) in enumerate(cv.split(X)):
            X_train = index_matrix(X, train_indices)
            X_test = index_matrix(X, test_indices)
            y_train = index_vector(y, train_indices)
            y_test = index_vector(y, test_indices)
            w_train = index_vector(w, train_indices)
            w_test = index_vector(w, test_indices)
            for base_learners in self.base_learner_grid:
                nuisance_model_factory = {
                    model_kind: base_learners[model_kind]
                    for model_kind in nuisance_models_no_propensity
                }
                treatment_model_factory = {
                    model_kind: base_learners[model_kind]
                    for model_kind in treatment_models
                }
                propensity_model_factory = base_learners.get(PROPENSITY_MODEL, None)

                param_grid = {
                    model_kind: self.base_learner_param_grids[
                        base_learners[model_kind].__name__
                    ]
                    for model_kind in all_models
                }
                for params in ParameterGrid(param_grid):
                    nuisance_model_params = {
                        model_kind: params[model_kind]
                        for model_kind in nuisance_models_no_propensity
                    }
                    treatment_model_params = {
                        model_kind: params[model_kind]
                        for model_kind in treatment_models
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
                            X_train=X_train,
                            y_train=y_train,
                            w_train=w_train,
                            X_test=X_test,
                            y_test=y_test,
                            w_test=w_test,
                            oos_method=oos_method,
                            scoring=self.scoring,
                            kwargs=kwargs,
                            cv_index=cv_index,
                        )
                    )

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        raw_results = parallel(delayed(_fit_and_score)(job) for job in jobs)
        self.raw_results_ = raw_results
        self.cv_results_ = _format_results(results=raw_results)
