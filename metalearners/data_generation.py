# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import wishart

from metalearners._typing import Matrix, Vector
from metalearners._utils import (
    check_probability,
    check_propensity_score,
    convert_and_pad_propensity_score,
    default_rng,
    get_n_variants,
    sigmoid,
)

_FORMATS = ["numpy", "pandas"]


def generate_categoricals(
    n_obs: int,
    n_features: int,
    n_categories: int | np.ndarray | None = None,
    n_uniform: int | None = None,
    p_binomial: float = 0.5,
    use_strings: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate a dataset of categorical features.

    Generates a dataset of ``n_obs`` observations and ``n_features`` categorical
    features. The first ``n_uniform`` features are sampled uniformly across their
    categories and the rest are sampled from a binomial distribution with parameters
    :math:`n = c_i` and :math:`p=` ``p_binomial`` where :math:`c_i` is the number of
    categories of feature :math:`i`.

    ``n_categories`` is the number of categories of the features, it can either be an int
    which is used for all the features or an array of length ``n_features``. If None,
    the number of categories for each feature is sampled from
    :math:`c_i \sim \mathcal{U}\{2,3,\dots,10\}`.

    In case ``n_uniform`` is None, all features are sampled uniformly.

    ``use_strings`` can be set to ``True`` if the wanted represantion of the variables
    are strings. If set to ``False`` it will return an array with dtype ``np.int64``.

    The function returns a ``np.ndarray`` with the sampled dataset and a ``np.ndarray``
    with the number of categories for each feature.
    """
    if rng is None:
        rng = default_rng

    check_probability(p_binomial)

    if n_categories is None:
        n_categories = rng.integers(low=2, high=10, size=n_features, endpoint=True)
    n_categories = np.broadcast_to(n_categories, n_features)

    if n_uniform is None:
        n_uniform = n_features

    dtype = str if use_strings else np.int64
    balanced_features = np.array([], dtype=dtype).reshape(n_obs, 0)  # type: ignore[var-annotated]
    unbalanced_features = np.array([], dtype=dtype).reshape(n_obs, 0)  # type: ignore[var-annotated]

    if n_uniform > 0:
        balanced_features = rng.integers(
            low=0, high=n_categories[:n_uniform], size=(n_obs, n_uniform)
        ).astype(dtype)

    if (n_non_uniform := n_features - n_uniform) > 0:
        unbalanced_features = rng.binomial(
            n_categories[n_uniform:] - 1,
            p_binomial,
            size=(n_obs, n_non_uniform),
        ).astype(dtype)
    return (
        np.concatenate([balanced_features, unbalanced_features], axis=1),
        n_categories,
    )


def generate_numericals(
    n_obs: int,
    n_features: int,
    mu: float | np.ndarray | None = None,
    wishart_scale: float = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Generate a dataset of numerical features.

    Generates a dataset of ``n_obs`` observations and ``n_features`` numerical features.
    These are sampled from :math:`\mathcal{N}(\mu, \Sigma)` where
    :math:`\mu \sim \mathcal{U}[-5,5]` unless specified in ``mu`` and
    :math:`\Sigma \sim \mathcal{W}(d, \sigma_w I_d)` where :math:`W` is the Wishart
    distribution and :math:`d` the number of features.


    ``mu`` can be either a float or an array of length ``n_features``.

    ``wishart_scale`` should be :math:`\geq 0` , in case it is 0 then :math:`\Sigma = I_d`.
    """
    if rng is None:
        rng = default_rng
    if mu is None:
        mu = rng.uniform(-5, 5, size=n_features)
    mu = np.broadcast_to(mu, n_features)
    if wishart_scale < 0:
        raise ValueError("wishart_scale needs to be >= 0")
    if wishart_scale > 0:
        cov_matrix = wishart.rvs(
            df=n_features,
            scale=wishart_scale * np.eye(n_features),
            random_state=rng,
        ).reshape(n_features, n_features)
    else:
        cov_matrix = np.eye(n_features)
    features = rng.multivariate_normal(mean=mu, cov=cov_matrix, size=n_obs)

    return features


# TODO: Add support for polars?
def generate_covariates(
    n_obs: int,
    n_features: int,
    n_categoricals: int = 0,
    format: Literal["pandas", "numpy"] = "pandas",
    mu: float | np.ndarray | None = None,
    wishart_scale: float = 1,
    n_categories: int | np.ndarray | None = None,
    n_uniform: int | None = None,
    p_binomial: float = 0.5,
    use_strings: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[Matrix, list[int], np.ndarray]:
    r"""Generates a dataset of covariates with both numerical and categorical features.

    Dataset is composed of ``n_obs`` observations and ``n_features`` features, with the
    first ``n_features - n_categoricals`` being numerical and the rest being categorical.
    Numerical features are generated using the function
    :func:`metalearners.data_generation.generate_numericals` and categorical features are
    generated using the function :func:`metalearners.data_generation.generate_categoricals`.

    By default, the generated dataset is returned as a Pandas DataFrame where categorical
    features are converted to ``pandas``\' `Categorical <https://pandas.pydata.org/docs/reference/api/pandas.Categorical.html#pandas.Categorical>`_
    type. Optionally, the dataset can be returned as a numpy array with dtype ``float64``
    with ``format = "numpy"``. If generating categorical variables, working with pandas
    DataFrames is preferred as they have support for category dtype.

    For ``mu`` and ``wishart_scale`` see the docstring for
    :func:`metalearners.data_generation.generate_numericals`.

    For ``n_categories``, ``n_uniform``, ``p_binomial`` and  ``use_strings``
    see the docstring for :func:`metalearners.data_generation.generate_categoricals`.

    ``use_strings`` can only be set to ``True`` when using ``format = "pandas"``.

    The function returns a tuple of three elements. The first element is the dataset
    generated (either a numpy array or a pandas DataFrame depending on ``format``). The
    second element is a list of indices indicating the columns of categorical features
    in the dataset. The third element is a ``np.ndarray`` with the number of categories
    for each feature.
    """
    if rng is None:
        rng = default_rng
    if format not in _FORMATS:
        raise ValueError(f"format needs to be one of {_FORMATS}")

    if format == "numpy" and use_strings:
        raise ValueError("if format is numpy then use_strings must be False")

    numerical_features = np.array([]).reshape(n_obs, 0)
    categorical_features = np.array([]).reshape(n_obs, 0)

    if n_features - n_categoricals > 0:
        numerical_features = generate_numericals(
            n_obs=n_obs,
            n_features=n_features - n_categoricals,
            mu=mu,
            wishart_scale=wishart_scale,
            rng=rng,
        )
    if n_categoricals > 0:
        categorical_features, n_categories = generate_categoricals(
            n_obs=n_obs,
            n_features=n_categoricals,
            n_categories=n_categories,
            n_uniform=n_uniform,
            p_binomial=p_binomial,
            use_strings=use_strings,
            rng=rng,
        )
    else:
        n_categories = np.array([])

    categorical_features_idx = list(range(n_features - n_categoricals, n_features))

    if format == "numpy":
        features = np.concatenate([numerical_features, categorical_features], axis=1)
    elif format == "pandas":
        numerical_features = pd.DataFrame(numerical_features)
        categorical_features = pd.DataFrame(categorical_features)
        features = pd.concat(
            [numerical_features, categorical_features], axis=1, ignore_index=True
        )
        features[categorical_features_idx] = features[categorical_features_idx].astype(
            "category"
        )
        for i, c in enumerate(categorical_features_idx):
            categories = list(range(n_categories[i]))
            if use_strings:
                categories = list(map(str, categories))  # type: ignore
            # We need to set the categories manually as there may be some unsampled categories,
            # and it may be possible that the user relies on having all of them when using OHE
            # for the potential outcomes function.
            features[c] = features[c].cat.set_categories(categories)
    return features, categorical_features_idx, n_categories


def insert_missing(
    X: Matrix,
    missing_probability: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Matrix:
    """Inserts missing values into the dataset.

    Each element of the dataset has a ``missing_probability`` chance of being replaced
    with a NaN, thus simulating a dataset with missing values.

    The function returns a copy of the original dataset, but with some elements replaced
    by NaNs.
    """
    if rng is None:
        rng = default_rng
    check_probability(missing_probability, zero_included=True)
    missing_mask = rng.binomial(1, p=missing_probability, size=X.shape).astype("bool")

    masked = X.copy()
    masked[missing_mask] = np.nan
    return masked


def generate_treatment(
    propensity_scores: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generates a treatment assignment based on the provided propensity scores.

    The function first determines the number of treatment variants based on the shape of
    the input propensity scores. If the propensity score array has a single dimension or
    only one column in the second dimension, there are two treatment variants (treated
    vs not-treated), and the value is interpreted as the treatment probability.
    Otherwise, the second dimension of the propensity scores array indicates the number
    of treatment variants.

    Each observation is assigned to a treatment group by drawing from a categorical
    distribution where the probability of each treatment group is given by the
    propensity scores.

    ``propensity_scores`` should be of size ``(n_obs,)`` or ``(n_obs, n_variants)``,
    where ``n_obs`` is the number of observations and ``n_variants`` is the number of
    treatment variants.

    The function return an array of shape ``(n_obs,)`` where each element indicates the
    treatment variant received.
    """
    if rng is None:
        rng = default_rng
    n_variants = get_n_variants(propensity_scores)
    propensity_scores = convert_and_pad_propensity_score(propensity_scores, n_variants)
    check_propensity_score(propensity_scores, n_variants=n_variants, sum_to_one=True)

    treatment = rng.multinomial(1, propensity_scores).argmax(axis=1)
    return treatment


def compute_experiment_outputs(
    mu: np.ndarray,
    treatment: Vector,
    sigma_y: float = 1,
    sigma_tau: float = 0.5,
    n_variants: int | None = None,
    is_classification: bool = False,
    positive_proportion: float = 0.5,
    return_probability_cate: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the experiment's observed outcomes y and the true CATE.

    This function generates experiment outputs and the true CATE values based on the
    given potential outcomes function and treatments. The treatment effect for each
    observation is computed as the difference in potential outcomes. Normally
    distributed noise is added to the response variable :math:`Y_i(0)` with standard
    deviation ``sigma_y`` and to each corresponding treatment effect to simulate
    real-world variance with standard deviation ``sigma_tau``.

    ``treatment`` must be a vector representing the treatment group assignment for each
    observation. Each element of the vector is an integer representing a treatment variant
    starting at 0.

    ``mu`` must be a matrix of size ``(n_obs, n_variants)`` containing the potential
    outcomes for each observation and treatment variant without added noise.

    ``n_variants`` can be passed to specify the number of treatment variants. If None,
    it is inferred from the maximum value in the 'treatment' vector plus one.

    ``is_classification`` determines if the problem to be simulated is a classification problem.
    If True, the function simulates a classification problem where the response variable is binary
    and the proportion of positive outputs is controlled by the ``positive_proportion`` parameter.
    It is important to notice that the potential outputs are passed through a sigmoid function and
    therefore the domain of them can be :math:`\mathbb{R}`. Classification problems are
    only implemented for binary treatments.

    In the case of a classification problem ``return_probability_cate`` specifies if the
    outputted CATE is the difference in probabilities between treating and not treating or
    if it samples from a Bernoulli distribution and the difference in samples is returned.

    The function returns a tuple containing the following elements:

    * ``y``: numpy array of the experiment's observed outcomes (response variable) after noise
      addition.

    * ``true_cate``: numpy array of the true CATE without any added noise.
    """
    if rng is None:
        rng = default_rng

    if isinstance(treatment, pd.Series):
        treatment = treatment.to_numpy()

    if n_variants is None:
        n_variants = np.max(treatment) + 1

    n_obs = mu.shape[0]

    if mu.shape[1] != n_variants:
        raise ValueError(
            f"mu should be a matrix where the second dimension has size n_variants. "
            f"n_variants is {n_variants} and mu is a matrix of shape {mu.shape}"
        )

    true_cate = mu[:, 1:] - mu[:, 0].reshape(-1, 1)
    true_y = mu[np.arange(n_obs), treatment]

    y_noise = rng.normal(loc=0, scale=sigma_y, size=n_obs)
    tau_noise = np.c_[
        np.zeros(n_obs),
        rng.normal(loc=0, scale=sigma_tau, size=(n_obs, n_variants - 1)),
    ]

    y = true_y + y_noise + tau_noise[np.arange(n_obs), treatment]

    if is_classification:
        if n_variants > 2:
            raise ValueError(
                "Generating classification problems is only implemented for binary treatments"
            )
        normalizer = np.quantile(true_y, 1 - positive_proportion)

        if return_probability_cate:
            true_cate = sigmoid(mu[:, 1] - normalizer) - sigmoid(mu[:, 0] - normalizer)
        else:
            true_cate = rng.binomial(
                n=1, p=sigmoid(mu[:, 1] - normalizer)
            ) - rng.binomial(n=1, p=sigmoid(mu[:, 0] - normalizer))
        true_cate = true_cate.reshape(-1, 1)
        y = rng.binomial(n=1, p=sigmoid(y - normalizer))

    return y, true_cate
