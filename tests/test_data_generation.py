# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from metalearners._utils import (
    convert_and_pad_propensity_score,
    get_linear_dimension,
    get_n_variants,
)
from metalearners.data_generation import (
    compute_experiment_outputs,
    generate_categoricals,
    generate_covariates,
    generate_numericals,
    generate_treatment,
    insert_missing,
)
from metalearners.outcome_functions import constant_treatment_effect


@pytest.mark.parametrize("n_obs", [100, 1_000_000])
@pytest.mark.parametrize(
    "n_features, n_categories, n_uniform",
    [
        (10, 5, 5),
        (10, 2, 10),
        (4, [2, 4, 5, 2], 0),
        (5, [2, 4, 2, 4, 6], None),
        (10, None, None),
    ],
)
@pytest.mark.parametrize("use_strings", [False, True])
@pytest.mark.parametrize("p_binomial", [0.1, 0.5, 0.8])
def test_generate_categoricals(
    n_obs, n_features, n_categories, n_uniform, p_binomial, use_strings, rng
):
    categoricals, n_categories_out = generate_categoricals(
        n_obs, n_features, n_categories, n_uniform, p_binomial, use_strings, rng=rng
    )
    assert categoricals.shape == (n_obs, n_features)
    assert (categoricals.astype(np.int64).min(axis=0) >= 0).all()
    if use_strings:
        assert categoricals.dtype.kind == "U"
    else:
        assert categoricals.dtype == np.int64
    if n_categories is not None:
        assert (categoricals.astype(np.int64).max(axis=0) < n_categories).all()
        assert (n_categories_out == n_categories).all()


@pytest.mark.parametrize("n_obs", [100_000, 1_000_000])
@pytest.mark.parametrize(
    "n_features, mu", [(10, None), (5, 2), (4, [1, 2, 3, 4]), (1, 5)]
)
@pytest.mark.parametrize("wishart_scale", [0, 0.5, 1])
def test_generate_numericals(n_obs, n_features, mu, wishart_scale, rng):
    numericals = generate_numericals(n_obs, n_features, mu, wishart_scale, rng=rng)
    assert numericals.shape == (n_obs, n_features)
    if mu is not None:
        np.testing.assert_allclose(numericals.mean(axis=0), mu, atol=1e-1)
    if wishart_scale == 0:
        cov = np.cov(numericals, rowvar=False)
        np.testing.assert_allclose(cov, np.eye(n_features), atol=1e-1)


@pytest.mark.parametrize("n_obs", [1000, 1_000_000])
@pytest.mark.parametrize("n_features", [5, 10, 100])
@pytest.mark.parametrize("n_categoricals", [0, 1, 5])
@pytest.mark.parametrize("format", ["numpy", "pandas"])
@pytest.mark.parametrize("use_strings", [False, True])
def test_generate_covariates(
    n_obs, n_features, n_categoricals, format, use_strings, rng
):
    context = (
        pytest.raises(
            ValueError, match="if format is numpy then use_strings must be False"
        )
        if format == "numpy" and use_strings
        else does_not_raise()
    )
    with context:
        features, categorical_features_idx, n_categories = generate_covariates(
            n_obs=n_obs,
            n_features=n_features,
            n_categoricals=n_categoricals,
            format=format,
            use_strings=use_strings,
            rng=rng,
        )
        assert features.shape == (n_obs, n_features)
        if format == "numpy":
            assert isinstance(features, np.ndarray)
            assert features.dtype == np.float64
            assert np.isnan(features).sum() == 0
        elif format == "pandas":
            assert isinstance(features, pd.DataFrame)
            assert (
                features.drop(categorical_features_idx, axis=1).dtypes == "float64"
            ).all()
            assert (features[categorical_features_idx].dtypes == "category").all()
            for i, c in enumerate(categorical_features_idx):
                categories = set(range(n_categories[i]))
                if use_strings:
                    categories = set(map(str, categories))  # type: ignore
                assert set(features[c].cat.categories) == categories
            assert pd.isna(features).sum().sum() == 0


@pytest.mark.parametrize("missing_probability", [0, 0.1])
@pytest.mark.parametrize("format", ["numpy", "pandas"])
def test_insert_missing(missing_probability, format, sample_size, rng):
    features, _, _ = generate_covariates(
        n_obs=sample_size,
        n_features=10,
        n_categoricals=5,
        format=format,
        rng=rng,
    )

    masked = insert_missing(features, missing_probability=missing_probability, rng=rng)

    assert masked.shape == features.shape
    rel_tol = 5e-2
    if format == "numpy":
        assert np.isnan(masked).mean() == pytest.approx(
            missing_probability, rel=rel_tol
        )
    elif format == "pandas":
        assert masked.isna().mean(None) == pytest.approx(  # type: ignore
            missing_probability, rel=rel_tol
        )


@pytest.mark.parametrize(
    "propensity_scores",
    [
        np.array([0.1, 0.5, 0.8, 0.9]),
        np.array([[0.2, 0.8], [0.1, 0.9], [0.7, 0.3], [0.9, 0.1]]),
        np.array(
            [
                [0.1, 0.5, 0.2, 0.2],
                [0.1, 0.4, 0.2, 0.3],
                [0.2, 0.3, 0.35, 0.15],
                [0.8, 0.05, 0.1, 0.05],
            ]
        ),
    ],
)
def test_generate_treatment_smoke(propensity_scores, rng):
    _ = generate_treatment(propensity_scores, rng)


@pytest.mark.parametrize(
    "propensity_scores",
    [
        np.broadcast_to(0.2, 1_000_000),
        np.broadcast_to(np.array([0.7]), 1_000_000),
        np.broadcast_to(np.array([0.2, 0.4, 0.1, 0.3]), (1_000_000, 4)),
    ],
)
def test_generate_treatment_known_result(propensity_scores, rng):
    treatment = generate_treatment(propensity_scores, rng)
    _, counts = np.unique(treatment, return_counts=True)
    n_variants = get_n_variants(propensity_scores)
    propensity_scores = convert_and_pad_propensity_score(propensity_scores, n_variants)
    np.testing.assert_allclose(
        counts / np.sum(counts), propensity_scores.mean(axis=0), atol=1e-3
    )


@pytest.mark.parametrize("dataset", ["numerical_covariates", "mixed_covariates"])
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("sigma_y", [0.5, 1])
@pytest.mark.parametrize("sigma_tau", [0.5, 1])
@pytest.mark.parametrize("use_pandas", [False, True])
def test_compute_experiment_outputs(
    dataset, n_variants, sigma_y, sigma_tau, use_pandas, request, rng
):
    features, categorical_features_idx, n_categories = request.getfixturevalue(dataset)
    treatment = rng.integers(low=0, high=n_variants, size=features.shape[0])
    if use_pandas:
        treatment = pd.Series(treatment)
    dim = get_linear_dimension(features)
    tau = np.array(range(1, n_variants))
    pof = constant_treatment_effect(dim, tau, rng=rng)
    mu = pof(features)
    y, true_cate = compute_experiment_outputs(
        mu,
        treatment,
        sigma_y=sigma_y,
        sigma_tau=sigma_tau,
        rng=rng,
    )

    assert y.shape == (features.shape[0],)
    assert true_cate.shape == (features.shape[0], n_variants - 1)
    expected_cate = tau.reshape(1, -1).repeat(features.shape[0], axis=0)
    np.testing.assert_allclose(true_cate, expected_cate)

    sigma_y_hat = (y - mu[:, 0])[treatment == 0].std()
    assert sigma_y_hat == pytest.approx(sigma_y, abs=1e-2)

    expected_std = np.sqrt(sigma_y**2 + sigma_tau**2)
    for k in range(1, n_variants):
        actual_std = (y - mu[:, k])[treatment == k].std()
        assert actual_std == pytest.approx(expected_std, abs=1e-2, rel=1e-1)


@pytest.mark.parametrize("dataset", ["numerical_covariates", "mixed_covariates"])
@pytest.mark.parametrize("sigma_y", [0.5, 1])
@pytest.mark.parametrize("sigma_tau", [0.5, 1])
@pytest.mark.parametrize("return_probability_cate", [False, True])
@pytest.mark.parametrize("positive_proportion", [0.1, 0.5, 0.9])
def test_compute_experiment_outputs_classification(
    dataset,
    sigma_y,
    sigma_tau,
    return_probability_cate,
    positive_proportion,
    request,
    rng,
):
    n_variants = 2
    features, categorical_features_idx, n_categories = request.getfixturevalue(dataset)
    treatment = rng.integers(low=0, high=n_variants, size=features.shape[0])
    dim = get_linear_dimension(features)
    tau = np.array(range(1, n_variants))
    pof = constant_treatment_effect(dim, tau, rng=rng)
    mu = pof(features)
    y, true_cate = compute_experiment_outputs(
        mu,
        treatment,
        sigma_y=sigma_y,
        sigma_tau=sigma_tau,
        n_variants=n_variants,
        is_classification=True,
        positive_proportion=positive_proportion,
        return_probability_cate=return_probability_cate,
        rng=rng,
    )

    assert y.shape == (features.shape[0],)
    assert true_cate.shape == (features.shape[0], n_variants - 1)
    assert y.mean() == pytest.approx(positive_proportion, abs=5e-2)

    if return_probability_cate:
        assert -1 < np.min(true_cate) <= np.max(true_cate) < 1
    else:
        assert set(np.unique(true_cate)) <= {-1, 0, 1}
