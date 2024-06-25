# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from metalearners._utils import get_linear_dimension
from metalearners.outcome_functions import (
    constant_treatment_effect,
    linear_treatment_effect,
    no_treatment_effect,
)


@pytest.mark.parametrize("ulow, uhigh", [(0, 1), (-1, 1), (2, 4.52)])
@pytest.mark.parametrize(
    "tau", [1, np.array([0.3, 2.5]), np.array([0, 1.4, 3.2, -0.4])]
)
@pytest.mark.parametrize("dataset", ["numerical_covariates", "mixed_covariates"])
def test_constant_treatment_effect(dataset, tau, rng, ulow, uhigh, request):
    features, _, _ = request.getfixturevalue(dataset)
    dim = get_linear_dimension(features)
    pof = constant_treatment_effect(dim, tau, ulow=ulow, uhigh=uhigh, rng=rng)
    mu_hat = pof(features)
    tau_hat = mu_hat[:, 1:] - mu_hat[:, 0].reshape(-1, 1)
    if isinstance(tau, np.ndarray):
        n_variants = len(tau) + 1
    else:
        n_variants = 2
    assert mu_hat.shape == (features.shape[0], n_variants)
    assert np.mean(tau_hat, axis=0) == pytest.approx(tau)


@pytest.mark.parametrize("ulow, uhigh", [(0, 1), (-1, 1), (2, 4.52)])
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("dataset", ["numerical_covariates", "mixed_covariates"])
def test_no_treatment_effect(dataset, n_variants, rng, ulow, uhigh, request):
    features, _, _ = request.getfixturevalue(dataset)
    dim = get_linear_dimension(features)
    pof = no_treatment_effect(
        dim, n_variants=n_variants, ulow=ulow, uhigh=uhigh, rng=rng
    )
    mu_hat = pof(features)
    tau_hat = mu_hat[:, 1:] - mu_hat[:, 0].reshape(-1, 1)
    assert mu_hat.shape == (features.shape[0], n_variants)
    assert np.mean(tau_hat, axis=0) == pytest.approx(0)


@pytest.mark.parametrize("ulow, uhigh", [(0, 1), (-1, 1), (2, 4.52)])
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("dataset", ["numerical_covariates", "mixed_covariates"])
def test_linear_treatment_effect_smoke(dataset, n_variants, rng, ulow, uhigh, request):
    features, _, _ = request.getfixturevalue(dataset)
    dim = get_linear_dimension(features)
    pof = linear_treatment_effect(
        dim, n_variants=n_variants, ulow=ulow, uhigh=uhigh, rng=rng
    )
    mu_hat = pof(features)
    assert mu_hat.shape == (features.shape[0], n_variants)


@pytest.mark.parametrize("beta_value", [0, 1, 10])
@pytest.mark.parametrize("n_variants", [2, 5])
@pytest.mark.parametrize("dataset", ["numerical_covariates", "mixed_covariates"])
def test_linear_treatment_effect_known_result(
    dataset, n_variants, rng, beta_value, request
):
    features, categorical_features_idx, n_categories = request.getfixturevalue(dataset)
    dim = get_linear_dimension(features)
    pof = linear_treatment_effect(
        dim, n_variants=n_variants, ulow=beta_value, uhigh=beta_value, rng=rng
    )
    mu_hat = pof(features)
    assert mu_hat.shape == (features.shape[0], n_variants)
    tau_hat = mu_hat[:, 1:] - mu_hat[:, 0].reshape(-1, 1)
    if isinstance(features, np.ndarray):
        numerical_features = features
    else:
        numerical_features = features.drop(categorical_features_idx, axis=1).to_numpy()
    expected = (
        ((numerical_features.sum(axis=1) + len(categorical_features_idx)) * beta_value)
        .reshape(-1, 1)
        .repeat(n_variants - 1, axis=1)
    )
    np.testing.assert_allclose(tau_hat, expected)
