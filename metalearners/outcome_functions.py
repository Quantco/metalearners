# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable

import numpy as np
import pandas as pd

from metalearners._typing import Matrix
from metalearners._utils import default_rng


def _beta(
    ulow: float, uhigh: float, dim: tuple | int, rng: np.random.Generator
) -> np.ndarray:
    return rng.uniform(low=ulow, high=uhigh, size=dim)


def constant_treatment_effect(
    dim: int,
    tau: float | np.ndarray,
    ulow: float = 0,
    uhigh: float = 1,
    rng: np.random.Generator | None = None,
) -> Callable:
    r"""Generate a potential outcomes function with constant treatment effect.

    .. math::
        f(x_i, w_i) = x_i' \beta_{control} + \sum_{k=1}^{n_v-1} \tau_k \cdot \mathcal{I}(\{w_i = k\})

    where :math:`x_i` is a vector of features, :math:`\tau` a vector of treatment effects,
    :math:`w_i` the treatment indicator, :math:`n_v` the number of variants and

    .. math::
        \beta_{control} \sim \mathcal{U}[u_l, u_h]

    ``dim`` indicates the dimension of :math:`\beta` and therefore it should be the
    number of numerical features plus the number of categories in all of the categorical
    features.

    ``tau`` expects to be of size :math:`n_v-1`.
    """
    if rng is None:
        rng = default_rng

    beta = _beta(ulow, uhigh, dim, rng)
    if isinstance(tau, int | float):
        tau = np.array([tau])
    tau = tau.reshape(1, -1)

    def f(X: Matrix) -> np.ndarray:
        ohe_encoded_features = (
            pd.get_dummies(X, dtype="float64").to_numpy()
            if isinstance(X, pd.DataFrame)
            else X
        )

        mu_0 = np.dot(ohe_encoded_features, beta)
        return np.c_[mu_0, mu_0.reshape(-1, 1) + tau]

    return f


def no_treatment_effect(
    dim: int,
    n_variants: int = 2,
    ulow: float = 0,
    uhigh: float = 1,
    rng: np.random.Generator | None = None,
) -> Callable:
    r"""Generate a potential outcomes function with no treatment effect.

    .. math::
        f(x_i, w_i) = x_i' \beta

    where :math:`x_i` is a vector of features and

    .. math::
        \beta \sim \mathcal{U}[u_l, u_h]

    ``dim`` indicates the dimension of :math:`\beta` and therefore the number of
    numerical features plus the number of categories in all of the categorical features.
    """
    if n_variants < 2:
        raise ValueError("n_variants needs to be an integer greater or equal to 2")

    tau = np.broadcast_to(0, n_variants - 1)

    return constant_treatment_effect(dim, tau=tau, ulow=ulow, uhigh=uhigh, rng=rng)


def linear_treatment_effect(
    dim: int,
    n_variants: int = 2,
    ulow: float = 0,
    uhigh: float = 1,
    rng: np.random.Generator | None = None,
) -> Callable:
    r"""Generate a potential outcomes function with linear treatment effect.

    .. math::
        f(x_i, w_i) = x_i' \beta_{control} + \sum_{k=1}^{n_v-1} \mathcal{I}(\{w_i = k\}) \cdot x_i' \beta^{(k)}

    where :math:`x_i` is a vector of features, :math:`w_i` the treatment indicator, and

    .. math::
        \beta_{control} \sim \mathcal{U}[u_l, u_h]
        \beta_{(k)} \sim \mathcal{U}[u_l, u_h]

    ``dim`` indicates the dimension of :math:`\beta_{control}` and :math:`\beta_{(k)}`
    therefore it should be the number of numerical features plus the number of categories
    in all of the categorical features.
    """
    if n_variants < 2:
        raise ValueError("n_variants needs to be an integer greater or equal to 2")

    if rng is None:
        rng = default_rng

    beta_control = _beta(ulow, uhigh, dim, rng)
    beta = _beta(ulow, uhigh, (dim, n_variants - 1), rng)

    def f(X: Matrix) -> np.ndarray:
        ohe_encoded_features = (
            pd.get_dummies(X, dtype="float64").to_numpy()
            if isinstance(X, pd.DataFrame)
            else X
        )

        mu_0 = np.dot(ohe_encoded_features, beta_control)
        return np.c_[mu_0, mu_0.reshape(-1, 1) + np.matmul(ohe_encoded_features, beta)]

    return f
