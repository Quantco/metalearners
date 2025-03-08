# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

from metalearners.slearner import (
    _TREATMENT,
    SLearner,
    _append_treatment_to_covariates_with_categorical,
    _append_treatment_to_covariates_with_one_hot_encoding,
    _np_to_dummies,
)


def test_feature_set_doesnt_raise(rng):
    slearner = SLearner(
        nuisance_model_factory=LinearRegression,
        is_classification=False,
        n_variants=2,
        feature_set=[0],
    )

    X = rng.standard_normal((100, 2))
    y = rng.standard_normal(100)
    w = rng.integers(0, 2, 100)
    slearner.fit(X, y, w)
    assert (
        slearner._nuisance_models["base_model"][0]._overall_estimator.n_features_in_  # type: ignore
        == 3
    )


@pytest.mark.parametrize(
    "x,categories",
    [
        (np.array([[1, 1, 1], [2, 2, 2]]), [1, 2]),
        (np.array([1, 2, 3]), [1]),
        (np.array([1, 2, 3]), [1, 2]),
    ],
)
def test_np_to_dummies_raises(x, categories):
    with pytest.raises(ValueError):
        _np_to_dummies(x, categories)


@pytest.mark.parametrize(
    "data,categories,expected",
    [
        (
            np.array([0, 1, 0, 1]),
            [0, 1],
            np.array(
                [
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                ]
            ),
        ),
        (
            np.array([0, 1, 0, 1]),
            [0, 1, 2],
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("drop_first", [True, False])
def test_np_to_dummies(data, categories, expected, drop_first):
    actual = _np_to_dummies(
        x=data,
        categories=categories,
        drop_first=drop_first,
    )
    expected = expected[:, drop_first:]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", ["np", "pd", "csr", "pl"])
def test_append_treatment_to_covariates_with_categorical(backend, request, sample_size):
    dataset_name = "mixed" if backend in {"pd", "pl"} else "numerical"
    covariates, _, _ = request.getfixturevalue(f"{dataset_name}_covariates")

    if backend == "pl":
        covariates = pl.from_pandas(covariates)

    if backend == "csr":
        covariates = csr_matrix(covariates)

    treatment = np.array([0] * sample_size)
    n_variants = 4
    categories = list(range(n_variants))

    X_with_w = _append_treatment_to_covariates_with_categorical(
        X=covariates,
        w=treatment,
        categories=categories,
    )

    assert X_with_w.shape[0] == sample_size

    if backend in ["np", "csr", "pd"]:
        assert isinstance(X_with_w, pd.DataFrame)
        assert X_with_w[_TREATMENT].dtype == "category"
        assert np.all(X_with_w[_TREATMENT].cat.categories == categories)

        if backend == "pd":
            covariates.columns = covariates.columns.astype(str)
            pd.testing.assert_frame_equal(
                X_with_w.drop(_TREATMENT, axis=1),
                covariates,
                check_dtype=False,
            )
        else:
            if backend == "csr":
                expected_df = pd.DataFrame.sparse.from_spmatrix(covariates)
            else:
                expected_df = pd.DataFrame(covariates)
            expected_df.columns = expected_df.columns.astype(str)
            pd.testing.assert_frame_equal(
                X_with_w.drop(_TREATMENT, axis=1),
                expected_df,
                check_dtype=False,
            )
    else:
        assert isinstance(X_with_w, pl.DataFrame)
        assert isinstance(X_with_w[_TREATMENT].dtype, pl.Categorical)
        assert set(X_with_w[_TREATMENT].cat.get_categories()) == {"0"}
        pl_assert_frame_equal(
            X_with_w.drop(_TREATMENT),
            covariates,
            check_dtypes=True,
        )


@pytest.mark.parametrize("backend", ["np", "pd", "csr", "pl"])
def test_append_treatment_to_covariates_with_one_hot_encoding(
    backend, request, sample_size
):
    dataset_name = "mixed" if backend in {"pd", "pl"} else "numerical"
    covariates, _, _ = request.getfixturevalue(f"{dataset_name}_covariates")

    if backend == "pl":
        covariates = pl.from_pandas(covariates)

    if backend == "csr":
        covariates = csr_matrix(covariates)

    treatment = np.array([0] * sample_size)
    n_variants = 4
    categories = list(range(n_variants))

    X_with_w = _append_treatment_to_covariates_with_one_hot_encoding(
        X=covariates,
        w=treatment,
        categories=categories,
    )

    assert X_with_w.shape[0] == sample_size
    assert isinstance(X_with_w, type(covariates))

    treatment_pd = pd.Series(treatment, dtype="category").cat.set_categories(categories)

    if backend == "np":
        expected_one_hot_encoding = pd.get_dummies(
            treatment_pd, dtype=int, drop_first=True
        ).values
        actual_one_hot_encoding = X_with_w[:, -(n_variants - 1) :]
        np.array_equal(expected_one_hot_encoding, actual_one_hot_encoding)

        assert np.array_equal(expected_one_hot_encoding, actual_one_hot_encoding)

        covariates_after = X_with_w[:, : -(n_variants - 1)]
        assert np.array_equal(covariates, covariates_after)

    elif backend == "csr":
        expected_one_hot_encoding = csr_matrix(
            pd.get_dummies(treatment_pd, dtype="int8", drop_first=True).values
        )
        actual_one_hot_encoding = X_with_w[:, -(n_variants - 1) :]
        assert (expected_one_hot_encoding != actual_one_hot_encoding).nnz == 0

        covariates_after = X_with_w[:, : -(n_variants - 1)]
        assert (covariates != covariates_after).nnz == 0

    else:
        treatment_columns = [f"{_TREATMENT}_{i}" for i in range(1, n_variants)]

        actual_one_hot_encoding = X_with_w[treatment_columns]

        if isinstance(X_with_w, pd.DataFrame):
            covariates.columns = covariates.columns.astype(str)
            expected_one_hot_encoding = pd.get_dummies(
                treatment_pd, dtype="int8", drop_first=True, prefix=_TREATMENT
            )
            pd.testing.assert_frame_equal(
                actual_one_hot_encoding, expected_one_hot_encoding
            )

            pd.testing.assert_frame_equal(
                X_with_w.drop(treatment_columns, axis=1),
                covariates,
            )

        elif isinstance(X_with_w, pl.DataFrame):

            expected_one_hot_encoding = pl.DataFrame(
                pd.get_dummies(
                    treatment_pd, dtype="int8", drop_first=True, prefix=_TREATMENT
                )
            )
            pl_assert_frame_equal(actual_one_hot_encoding, expected_one_hot_encoding)

            pl_assert_frame_equal(
                X_with_w.drop(treatment_columns),
                covariates,
                check_dtypes=True,
            )
