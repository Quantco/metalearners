# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

from metalearners.slearner import SLearner, _append_treatment_to_covariates


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
    "model, supports_categoricals", [(LinearRegression, False), (LGBMRegressor, True)]
)
@pytest.mark.parametrize("backend", ["np", "pd", "csr"])
def test_append_treatment_to_covariates(
    model,
    supports_categoricals,
    backend,
    sample_size,
    request,
):
    dataset_name = "mixed" if backend == "pd" else "numerical"
    covariates, _, _ = request.getfixturevalue(f"{dataset_name}_covariates")

    if backend == "csr":
        covariates = csr_matrix(covariates)

    treatment = np.array([0] * sample_size)
    n_variants = 4
    X_with_w = _append_treatment_to_covariates(
        covariates, treatment, supports_categoricals, n_variants
    )
    assert X_with_w.shape[0] == sample_size

    treatment_pd = pd.Series(treatment, dtype="category").cat.set_categories(
        list(range(n_variants))
    )

    if backend in ["np", "csr"] and not supports_categoricals:
        if backend == "np":
            assert isinstance(X_with_w, np.ndarray)
        elif backend == "csr":
            assert isinstance(X_with_w, csr_matrix)
        assert (
            (
                X_with_w[:, -3:]
                == pd.get_dummies(treatment_pd, dtype=int, drop_first=True).values
            )
            .all()
            .all()
        )
        assert (X_with_w[:, :-3] != covariates).sum() < 1
    else:
        assert isinstance(X_with_w, pd.DataFrame)
        if backend == "np":
            covariates_pd = pd.DataFrame(covariates)
        elif backend == "csr":
            covariates_pd = pd.DataFrame.sparse.from_spmatrix(covariates)
        else:
            covariates_pd = covariates
        covariates_pd.columns = covariates_pd.columns.astype(str)
        if not supports_categoricals:
            assert X_with_w[["treatment_1", "treatment_2", "treatment_3"]].equals(
                pd.get_dummies(
                    treatment_pd, dtype=int, drop_first=True, prefix="treatment"
                )
            )

            assert X_with_w.drop(
                ["treatment_1", "treatment_2", "treatment_3"], axis=1
            ).equals(covariates_pd)
        else:
            assert X_with_w["treatment"].dtype == "category"
            assert np.all(X_with_w["treatment"].cat.categories == [0, 1, 2, 3])

            assert X_with_w.drop("treatment", axis=1).equals(covariates_pd)
