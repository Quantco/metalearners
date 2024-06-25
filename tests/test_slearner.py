# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
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
@pytest.mark.parametrize("use_pd", [False, True])
def test_append_treatment_to_covariates(
    model,
    supports_categoricals,
    use_pd,
    sample_size,
    request,
):
    dataset_name = "mixed" if use_pd else "numerical"
    covariates, _, _ = request.getfixturevalue(f"{dataset_name}_covariates")
    treatment = np.array([0] * sample_size)
    n_variants = 4
    X_with_w = _append_treatment_to_covariates(
        covariates, treatment, supports_categoricals, n_variants
    )
    assert X_with_w.shape[0] == sample_size

    treatment_pd = pd.Series(treatment, dtype="category").cat.set_categories(
        list(range(n_variants))
    )

    if not use_pd and not supports_categoricals:
        assert isinstance(X_with_w, np.ndarray)
        assert (
            (
                X_with_w[:, -3:]
                == pd.get_dummies(treatment_pd, dtype=int, drop_first=True)
            )
            .all()
            .all()
        )
        assert np.all(X_with_w[:, :-3] == covariates)
    else:
        assert isinstance(X_with_w, pd.DataFrame)
        covariates_pd = pd.DataFrame(covariates) if not use_pd else covariates
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
