# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression

from metalearners.slearner import SLearner, _append_treatment_to_covariates


def test_feature_set_raise():
    with pytest.raises(
        ValueError, match="SLearner does not support feature set definition."
    ):
        SLearner(LinearRegression, False, 2, feature_set="")


def test_validate_models():
    with pytest.raises(
        ValueError,
        match="is_classification is set to True but the base_model is not a classifier.",
    ):
        SLearner(LGBMRegressor, True, 2)
    with pytest.raises(
        ValueError,
        match="is_classification is set to False but the base_model is not a regressor.",
    ):
        SLearner(LGBMClassifier, False, 2)


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
