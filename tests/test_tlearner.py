# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import pytest
from lightgbm import LGBMClassifier, LGBMRegressor

from metalearners.tlearner import TLearner


def test_validate_models():
    with pytest.raises(
        ValueError,
        match="is_classification is set to True but the treatment_model is not a classifier.",
    ):
        TLearner(LGBMRegressor, True, 2)
    with pytest.raises(
        ValueError,
        match="is_classification is set to False but the treatment_model is not a regressor.",
    ):
        TLearner(LGBMClassifier, False, 2)
