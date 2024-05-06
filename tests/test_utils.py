# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import pytest
from lightgbm import LGBMRegressor

from metalearners.metalearner import MetaLearner
from metalearners.utils import metalearner_factory


@pytest.mark.parametrize("prefix", ["T"])
def test_metalearner_factory_smoke(prefix):
    factory = metalearner_factory(prefix)
    model = factory(nuisance_model_factory=LGBMRegressor, is_classification=False)
    assert isinstance(model, MetaLearner)


@pytest.mark.parametrize("prefix", ["", "H", None])
def test_metalearner_factory_raises(prefix):
    with pytest.raises(ValueError, match="No MetaLearner implementation found"):
        metalearner_factory(prefix)
