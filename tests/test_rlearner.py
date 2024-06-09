# # Copyright (c) QuantCo 2024-2024
# # SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import pytest

from metalearners.rlearner import r_loss


@pytest.mark.parametrize("use_pandas", [True, False])
def test_r_loss(use_pandas):
    factory = pd.Series if use_pandas else np.array
    cate_estimates = factory([2, 2])
    outcomes = factory([6.1, 6.1])
    outcome_estimates = factory([3.1, 3.1])
    treatments = factory([1, 1])
    propensity_scores = factory([0.5, 0.5])
    # (6.1 - 3.1) - 2(1 -.5)
    # = 3 - 1 = 2
    result = r_loss(
        cate_estimates=cate_estimates,
        outcomes=outcomes,
        outcome_estimates=outcome_estimates,
        treatments=treatments,
        propensity_scores=propensity_scores,
    )
    assert result == pytest.approx(2, abs=1e-4, rel=1e-4)
