# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from metalearners._narwhals_utils import (
    infer_native_namespace,
    nw_to_dummies,
    vector_to_nw,
)


@pytest.mark.parametrize("backend", [pl, pd])
def test_infer_native_namespace(backend):
    raw_data = {"a": [1, 2, 3], "b": ["a", "b", "c"]}
    df = backend.DataFrame(raw_data)
    df_nw = nw.from_native(df, eager_only=True)
    assert infer_native_namespace(df_nw) == backend


@pytest.mark.parametrize("native_namespace", [pl, pd])
@pytest.mark.parametrize(
    "raw_data",
    [
        [1, 2, 3],
        [1.1, 2.2, 3.3],
        [True, False, False],
        list("meta"),
    ],
)
def test_vector_to_nw_np(native_namespace, raw_data):
    original_vector = np.array(raw_data)
    vector_nw = vector_to_nw(original_vector, native_namespace=native_namespace)
    assert isinstance(vector_nw, nw.Series)

    new_vector = vector_nw.to_native()
    assert isinstance(new_vector, native_namespace.Series)


@pytest.mark.parametrize("backend", [pd, pl])
@pytest.mark.parametrize(
    "raw_data",
    [
        [1, 2, 3],
        [1.1, 2.2, 3.3],
        [True, False, False],
        list("meta"),
    ],
)
def test_vector_to_nw(backend, raw_data):
    original_vector = backend.Series(raw_data)
    vector_nw = vector_to_nw(original_vector)
    assert isinstance(vector_nw, nw.Series)

    new_vector = vector_nw.to_native()
    assert all(original_vector == new_vector)


_TEST_COLUMN = "test_column"


@pytest.mark.parametrize(
    "data,categories,expected",
    [
        (
            [0, 1, 2],
            [0, 1, 2],
            pd.DataFrame(
                {
                    f"{_TEST_COLUMN}_0": [1, 0, 0],
                    f"{_TEST_COLUMN}_1": [0, 1, 0],
                    f"{_TEST_COLUMN}_2": [0, 0, 1],
                },
                dtype="int8",
            ),
        ),
        (
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            pd.DataFrame(
                {
                    f"{_TEST_COLUMN}_0.0": [1, 0, 0],
                    f"{_TEST_COLUMN}_1.0": [0, 1, 0],
                    f"{_TEST_COLUMN}_2.0": [0, 0, 1],
                },
                dtype="int8",
            ),
        ),
        (
            [0, 0, 0],
            [0, 1, 2],
            pd.DataFrame(
                {
                    f"{_TEST_COLUMN}_0": [1, 1, 1],
                    f"{_TEST_COLUMN}_1": [0, 0, 0],
                    f"{_TEST_COLUMN}_2": [0, 0, 0],
                },
                dtype="int8",
            ),
        ),
    ],
)
@pytest.mark.parametrize("drop_first", [True, False])
def test_nw_to_dummies(data, categories, expected, drop_first):
    column_name = _TEST_COLUMN
    series_nw = nw.from_native(pd.Series(data, name=column_name), series_only=True)
    dummies_nw = nw_to_dummies(
        x=series_nw,
        categories=categories,
        column_name=column_name,
        drop_first=drop_first,
    )
    assert isinstance(dummies_nw, nw.DataFrame)

    expected = expected.iloc[:, int(drop_first) :]

    # We couldn't find a good way to compare narwhals objects for equality.
    # Therefore, we convert back to the backend.
    dummies = dummies_nw.to_native()
    pd.testing.assert_frame_equal(expected, dummies)
