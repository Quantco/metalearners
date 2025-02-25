# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from types import ModuleType

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
from narwhals.dependencies import is_into_series

from metalearners._typing import Vector


def nw_to_dummies(
    x: nw.Series, categories: Sequence, column_name: str, drop_first: bool = True
) -> nw.DataFrame:
    """Turn a vector into a matrix with dummies.

    This operation is also referred to as one-hot-encoding.

    ``x`` is expected to have values which can be cast to integer.
    """
    relevant_categories = categories[1:] if drop_first else categories
    return x.to_frame().select(
        [
            (nw.col(column_name) == cat).cast(nw.Int8).name.suffix(f"_{cat}")
            for cat in relevant_categories
        ]
    )


def vector_to_nw(x: Vector, native_namespace=None) -> nw.Series:
    if isinstance(x, np.ndarray):
        if native_namespace is None:
            raise ValueError(
                "x is a numpy object but no native_namespace was provided to "
                "load it into narwhals."
            )
        # narwhals doesn't seem to like 1d numpy arrays. Therefore we first convert to
        # a 2d np array and then convert the narwhals DataFrame to a narwhals Series.
        return nw.from_numpy(x.reshape(-1, 1), native_namespace=native_namespace)[
            "column_0"
        ]
    if is_into_series(x):
        return nw.from_native(x, series_only=True, eager_only=True)
    raise TypeError(f"Unexpected type {type(x)} for Vector.")


def infer_native_namespace(X_nw: nw.DataFrame) -> ModuleType:
    if X_nw.implementation.name == "PANDAS":
        return pd
    if X_nw.implementation.name == "POLARS":
        return pl
    raise TypeError("Couldn't infer native_namespace of matrix.")
