# Copyright (c) QuantCo 2024-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from types import ModuleType

import narwhals.stable.v1 as nw
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
    if len(categories) < 2:
        raise ValueError(
            "categories to be used for nw_to_dummies must have at least two "
            "distinct values."
        )

    if set(categories) < set(x.unique()):
        raise ValueError("We observed a value which isn't part of the categories.")

    relevant_categories = categories[1:] if drop_first else categories
    return x.to_frame().select(
        [
            (nw.col(column_name) == cat).cast(nw.Int8).name.suffix(f"_{cat}")
            for cat in relevant_categories
        ]
    )


def vector_to_nw(x: Vector, native_namespace: ModuleType | None = None) -> nw.Series:
    if isinstance(x, np.ndarray):
        if native_namespace is None:
            raise ValueError(
                "x is a numpy object but no native_namespace was provided to "
                "load it into narwhals."
            )
        return nw.new_series(name="column_0", values=x, native_namespace=native_namespace)
    if is_into_series(x):
        return nw.from_native(x, series_only=True, eager_only=True)
    raise TypeError(f"Unexpected type {type(x)} for Vector.")


def infer_native_namespace(df_nw: nw.DataFrame) -> ModuleType:
    return df_nw.implementation.to_native_namespace()


def stringify_column_names(df_nw: nw.DataFrame) -> nw.DataFrame:
    return df_nw.rename({column: str(column) for column in df_nw.columns})
