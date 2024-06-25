# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import importlib.metadata
import warnings

from .drlearner import DRLearner
from .rlearner import RLearner
from .slearner import SLearner
from .tlearner import TLearner
from .xlearner import XLearner

__all__ = ["DRLearner", "RLearner", "SLearner", "TLearner", "XLearner"]


try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError as e:  # pragma: no cover
    warnings.warn(f"Could not determine version of {__name__}\n{e!s}", stacklevel=2)
    __version__ = "unknown"
