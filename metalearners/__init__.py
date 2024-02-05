# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except Exception:
    __version__ = "unknown"
