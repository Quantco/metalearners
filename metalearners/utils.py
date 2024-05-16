# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from metalearners.metalearner import MetaLearner
from metalearners.rlearner import RLearner
from metalearners.slearner import SLearner
from metalearners.tlearner import TLearner
from metalearners.xlearner import XLearner


def metalearner_factory(metalearner_prefix: str) -> type[MetaLearner]:
    match metalearner_prefix:
        case "T":
            return TLearner
        case "S":
            return SLearner
        case "X":
            return XLearner
        case "R":
            return RLearner
        case _:
            raise ValueError(
                f"No MetaLearner implementation found for prefix {metalearner_prefix}."
            )
