# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from metalearners.metalearner import MetaLearner
from metalearners.tlearner import TLearner


def metalearner_factory(metalearner_prefix: str) -> type[MetaLearner]:
    match metalearner_prefix:
        case "T":
            return TLearner
        case _:
            raise ValueError(
                f"No MetaLearner implementation found for prefix {metalearner_prefix}."
            )
