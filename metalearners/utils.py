# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from metalearners.drlearner import DRLearner
from metalearners.metalearner import MetaLearner
from metalearners.rlearner import RLearner
from metalearners.slearner import SLearner
from metalearners.tlearner import TLearner
from metalearners.xlearner import XLearner


def metalearner_factory(metalearner_prefix: str) -> type[MetaLearner]:
    """Returns the MetaLearner class corresponding to the given prefix.

    The accepted ``metalearner_prefix`` values are:

    * ``"S"`` for :class:`~metalearners.slearner.SLearner`
    * ``"T"`` for :class:`~metalearners.tlearner.TLearner`
    * ``"X"`` for :class:`~metalearners.xlearner.XLearner`
    * ``"R"`` for :class:`~metalearners.rlearner.RLearner`
    * ``"DR"`` for :class:`~metalearners.drlearner.DRLearner`
    """
    match metalearner_prefix:
        case "T":
            return TLearner
        case "S":
            return SLearner
        case "X":
            return XLearner
        case "R":
            return RLearner
        case "DR":
            return DRLearner
        case _:
            raise ValueError(
                f"No MetaLearner implementation found for prefix {metalearner_prefix}."
            )


def simplify_output(tensor: np.ndarray) -> np.ndarray:
    """Reduces dimensions of a CATE estimation tensor if possible.

    The returned results will be of shape

    * :math:`(n_{obs})` if there are 2 tratment variants and and the outcome is either
      a regression outcome or a binary classification outcome.

    * :math:`(n_{obs}, n_{classes})` if there are 2 treatment variants and and the outcome
      is a classification outcome with at least 3 classes.

    * :math:`(n_{obs}, n_{variants} - 1)` if there are at least 3
      variants and the outcome is either a regression outcome or a binary classification
      outcome.

    * :math:`(n_{obs}, n_{variants} - 1, n_{classes})` if there are at least 3
      variants and and the outcome is a classification outcome with at least 3 classes.
    """
    if (n_dim := len(tensor.shape)) != 3:
        raise ValueError(
            f"Output needs to be 3-dimensional but is {n_dim}-dimensional."
        )
    n_obs, n_variants, n_outputs = tensor.shape
    if n_variants == 1 and n_outputs == 1:
        return tensor.reshape(n_obs)
    if n_variants == 1 and n_outputs == 2:
        return tensor[:, 0, 1].reshape(n_obs)
    if n_variants == 1:
        return tensor.reshape(n_obs, n_outputs)
    if n_outputs == 1:
        return tensor.reshape(n_obs, n_variants)
    if n_outputs == 2:
        return tensor[:, :, 1].reshape(n_obs, n_variants)
    return tensor
