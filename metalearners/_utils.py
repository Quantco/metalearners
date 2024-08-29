# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Callable
from inspect import signature
from operator import le, lt
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from sklearn.base import check_array, check_X_y, is_classifier, is_regressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from metalearners._typing import Matrix, PredictMethod, Vector, _ScikitModel

_PREDICT = "predict"
_PREDICT_PROBA = "predict_proba"
ONNX_PROBABILITIES_OUTPUTS = ["probabilities", "output_probability"]

default_rng = np.random.default_rng()


def safe_len(X: Matrix) -> int:
    """Determine the length of a Matrix."""
    if scipy.sparse.issparse(X):
        return X.shape[0]
    return len(X)


def index_matrix(matrix: Matrix, rows: Vector) -> Matrix:
    """Subselect certain rows from a matrix."""
    if isinstance(rows, pd.Series):
        rows = rows.to_numpy()
    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[rows]
    return matrix[rows, :]


def index_vector(vector: Vector, rows: Vector) -> Vector:
    """Subselect certain rows from a vector."""
    if isinstance(rows, pd.Series):
        rows = rows.to_numpy()
    if isinstance(vector, pd.Series):
        return vector.iloc[rows]
    return vector[rows]


def are_pd_indices_equal(*args: pd.DataFrame | pd.Series) -> bool:
    if len(args) < 2:
        return True
    reference_index = args[0].index
    for data_structure in args[1:]:
        if any(data_structure.index != reference_index):
            return False
    return True


def is_pd_df_or_series(arg) -> bool:
    return isinstance(arg, pd.DataFrame) or isinstance(arg, pd.Series)


def validate_all_vectors_same_index(*args: Vector) -> None:
    if len(args) < 2:
        return None
    pd_args = list(filter(is_pd_df_or_series, args))
    if len(pd_args) > 1:
        if not are_pd_indices_equal(*pd_args):
            raise ValueError(
                "All inputs provided as pandas data structures are expected to rely on "
                "the same index. Yet, at least two data structures have a different index."
            )
    if 0 < len(pd_args) < len(args):
        if any(pd_args[0].index != range(0, len(pd_args[0]))):  # type: ignore
            raise ValueError(
                "In order to mix numpy np.ndarray  and pd.Series objects, the pd.Series objects "
                "should have an index of 0 to n-1."
            )


def validate_number_positive(
    value: int | float, name: str, strict: bool = True
) -> None:
    """Validates that a number is positive.

    If ``strict = True`` then it validates that the number is strictly positive.
    """
    if strict:
        if value <= 0:
            raise ValueError(
                f"{name} was expected to be strictly positive but was {value}."
            )
    else:
        if value < 0:
            raise ValueError(
                f"{name} was expected to be positive or zero but was {value}."
            )


def check_propensity_score(
    propensity_scores: Matrix,
    features: Matrix | None = None,
    n_variants: int = 2,
    sum_to_one: bool = False,
    check_kwargs: dict | None = None,
    sum_tolerance: float = 0.001,
) -> None:
    """Ensure propensity scores match assumptions.

    The function ensures that ``propensity_scores`` and, if provided, ``features`` are:
     * shape-compatible and contain no missing or invalid entries;
     * the shape of ``propensity_scores`` is (nobs, ``n_variants``);
     * propensity scores are between 0 and 1;
     * if ``sum_to_one`` is ``True``, ``propensity_scores`` sums up to 1 for each observation.
    Shape compatibility is checked through scikit-learn's :func:`check_X_y` and
    :func:`check_array`; optional parameters for that method can be supplied via
    ``check_kwargs``.
    """
    if check_kwargs is None:
        check_kwargs = {}

    if len(propensity_scores.shape) < 2 or propensity_scores.shape[1] != n_variants:
        raise ValueError(
            f"One propensity score must be provided for each variant. "
            f"There are {n_variants} but the shape of the propensity "
            f"scores is {propensity_scores.shape}."
        )

    if features is not None:
        check_X_y(features, propensity_scores, multi_output=True, **check_kwargs)
    else:
        check_array(propensity_scores, **check_kwargs)

    if not 0.0 < np.min(propensity_scores) <= np.max(propensity_scores) < 1.0:
        raise ValueError(
            f"Propensity scores have to be between 0 and 1. Minimum is "
            f"{np.min(propensity_scores):.4f} and maximum is "
            f"{np.max(propensity_scores):.4f}."
        )

    if sum_to_one:
        min_sum = np.min(np.sum(propensity_scores, axis=1))
        max_sum = np.max(np.sum(propensity_scores, axis=1))
        if not 1 - sum_tolerance < min_sum <= max_sum < 1 + sum_tolerance:
            raise ValueError(
                f"Propensity scores for all observations must sum to 1. "
                f"Minimum is {min_sum:.4f} and maximum is {max_sum:.4f}."
            )


def convert_and_pad_propensity_score(
    propensity_scores: Vector | Matrix, n_variants: int
) -> np.ndarray:
    """Convert to ``np.ndarray`` and pad propensity scores, if necessary.

    Taking in a matrix or vector of propensity scores ``propensity_scores``, the function
    performs two things:
    * convert ``propensity_scores`` to an ``np.ndarray``.
    * if ``propensity_scores`` is a vector, the function will expand it to a matrix in
     case the number of variants ``n_variants`` is 2. That ensures there  is one
     propensity score per variant. The expansion assumes that the provided scores are
     those for the second variant.
    """
    if isinstance(propensity_scores, pd.Series) or isinstance(
        propensity_scores, pd.DataFrame
    ):
        propensity_scores = propensity_scores.to_numpy()
    p_is_1d = len(propensity_scores.shape) == 1 or propensity_scores.shape[1] == 1
    if n_variants == 2 and p_is_1d:
        propensity_scores = np.c_[1 - propensity_scores, propensity_scores]
    return propensity_scores


def get_n_variants(propensity_scores: Matrix) -> int:
    """Returns the number of treatment variants based on the shape of
    ``propensity_scores``.

    If the propensity scores array has a single dimension (i.e., it's a 1D array) or
    only one column in the second dimension, the function returns 2 (representing a binary
    treatment scenario, typically treated vs. not-treated). Otherwise, the number of treatment
    variants is assumed to be the size of the second dimension of the ``propensity_scores``
    array.

    propensity_score can be either a ``np.ndarray`` or a ``pd.DataFrame`` of propensity
    scores. The expected shape is ``(n_obs,)`` for binary treatment, or
    ``(n_obs, n_variants)`` for multiple treatments.
    """
    n_variants = (
        2
        if len(propensity_scores.shape) == 1 or propensity_scores.shape[1] == 1
        else propensity_scores.shape[1]
    )
    return n_variants


def get_linear_dimension(X: Matrix) -> int:
    """Calculates the required dimensionality of a vector in order to perform a linear
    transformation of the given data matrix.

    If the matrix consists only of numerical variables, the linear dimension is the
    number of features in the matrix. However, if there are categorical variables, they
    are expanded into dummy/indicator variables (one-hot encoding) for each category.
    Thus, the dimension, in this case, is the total number of numerical features plus
    the number of individual categories across all categorical features.
    """
    if isinstance(X, pd.DataFrame):
        categorical_features = X.select_dtypes(include="category")
        n_categories = 0
        for c in categorical_features.columns:
            n_categories += len(X[c].cat.categories)
        return len(set(X.columns) - set(categorical_features)) + n_categories
    return X.shape[1]


def sigmoid(x: np.ndarray) -> np.ndarray:
    r"""Sigmoid function.

    .. math::
        \sigma (x) = \frac{1}{1+e^{-x}}
    """
    return 1 / (1 + np.exp(-x))


def check_probability(p: float, zero_included=False, one_included=False) -> None:
    r"""Checks whether the provided probability p lies within a valid range.

    The valid range, from 0 to 1, which can be inclusive or exclusive, depending on
    the flags ``zero_included`` and ``one_included`` respectively.
    """

    if np.isnan(p):
        raise ValueError("Invalid input! Probability p should not be NaN.")

    left_operator = le if zero_included else lt
    right_operator = le if one_included else lt

    if not left_operator(0, p):
        raise ValueError("Probability p must be greater than or equal to 0.")
    if not right_operator(p, 1):
        raise ValueError("Probability p must be less than or equal to 1.")


def convert_treatment(treatment: Vector) -> np.ndarray:
    """Convert to ``np.ndarray`` and adapt dtype, if necessary."""
    if isinstance(treatment, np.ndarray):
        new_treatment = treatment.copy()
    elif isinstance(treatment, pd.Series):
        new_treatment = treatment.to_numpy()
    if new_treatment.dtype == bool:
        return new_treatment.astype(int)
    if new_treatment.dtype == float and all(x.is_integer() for x in new_treatment):
        return new_treatment.astype(int)

    if not pd.api.types.is_integer_dtype(new_treatment):
        raise TypeError(
            "Treatment must be boolean, integer or float with integer values."
        )
    return new_treatment


def supports_categoricals(model: _ScikitModel) -> bool:
    if (
        isinstance(model, HistGradientBoostingClassifier)
        or isinstance(model, HistGradientBoostingRegressor)
    ) and model.categorical_features == "from_dtype":
        return True
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        if isinstance(model, LGBMClassifier | LGBMRegressor):
            return True
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier, XGBRegressor

        if isinstance(model, XGBClassifier | XGBRegressor) and model.enable_categorical:
            return True
    except (ImportError, AttributeError):  # enable_categorical was added in v1.5.0
        pass
    try:
        from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV

        if isinstance(model, GeneralizedLinearRegressor | GeneralizedLinearRegressorCV):
            return True
    except ImportError:
        pass
    # TODO: Add support for Catboost? The problem is that we need the cat features names and reinit the model
    return False


def function_has_argument(func: Callable, argument: str) -> bool:
    return argument in signature(func).parameters


def validate_model_and_predict_method(
    model_factory: type[_ScikitModel],
    predict_method: PredictMethod,
    name: str = "model",
) -> None:
    if is_classifier(model_factory) and predict_method == _PREDICT:
        raise ValueError(
            f"The {name} is supposed to be used with the predict "
            "method 'predict' but it is a classifier."
        )
    if is_regressor(model_factory) and predict_method == _PREDICT_PROBA:
        raise ValueError(
            f"The {name} is supposed to be used with the predict "
            "method 'predict_proba' but it is not a classifier."
        )


def clip_element_absolute_value_to_epsilon(vector: Vector, epsilon: float) -> Vector:
    """Clip in accordance with the sign of each element if the element's absolute value
    is below ``epsilon``.

    For instance, if ``vector`` equals ``[0, 1, -1, .09, -.09]`` and ``epsilon`` equals ``.1``,
    this function will return ``[0, 1, -1, .1, -.1]``.

    Typically, this is done in order to avoid numerical problems when there is an element-wise
    division by ``vector`` and that the elements of ``vector`` are very close to 0.
    """
    bound = np.where(vector < 0, -1, 1) * epsilon
    return np.where(np.abs(vector) < epsilon, bound, vector)


def validate_valid_treatment_variant_not_control(
    treatment_variant: int, n_variants: int
) -> None:
    if treatment_variant >= n_variants:
        raise ValueError(
            f"MetaLearner was initialized to have {n_variants} 0-index variants but tried "
            f"to index variant {treatment_variant}."
        )
    if treatment_variant < 1:
        raise ValueError(
            "pseudo outcomes can only be computed for a treatment variant which isn't "
            "considered to be control."
        )


def load_mindset_data(
    path: Path,
) -> tuple[pd.DataFrame, str, str, list[str], list[str]]:
    # TODO: Optionally make this function work with a URL instead of a file system reference.
    # That way, we don't need to package the data for someone to be able to use this function.
    df = pd.read_csv(path)
    outcome_column = "achievement_score"
    treatment_column = "intervention"
    feature_columns = [
        column
        for column in df.columns
        if column not in [outcome_column, treatment_column]
    ]
    categorical_feature_columns = [
        "ethnicity",
        "gender",
        "frst_in_family",  # spellchecker:disable-line
        "school_urbanicity",
        "schoolid",
    ]
    # Note that explicitly setting the dtype of these features to category
    # allows both lightgbm as well as shap plots to
    # 1. Operate on features which are not of type int, bool or float
    # 2. Correctly interpret categoricals with int values to be
    #    interpreted as categoricals, as compared to ordinals/numericals.
    for categorical_feature_column in categorical_feature_columns:
        df[categorical_feature_column] = df[categorical_feature_column].astype(
            "category"
        )
    return (
        df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
    )


def load_twins_data(
    path: Path,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, str, str, list[str], list[str], str]:
    # TODO: Optionally make this function work with a URL instead of a file system reference.
    # That way, we don't need to package the data for someone to be able to use this function.
    df = pd.read_csv(path)
    drop_columns = [
        "bord",
        "brstate_reg",
        "stoccfipb_reg",
        "mplbir_reg",
        "infant_id",
        "wt",
    ]
    # We remove wt (weight) and bord (birth order) as they are different for each twin.
    # We remove _reg variables as they are already represented by the corresponding
    # variable without _reg and this new only groups them in bigger regions.
    # We remove infant_id as it's a unique identifier for each infant.
    df = df.drop(drop_columns, axis=1)
    outcome_column = "outcome"
    treatment_column = "treatment"
    true_cate_column = "true_cate"
    feature_columns = [
        column
        for column in df.columns
        if column not in [outcome_column, treatment_column]
    ]
    assert len(feature_columns) == 45

    ordinary_feature_columns = [
        "dlivord_min",
        "dtotord_min",
    ]
    categorical_feature_columns = [
        column for column in feature_columns if column not in ordinary_feature_columns
    ]
    for categorical_feature_column in categorical_feature_columns:
        df[categorical_feature_column] = df[categorical_feature_column].astype(
            "category"
        )

    n_twins_pairs = df.shape[0] // 2
    chosen_twin = rng.binomial(n=1, p=0.3, size=n_twins_pairs)

    selected_rows = []
    for i in range(0, len(df), 2):
        pair_idx = i // 2
        selected_row_idx = i + chosen_twin[pair_idx]
        selected_rows.append(selected_row_idx)

    chosen_df = df.iloc[selected_rows].reset_index(drop=True)

    mu_0 = df[df[treatment_column] == 0][outcome_column].reset_index(drop=True)
    mu_1 = df[df[treatment_column] == 1][outcome_column].reset_index(drop=True)
    chosen_df["mu_0"] = mu_0
    chosen_df["mu_1"] = mu_1
    chosen_df[true_cate_column] = mu_1 - mu_0

    return (
        chosen_df,
        outcome_column,
        treatment_column,
        feature_columns,
        categorical_feature_columns,
        true_cate_column,
    )


def get_one(*args, **kwargs) -> int:
    return 1


def get_predict(*args, **kwargs) -> PredictMethod:
    return "predict"


def get_predict_proba(*args, **kwargs) -> PredictMethod:
    return "predict_proba"


def simplify_output_2d(tensor: np.ndarray) -> np.ndarray:
    """Reduces the third dimension of a CATE estimation tensor.

    In the case of a classification task it only works in the binary classification
    outcome and returns the CATE of the positive class.

    The returned array will be of shape :math:`(n_{obs}, n_{variants} - 1)`.
    """
    if (n_dim := len(tensor.shape)) != 3:
        raise ValueError(
            f"Output needs to be 3-dimensional but is {n_dim}-dimensional."
        )
    n_obs, n_variants, n_outputs = tensor.shape
    if n_outputs == 1:
        return tensor[:, :, 0]
    elif n_outputs == 2:
        return tensor[:, :, 1]
    else:
        raise ValueError(
            "This function requires a regression or a classification with binary outcome "
            "task."
        )


# Taken from https://stackoverflow.com/questions/13741998/is-there-a-way-to-let-classes-inherit-the-documentation-of-their-superclass-with
def copydoc(fromfunc, sep="\n"):
    """
    Decorator: Copy the docstring of ``fromfunc``
    """

    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__ is None:
            func.__doc__ = sourcedoc
        else:
            func.__doc__ = sep.join([sourcedoc, func.__doc__])
        return func

    return _decorator


def default_metric(predict_method: PredictMethod) -> str:
    if predict_method == _PREDICT_PROBA:
        return "neg_log_loss"
    return "neg_root_mean_squared_error"


def check_onnx_installed() -> None:
    """Ensures that ``onnx`` is available."""
    try:
        import onnx  # noqa F401
    except ImportError:
        raise ImportError(
            "onnx is not installed. Please install onnx to use this feature."
        )


def check_spox_installed() -> None:
    """Ensures that ``spox`` is available."""
    try:
        import spox  # noqa F401
    except ImportError:
        raise ImportError(
            "spox is not installed. Please install spox to use this feature."
        )


def infer_dtype_and_shape_onnx(tensor) -> tuple[np.dtype, tuple]:
    """Returns the ``np.dtype`` and shape of an ONNX tensor."""
    check_onnx_installed()
    import onnx

    dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.type.tensor_type.elem_type)
    shape = tuple(
        d.dim_value if d.HasField("dim_value") else None
        for d in tensor.type.tensor_type.shape.dim
    )
    return dtype, shape


def infer_probabilities_output(model) -> tuple[int, str]:
    """Returns the index and name of the output which contains the probabilities outcome
    in a ONNX classifier."""
    check_onnx_installed()
    for i, output in enumerate(model.graph.output):
        if output.name in ONNX_PROBABILITIES_OUTPUTS:
            return i, output.name
    raise ValueError("No probabilities output was found.")


def infer_input_dict(model) -> dict:
    """Returns a dict where the keys are the input names of the model and the values are
    ``spox.Var`` with the corresponding shape and type."""
    check_spox_installed()
    from spox import Tensor, Var, argument

    input_dict: dict[str, Var] = {}
    for input_tensor in model.graph.input:
        input_dtype, input_shape = infer_dtype_and_shape_onnx(input_tensor)
        input_dict[input_tensor.name] = argument(Tensor(input_dtype, input_shape))

    return input_dict


def warning_experimental_feature(function_name: str) -> None:
    warnings.warn(
        f"{function_name} is an experimental feature. Use it at your own risk!",
        stacklevel=2,
    )
