<!-- Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug -->

# Changelog

## 0.14.0 (2025-07-xx)

- Remove `polars` as a run dependency.
- Make [`MetaLearner.feature_importances`][metalearners.metalearner.MetaLearner.feature_importances] and
  [`Explainer.feature_importances`][metalearners.metalearner.Explainer.feature_importances] return a list
  of `dict`s, rather than a list of `pandas.DataFrame`s.

## 0.13.0 (2025-05-19)

## New features

- Add support for `polars` input in all `fit*` and `predict*` functions.

## 0.12.0 (2025-01-29)

### Other changes

- Comply with `scikit-learn` versions 1.6 and higher.

## 0.11.0 (2024-09-05)

### New features

- Add support for using `scipy.sparse.csr_matrix` as datastructure for covariates `X`.

## 0.10.0 (2024-08-13)

### New features

- Add abstract method [`MetaLearner.predict_conditional_average_outcomes`][metalearners.metalearner.MetaLearner.predict_conditional_average_outcomes] to [`metalearners.metalearner.MetaLearner`][metalearners.metalearner.MetaLearner].
- Implement [`RLearner.predict_conditional_average_outcomes`][metalearners.rlearner.RLearner.predict_conditional_average_outcomes] for [`metalearners.rlearner.RLearner`][metalearners.rlearner.RLearner].

### Bug fixes

- Fix bug in which the [`metalearners.slearner.SLearner`][metalearners.slearner.SLearner]'s inference step would have some leakage in the in-sample scenario.

## 0.9.0 (2024-08-02)

### New features

- Add [`MetaLearner.init_args`][metalearners.metalearner.MetaLearner.init_args].
- Add [`FixedBinaryPropensity`][metalearners.utils.FixedBinaryPropensity].
- Add `MetaLearner._build_onnx` to [`metalearners.MetaLearner`][metalearners.metalearner.MetaLearner] abstract class and implement it for [`TLearner`][metalearners.tlearner.TLearner], [`XLearner`][metalearners.xlearner.XLearner], [`RLearner`][metalearners.rlearner.RLearner], and [`DRLearner`][metalearners.drlearner.DRLearner].
- Add `MetaLearner._necessary_onnx_models`.
- Add [`DRLearner.average_treatment_effect`][metalearners.drlearner.DRLearner.average_treatment_effect] to compute the AIPW point estimate and standard error for average treatment effects (ATE) without requiring a full model fit.

## 0.8.0 (2024-07-22)

### New features

- Add [`MetaLearner.fit_all_nuisance`][metalearners.metalearner.MetaLearner.fit_all_nuisance] and [`MetaLearner.fit_all_treatment`][metalearners.metalearner.MetaLearner.fit_all_treatment].
- Add optional `store_raw_results` and `store_results` parameters to [`MetaLearnerGridSearch`][metalearners.grid_search.MetaLearnerGridSearch].
- Renamed `_GSResult` to [`GSResult`][metalearners.grid_search.GSResult].
- Added `grid_size_` attribute to [`MetaLearnerGridSearch`][metalearners.grid_search.MetaLearnerGridSearch].
- Implement [`CrossFitEstimator.score`][metalearners.cross_fit_estimator.CrossFitEstimator.score].

### Bug fixes

- Fixed a bug in [`MetaLearner.evaluate`][metalearners.metalearner.MetaLearner.evaluate] where it failed in the case of `feature_set` being different from `None`.

## 0.7.0 (2024-07-12)

### New features

- Add optional `adaptive_clipping` parameter to [`DRLearner`][metalearners.drlearner.DRLearner].

### Other changes

- Change the index columns order in `MetaLearnerGridSearch.results_`.
- Raise a custom error if only one class is present in a classification outcome.
- Raise a custom error if there are some treatment variants which have seen classification outcomes that have not appeared for some other treatment variant.

## 0.6.0 (2024-07-08)

### New features

- Implement [`MetaLearnerGridSearch`][metalearners.grid_search.MetaLearnerGridSearch].
- Add a `scoring` parameter to [`MetaLearner.evaluate`][metalearners.metalearner.MetaLearner.evaluate] and implement the abstract method for [`XLearner`][metalearners.xlearner.XLearner] and [`DRLearner`][metalearners.drlearner.DRLearner].

### Other changes

- Increase the lower bound on `scikit-learn` from 1.3 to 1.4.
- Drop the run dependency on `git_root`.

## 0.5.0 (2024-06-18)

- No longer raise an error if `feature_set` is provided to [`SLearner`][metalearners.slearner.SLearner].
- Fix a bug where base model dictionaries -- e.g., `n_folds` or `feature-set` -- were improperly initialized if the provided dictionary's keys were a strict superset of the expected keys.

## 0.4.2 (2024-06-18)

- Ship license file.

## 0.4.1 (2024-06-18)

- Fix dependencies for pip.

## 0.4.0 (2024-06-18)

- Implemented [`CrossFitEstimator.clone`][metalearners.cross_fit_estimator.CrossFitEstimator.clone].
- Added `n_jobs_base_learners` to [`MetaLearner.fit`][metalearners.metalearner.MetaLearner.fit].
- Renamed [`Explainer.feature_importances`][metalearners.explainer.Explainer.feature_importances]. Note this is a breaking change.
- Renamed [`MetaLearner.feature_importances`][metalearners.metalearner.MetaLearner.feature_importances]. Note this is a breaking change.
- Renamed [`Explainer.shap_values`][metalearners.explainer.Explainer.shap_values]. Note this is a breaking change.
- Renamed [`MetaLearner.shap_values`][metalearners.metalearner.MetaLearner.shap_values]. Note this is a breaking change.
- Renamed [`MetaLearner.explainer`][metalearners.metalearner.MetaLearner.explainer]. Note this is a breaking change.
- Implemented `synchronize_cross_fitting` parameter for [`MetaLearner.fit`][metalearners.metalearner.MetaLearner.fit].
- Implemented `cv` parameter for [`CrossFitEstimator.fit`][metalearners.cross_fit_estimator.CrossFitEstimator.fit].

## 0.3.0 (2024-06-03)

- Implemented [`Explainer`][metalearners.explainer.Explainer] with support for binary classification and regression outcomes and discrete treatment variants.
- Integration of [`Explainer`][metalearners.explainer.Explainer] with [`MetaLearner`][metalearners.metalearner.MetaLearner] for feature importance and SHAP values calculations.
- Implemented model reuse through the `fitted_nuisance_models` and `fitted_propensity_model` parameters of [`MetaLearner`][metalearners.metalearner.MetaLearner].
- Allow for `fit_params` in [`MetaLearner.fit`][metalearners.metalearner.MetaLearner.fit].

## 0.2.0 (2024-05-28)

Beta release with:

- [`DRLearner`][metalearners.drlearner.DRLearner] with support for binary classification and regression outcomes and discrete treatment variants.
- Generalization of [`TLearner`][metalearners.tlearner.TLearner], [`XLearner`][metalearners.xlearner.XLearner], and [`RLearner`][metalearners.rlearner.RLearner] to allow for more than two discrete treatment variants.
- Unification of shapes returned by `predict` methods.
- [`simplify_output`][metalearners.utils.simplify_output] and [`metalearner_factory`][metalearners.utils.metalearner_factory].

## 0.1.0 (2024-05-16)

Alpha release with:

- [`TLearner`][metalearners.tlearner.TLearner] with support for binary classification and regression outcomes and binary treatment variants.
- [`SLearner`][metalearners.slearner.SLearner] with support for binary classification and regression outcomes and discrete treatment variants.
- [`XLearner`][metalearners.xlearner.XLearner] with support for binary classification and regression outcomes and binary treatment variants.
- [`RLearner`][metalearners.rlearner.RLearner] with support for binary classification and regression outcomes and binary treatment variants.
