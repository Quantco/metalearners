.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

0.11.0 (2024-09-05)
-------------------

**New features**

* Add support for using ``scipy.sparse.csr_matrix`` as datastructure for covariates ``X``.


0.10.0 (2024-08-13)
-------------------

**New features**

* Add abstract method
  :meth:`~metalearners.metalearner.MetaLearner.predict_conditional_average_outcomes`
  to :class:`~metalearners.metalearner.MetaLearner`.

* Implement
  :meth:`~metalearners.rlearner.RLearner.predict_conditional_average_outcomes`
  for :class:`~metalearners.rlearner.RLearner`.

**Bug fixes**

* Fix bug in which the :class:`~metalearners.slearner.SLearner`'s
  inference step would have some leakage in the in-sample scenario.


0.9.0 (2024-08-02)
------------------

**New features**

* Add :meth:`metalearners.metalearner.MetaLearner.init_args`.

* Add :class:`metalearners.utils.FixedBinaryPropensity`.

* Add ``_build_onnx`` to :class:`metalearners.MetaLearner` abstract class and implement it
  for :class:`metalearners.TLearner`, :class:`metalearners.XLearner`, :class:`metalearners.RLearner`
  and :class:`metalearners.DRLearner`.

* Add ``_necessary_onnx_models`` to :class:`metalearners.MetaLearner`.

* Add :meth:`metalearners.metalearner.DRLearner.average_treatment_effect` to
  compute the AIPW point estimate and standard error for
  _average treatment effects (ATE)_ without requiring a full model fit.


0.8.0 (2024-07-22)
------------------

**New features**

* Add :meth:`metalearners.metalearner.MetaLearner.fit_all_nuisance` and
  :meth:`metalearners.metalearner.MetaLearner.fit_all_treatment`.

* Add optional ``store_raw_results`` and ``store_results`` parameters to :class:`metalearners.grid_search.MetaLearnerGridSearch`.

* Renamed :class:`metalearners.grid_search._GSResult` to :class:`metalearners.grid_search.GSResult`.

* Added ``grid_size_`` attribute to :class:`metalearners.grid_search.MetaLearnerGridSearch`.

* Implement :meth:`metalearners.cross_fit_estimator.CrossFitEstimator.score`.

**Bug fixes**

* Fixed a bug in :meth:`metalearners.metalearner.MetaLearner.evaluate` where it failed
  in the case of ``feature_set`` being different from ``None``.


0.7.0 (2024-07-12)
------------------

**New features**

* Add optional ``adaptive_clipping`` parameter to :class:`metalearners.DRLearner`.

**Other changes**

* Change the index columns order in ``MetaLearnerGridSearch.results_``.

* Raise a custom error if only one class is present in a classification outcome.

* Raise a custom error if there are some treatment variants which have seen classification outcomes which have not appeared for some other treatment variant.


0.6.0 (2024-07-08)
------------------

**New features**

* Implement :class:`metalearners.grid_search.MetaLearnerGridSearch`.

* Add a ``scoring`` parameter to :meth:`metalearners.metalearner.MetaLearner.evaluate` and
  implement the abstract method for the :class:`metalearners.XLearner` and
  :class:`metalearners.DRLearner`.

**Other changes**

* Increase lower bound on ``scikit-learn`` from 1.3 to 1.4.

* Drop the run dependency on ``git_root``.


0.5.0 (2024-06-18)
------------------

* No longer raise an error if ``feature_set`` is provided to
  :class:`metalearners.SLearner`.

* Fix a bug where base model dictionaries -- e.g. ``n_folds`` or
  ``feature-set`` -- were improperly initialized if the provided
  dictionary's keys were a strict superset of the expected keys.

0.4.2 (2024-06-18)
------------------

* Ship license file.

0.4.1 (2024-06-18)
------------------

* Fix dependencies for pip.

0.4.0 (2024-06-18)
------------------

* Implemented :meth:`metalearners.cross_fit_estimator.CrossFitEstimator.clone`.

* Added ``n_jobs_base_learners`` to :meth:`metalearners.metalearner.MetaLearner.fit`.

* Renamed :meth:`metalearners.explainer.Explainer.feature_importances`. Note this is
  a breaking change.

* Renamed :meth:`metalearners.metalearner.MetaLearner.feature_importances`. Note this
  is a breaking change.

* Renamed :meth:`metalearners.explainer.Explainer.shap_values`. Note this is
  a breaking change.

* Renamed :meth:`metalearners.metalearner.MetaLearner.shap_values`. Note this
  is a breaking change.

* Renamed :meth:`metalearners.metalearner.MetaLearner.explainer`. Note this is
  a breaking change.

* Implemented ``synchronize_cross_fitting`` parameter for
  :meth:`metalearners.metalearner.MetaLearner.fit`.

* Implemented ``cv`` parameter for :meth:`metalearners.cross_fit_estimator.fit`.


0.3.0 (2024-06-03)
------------------

* Implemented :class:`metalearners.explainer.Explainer` with support for binary
  classification and regression outcomes and discrete treatment
  variants.

* Integration of :class:`metalearners.explainer.Explainer` with :class:`metalearners.metalearner.MetaLearner`
  for feature importance and SHAP values calculations.

* Implemented model reusage through the ``fitted_nuisance_models`` and ``fitted_propensity_model``
  parameters of :class:`metalearners.metalearner.MetaLearner`.

* Allow for ``fit_params`` in :meth:`metalearners.metalearner.MetaLearner.fit`.

0.2.0 (2024-05-28)
------------------

Beta release with

* :class:`metalearners.DRLearner` with support for binary
  classification and regression outcomes and discrete treatment
  variants.

* Generalization of :class:`metalearners.TLearner`,
  :class:`metalearners.XLearner` and :class:`metalearners.RLearner`
  to allow for more than two discrete treatment variants.

* Unification of shapes returned by ``predict`` methods.

* :func:`metalearners.utils.simplify_output` and :func:`metalearners.utils.metalearner_factory`.


0.1.0 (2024-05-16)
------------------

Alpha release with

* :class:`metalearners.TLearner` with support for binary
  classification and regression outcomes and binary treatment
  variants.

* :class:`metalearners.SLearner` with support for binary
  classification and regression outcomes and discrete treatment
  variants.

* :class:`metalearners.XLearner` with support for binary
  classification and regression outcomes and binary treatment
  variants.

* :class:`metalearners.RLearner` with support for binary
  classification and regression otucomes and binary treatment variants.
