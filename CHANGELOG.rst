.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

0.2.0 (2024-xx-xx)
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
