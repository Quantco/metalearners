.. _example-reuse :

=============================
 Example: Reusing base models
=============================

Motivation
----------

In our :ref:`Why MetaLearners <motivation-why-metalearners>` section
we praise the modularity of MetaLearners. Part of the reason why
modularity is useful is because we can actively decouple different
parts of the CATE estimation process.

Concretely, this decoupling allows for saving lots of compute
resources: if we know that we merely want to change *some parts* of a
MetaLearner, we may as well reuse the parts that we don't want to
change. Enabling this kind of base model reuse was one of the
requirements on ``metalearners``, see :ref:`Why not causalml or econml
<motivation-why-not-causalml-or-econml>`.

For instance, imagine trying to tune an R-Learner's -- consisting of two
nuisance models, a propensity model and an outcome model -- propensity
model with respect to its R-Loss. In such a scenario we would like to
reuse the same outcome model because it isn't affected by the
propensity model and thereby save a lot of redundant compute.

Example
-------

Loading the data
""""""""""""""""

Just like in our :ref:`example on estimating CATEs with a MetaLearner
<example-basic>`, we will first load some experiment data:

.. code-block:: python

   import pandas as pd
   from pathlib import Path
   from git_root import git_root

   df = pd.read_csv(git_root("data/learning_mindset.zip"))
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
       "frst_in_family",   # spellchecker:disable-line
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

Now that we've loaded the experiment data, we can train a MetaLearner.


Training a first MetaLearner
""""""""""""""""""""""""""""

Again, mirroring our :ref:`example on estimating CATEs with a MetaLearner
<example-basic>`, we can train an
:class:`~metalearners.rlearner.RLearner` as follows:

.. code-block:: python

  rlearner = RLearner(
      nuisance_model_factory=LGBMRegressor,
      propensity_model_factory=LGBMClassifier,
      treatment_model_factory=LGBMRegressor,
      is_classification=False,
      n_variants=2,
  )

  rlearner.fit(
      X=df[feature_columns],
      y=df[outcome_column],
      w=df[treatment_column],
  )

By virtue of having fitted the 'overall' MetaLearner, we fitted
the base model, too. Thereby we can now reuse some of them if we wish to.

Extracting a basel model from a trained MetaLearner
"""""""""""""""""""""""""""""""""""""""""""""""""""

In order to reuse a base model from one MetaLearner for another
MetaLearner, we first have to from the former. If, for instance, we
are interested in reusing the outcome nuisance model of the
:class:`~metalearners.rlearner.RLearner` we just trained, we can
access it via its ``_nuisance_models`` attribute:

.. code-block:: python

  rlearner._nuisance_models

  >>> {'propensity_model': [CrossFitEstimator(n_folds=10, ...)], 'outcome_model': [CrossFitEstimator(n_folds=10, ...)]}

We notice that the :class:`~metalearners.rlearner.RLearner` has two
kinds of nuisance models: ``"propensity_model"`` and ``"outcome_model"``. Note
that we could've figured this out by calling its
:meth:`~metalearners.rlearner.RLearner.nuisance_model_specifications()` method,
too.

Therefore, we now know how to fetch our outcome model:

.. code-block:: python

  outcome_model = rlearner._nuisance_models["outcome_model"]


Training a second MetaLearner by reusing a base model
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Given that we know have an already trained outcome model, we can reuse
for another 'kind' of :class:`~metalearners.rlearner.RLearner` on the
same data. Concretely, we will now want to use a different
``propensity_model_factory`` and ``nuisance_model_factory``. Note that
this time, we do not specify a ``nuisance_model_factory`` in the
initialization of the :class:`~metalearners.rlearner.RLearner` since
the :class:`~metalearners.rlearner.RLearner` only relies on a single
non-propensity nuisance model. This might vary for other MetaLearners,
such as the :class:`~metalearners.drlearner.DRLearner`.

.. code-block:: python

  rlearner_new = RLearner(
      propensity_model_factory=LogisticRegression,
      treatment_model_factory=LinearRegression,
      is_classification=False,
      fitted_nuisance_models={"outcome_model": outcome_model},
      n_variants=2,
  )

  rlearner_new.fit(
      X=df[feature_columns],
      y=df[outcome_column],
      w=df[treatment_column],
  )

What's more is that we can also reuse models between different kinds
of MetaLearner architectures. A propensity model, for instance, is
used in many scenarios. Let's reuse it for a :class:`~metalearners.drlearner.DRLearner`:

.. code-block:: python

  trained_propensity_model = rlearner._nuisance_models["propensity_model"]

  drlearner = DRLearner(
      nuisance_model_factory=LGBMRegressor,
      treatment_model_factory=LGBMRegressor,
      fitted_nuisance_models={"propensity_model": trained_propensity_model},
      is_classification=False,
      n_variants=2,
  )

  rlearner_new.fit(
      X=df[feature_columns],
      y=df[outcome_column],
      w=df[treatment_column],
  )


Further comments
""""""""""""""""

* Note that the nuisance models are always expected to be of type
  :class:`~metalearners.cross_fit_estimator.CrossFitEstimator`. More
  precisely, the when extracting or passing a particular model kind,
  we always pass a list of
  :class:`~metalearners.cross_fit_estimator.CrossFitEstimator`.
* In the examples above we reused nuisance models trained as part of a
  call to a MetaLearners overall :meth:`~metalearners.metalearner.MetaLearner.fit` method. If one wants to train a nuisance model in isolation (i.e. not
  through a MetaLearner) to be used in a MetaLearner afterwards, one
  should do it by instantiating
  :class:`~metalearners.cross_fit_estimator.CrossFitEstimator`.
* Additionally, individual nuisance models can be trained via a
  MetaLearner's :meth:`~metalearners.metalearner.MetaLearner.fit_nuisance`
  method.
* We strongly recommend only reusing base models if they have been trained on
  exactly the same data. If this is not the case, some functionalities
  will probably not work as hoped for.
* Note that only :term:`nuisance models <Nuisance model>` can be reused, not :term:`treatment
  models <Treatment effect model>`.
