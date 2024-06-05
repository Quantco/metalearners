.. _example-optuna :

=========================================
 Example: Model selection with optuna
=========================================

Motivation
----------

We know that model selection and/or hyperparameter optimization (HPO) can
have massive impacts on the prediction quality in regular Machine
Learning. Yet, it seems that model selection and hyperparameter
optimization are  of substantial importance for CATE estimation with
MetaLearners, too, see e.g. `Machlanski et. al <https://arxiv.org/abs/2303.01412>`_.

However, model selection and HPO for MetaLearners look quite different from what we're used to from e.g. simple supervised learning problems. Concretely,

* In terms of a MetaLearners's option space, there are several levels
  to optimize for:

  1. The MetaLearner architecture, e.g. R-Learner vs DR-Learner
  2. The model to choose per base estimator of said MetaLearner architecture, e.g. ``LogisticRegression`` vs ``LGBMClassifier``
  3. The model hyperparameters per base model

*  On a conceptual level, it's not clear how to measure model quality
   for MetaLearners. As a proxy for the underlying quantity of
   interest one might look into base model performance, the R-Loss of
   the CATE estimates or some more elaborate approaches alluded to by
   `Machlanski et. al <https://arxiv.org/abs/2303.01412>`_.

We think that HPO can be divided into two camps:

* Exploration of (hyperparameter, metric evaluation) pairs where the
  pairs do not influence each other (e.g. grid search, random search)

* Exploration of (hyperparameter, metric evaluation) pairs where the
  pairs do influence each other (e.g. Bayesian optimization,
  evolutionary algorithms); in other words, there is a feedback-loop between
  sample result and sample

In this example, we will illustrate the latter camp based on an
application of `optuna <https://github.com/optuna/optuna>`_ -- a
popular framework for HPO -- in interplay with ``metalearners``.

Installation
------------

In order to use ``optuna``, we first need to install the package.
We can do so either via conda and conda-forge

.. code-block:: console

   $ conda install optuna -c conda-forge

or via pip and PyPI

.. code-block:: console

   $ pip install optuna

Usage
-----

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

Now that we've loaded the experiment data, we can split it up into
train and validation data:

.. code-block:: python

  from sklearn.model_selection import train_test_split

  X_train, X_validation, y_train, y_validation, w_train, w_validation = train_test_split(
      df[feature_columns], df[outcome_column], df[treatment_column], test_size=0.25
  )


Optimizing base model hyperparameters
"""""""""""""""""""""""""""""""""""""

Let's say that we want to work with an
:class:`~metalearners.rlearner.RLearner` and ``LightGBM`` estimators
for base models. We will seek two optimize three hyperparameters of
our base models:

* The number of estimators ``n_estimators`` of our outcome model.
* The max depth ``max_depth`` of our outcome model.
* The number of estimators ``n_estimators`` of our treatment effect
  model.

We can mold this ambition into the following simple script creating an
``optuna`` ``study``:

.. code-block:: python

  import optuna
  from metalearners.rlearner import r_loss
  from metalearners.utils import simplify_output
  from metalearners import RLearner
  from lightgbm import LGBMRegressor, LGBMClassifier

  def objective(trial):

      n_estimators_nuisance = trial.suggest_int("n_estimators_nuisance", 5, 250)
      max_depth_nuisance = trial.suggest_int("max_depth_nuisance", 3, 30)
      n_estimators_treatment = trial.suggest_int("n_estimators_treatment", 5, 100)

      rlearner = RLearner(
	  nuisance_model_factory=LGBMRegressor,
	  nuisance_model_params={"n_estimators": n_estimators_nuisance, "max_depth": max_depth_nuisance, "verbosity": -1},
	  propensity_model_factory=LGBMClassifier,
	  propensity_model_params={"n_estimators": 5, "verbosity": -1},
	  treatment_model_factory=LGBMRegressor,
	  treatment_model_params={"n_estimators": n_estimators_treatment, "verbosity": -1},
	  is_classification=False,
	  n_variants=2,
      )

      rlearner.fit(X=X_train, y=y_train, w=w_train)

      return rlearner.evaluate(
	  X=X_validation,
	  y=y_validation,
	  w=w_validation,
	  is_oos=True,
      )["r_loss_1_vs_0"]

  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=100)

Note that the metric to be optimized is the R-Loss here. We can obtain
it -- among other metrics -- via the
:class:`~metalearners.rlearner.RLearner`'s
:meth:`~metalearners.rlearner.RLearner.evaluate` method. We can see it
evolve as follows across the 100 trials:

.. code-block:: console

   [I 2024-06-03 19:20:13,029] A new study created in memory with name: no-name-342fd485-82a8-403b-9201-c98d01274024
   [I 2024-06-03 19:20:18,481] Trial 0 finished with value: 0.8058146296845027 and parameters: {'n_estimators_nuisance': 78, 'max_depth_nuisance': 7, 'n_estimators_treatment': 28}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:20:25,886] Trial 1 finished with value: 0.8158949566481672 and parameters: {'n_estimators_nuisance': 110, 'max_depth_nuisance': 5, 'n_estimators_treatment': 90}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:20:31,482] Trial 2 finished with value: 0.8059918319487219 and parameters: {'n_estimators_nuisance': 108, 'max_depth_nuisance': 24, 'n_estimators_treatment': 10}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:20:37,337] Trial 3 finished with value: 0.8202956725566805 and parameters: {'n_estimators_nuisance': 16, 'max_depth_nuisance': 10, 'n_estimators_treatment': 99}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:20:47,279] Trial 4 finished with value: 0.8289376079649482 and parameters: {'n_estimators_nuisance': 168, 'max_depth_nuisance': 16, 'n_estimators_treatment': 58}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:20:58,013] Trial 5 finished with value: 0.8301790091587447 and parameters: {'n_estimators_nuisance': 194, 'max_depth_nuisance': 26, 'n_estimators_treatment': 64}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:21:09,459] Trial 6 finished with value: 0.8295207764791498 and parameters: {'n_estimators_nuisance': 239, 'max_depth_nuisance': 27, 'n_estimators_treatment': 39}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:21:17,956] Trial 7 finished with value: 0.81843425756809 and parameters: {'n_estimators_nuisance': 195, 'max_depth_nuisance': 4, 'n_estimators_treatment': 93}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:21:28,087] Trial 8 finished with value: 0.8281067745873746 and parameters: {'n_estimators_nuisance': 143, 'max_depth_nuisance': 23, 'n_estimators_treatment': 100}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:21:36,216] Trial 9 finished with value: 0.8174537107742403 and parameters: {'n_estimators_nuisance': 82, 'max_depth_nuisance': 15, 'n_estimators_treatment': 87}. Best is trial 0 with value: 0.8058146296845027.
   [I 2024-06-03 19:21:39,475] Trial 10 finished with value: 0.7953483370042609 and parameters: {'n_estimators_nuisance': 37, 'max_depth_nuisance': 10, 'n_estimators_treatment': 14}. Best is trial 10 with value: 0.7953483370042609.
   [I 2024-06-03 19:21:42,757] Trial 11 finished with value: 0.7952682807538822 and parameters: {'n_estimators_nuisance': 37, 'max_depth_nuisance': 10, 'n_estimators_treatment': 13}. Best is trial 11 with value: 0.7952682807538822.
   [I 2024-06-03 19:21:44,104] Trial 12 finished with value: 0.8249424971850418 and parameters: {'n_estimators_nuisance': 9, 'max_depth_nuisance': 11, 'n_estimators_treatment': 5}. Best is trial 11 with value: 0.7952682807538822.
   [I 2024-06-03 19:21:48,302] Trial 13 finished with value: 0.7984298284740843 and parameters: {'n_estimators_nuisance': 38, 'max_depth_nuisance': 12, 'n_estimators_treatment': 26}. Best is trial 11 with value: 0.7952682807538822.
   ...
   [I 2024-06-03 19:27:47,641] Trial 97 finished with value: 0.7884468426483208 and parameters: {'n_estimators_nuisance': 135, 'max_depth_nuisance': 3, 'n_estimators_treatment': 12}. Best is trial 73 with value: 0.7839636658643124.
   [I 2024-06-03 19:27:50,443] Trial 98 finished with value: 0.7880500972663745 and parameters: {'n_estimators_nuisance': 133, 'max_depth_nuisance': 3, 'n_estimators_treatment': 12}. Best is trial 73 with value: 0.7839636658643124.
   [I 2024-06-03 19:27:53,078] Trial 99 finished with value: 0.7869700632283656 and parameters: {'n_estimators_nuisance': 139, 'max_depth_nuisance': 3, 'n_estimators_treatment': 8}. Best is trial 73 with value: 0.7839636658643124.


Alternatively, if we'd like to optimize a base model in light of its
individual metric -- in this case an RMSE on the observed outcomes for an
the outcome model -- we can easily do that, too:

.. code-block :: python

  from sklearn.metrics import root_mean_squared_error

  def objective_individual(trial):

      n_estimators_nuisance = trial.suggest_int("n_estimators_nuisance", 5, 250)
      max_depth_nuisance = trial.suggest_int("max_depth_nuisance", 3, 30)

      rlearner = RLearner(
	  nuisance_model_factory=LGBMRegressor,
	  nuisance_model_params={"n_estimators": n_estimators_nuisance, "max_depth": max_depth_nuisance, "verbosity": -1},
	  propensity_model_factory=LGBMClassifier,
	  treatment_model_factory=LGBMRegressor,
	  is_classification=False,
	  n_variants=2,
      )

      rlearner.fit_nuisance(X=X_train, y=y_train, model_kind="outcome_model", model_ord=0)

      outcome_predictions = rlearner.predict_nuisance(X=X_validation, model_kind="outcome_model", model_ord=0, is_oos=True)

      return root_mean_squared_error(y_validation, outcome_predictions)

  study_individual = optuna.create_study(direction='minimize')
  study_individual.optimize(objective_individual, n_trials=100)


Leading to the following output

.. code-block:: console

   [I 2024-06-03 19:35:14,137] A new study created in memory with name: no-name-94241529-da38-41bd-a486-040308b1f023
   [I 2024-06-03 19:35:18,853] Trial 0 finished with value: 0.8203088574399058 and parameters: {'n_estimators_nuisance': 96, 'max_depth_nuisance': 28}. Best is trial 0 with value: 0.8203088574399058.
   [I 2024-06-03 19:35:27,959] Trial 1 finished with value: 0.8383203265114683 and parameters: {'n_estimators_nuisance': 236, 'max_depth_nuisance': 13}. Best is trial 0 with value: 0.8203088574399058.
   [I 2024-06-03 19:35:32,507] Trial 2 finished with value: 0.821097910375138 and parameters: {'n_estimators_nuisance': 100, 'max_depth_nuisance': 19}. Best is trial 0 with value: 0.8203088574399058.
   [I 2024-06-03 19:35:35,020] Trial 3 finished with value: 0.8069517664852458 and parameters: {'n_estimators_nuisance': 227, 'max_depth_nuisance': 3}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:35:37,044] Trial 4 finished with value: 0.8108314861878544 and parameters: {'n_estimators_nuisance': 36, 'max_depth_nuisance': 7}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:35:45,037] Trial 5 finished with value: 0.8324009294214451 and parameters: {'n_estimators_nuisance': 189, 'max_depth_nuisance': 23}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:35:51,032] Trial 6 finished with value: 0.8255894318735717 and parameters: {'n_estimators_nuisance': 134, 'max_depth_nuisance': 23}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:35:57,481] Trial 7 finished with value: 0.8295098178376358 and parameters: {'n_estimators_nuisance': 160, 'max_depth_nuisance': 26}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:36:04,078] Trial 8 finished with value: 0.8301142921842086 and parameters: {'n_estimators_nuisance': 165, 'max_depth_nuisance': 19}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:36:12,468] Trial 9 finished with value: 0.8353268112420604 and parameters: {'n_estimators_nuisance': 213, 'max_depth_nuisance': 23}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:36:12,604] Trial 10 finished with value: 0.8941068693906029 and parameters: {'n_estimators_nuisance': 5, 'max_depth_nuisance': 3}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:36:12,968] Trial 11 finished with value: 0.8195432140897054 and parameters: {'n_estimators_nuisance': 21, 'max_depth_nuisance': 3}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:36:15,676] Trial 12 finished with value: 0.8120941736598022 and parameters: {'n_estimators_nuisance': 51, 'max_depth_nuisance': 9}. Best is trial 3 with value: 0.8069517664852458.
   [I 2024-06-03 19:36:19,044] Trial 13 finished with value: 0.8157459078823713 and parameters: {'n_estimators_nuisance': 64, 'max_depth_nuisance': 8}. Best is trial 3 with value: 0.8069517664852458.
   ...
   [I 2024-06-03 19:40:05,394] Trial 97 finished with value: 0.8121024415590181 and parameters: {'n_estimators_nuisance': 55, 'max_depth_nuisance': 6}. Best is trial 25 with value: 0.8022308112279057.
   [I 2024-06-03 19:40:06,124] Trial 98 finished with value: 0.8039230485915543 and parameters: {'n_estimators_nuisance': 47, 'max_depth_nuisance': 3}. Best is trial 25 with value: 0.8022308112279057.
   [I 2024-06-03 19:40:07,671] Trial 99 finished with value: 0.8056511905287118 and parameters: {'n_estimators_nuisance': 67, 'max_depth_nuisance': 4}. Best is trial 25 with value: 0.8022308112279057.


Optimizing over architectures
"""""""""""""""""""""""""""""

``optuna``'s flexibility allows for not only the search over classical
hyperparameters of a given estimator but also to iterate over the
choice of base estimator architectures. Pushing it a step further, one
can even optimize over the space of MetaLearner architectures.

In the following example we will attempt to optimize over the
following search space:

1. MetaLearner: R-Learner vs DR-Learner
2. Nuisance model: ``LGBMRegressor`` vs ``Ridge``
3. Hyperparameter: ``n_estimators`` if ``LGBMRegressor`` and ``alpha``
   if ``Ridge``

Note that the choice of the base learner in the second step should be
conditioned on the choice in the first step. In other words, we do not
want to update our belief system on outcome learners for the R-Learner
by observing outcome learner for the DR-Learner. The same idea applies
to the interplay between steps two and three. This conditioning
becomes apparent in the source code below via the underscores,
e.g. ``nuisance_r``, which is only sampled (and thereby updated) if we
are using an R-Learner and and ``nuisance_dr``, which is only sampled
(and thereby updated) if we are using a DR-Learner.

.. code-block:: python

   import optuna
   from metalearners.utils import metalearner_factory, simplify_output
   from sklearn.linear_model import Ridge
   from metalearners.rlearner import r_loss


   # Arbitrary models for R-Loss
   outcome_estimates = LGBMRegressor().fit(X_train, y_train).predict(X_validation)
   propensity_scores = LGBMClassifier().fit(X_train, w_train).predict(X_validation)

   def objective_overall(trial):

       ### SAMPLING

       # Highest level of granularity: we sample the MetaLearner architecture.
       architecture = trial.suggest_categorical('architecture', ['R', 'DR'])

       # We distinguish cases because we do not want a DR-Learner run to influence
       # the optimizing process of R-Learner-related parameters.
       if architecture == 'R':

	   # Second level of granularity: we sample the nuisance base model.
	   # Note that this is conditioned on using the R-Learner.
	   nuisance_r = trial.suggest_categorical('nuisance_r', ['LGBMRegressor', 'Ridge'])
	   nuisance_dr = None
	   nuisance_dr_log_reg_alpha = None
	   nuisance_dr_lgbm_n_estimators = None

	   if nuisance_r == "LGBMRegressor":

	       # Lowest level of granularity: we sample the nuisance base model hyperparameters.
	       nuisance_r_lgbm_n_estimators = trial.suggest_int('nuisance_r_lgbm_n_estimators', 5, 250)
	       nuisance_r_lin_reg_alpha = None

	       nuisance_params = {"n_estimators": nuisance_r_lgbm_n_estimators, "verbose": -1}
	   else:
	       nuisance_r_lin_reg_alpha = trial.suggest_float('nuisance_r_lin_reg_alpha', 0, 10)
	       nuisance_r_lgbm_n_estimators = None

	       nuisance_params = {"alpha": nuisance_r_lin_reg_alpha}

       else:
	   nuisance_dr = trial.suggest_categorical('nuisance_dr', ['LGBMRegressor', 'Ridge'])
	   nuisance_r = None
	   nuisance_r_lin_reg_alpha = None
	   nuisance_r_lgbm_n_estimators = None

	   if nuisance_dr == "LGBMRegressor":

	       # Lowest level of granularity: we sample the nuisance base model hyperparameters.
	       nuisance_dr_lgbm_n_estimators = trial.suggest_int('nuisance_dr_lgbm_n_estimators', 5, 250)
	       nuisance_dr_lin_reg_alpha = None

	       nuisance_params = {"n_estimators": nuisance_dr_lgbm_n_estimators, "verbose": -1}

	   else:
	       nuisance_dr_lin_reg_alpha = trial.suggest_float('nuisance_dr_lin_reg_alpha', 0, 10)
	       nuisance_dr_lgbm_n_estimators = None

	       nuisance_params = {"alpha": nuisance_dr_lin_reg_alpha}

       ### LEARNING

       _metalearner_factory = metalearner_factory(architecture)
       # We know that only one of them is not None, therefore we can use or.
       nuisance_model_type = nuisance_r or nuisance_dr
       metalearner = _metalearner_factory(
	   nuisance_model_factory=LGBMRegressor if nuisance_model_type == "LGBMRegressor" else Ridge,
	   nuisance_model_params=nuisance_params,
	   propensity_model_factory=LGBMClassifier,
	   propensity_model_params={"n_estimators": 5, "max_depth": 5, "verbose": -1},
	   treatment_model_factory=LGBMRegressor,
	   treatment_model_params={"n_estimators": 5, "max_depth": 5, "verbose": -1},
	   is_classification=False,
	   n_variants=2,
       )
       metalearner.fit(X_train, y_train, w_train)

       ### EVALUATING

       cate_estimates = simplify_output(metalearner.predict(X_validation, is_oos=True))

       return r_loss(
	   cate_estimates=cate_estimates,
	   outcome_estimates=outcome_estimates,
	   propensity_scores=propensity_scores,
	   outcomes=y_validation,
	   treatments=w_validation,
       )

   study_overall = optuna.create_study(direction='minimize')
   study_overall.optimize(objective_overall, n_trials=100)

Leading to the following output

.. code-block:: console

   [I 2024-06-03 18:58:24,270] A new study created in memory with name: no-name-1bd16fa2-dee8-4d20-9505-1e272af0f8f9
   [I 2024-06-03 18:58:24,919] Trial 0 finished with value: 0.8147798597769813 and parameters: {'architecture': 'DR', 'nuisance_dr': 'Ridge', 'nuisance_dr_lin_reg_alpha': 2.2902267246714603}. Best is trial 0 with value: 0.8147798597769813.
   [I 2024-06-03 18:58:25,526] Trial 1 finished with value: 0.8133275607687053 and parameters: {'architecture': 'R', 'nuisance_r': 'Ridge', 'nuisance_r_lin_reg_alpha': 4.637862092794302}. Best is trial 1 with value: 0.8133275607687053.
   [I 2024-06-03 18:58:35,532] Trial 2 finished with value: 0.8147846754978962 and parameters: {'architecture': 'R', 'nuisance_r': 'LGBMRegressor', 'nuisance_r_lgbm_n_estimators': 225}. Best is trial 1 with value: 0.8133275607687053.
   [I 2024-06-03 18:58:53,424] Trial 3 finished with value: 0.8142647285796442 and parameters: {'architecture': 'DR', 'nuisance_dr': 'LGBMRegressor', 'nuisance_dr_lgbm_n_estimators': 217}. Best is trial 1 with value: 0.8133275607687053.
   [I 2024-06-03 18:58:54,052] Trial 4 finished with value: 0.8132955713927359 and parameters: {'architecture': 'R', 'nuisance_r': 'Ridge', 'nuisance_r_lin_reg_alpha': 9.719207974019277}. Best is trial 4 with value: 0.8132955713927359.
   [I 2024-06-03 18:58:54,696] Trial 5 finished with value: 0.8133872323810601 and parameters: {'architecture': 'DR', 'nuisance_dr': 'Ridge', 'nuisance_dr_lin_reg_alpha': 2.9620885067837213}. Best is trial 4 with value: 0.8132955713927359.
   [I 2024-06-03 18:58:55,317] Trial 6 finished with value: 0.8131459701081047 and parameters: {'architecture': 'R', 'nuisance_r': 'Ridge', 'nuisance_r_lin_reg_alpha': 7.8677812175017605}. Best is trial 6 with value: 0.8131459701081047.
   [I 2024-06-03 18:58:55,966] Trial 7 finished with value: 0.8143004222591679 and parameters: {'architecture': 'DR', 'nuisance_dr': 'Ridge', 'nuisance_dr_lin_reg_alpha': 8.706319710803129}. Best is trial 6 with value: 0.8131459701081047.
   [I 2024-06-03 18:58:56,596] Trial 8 finished with value: 0.814613404338131 and parameters: {'architecture': 'DR', 'nuisance_dr': 'Ridge', 'nuisance_dr_lin_reg_alpha': 0.33666864458317125}. Best is trial 6 with value: 0.8131459701081047.
   [I 2024-06-03 18:59:12,479] Trial 9 finished with value: 0.8141717774602024 and parameters: {'architecture': 'DR', 'nuisance_dr': 'LGBMRegressor', 'nuisance_dr_lgbm_n_estimators': 178}. Best is trial 6 with value: 0.8131459701081047.
   [I 2024-06-03 18:59:13,099] Trial 10 finished with value: 0.8139938540534846 and parameters: {'architecture': 'R', 'nuisance_r': 'Ridge', 'nuisance_r_lin_reg_alpha': 9.451770473441274}. Best is trial 6 with value: 0.8131459701081047.
   [I 2024-06-03 18:59:13,710] Trial 11 finished with value: 0.8131742628076631 and parameters: {'architecture': 'R', 'nuisance_r': 'Ridge', 'nuisance_r_lin_reg_alpha': 9.829822261109065}. Best is trial 6 with value: 0.8131459701081047.
   [I 2024-06-03 18:59:14,335] Trial 12 finished with value: 0.8132605930727189 and parameters: {'architecture': 'R', 'nuisance_r': 'Ridge', 'nuisance_r_lin_reg_alpha': 7.76020864716155}. Best is trial 6 with value: 0.8131459701081047.
   [I 2024-06-03 18:59:14,958] Trial 13 finished with value: 0.8130793186549206 and parameters: {'architecture': 'R', 'nuisance_r': 'Ridge', 'nuisance_r_lin_reg_alpha': 0.5322693901044904}. Best is trial 13 with value: 0.8130793186549206.

   ...
   [I 2024-06-03 19:01:16,545] Trial 97 finished with value: 0.8130319461209982 and parameters: {'architecture': 'R', 'nuisance_r': 'Ridge', 'nuisance_r_lin_reg_alpha': 2.591358482723367}. Best is trial 69 with value: 0.8124600668314831.
   [I 2024-06-03 19:01:20,357] Trial 98 finished with value: 0.8144984717668121 and parameters: {'architecture': 'R', 'nuisance_r': 'LGBMRegressor', 'nuisance_r_lgbm_n_estimators': 64}. Best is trial 69 with value: 0.8124600668314831.
   [I 2024-06-03 19:01:20,977] Trial 99 finished with value: 0.8125431242342713 and parameters: {'architecture': 'R', 'nuisance_r': 'Ridge', 'nuisance_r_lin_reg_alpha': 8.490782741739888}. Best is trial 69 with value: 0.8124600668314831.
