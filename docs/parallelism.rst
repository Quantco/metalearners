.. _parallelism:

What about parallelism?
************************

In the context of the topic outlined in :ref:`Motivation_multiprocessing`, one of the factors
motivating the implementation of this library is the introduction of parallelism in ``metalearners``.
We've discovered three potential levels for executing parallelism:

#.  **Base model level**: Certain :term:`base models<Base model>` implement the option to use multiple threads
    during their training. Examples of these models include
    `LightGBM <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor>`_
    or `RandomForest from sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_.
    These models can be instantiated with the ``n_jobs`` parameter to use multi-threading.

    To use parallelism at this level one can use the ``nuisance_model_params``,
    ``treatment_model_params`` and ``propensity_model_params`` parameters when instantiating
    the metalearner.

#.  **Cross-fitting level**: As explained in the :ref:`cross-fit FAQ <Cross-fit-faq>`
    cross-fitting is employed for all stand-alone nuisance and treatment models, irrespective
    of the MetaLearner. This introduces a new possible level of parallelism, as the model
    associated with each fold can be trained independently from the others.

    To use parallelism at this level one can use the ``n_jobs_cross_fitting`` parameter of the
    :py:meth:`~metalearners.metalearner.MetaLearner.fit` method of the metalearner.

#.  **Stage level**: A majority of MetaLearners entails multiple
    :term:`nuisance<Nuisance model>` and/or :term:`treatment models<Treatment effect model>`.
    Within an individual stage, these models are independent of each other, an example of
    this would be one :term:`propensity model<Propensity model>` and
    an :term:`outcome model<Outcome model>` for each treatment variant.
    This independence translates into another possibility for parallelism.

    To use parallelism at this level one can use the ``n_jobs_base_learners`` parameter of the
    :py:meth:`~metalearners.metalearner.MetaLearner.fit` method of the MetaLearner.

Our experiments leveraging parallelism at various levels reveal that there is not a
'one-size-fits-all' setting; the optimal configuration varies significantly based on factors
such as the choice of base models, the number of variants, the number of folds, or the
number of observations.

We suggest assessing several configurations to determine the most effective approach for
your specific use case. It is essential to remember that the most efficient configuration
for one scenario may not work as effectively for another.
