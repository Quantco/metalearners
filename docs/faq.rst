FAQ
===

* **What is the difference between Average treatment effect estimation
  and conditional average treatment effect estimation?**

    The Average Treatment Effect (ATE) is a scalar summary statistic that summarises the effect of an
    intervention on a population. The conditional average treatment effect (CATE)
    summarises the affect of the intervention an individual or subgroup, as defined
    by the conditioning covariate.

    When conditioning covariates are discrete then the CATE is finite dimensional,
    e.g. there is a a single CATE value for each population subgroup with the same covariate values.
    However, when the conditioning covariates are continuous, then the CATE is infinite dimensional,
    i.e. it is a function, similar to a regression function, that maps inputs to a predicted/ expected
    treatment effect for each individual.

    This distinction is important since the statistical methods used to estimate finite dimensional
    statistics may be different to those that we use to estimate regression functions.
    In particular, Double/ debiased machine learning was developed for statistical inference of finite
    quantities (unbiased estimation, standard error estimation, confidence interval estimation), whilst
    metalearning (e.g. R-, S-, T- DR- learning) is used for function learning.

    In the context of casual inference, however, the line between function learning
    and estimating summary statistics is somewhat blurred since summary estimators usually rely on
    some preliminary function learning.

* **What is Double/ Debiased Machine Learning?**
    Double machine learning is an ATE estimation technique, pioneered by
    `Chernozhukov et al. (2016) <https://arxiv.org/abs/1608.00060>`_.
    It is 'double' in the sense that it relies on two preliminary models: one for the probability of
    receiving treatment given covariates (the propensity score), and one for the outcome given covariates and
    optionally the (discrete) treatment.

    Double ML is also referred to as 'debiased' ML, since the propensity score model is used to 'debias'
    a naive estimator that uses the outcome model to predict the expected outcome under treatment, and under no treatment,
    for each individual.

    Double/ Debiased ML estimators of the ATE are also called 'double robust' since the error in the final ATE estimate
    may converges 'quickly' provided that the one of the propensity score or outcome learners is sufficiently accurate.
    This property is desirable since it means that one can trade-off accuracy in the propensity score and outcome learners,
    and there are some settings where the propensity score model is known, e.g. in a randomised experiment.

    Some implementations of Double Machine Learning include the
    `DoubleML Library <https://docs.doubleml.org/stable/index.html>`_ or the
    `DML module of EconML <https://econml.azurewebsites.net/_autosummary/econml.dml.DML.html>`_.

* **What is MetaLearning?**
    Following the definition by `Kunzel et al. <https://doi.org/10.1073/pnas.1804597116>`_, MetaLearning
    refers to any technique where a functional learning problem is decomposed into a sequence of
    simpler learning/ regression tasks.

    For instance, there are several ways to represent the CATE as the minimiser of a loss function, but these losses
    usually include unknown functions, such as the predicted mean outcome or the propensity score.
    A metalearner might therefore first learn the components of the unknown loss, then minimise the estimated loss in a second-stage.

    There maybe several theoretical or practical reasons to prefer one CATE decomposition over another, which
    is why so many CATE metalearners have been developed.

* **What MetaLearners are available for CATE estimation?**
    CATE metalearners usually follow the naming convention of `Kunzel et al. <https://doi.org/10.1073/pnas.1804597116>`_,
    who proposed the S-, T-, and X-learners, and, in a `longer version of their paper <https://arxiv.org/abs/1706.03461>`_,
    the U- and F-learners. Of these, the S- and T-learners are the simplest, which respectively fit a
    (S-) single outcome learner in the whole population and (T-) two outcome learners for each of the treated an untreated subpopulations.

    The R-Learner, due to `Nie and Wager (2017) <https://arxiv.org/abs/1712.04912>`_, uses a loss based on residuals from
    an outcome learner (fitted on covariates but not treatment), and residuals from a propensity score learner. The R-Learner
    loss is orthogonal, in the sense described by `Foster and Syrgkanis <https://arxiv.org/abs/1901.09036>`_.
    This orthogonality means that the error bound for the CATE may decay faster than the error bounds of the component models.

    The DR-learner, due to `Kennedy (2020) <https://arxiv.org/abs/2004.14497>`_,  uses a mean squared error loss, where the outcome
    is an estimated pseudo-outcome based an initial outcome learner (fitted on covariates and treatment), and a propensity score learner.
    The DR-learner is so called because it has similar double-robust properties to the Double/ debiased ML estimator,
    but in the function estimation setting.

    Some other implementations of Metalearning include `causalml <https://github.com/uber/causalml>`_.

.. _Cross-fit-faq:

* **Why do we cross-fit for all MetaLearners?**
    Cross-fitting is described as a central part of the learning process
    for some metaLearners. For instance, the outcome and propensity score models that make up the R-Learner loss for a single
    observation, are usually obtained a split of the data that does not include that observation.

    Some metalearners, such as the T-Learner, do not require cross-fitting, but are often cross-fitted when the resulting predictions
    are intended to be used in Double/ Debiased ML estimators (e.g. for the ATE). This is because, Debiased ML estimators usually
    require cross-fitting to control for biases related to overfitting of the component models,
    when component models are used for 'in-sample' prediction.

    We suggest to use cross-fitting when estimating the CATEs
    'in-sample' (on the training data) in order to avoid
    overfitting. When estimating CATEs 'out-of-sample' (on test or
    production data), we allow for the usage of the cross-fitted
    models via consensus algorithms: typically the mean or median.
    `Jacob (2020) <https://arxiv.org/pdf/2007.02852>`_ and
    `Chernozhukov (2018) <https://academic.oup.com/ectj/article/21/1/C1/5056401>`_
    discuss this approach in further detail.

    See :class:`metalearners.cross_fit_estimator.CrossFitEstimator`
    for our implementation of cross-fitting.

* **How do the MetaLearners work with classification outcomes?**
    If the outcome of an experiment is a class and not a scalar, e.g. conversion,
    we have two cases:

    * **Binary classification**: All MetaLearners can handle this scenario. In this situation
      :math:`\mathbb{E}[Y | X = x] = \mathbb{P}[Y = 1 | X = x]`. Thanks to the binary nature
      of the class, operations can be conducted combining the estimated probability for class 1 and observed outcomes
      (:math:`Y \in \{0,1\}`). This happens in the X, R, and DR-Learner methods. Although
      the result of these operations is treated as the difference in probabilities, this
      represents an approximation since the observed outcome probabilities are unavailable.
      Nonetheless, this approach provides the best approximation possible in such circumstances.
    * **Multiclass classification**: The S and T-Learners are the only MetaLearners capable
      of handling multiclass classification. In this case, the nuisance models of the S
      and T-Learners can predict the probability assigned to each outcome class.
      Followingly, the CATE can be estimated by computing the difference between the per-class probabilities of different variants.
