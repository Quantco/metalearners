FAQ
========

* **What's the difference between Double Machine Learning, Debiased Machine Learning,
  Orthogonal Machine Learning, Doubly-Robust Machine Learning and the R-Learner?**

    The Double Machine Learning blueprint relies on estimating two nuisance models in its
    first stage: a propensity model as well as an outcome model, both depending only on X.
    These models are then used to calculate the treatment and outcome residuals and finally
    the outcome residuals are regressed against the treatment residuals. See
    `Chernozhukov et al. (2016) <https://arxiv.org/abs/1608.00060>`_ for more details.
    Some implementations of Double Machine Learning are the
    `DoubleML Library <https://docs.doubleml.org/stable/index.html>`_ or the
    `DML module of EconML <https://econml.azurewebsites.net/_autosummary/econml.dml.DML.html>`_.

    Debiased Machine Learning and Orthogonal Machine Learning usually refer to the same
    algorithms as Double Machine Learning as their goal is to *debias* the used nuisance
    models with *orthogonalization*.

    The R-Learner is a generalization of the Double Machine Learning framework where instead
    of regressing the outcome residuals against the treatment residuals, usually with
    Linear Regression in DML, these are used
    to build a loss function which can be used with any Machine Learning which supports
    weighted loss functions. See `Nie and Wager (2017) <https://arxiv.org/abs/1712.04912>`_
    for more details. The main drawback of this method against Double Machine Learning is
    the fact that in its standard form it can only be used with binary or single-dimensional
    continuous treatments, some adaptations can be done to work with categorical treatments
    but model choice for the final model becomes highly restricted. On the other hand,
    Double Machine Learning can be used with categorical or continuous treatment.

    On the other hand, Doubly-Robust Machine Learning differs from all the previous methods
    by the fact that the outcome nuisance models depends not only on the variables X,
    but also on the treatments T. Then these models are used to build pseudo outcomes which
    their expected value is the true CATE and a final model is used to learn the CATE
    from them. See `Kennedy (2020) <https://arxiv.org/abs/2004.14497>`_ for more details.
